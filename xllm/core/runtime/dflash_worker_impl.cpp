/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "runtime/dflash_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/spec_verify.h"
#if defined(USE_NPU) || defined(USE_MLU)
#include "core/kv_cache/transfer/mooncake_kv_transfer.h"
#endif
#if defined(USE_NPU)
#include "core/kv_cache/transfer/kv_transfer_completion.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#endif
#include "core/common/types.h"
#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/model_type.h"
#include "core/framework/model/seq_lens.h"
#include "core/framework/model_loader/hf_model_config.h"
#include "core/framework/speculative/draft_extend_input.h"
#include "core/platform/platform.h"
#include "core/runtime/speculative_worker_impl.h"
#include "core/util/tensor_helper.h"
#include "runtime/llm_worker_impl.h"
#include "util/json_reader.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
namespace {
// Expands the accepted [batch, width] block into the compact flat index set
// plus the matching positions and cache slots, on the current stream, so the
// scatter stays ordered behind rejection sampling with no host wait.
struct AcceptedContextRowsDevice {
  torch::Tensor accepted_index;  // [num_valid] int64, flat embedding rows
  torch::Tensor positions;       // [num_valid] model positions
  torch::Tensor slots;           // [num_valid] absolute cache slots
  int64_t num_valid = 0;
};

// cache_position → physical slot via the flat block table, following
// append_decode_row semantics.
torch::Tensor gather_cache_slots(const torch::Tensor& block_tables,
                                 const torch::Tensor& flat_row_indices,
                                 const torch::Tensor& cache_positions,
                                 int64_t block_size) {
  const torch::Tensor block_index = cache_positions / block_size;
  const torch::Tensor block_base =
      block_tables.to(torch::kLong)
          .index_select(0,
                        flat_row_indices * block_tables.size(1) + block_index);
  return block_base * block_size + cache_positions % block_size;
}

AcceptedContextRowsDevice build_accepted_context_rows_device(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& positions_base,
    const torch::Tensor& kv_lens_base,
    const torch::Tensor& block_tables_device,
    int32_t block_size) {
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(accepted_tokens.scalar_type(), torch::kLong);
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t token_width = accepted_tokens.size(1);
  CHECK_EQ(positions_base.numel(), batch_size);
  CHECK_EQ(kv_lens_base.numel(), batch_size);
  CHECK(block_tables_device.dim() == 2);
  CHECK_EQ(block_tables_device.size(0), batch_size);

  const torch::TensorOptions int64_options =
      accepted_tokens.options().dtype(torch::kLong);
  const torch::Tensor valid = accepted_tokens.ge(0);
  // Accepted rows are prefix-valid: keep the compact set of (row, col)
  // pairs whose token is committed.
  const torch::Tensor flat_valid = valid.flatten();
  const torch::Tensor flat_index =
      torch::nonzero(flat_valid).reshape({-1}).to(torch::kLong);
  const torch::Tensor cols = flat_index % token_width;

  const torch::Tensor base_positions =
      positions_base.to(int64_options).reshape({batch_size});
  const torch::Tensor base_kv_lens =
      kv_lens_base.to(int64_options).reshape({batch_size});
  const torch::Tensor row_starts = flat_index / token_width;
  AcceptedContextRowsDevice rows;
  rows.num_valid = flat_index.numel();
  if (rows.num_valid == 0) {
    return rows;
  }
  rows.accepted_index = flat_index;
  rows.positions = base_positions.index_select(0, row_starts) + cols;
  // cache_position = kv_len + col - 1 (append_decode_row semantics).
  const torch::Tensor cache_positions =
      base_kv_lens.index_select(0, row_starts) + cols - 1;
  rows.slots = gather_cache_slots(
      block_tables_device, row_starts, cache_positions, block_size);
  return rows;
}

// Device-side next-round query rows for the SEQUENCE layout: one
// [batch, query_width] block per sequence (column 0 = accepted anchor, rest
// masks), derived from the producing step's row bases plus the device-side
// accepted length, so the query forward launches behind the rejection
// sampling with no host wait. Slot ids stay inside the scheduler's
// speculative reservation (committed + 2 * width in overlap mode), like the
// host builder.
struct NextQueryRowsDevice {
  torch::Tensor token_ids;    // [B * W] int64 (anchor + mask)
  torch::Tensor positions;    // [B * W]
  torch::Tensor slots;        // [B * W]
  torch::Tensor kv_seq_lens;  // [B]
  torch::Tensor anchors;      // [B] last accepted token per sequence
};

NextQueryRowsDevice build_next_query_rows_device(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& positions_base,
    const torch::Tensor& kv_lens_base,
    const torch::Tensor& block_tables_device,
    int32_t mask_token_id,
    int32_t block_size,
    int32_t query_width) {
  CHECK_EQ(accepted_tokens.dim(), 2);
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t token_width = accepted_tokens.size(1);
  CHECK_EQ(positions_base.numel(), batch_size);
  CHECK_EQ(kv_lens_base.numel(), batch_size);
  CHECK(block_tables_device.dim() == 2);
  CHECK_EQ(block_tables_device.size(0), batch_size);

  const torch::TensorOptions int64_options =
      accepted_tokens.options().dtype(torch::kLong);
  // Last valid (non-negative) token per row = the next round's anchor.
  const torch::Tensor last_col = last_accepted_columns(accepted_tokens);
  const torch::Tensor row_arange =
      torch::arange(batch_size, int64_options) * token_width;
  const torch::Tensor anchors =
      accepted_tokens.clamp_min(0).flatten().index_select(
          0, row_arange + last_col);

  // Per-(seq, col) grid: col 0 = anchor, col > 0 = mask.
  const torch::Tensor cols =
      torch::arange(query_width, int64_options).repeat({batch_size});
  const torch::Tensor row_ids =
      torch::arange(batch_size, int64_options).repeat_interleave(query_width);
  NextQueryRowsDevice rows;
  rows.token_ids = torch::where(cols.ge(1),
                                torch::full_like(cols, mask_token_id),
                                anchors.index_select(0, row_ids))
                       .to(torch::kInt32);

  const torch::Tensor base_positions =
      positions_base.to(int64_options).reshape({batch_size});
  const torch::Tensor accepted_lens =
      accepted_tokens.ge(0).sum(/*dim=*/1).to(torch::kLong);
  // The next round's row bases: position and kv advance by accepted_len.
  // Add per row first, then expand over the grid (one [B] add + one
  // gather instead of two [B*W] gathers).
  const torch::Tensor next_row_bases = base_positions + accepted_lens;
  const torch::Tensor next_positions =
      next_row_bases.index_select(0, row_ids) + cols;
  const torch::Tensor next_kv =
      kv_lens_base.to(int64_options).reshape({batch_size}) + accepted_lens;
  rows.positions = next_positions.to(torch::kInt32);
  rows.kv_seq_lens = next_kv.to(torch::kInt32);

  // cache_position = next_kv + col - 1 (append_decode_row semantics); the
  // anchor row (col 0) re-writes the last context slot, same as the host
  // builder.
  const torch::Tensor cache_positions =
      next_kv.index_select(0, row_ids) + cols - 1;
  rows.slots = gather_cache_slots(
                   block_tables_device, row_ids, cache_positions, block_size)
                   .to(torch::kInt32);
  rows.anchors = anchors;
  return rows;
}

runtime::Options draft_options(const runtime::Options& options) {
  // DSpark sizes its attention window from num_speculative_tokens; other
  // DFlash-style drafts still run one step at a time.
  const bool is_dspark =
      SpeculativeConfig::is_dspark_algorithm(options.speculative_algorithm());
  const bool is_dflash2 =
      SpeculativeConfig::is_dflash2_algorithm(options.speculative_algorithm());
  const int32_t draft_num_speculative_tokens =
      is_dspark ? options.num_speculative_tokens() : 0;
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(true)
      .num_decoding_tokens(is_dflash2 ? options.num_speculative_tokens() + 1
                                      : 1)
      .num_speculative_tokens(draft_num_speculative_tokens)
      .enable_graph_aux_hidden_states(false);
  return opts;
}

}  // namespace

DFlashWorkerImpl::DFlashWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : DraftModelSpecWorkerImpl<LlmForwardInput>(
          parallel_args,
          device,
          options,
          speculative_target_options(options),
          WorkerType::LLM,
          [&parallel_args, &device, &options] {
            // The draft consumes the target's aux hidden states, which the
            // worker does not expose under CP; reject cp_size > 1 except
            // DeepSeek-V4 on NPU.
            if (parallel_args.cp_size() > 1) {
              CHECK(Platform::is_npu() &&
                    is_deepseek_v4_family(get_model_type(options.model_path(),
                                                         options.backend())))
                  << "Block-diffusion speculative decoding does not support "
                     "context parallelism (cp_size > 1).";
            }
            return std::make_unique<LLMWorkerImpl>(
                parallel_args, device, draft_options(options));
          }) {}

DFlashWorkerImpl::~DFlashWorkerImpl() {
  // Keep launch-thread-owned context alive until queued execution completes.
  drain_pending_execution();
}

bool DFlashWorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  MasterStatus master_status) {
  // DFlash draft attends each block non-causally, which the shared QWen3 model
  // only wires up on the chunked-prefill mask path. Without it the draft falls
  // back to a causal mask and proposal quality silently degrades, so require
  // the flag rather than accept a misconfigured run.
  CHECK(::xllm::SchedulerConfig::get_instance().enable_chunked_prefill())
      << "Block-diffusion speculative decoding requires "
         "--enable_chunked_prefill=true.";
  CHECK(SpeculativeConfig::is_probabilistic_draft_sampling_supported(
            options_.speculative_algorithm()) ||
        !is_probabilistic_draft_)
      << "This drafter only supports greedy drafts.";
  // Draft config's use_sliding_window / sliding_window is intentionally
  // ignored: xLLM NPU FIA hard-codes pre_tokens=INT_MAX, next_tokens=0
  // and never plumbs sliding_window into the aclnn call. Attended kv_len
  // on the draft path is bounded by the target sequence length, not by
  // block_size, so enforcing a block_size-vs-window relationship here
  // would compare the wrong quantities.
  const bool result = DraftModelSpecWorkerImpl<LlmForwardInput>::init_model(
      model_weights_path, random_seed, master_status);

  if (impl_->get_status() == WorkerImpl::Status::LOADED) {
    target_is_hybrid_recurrent_ =
        has_linear_attention_layers(impl_->context_.get_model_args());
  }

  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
    JsonReader reader;
    const std::string config_path = model_weights_path + "/config.json";
    CHECK(reader.parse(config_path))
        << "Failed to parse block-diffusion draft config: " << config_path;
    mask_token_id_ = reader.value_or<int32_t>({"dflash_config.mask_token_id",
                                               "mask_token_id",
                                               "dspark_noise_token_id"},
                                              /*default=*/-1);
    CHECK_GE(mask_token_id_, 0)
        << "Block-diffusion draft config requires mask_token_id, "
           "dflash_config.mask_token_id, or dspark_noise_token_id.";

    // DSpark checkpoints carry trained mtp.0.embed / mtp.<last>.head tensors
    // (with the draft-side QuaRot transform); sharing the target modules here
    // makes the draft backbone/Markov head project through the wrong
    // vocabulary basis and reduces acceptance to near-random levels.
    const bool uses_own_head_and_embedding =
        draft_args.model_type() == "deepseek_v4_dspark";
    if (!uses_own_head_and_embedding) {
      draft_impl_->get_causal_lm()->share_lm_head(*impl_->get_causal_lm());
      draft_impl_->get_causal_lm()->share_embed_tokens(*impl_->get_causal_lm());
    }
    const ModelArgs& target_args = impl_->context_.get_model_args();
    CHECK_LT(mask_token_id_, target_args.vocab_size())
        << "Block-diffusion mask_token_id (" << mask_token_id_
        << ") must be < target vocab_size (" << target_args.vocab_size()
        << ").";
    if (SpeculativeConfig::is_dflash2_algorithm(
            options_.speculative_algorithm())) {
      const int32_t requested_block_size =
          options_.num_speculative_tokens() + 1;
      CHECK_EQ(requested_block_size, draft_args.block_size())
          << "DFlash2 runtime block size must match the checkpoint's trained "
             "dflash_config.block_size.";
      CHECK(!target_is_hybrid_recurrent_ ||
            ::xllm::ExecutionConfig::get_instance().enable_graph())
          << "DFlash2 with a hybrid recurrent target requires ACL Graph: the "
             "expanded spec-verify replay path preserves the accepted GDN "
             "checkpoint, while eager validation is not lossless.";
    }
    const int64_t num_target_layers =
        static_cast<int64_t>(target_args.layers_to_capture().size());
    CHECK_GT(num_target_layers, 0)
        << "Block-diffusion draft config requires dspark_target_layer_ids, "
           "target_layer_ids, dflash_config.target_layer_ids, or "
           "aux_hidden_state_layer_ids.";
    expected_context_hidden_size_ =
        static_cast<int64_t>(target_args.hidden_size()) * num_target_layers;
  }
  return result;
}

std::tuple<int64_t, int64_t> DFlashWorkerImpl::estimate_kv_cache_capacity() {
  CHECK(impl_ != nullptr);
  return estimate_kv_cache_capacity_with_draft(
      speculative_target_options(options_), draft_options(options_));
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_empty(
    const LlmForwardInput& input) {
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    LlmForwardInput target_forward_input;
    std::optional<ForwardOutput> output =
        run_worker_no_sync_impl(*impl_,
                                input,
                                target_forward_input,
                                *prepare_stream_,
                                *compute_stream_);
    // Active prefill ranks write the draft context KV without a draft forward.
    // Keep idle ranks symmetric: a draft MoE forward here would enter EP
    // collectives that active ranks never join and deadlock the whole group.
    // Sync the target forward before its staged input can be reused.
    compute_stream_->synchronize();
    if (output.has_value()) {
      clear_all_output_embeddings(output.value());
    }
    return output;
  }

  const DecodeExpansionOptions decode_options = prepare_for_draft();
  LlmForwardInput query_input = input.clone();
  dflash_detail::invalidate_draft_model_geometry(query_input.input_params);
  const int32_t num_sequences = query_input.input_params.meta.num_sequences;
  if (!prepare_parallel_draft_metadata(query_input, decode_options)) {
    const int32_t rows_per_sequence =
        decode_options.per_request
            ? 1
            : (decode_options.num_speculative_tokens + 1);
    update_decode_batch_metadata(query_input.input_params,
                                 decode_options,
                                 num_sequences * rows_per_sequence);
  }
  LlmForwardInput draft_forward_input;
  run_worker_no_sync_impl(*draft_impl_,
                          query_input,
                          draft_forward_input,
                          *prepare_stream_,
                          *compute_stream_);

  LlmForwardInput validate_input = input.clone();
  // DSpark's N-wide draft geometry must be rescaled to (N+1) for the target's
  // anchor + drafts forward.
  scale_speculative_parallel_token_counts(validate_input.input_params,
                                          options_.num_speculative_tokens());
  LlmForwardInput target_forward_input;
  ForwardOutput output = run_worker_no_sync_impl(*impl_,
                                                 validate_input,
                                                 target_forward_input,
                                                 *prepare_stream_,
                                                 *compute_stream_)
                             .value();
  compute_stream_->synchronize();
  clear_all_output_embeddings(output);
  return output;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_prefill(
    const LlmForwardInput& input) {
  LlmForwardInput target_forward_input;
  ForwardOutput output;
  {
    AUTO_COUNTER(speculative_execution_latency_seconds_target);
    output = run_worker_no_sync_impl(*impl_,
                                     input,
                                     target_forward_input,
                                     *prepare_stream_,
                                     *compute_stream_,
                                     ForwardCompletion::CALLER_MANAGED)
                 .value();
  }

  const torch::Tensor& embeddings = output.sample_output.embeddings;
  if (embeddings.defined()) {
    const Slice<int32_t> positions =
        tensor_slice(target_forward_input.positions_host);
    CHECK_EQ(positions.size(), static_cast<size_t>(embeddings.size(0)))
        << "DFlash prefill hidden/position count mismatch.";
    torch::Tensor context_cache_slots =
        target_forward_input.input_params.attention.device.new_cache_slots;
    if (!target_forward_input.input_params.multi_block_tables.empty()) {
      const DecodeBatchContext ctx =
          make_decode_batch_context(target_forward_input);
      std::vector<int32_t> grouped_swa_slots =
          build_grouped_prefill_swa_slots(ctx, options_.block_size());
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      context_cache_slots =
          async_h2d_tensor(grouped_swa_slots, device_.unwrap());
    }
    CHECK(context_cache_slots.defined())
        << "DFlash prefill requires context cache slots.";
    CHECK_EQ(static_cast<size_t>(context_cache_slots.numel()), positions.size())
        << "DFlash prefill hidden/cache slot count mismatch.";

    AUTO_COUNTER(speculative_execution_latency_seconds_draft);
    append_failed_request_ids(output,
                              write_context_kv(target_forward_input,
                                               embeddings,
                                               target_forward_input.positions,
                                               context_cache_slots));
  }

  if (input.sampling_params.num_sample_seqs() > 0) {
    prepare_first_draft_inputs(output);
    target_state_store_->write_prefill_target_context(
        input.input_params.embedding.batch_rows(),
        output.sample_output.next_tokens);
  } else {
    clear_all_output_embeddings(output);
  }

  if (!publishes_step_output()) {
    return make_transfer_failure_output(std::move(output.failed_request_ids));
  }
  return output;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_decode(
    const LlmForwardInput& raw_input) {
  LlmForwardInput input = raw_input.clone();
  LlmForwardInput validate_input;

  const std::vector<TargetDecodeStateStore::DecodeRow> last_states =
      target_state_store_->read_decode_states(
          input.input_params.embedding.batch_rows());
  // Positions track position+1; a fake token with no cached state keeps the
  // raw input id instead of falling back to the recorded state token.
  const DecodeRowPolicy policy;
  rewrite_decode_rows_from_target_state(input, last_states, policy);
  LlmForwardInput draft_forward_input;
  auto [draft_token_ids, draft_logits] =
      run_decode_draft(input, validate_input, draft_forward_input);
  return run_validate(input, draft_token_ids, draft_logits, validate_input);
}

std::optional<ForwardOutput> DFlashWorkerImpl::launch_query_forward(
    const LlmForwardInput& input,
    torch::Tensor& anchors) {
  PendingQueryRound pending;
  if (take_pending_query(input, pending)) {
    anchors = std::move(pending.anchors);
    return std::move(pending.output);
  }
  anchors = anchor_token_ids(input);
  LlmForwardInput query_input;
  prepare_query_inputs(input, query_input);
  query_input.skip_sampling_for_logits_only = query_forward_logits_only();
  query_input.return_selected_hidden = query_forward_returns_selected_hidden();
  return detail::run_worker_no_sync(
      *draft_impl_, query_input, *prepare_stream_, *compute_stream_);
}

std::pair<torch::Tensor, torch::Tensor> DFlashWorkerImpl::run_decode_draft(
    const LlmForwardInput& input,
    LlmForwardInput& validate_input,
    LlmForwardInput& draft_forward_input) {
  Timer draft_timer;

  torch::Tensor anchors;
  ForwardOutput draft_output = launch_query_forward(input, anchors).value();
  prepare_validate_inputs(input, validate_input);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());
  return {draft_output.sample_output.next_tokens.view(
              {-1, options_.num_speculative_tokens()}),
          torch::Tensor()};
}

std::optional<ForwardOutput> DFlashWorkerImpl::run_validate(
    const LlmForwardInput& input,
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_logits,
    LlmForwardInput& validate_input) {
  Timer target_timer;
  {
    const int32_t num_speculative_tokens = options_.num_speculative_tokens();
    const int32_t num_val_tokens = num_speculative_tokens + 1;

    const int64_t total_num_val_tokens = validate_input.token_ids.numel();
    const int64_t num_sequences = total_num_val_tokens / num_val_tokens;
    const torch::TensorOptions token_options =
        validate_input.token_ids.options();
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    CHECK(compute_stream_->wait_event(
        validate_input.runtime.metadata_ready_event))
        << "failed to wait DFlash metadata ready event";
    torch::Tensor validate_token_rows =
        validate_input.token_ids.view({num_sequences, num_val_tokens});

    validate_input.runtime.device_tensors_ready = false;
    torch::Tensor draft_tokens = safe_to(
        draft_token_ids.contiguous(), token_options, /*non_blocking=*/true);
    validate_token_rows.slice(/*dim=*/1, /*start=*/1, /*end=*/num_val_tokens)
        .copy_(draft_tokens, /*non_blocking=*/true);
    validate_input.runtime.device_tensors_ready = true;
    // Graph staging on the prepare stream waits for this token update.
    validate_input.runtime.metadata_ready_event =
        compute_stream_->record_event_or_sync();
  }

  LlmForwardInput target_forward_input;
  ForwardOutput target_output = run_worker_no_sync_impl(*impl_,
                                                        validate_input,
                                                        target_forward_input,
                                                        *prepare_stream_,
                                                        *compute_stream_)
                                    .value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              target_timer.elapsed_seconds());

  Timer validation_timer;
  SampleOutput val_output =
      run_rejection_sampling(input.sampling_params,
                             draft_token_ids,
                             draft_logits,
                             target_output.logits,
                             target_output.sample_output.embeddings);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              validation_timer.elapsed_seconds());

  // The context-KV scatter stays ordered behind the rejection sampling, so
  // the step no longer waits on the full compute stream; the store write
  // rides the pinned handoff. One row-bases upload serves both the scatter
  // and the next-query launch.
  broadcast_committed_tokens(val_output.next_tokens);
  const RowBases row_bases = upload_row_bases(input);
  const StreamEventPtr row_bases_ready =
      prepare_stream_->record_event_or_sync();
  append_failed_request_ids(
      target_output,
      scatter_context_kv(input, val_output, row_bases, row_bases_ready));
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    torch::Tensor accepted_tokens_host =
        target_context_handoff_.acquire_host_buffer(val_output.next_tokens);
    accepted_tokens_host.copy_(val_output.next_tokens,
                               /*non_blocking=*/true);
    // The event covers both the token consensus and the D2H copy.
    StreamEventPtr accepted_ready = compute_stream_->record_event();
    if (accepted_ready == nullptr) {
      CHECK_EQ(compute_stream_->synchronize(), 0)
          << "failed to synchronize DFlash target context";
    }
    target_context_handoff_.stage(input.input_params.embedding.batch_rows(),
                                  accepted_ready,
                                  std::move(accepted_tokens_host));
  }
  launch_next_query(input, val_output, row_bases, row_bases_ready);
  if (!publishes_step_output()) {
    return make_transfer_failure_output(
        std::move(target_output.failed_request_ids));
  }
  return finalize_verify_output(std::move(target_output),
                                std::move(val_output),
                                input.json_object_states);
}

bool DFlashWorkerImpl::take_pending_query(const LlmForwardInput& input,
                                          PendingQueryRound& out) {
  if (!pending_query_.has_value()) {
    return false;
  }
  if (input.input_params.embedding.embedding_ids !=
          pending_query_->rows_.embedding_ids ||
      input.input_params.embedding.request_ids !=
          pending_query_->rows_.request_ids) {
    // Batch transition: the launched query wrote KV slots that stay inside
    // the producing batch's reservation, so dropping the round is safe.
    pending_query_.reset();
    return false;
  }
  out = std::move(*pending_query_);
  pending_query_.reset();
  return true;
}

void DFlashWorkerImpl::launch_next_query(const LlmForwardInput& input,
                                         const SampleOutput& val_output,
                                         const RowBases& bases,
                                         const StreamEventPtr& bases_ready) {
  // Prelaunch only benefits the sequential one-row-per-token layout: block
  // layouts rebuild their geometry from the accepted block each round.
  if (draft_input_per_request_ || sample_from_anchor_ ||
      input.input_params.meta.is_graph_warmup || target_is_hybrid_recurrent_ ||
      !enable_schedule_overlap()) {
    return;
  }
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  const int32_t query_width = options_.num_speculative_tokens() + 1;

  // Row math runs on the compute stream ordered behind the rejection
  // sampling; the [B]/[B*W] host mirrors copy back behind the event. The host
  // wait lands on the same visibility point the verify tail already needs, so
  // the query forward still leaves the next step's critical path.
  torch::Tensor token_ids;
  torch::Tensor positions;
  torch::Tensor slots_host;
  torch::Tensor kv_seq_lens_host;
  torch::Tensor anchors;
  StreamEventPtr rows_ready;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    if (bases_ready != nullptr) {
      CHECK(compute_stream_->wait_event(bases_ready))
          << "failed to wait DFlash query row bases ready event";
    }
    const NextQueryRowsDevice rows =
        build_next_query_rows_device(val_output.next_tokens,
                                     bases.positions,
                                     bases.kv_lens,
                                     bases.block_tables,
                                     mask_token_id_,
                                     options_.block_size(),
                                     query_width);
    token_ids = rows.token_ids;
    positions = rows.positions;
    anchors = rows.anchors;
    slots_host = torch::empty(rows.slots.sizes(),
                              pinned_cpu_options(rows.slots.scalar_type()));
    slots_host.copy_(rows.slots, /*non_blocking=*/true);
    kv_seq_lens_host =
        torch::empty(rows.kv_seq_lens.sizes(),
                     pinned_cpu_options(rows.kv_seq_lens.scalar_type()));
    kv_seq_lens_host.copy_(rows.kv_seq_lens, /*non_blocking=*/true);
    rows_ready = compute_stream_->record_event();
  }
  if (rows_ready == nullptr) {
    CHECK_EQ(compute_stream_->synchronize(), 0)
        << "failed to synchronize DFlash next-query rows";
  } else {
    CHECK(rows_ready->synchronize()) << "failed to wait DFlash next-query rows";
  }

  LlmForwardInput query_input = input.clone_for_draft_step();
  query_input.token_ids = token_ids;
  query_input.positions = positions;
  query_input.sampling_params = input.sampling_params;
  prepare_draft_sampling(/*probabilistic=*/false, query_input.sampling_params);
  query_input.sampling_params.logits_indices() = anchors;
  LlmModelParams& params = query_input.input_params;
  params.embedding.input_embedding = torch::Tensor();
  dflash_detail::invalidate_draft_model_geometry(params);
  params.attention.host.new_cache_slots = tensor_to_vector<int32_t>(slots_host);
  params.attention.host.kv_seq_lens =
      tensor_to_vector<int32_t>(kv_seq_lens_host);
  params.attention.host.q_seq_lens =
      std::vector<int32_t>(num_sequences, query_width);
  params.attention.host.q_cu_seq_lens =
      util::cu_num_tokens(params.attention.host.q_seq_lens);
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.meta.num_sequences = num_sequences * query_width;
  params.meta.q_max_seq_len = query_width;
  params.attention.rebuild_device_buffer(device_.unwrap());
  query_input.skip_sampling_for_logits_only = query_forward_logits_only();
  query_input.return_selected_hidden = query_forward_returns_selected_hidden();

  PendingQueryRound pending;
  pending.rows_.embedding_ids = input.input_params.embedding.embedding_ids;
  pending.rows_.request_ids = input.input_params.embedding.request_ids;
  pending.anchors = anchors;
  pending.output = detail::run_worker_no_sync(*draft_impl_,
                                              query_input,
                                              *prepare_stream_,
                                              *compute_stream_,
                                              &pending.prepared_input)
                       .value();
  pending_query_ = std::move(pending);
}

torch::Tensor DFlashWorkerImpl::anchor_token_ids(
    const LlmForwardInput& input) const {
  const int32_t batch_size = input.input_params.meta.num_sequences;
  CHECK(input.token_ids_host.defined())
      << "block drafts require host token ids for the anchor.";
  CHECK_GE(input.token_ids_host.numel(), batch_size)
      << "anchor token count must cover the decode batch.";
  return input.token_ids_host.slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size)
      .to(draft_impl_->device(), torch::kLong);
}
void DFlashWorkerImpl::broadcast_committed_tokens(torch::Tensor& tokens) {
  if (get_optimization_config().enable_spec_token_broadcast) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    broadcast_spec_tokens(tokens);
  }
}

void DFlashWorkerImpl::prepare_validate_inputs(
    const LlmForwardInput& input,
    LlmForwardInput& validate_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  LlmForwardInput prepared_input = input.clone();
  prepared_input.runtime.metadata_ready_event.reset();
  const bool use_linear_spec_verify = target_is_hybrid_recurrent_;
  prepared_input.input_params.is_spec_verify = use_linear_spec_verify;
  SpeculativeWorkerImpl<LlmForwardInput>::prepare_validate_inputs(
      prepared_input, validate_input);
  if (use_linear_spec_verify) {
    LlmModelParams& input_params = validate_input.input_params;
    apply_chunked_prefill_spec_verify_tail(
        input,
        input_params,
        input.input_params.meta.num_sequences,
        validate_input.token_ids.options().dtype(torch::kInt32),
        /*publish_device_tensor=*/true);
    // The generic builder materializes the device buffer before the hybrid
    // cumulative lengths above are canonicalized.  Rebind it so GDN sees
    // [0, q_len_0, ...] rather than the stale pre-verify host layout.
    input_params.attention.rebuild_device_buffer(device_);
#if defined(USE_NPU)
    if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
      build_expanded_spec_verify_graph_input(
          input_params, device_.unwrap(), options_.block_size());
    }
#endif
  }
  validate_input.runtime.metadata_ready_event =
      prepare_stream_->record_event_or_sync();
}

DecodeExpansionOptions DFlashWorkerImpl::prepare_for_draft() const {
  DecodeExpansionOptions decode_options;
  decode_options.num_speculative_tokens =
      options_.num_speculative_tokens() -
      static_cast<int32_t>(sample_from_anchor_);
  decode_options.block_size = options_.block_size();
  decode_options.per_request = draft_input_per_request_;
  decode_options.causal = false;
  decode_options.forward_type = draft_forward_type_;
  decode_options.sample_from_anchor = sample_from_anchor_;
  decode_options.mask_token_id = mask_token_id_;
  return decode_options;
}

bool DFlashWorkerImpl::prepare_parallel_draft_metadata(
    LlmForwardInput& query_input,
    const DecodeExpansionOptions& decode_options) {
  scale_speculative_parallel_token_counts(
      query_input.input_params, decode_options.num_speculative_tokens);
  if (!draft_needs_dummy_tokens_) {
    return false;
  }
  auto& params = query_input.input_params;
  std::replace(params.parallel.dp_global_token_nums.begin(),
               params.parallel.dp_global_token_nums.end(),
               0,
               (decode_options.num_speculative_tokens + 1));
  if (params.meta.num_sequences != 0) {
    return false;
  }

  build_idle_decode_input(query_input, decode_options);
  params.meta.actual_num_sequences = 0;
  query_input.runtime.input_host_buffer_has_layout = false;
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  query_input.input_params.attention.rebuild_device_buffer(device_.unwrap());
  return true;
}

void DFlashWorkerImpl::prepare_query_inputs(const LlmForwardInput& input,
                                            LlmForwardInput& query_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  query_input = input.clone();
  query_input.runtime.device_tensors_ready = false;
  LlmModelParams& input_params = query_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  dflash_detail::invalidate_draft_model_geometry(input_params);

  const DecodeExpansionOptions decode_options = prepare_for_draft();
  const int32_t query_width = (decode_options.num_speculative_tokens + 1);
  const bool idle_rank =
      prepare_parallel_draft_metadata(query_input, decode_options);
  if (idle_rank) {
    query_input.sampling_params.logits_indices() = torch::Tensor();
  } else {
    DecodeBatchContext batch_ctx = make_decode_batch_context(input);
    DecodeBuildBuffers buf = expand_decode_inputs(batch_ctx, decode_options);
    const int32_t num_logits_per_request =
        decode_options.sample_from_anchor ? query_width : query_width - 1;
    const int32_t num_logits = batch_ctx.num_requests * num_logits_per_request;
    const torch::Tensor& draft_query_indices =
        draft_logits_indices_.get(num_logits, [&]() {
          return decode_options.sample_from_anchor
                     ? arange_indices(num_logits, device_)
                     : torch::arange(
                           static_cast<int64_t>(batch_ctx.num_requests) *
                               query_width,
                           torch::TensorOptions()
                               .dtype(torch::kInt)
                               .device(device_))
                           .view({batch_ctx.num_requests, query_width})
                           .narrow(/*dim=*/1, 1, query_width - 1)
                           .reshape({-1});
        });
    build_decode_input(query_input, buf, decode_options, device_);
    query_input.sampling_params.logits_indices() = draft_query_indices;
  }
  prepare_draft_sampling(/*probabilistic=*/false, query_input.sampling_params);
  for (const auto& field : kSamplingFields) {
    if (!field.per_request) {
      repeat_interleave_if_defined(field.member(query_input.sampling_params),
                                   options_.num_speculative_tokens());
    }
  }
  query_input.runtime.device_tensors_ready = true;
}

std::vector<std::string> DFlashWorkerImpl::write_context_kv(
    const LlmForwardInput& input,
    const torch::Tensor& context_hidden,
    const torch::Tensor& positions_device,
    const torch::Tensor& new_cache_slots_device) {
  CHECK(context_hidden.defined()) << "DFlash context hidden is undefined.";
  CHECK_EQ(context_hidden.dim(), 2) << "DFlash context hidden must be 2D.";
  CHECK_EQ(context_hidden.size(1), expected_context_hidden_size_)
      << "DFlash context hidden size must be hidden_size * "
      << "target_layer_ids.size().";

  CHECK(context_hidden.device() == device_.unwrap())
      << "DFlash context hidden must already be on the compute device.";

  // Both the target forward that produced context_hidden and this pass run
  // on compute_stream_, so no explicit event dance is needed — the stream
  // orders them. Model methods below use torch ops on the same stream.
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  CHECK(input.input_params.synchronize_draft_layer())
      << "Failed to wait for Draft Host KV load completion.";

#if defined(USE_NPU)
  // PD PUSH: the draft context-KV scattered below is not covered by the target
  // push, so wire the draft push here. Overwrite layer_synchronizer with a
  // draft-sized one (one event per draft layer); the target's was already
  // waited on before this runs, so the overwrite is safe. No-op when
  // transfer_kv_infos is empty. NPU-only: the scatter's record_event loop is
  // NPU-gated, so a non-NPU push would stall on events that never record.
  KVTransferCompletion kv_transfers = [&]() {
    if (options_.kv_push_enabled() && !input.transfer_kv_infos.empty()) {
      std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
          std::make_shared<NPULayerSynchronizerImpl>(
              draft_impl_->context_.get_model_args().n_layers());
      input.input_params.parallel.layer_synchronizer = layer_synchronizer;
      auto request_ids =
          KVTransfer::unique_request_ids(input.transfer_kv_infos);
      return KVTransferCompletion(kv_transfer_->push_kv_blocks_async(
                                      input.transfer_kv_infos,
                                      draft_impl_->context_.get_parallel_args(),
                                      layer_synchronizer,
                                      /*is_spec_draft=*/true),
                                  std::move(request_ids));
    }
    return KVTransferCompletion{};
  }();
#endif

  ModelOutput scatter_output =
      draft_impl_->write_context_kv(context_hidden,
                                    positions_device,
                                    new_cache_slots_device,
                                    input.input_params);
  // Model returns an empty ModelOutput (hidden_states undefined) when the
  // per-layer NPULayerSynchronizer::record_event fails mid-scatter under
  // PD-PUSH transfer. Fail fast instead of silently continuing: subsequent
  // layers won't have been scattered and the PD transfer side would block
  // forever waiting on the missing per-layer event.
  CHECK(scatter_output.hidden_states.defined())
      << "DFlash context-KV scatter failed (layer_synchronizer record_event "
         "returned false); PD-PUSH transfer would deadlock.";

  // The scatter's consumers are FIFO-ordered behind it on compute_stream_, so
  // no host wait is needed here; the PD-PUSH wait below owns its own
  // completion.
#if defined(USE_NPU)
  // Wait for the draft KV push (if any) so the source draft cache is not
  // overwritten by the next step while the transfer is still reading it
  // (mirrors step_internal's wait_kv_push()). No-op when no push was issued.
  return kv_transfers.wait(
      draft_impl_->context_.get_parallel_args().kv_transfer_groups(), device_);
#else
  return {};
#endif
}

// Uploads the per-row bases (positions, kv lengths, block tables) the device
// row builders consume, on prepare_stream_ in parallel with the verify tail.
DFlashWorkerImpl::RowBases DFlashWorkerImpl::upload_row_bases(
    const LlmForwardInput& input) {
  RowBases bases;
  c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
  bases.positions = input.positions_host
                        .slice(/*dim=*/0,
                               /*start=*/0,
                               /*end=*/input.input_params.meta.num_sequences)
                        .to(torch::kInt32)
                        .to(device_.unwrap(), /*non_blocking=*/true);
  bases.kv_lens = async_h2d_tensor(
      input.input_params.attention.host.kv_seq_lens, device_.unwrap());
  bases.block_tables =
      input.input_params.attention.host.block_tables.to(torch::kInt32)
          .to(device_.unwrap(), /*non_blocking=*/true);
  return bases;
}

std::vector<std::string> DFlashWorkerImpl::scatter_context_kv(
    const LlmForwardInput& input,
    const SampleOutput& validate_output,
    const RowBases& bases,
    const StreamEventPtr& bases_ready) {
  const torch::Tensor& accepted_embeddings = validate_output.embeddings;
  CHECK(accepted_embeddings.defined())
      << "DFlash validate target embeddings are undefined.";
  CHECK_EQ(accepted_embeddings.dim(), 3)
      << "DFlash validate target embeddings must be [batch,width,hidden].";

  CHECK(validate_output.next_tokens.defined())
      << "DFlash accepted tokens are undefined.";
  const torch::Tensor& accepted_tokens = validate_output.next_tokens;
  CHECK_EQ(accepted_tokens.dim(), 2)
      << "DFlash accepted tokens must be [batch,width].";
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t token_width = accepted_tokens.size(1);
  CHECK_EQ(accepted_embeddings.size(0), batch_size)
      << "DFlash accepted token/embedding batch mismatch.";
  CHECK_EQ(accepted_embeddings.size(1), token_width)
      << "DFlash accepted token/embedding width mismatch.";

  // Row geometry is derived on device from the accepted block, so the whole
  // scatter stays ordered behind the rejection sampling without any host
  // wait; the row-base H2D uploads were issued by the caller on
  // prepare_stream_.
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  if (bases_ready != nullptr) {
    CHECK(compute_stream_->wait_event(bases_ready))
        << "failed to wait DFlash context row bases ready event";
  }
  const AcceptedContextRowsDevice rows =
      build_accepted_context_rows_device(accepted_tokens,
                                         bases.positions,
                                         bases.kv_lens,
                                         bases.block_tables,
                                         options_.block_size());
  CHECK_GT(rows.num_valid, 0) << "DFlash accepted context must not be empty.";
  const torch::Tensor flat_embeddings = accepted_embeddings.reshape(
      {batch_size * token_width, accepted_embeddings.size(/*dim=*/2)});
  const torch::Tensor context_hidden =
      flat_embeddings.index_select(/*dim=*/0, rows.accepted_index);
  return write_context_kv(input,
                          context_hidden,
                          rows.positions.to(torch::kInt32),
                          rows.slots.to(torch::kInt32));
}

}  // namespace xllm
