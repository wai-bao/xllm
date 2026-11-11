/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/runtime/mtp_worker_impl.h"

#include <glog/logging.h>

#include "runtime/vlm_forward_params.h"

#if defined(USE_NPU)
#include <acl/acl.h>
#include <c10/core/DeviceType.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>

#include "common/metrics.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#endif
#include "core/common/types.h"
#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_extend_input.h"
#include "core/framework/speculative/draft_input_sampling.h"
#include "core/framework/speculative/draft_proposal.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/layers/common/dsa_topk_share_plan.h"
#include "core/platform/platform.h"
#include "core/runtime/spec_verify_input_layout.h"
#include "core/util/tensor_helper.h"
#include "models/model_registry.h"
#include "runtime/llm_worker_impl.h"
#include "util/slice.h"
#include "util/timer.h"

namespace xllm {
namespace {
constexpr int64_t kMaxSpecVerifyGraphUpdateBlockTableWidth = (1 << 15) - 1;

void record_output_ready_event(ForwardOutput& output, Stream& stream) {
  output.ready_event = stream.record_event_or_sync();
}

void finalize_output_on_stream(ForwardOutput& output,
                               Stream& stream,
                               bool allow_async) {
  if (allow_async) {
    record_output_ready_event(output, stream);
    return;
  }
  const int32_t ret = stream.synchronize();
  CHECK_EQ(ret, 0) << "failed to synchronize MTP compute stream, ret=" << ret;
  output.retained_inputs.clear();
}

template <typename TargetInput>
std::vector<size_t> rows_for_failed_sequence_ids(
    const TargetInput& input,
    const std::unordered_set<std::string>& failed_ids) {
  std::vector<size_t> rows;
  rows.reserve(failed_ids.size());
  for (size_t row = 0; row < input.sample_sequence_ids.size(); ++row) {
    if (failed_ids.contains(input.sample_sequence_ids[row])) {
      rows.emplace_back(row);
    }
  }
  return rows;
}

// Appends settle-produced errors without clobbering errors the verify pass
// already attached to the same output.
inline void append_json_errors(std::vector<JsonObjectOutputError>& dst,
                               std::vector<JsonObjectOutputError> src) {
  dst.insert(dst.end(),
             std::make_move_iterator(src.begin()),
             std::make_move_iterator(src.end()));
}

template <typename TargetInput>
std::vector<size_t> failed_draft_input_rows(
    const TargetInput& input,
    const std::vector<JsonObjectOutputError>& output_errors) {
  if (input.json_object_errors.empty() && output_errors.empty()) {
    return {};
  }
  return rows_for_failed_sequence_ids(
      input,
      detail::failed_sequence_ids_from_errors(input.json_object_errors,
                                              output_errors,
                                              /*require_non_empty_id=*/false));
}

template <typename TargetInput>
void stabilize_decode_host_tensors(TargetInput& input) {
  stabilize_host_views(input);
  for (torch::Tensor& block_table : input.input_params.multi_block_tables) {
    block_table = clone_host_block_tables(block_table);
  }
}

template <typename TargetInput>
void set_token_ids_device_tensor(TargetInput& input,
                                 const torch::Tensor& token_ids,
                                 const torch::TensorOptions& token_options,
                                 Stream& compute_stream) {
  CHECK(token_ids.defined()) << "draft token_ids must be defined";
  torch::Tensor flat_token_ids = token_ids.flatten();
  CHECK_EQ(flat_token_ids.numel(), input.input_params.meta.num_sequences)
      << "draft token count must match num_sequences";

  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  input.runtime.device_tensors_ready = false;
  input.token_ids_host = torch::Tensor();
  input.token_ids =
      safe_to(flat_token_ids, token_options, /*non_blocking=*/true);
  input.runtime.device_tensors_ready = true;
}

void check_mtp_decode_states(
    const std::vector<TargetDecodeStateStore::DecodeRow>& states,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& token_ids_host,
    bool allow_overlap_fake_token) {
  CHECK(!request_ids.empty())
      << "MTP decode requires request ids for bootstrap state validation";
  CHECK_EQ(states.size(), request_ids.size())
      << "MTP decode request/state count mismatch";
  CHECK_GE(token_ids_host.numel(), static_cast<int64_t>(states.size()))
      << "MTP decode token/state count mismatch";

  Slice<int32_t> token_ids = tensor_slice(token_ids_host);
  for (int32_t i = 0; i < static_cast<int32_t>(states.size()); ++i) {
    const TargetDecodeStateStore::DecodeRow& state = states[i];
    const int32_t token_id = token_ids[i];
    CHECK(state.valid) << "MTP decode missing target state, request_id="
                       << request_ids[i];
    if (token_id < 0) {
      CHECK(allow_overlap_fake_token)
          << "MTP decode fake token is only allowed with schedule overlap, "
          << "request_id=" << request_ids[i];
      CHECK_GE(state.token_id, 0)
          << "MTP decode fake token requires a valid cached target token, "
          << "request_id=" << request_ids[i];
      continue;
    }
    CHECK_EQ(state.token_id, token_id)
        << "MTP decode target state token mismatch, request_id="
        << request_ids[i];
  }
}

template <typename TargetInput>
void replace_host_token_placeholders(TargetInput& input,
                                     int32_t placeholder,
                                     const torch::Tensor& replacements,
                                     const torch::TensorOptions& token_options,
                                     bool refresh_device = true) {
  input.runtime.device_tensors_ready = false;
  torch::Tensor replacement_cpu =
      to_cpu_contiguous(replacements.flatten(), torch::kInt);
  int32_t* token_ids = input.token_ids_host.template data_ptr<int32_t>();
  const size_t num_token_ids =
      static_cast<size_t>(input.token_ids_host.numel());
  Slice<int32_t> replacement_ids = tensor_slice(replacement_cpu);

  size_t replacement_idx = 0;
  for (size_t i = 0; i < num_token_ids; ++i) {
    if (token_ids[i] != placeholder) {
      continue;
    }
    CHECK_LT(replacement_idx, replacement_ids.size())
        << "not enough speculative replacement tokens";
    token_ids[i] = replacement_ids[replacement_idx++];
  }
  CHECK_EQ(replacement_idx, replacement_ids.size())
      << "unused speculative replacement tokens";

  if (refresh_device) {
    input.token_ids =
        safe_to(input.token_ids_host, token_options, /*non_blocking=*/true);
    input.runtime.device_tensors_ready = true;
  }
}

template <typename TargetInput>
void set_positions_tensor(TargetInput& input,
                          const torch::Tensor& positions_host,
                          const torch::TensorOptions& device_options) {
  input.runtime.device_tensors_ready = false;
  input.positions_host = positions_host;
  input.positions =
      safe_to(input.positions_host, device_options, /*non_blocking=*/true);
  input.runtime.device_tensors_ready = true;
}

ParallelArgs mtp_draft_parallel_args(const ParallelArgs& parallel_args,
                                     const runtime::Options& options) {
  if (!options.enable_mtp_draft_body_tp1()) {
    return parallel_args;
  }
  CHECK(parallel_args.single_rank_group_ != nullptr)
      << "MTP draft body TP1 requires a single-rank process group";
  ParallelArgs draft_args = parallel_args;
  draft_args.kv_transfer_groups_ = parallel_args.kv_transfer_groups();
  draft_args.rank(0)
      .world_size(1)
      .dp_size(1)
      .ep_size(1)
      .cp_size(1)
      .tp_size(1)
      .sp_size(1);
  draft_args.mapping_data(nlohmann::json{});
  draft_args.process_group_ = parallel_args.single_rank_group_;
  draft_args.dp_local_process_group_ = parallel_args.single_rank_group_;
  draft_args.lm_head_group_ = parallel_args.tp_group_;
  draft_args.tp_group_ = parallel_args.single_rank_group_;
  draft_args.cp_group_ = parallel_args.single_rank_group_;
  draft_args.moe_ep_group_ = parallel_args.single_rank_group_;
  draft_args.moe_tp_group_ = parallel_args.single_rank_group_;
  return draft_args;
}

}  // namespace

template <typename TargetInput>
MTPWorkerImpl<TargetInput>::MTPWorkerImpl(const ParallelArgs& parallel_args,
                                          const torch::Device& device,
                                          const runtime::Options& options,
                                          WorkerType worker_type)
    : MTPWorkerImpl<TargetInput>(parallel_args,
                                 device,
                                 options,
                                 speculative_target_options(options),
                                 speculative_draft_options(options),
                                 worker_type) {}

template <typename TargetInput>
MTPWorkerImpl<TargetInput>::MTPWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    const runtime::Options& target_options,
    const runtime::Options& draft_options,
    WorkerType worker_type)
    : DraftModelSpecWorkerImpl<TargetInput>(
          parallel_args,
          device,
          options,
          target_options,
          worker_type,
          [&parallel_args, &device, &options, &draft_options] {
            return std::make_unique<LLMWorkerImpl>(
                mtp_draft_parallel_args(parallel_args, options),
                device,
                draft_options);
          }) {}

template <typename TargetInput>
MTPWorkerImpl<TargetInput>::~MTPWorkerImpl() {
  DraftModelSpecWorkerImpl<TargetInput>::drain_pending_execution();
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::init_model(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  const bool result = DraftModelSpecWorkerImpl<TargetInput>::init_model(
      model_weights_path, random_seed, master_status);

  if (impl_ != nullptr && impl_->get_status() == WorkerImpl::Status::LOADED) {
    context_ = impl_->context_;
    target_verify_capabilities_ =
        spec_verify::resolve_target_spec_verify_capabilities(
            ModelRegistry::get_target_spec_verify_mode(
                context_.get_model_args().model_type()),
            ModelConfig::is_python_model_impl(context_.get_model_impl()));
  }

  if (draft_impl_ != nullptr &&
      draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    const bool keep_draft_owned_weights =
        options_.enable_mtp_draft_body_tp1() &&
        draft_impl_->owns_mtp_draft_weights();
    if (!keep_draft_owned_weights) {
      const bool python_weights_shared =
          draft_impl_->share_weights_from(*impl_);
      if (!python_weights_shared) {
        const bool share_lm_head = share_target_lm_head_with_draft();
        if (share_lm_head) {
          auto head = impl_->get_lm_head();
          draft_impl_->set_lm_head(head);
        }
        auto word_embedding = impl_->get_word_embedding();
        draft_impl_->set_word_embedding(word_embedding);
      }
    }
  }
#if defined(USE_NPU)
  if (result && supports_explicit_spec_verify_replay_update()) {
    CHECK_EQ(::xllm::KernelConfig::get_instance().npu_kernel_backend(), "TORCH")
        << "Expanded MTP verify requires the NPU Torch backend";
  }
#endif
  return result;
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::on_kv_pools_allocated(int64_t num_blocks) {
  DraftModelSpecWorkerImpl<TargetInput>::on_kv_pools_allocated(num_blocks);
  draft_input_state_store_ = std::make_unique<DraftInputStateStore>(num_blocks);
}

template <typename TargetInput>
std::tuple<int64_t, int64_t>
MTPWorkerImpl<TargetInput>::estimate_kv_cache_capacity() {
  CHECK(impl_ != nullptr);
  return estimate_kv_cache_capacity_with_draft(
      speculative_target_options(options_),
      speculative_draft_options(options_));
}

template <typename TargetInput>
KVCacheShape MTPWorkerImpl<TargetInput>::draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape) const {
  if (options_.enable_mtp_draft_body_tp1()) {
    return this->build_draft_kv_cache_shape(target_kv_cache_shape,
                                            /*draft_world_size=*/1);
  }
  if (KVCacheConfig::get_instance().enable_vmm_kvcache() ||
      target_kv_cache_shape.shared_draft_layers() > 0) {
    return build_draft_kv_cache_shape(target_kv_cache_shape);
  }
  return target_kv_cache_shape;
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::positions_are_decoupled_from_kv_length()
    const {
  if (impl_ == nullptr) {
    return false;
  }
  const ModelArgs& target_args = impl_->context_.get_model_args();
  return !target_args.rope_scaling_mrope_section().empty();
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::should_use_explicit_spec_verify_replay_update(
    const TargetInput& input) const {
#if defined(USE_NPU)
  const torch::Tensor& block_tables =
      input.input_params.attention.host.block_tables;
  const int64_t num_sequences = input.input_params.meta.num_sequences;
  if (!::xllm::ExecutionConfig::get_instance().enable_graph() ||
      !::xllm::ExecutionConfig::get_instance()
           .enable_graph_mode_decode_no_padding() ||
      !supports_explicit_spec_verify_replay_update() ||
      ModelConfig::is_python_model_impl(context_.get_model_impl()) ||
      impl_ == nullptr || options_.num_speculative_tokens() <= 0 ||
      options_.block_size() <= 0 ||
      !spec_verify::has_speculative_verify_block_table_layout(
          block_tables, input.input_params.multi_block_tables, num_sequences)) {
    return false;
  }
  const int64_t block_table_width =
      block_tables.defined()
          ? spec_verify_block_table_width(block_tables)
          : spec_verify::speculative_verify_block_table_capacity(
                impl_->context_.get_model_args().max_position_embeddings(),
                logical_block_size());
  if (block_table_width <= 0 ||
      block_table_width > kMaxSpecVerifyGraphUpdateBlockTableWidth) {
    return false;
  }
  const int64_t spec_width =
      static_cast<int64_t>(options_.num_speculative_tokens()) + 1;
  return kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
      spec_width, options_.block_size());
#else
  (void)input;
  return false;
#endif
}

template <typename TargetInput>
int64_t MTPWorkerImpl<TargetInput>::spec_verify_block_table_width(
    const torch::Tensor& block_tables) const {
  CHECK(block_tables.defined() && block_tables.dim() == 2);
  int64_t required_width = block_tables.size(1);
  if (impl_ != nullptr) {
    const int64_t declared_capacity =
        spec_verify::speculative_verify_block_table_capacity(
            impl_->context_.get_model_args().max_position_embeddings(),
            logical_block_size());
    CHECK_LE(required_width, declared_capacity)
        << "block table width exceeds the model position capacity";
    required_width = declared_capacity;
  }
  return required_width;
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::use_chunked_prefill_spec_verify_path() const {
  return target_verify_capabilities_.uses_chunked_prefill ||
         supports_explicit_spec_verify_replay_update();
}

template <typename TargetInput>
int32_t MTPWorkerImpl<TargetInput>::logical_block_size() const {
  CHECK_GT(options_.block_size(), 0);
  const int32_t kv_split_size = parallel_args_.kv_split_size_effective();
  CHECK_GT(kv_split_size, 0);
  return options_.block_size() * kv_split_size;
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::owns_npu_parallel_input_prepare() const {
  return false;
}

template <typename TargetInput>
std::optional<ForwardOutput> MTPWorkerImpl<TargetInput>::step_empty(
    const TargetInput& input) {
  flush_pending_target_context();
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    TargetInput target_prepared;
    TargetInput draft_prepared;
    ForwardOutput output =
        detail::run_worker_no_sync(
            *impl_, input, *compute_stream_, *compute_stream_, &target_prepared)
            .value();
    ForwardOutput draft_output =
        dispatch_draft_worker(input, draft_prepared).value();
    transfer_retained_inputs(output, draft_output);
    clear_all_output_embeddings(output);
    finalize_output_on_stream(
        output, *compute_stream_, this->enable_schedule_overlap());
    return output;
  }

  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  std::vector<TargetInput> prepared(num_speculative_tokens - 1);
  std::vector<ForwardOutput> draft_outputs;
  draft_outputs.reserve(num_speculative_tokens - 1);
  // Idle DP ranks must enter the same model collectives as active peers:
  // K-1 recursive drafts, target verification, then one target-conditioned
  // extend.
  for (int32_t draft_idx = 1; draft_idx < num_speculative_tokens; ++draft_idx) {
    draft_outputs.emplace_back(
        dispatch_draft_worker(input, prepared[draft_idx - 1]).value());
  }

  // Idle DP ranks never read the grammar states or per-request payloads this
  // clone carries (O(seq_len) copies); clone_for_draft_step skips them.
  TargetInput expanded = input.clone_for_draft_step();
  scale_speculative_parallel_token_counts(expanded.input_params,
                                          num_speculative_tokens);
  TargetInput target_prepared;
  ForwardOutput output = detail::run_worker_no_sync(*impl_,
                                                    expanded,
                                                    *compute_stream_,
                                                    *compute_stream_,
                                                    &target_prepared)
                             .value();

  expanded.input_params.clear_linear_attention_state();
  expanded.input_params.is_spec_verify = false;
  expanded.input_params.embedding.input_embedding =
      output.sample_output.embeddings;
  if (use_chunked_prefill_spec_verify_path()) {
    expanded.input_params.meta.batch_forward_type =
        BatchForwardType::CHUNKED_PREFILL;
  }
  prepare_draft_sampling(is_probabilistic_draft_, expanded.sampling_params);
  TargetInput extend_prepared;
  ForwardOutput extended =
      dispatch_draft_worker(expanded, extend_prepared).value();
  transfer_retained_inputs(output, extended);
  for (ForwardOutput& draft_output : draft_outputs) {
    transfer_retained_inputs(output, draft_output);
  }
  clear_all_output_embeddings(output);
  finalize_output_on_stream(
      output, *compute_stream_, this->enable_schedule_overlap());
  return output;
}

template <typename TargetInput>
std::optional<ForwardOutput> MTPWorkerImpl<TargetInput>::step_prefill(
    const TargetInput& input) {
  flush_pending_target_context();

  Timer target_timer;
  TargetInput target_prepared;
  TargetInput draft_prepared;

  ForwardOutput output =
      detail::run_worker_no_sync(
          *impl_, input, *compute_stream_, *compute_stream_, &target_prepared)
          .value();
  std::vector<size_t> prior_input_failures =
      failed_draft_input_rows(input, output.json_object_errors);
  mark_invalid_draft_input_rows(
      input,
      draft_input_state_store_->failed_draft_input_mask(
          input.input_params.embedding.batch_rows()),
      &prior_input_failures,
      output.json_object_errors);
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              target_timer.elapsed_seconds());

  TargetInput prefill_input;
  prepare_prefill_inputs(input, prefill_input);
  std::optional<DraftInputSamplingState> draft_input_sampling_state;
  torch::Tensor root_tokens_host;
  std::vector<uint8_t> vocabulary_invalid_rows;

  auto& embeddings = output.sample_output.embeddings;
  check_draft_input_embedding(embeddings, "prefill");

  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // Target prefill seeds the MTP decode cache. Under orthogonal CP x TP each
    // CP shard samples independently unless this token is synchronized across
    // both axes, which would cache divergent seeds.
    maybe_broadcast_spec_tokens(
        output.sample_output.next_tokens,
        input.sampling_params.sampling_mode() == SamplingMode::GREEDY);
    if (embeddings.defined()) {
      prefill_input.input_params.embedding.input_embedding = embeddings.clone();
    }
    if (output.sample_output.next_tokens.defined()) {
      // Share the post-consensus host snapshot with all prefill consumers.
      root_tokens_host =
          to_cpu_contiguous(output.sample_output.next_tokens, torch::kLong);
      replace_host_token_placeholders(prefill_input,
                                      -1,
                                      root_tokens_host,
                                      prefill_input.token_ids.options());
    }
    if (input.sampling_params.logits_indices().defined() &&
        output.sample_output.next_tokens.defined()) {
      draft_input_sampling_state = draft_input_state_store_->prepare(
          input.input_params.embedding.batch_rows(),
          input.sampling_params,
          input.json_object_states,
          root_tokens_host.view({-1, 1}),
          device_);
      prefill_input.sampling_params =
          draft_input_sampling_state->sampling_params;
      prepare_draft_proposal_sampling(is_probabilistic_draft_,
                                      prefill_input.sampling_params);
      prefill_input.json_object_states =
          draft_input_sampling_state->json_object_states;
      vocabulary_invalid_rows = compute_vocabulary_invalid_rows(
          prefill_input.sampling_params,
          draft_input_sampling_state->histories,
          prefill_input.json_object_states,
          draft_input_sampling_state->invalid_rows);
    }
    if (embeddings.defined() || output.sample_output.next_tokens.defined()) {
      record_current_metadata_ready_event(prefill_input, *compute_stream_);
    }
  }
  Timer draft_timer;
  ForwardOutput draft_output =
      dispatch_draft_worker(prefill_input, draft_prepared).value();
  std::optional<DraftInput> draft_input;
  if (draft_input_sampling_state.has_value()) {
    draft_input =
        assemble_draft_input(draft_output,
                             draft_prepared.sampling_params,
                             /*require_confidence=*/is_pd_prefill_exporter());
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());

  if (input.sampling_params.num_sample_seqs() > 0) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    target_state_store_->write_prefill_target_context(
        input.input_params.embedding.batch_rows(), root_tokens_host);
    output.sample_output.next_tokens = root_tokens_host;
    CHECK(draft_input_sampling_state.has_value());
    mark_invalid_draft_input_rows(input,
                                  vocabulary_invalid_rows,
                                  &prior_input_failures,
                                  output.json_object_errors);
    if (is_pd_prefill_exporter()) {
      output.draft_input = std::move(*draft_input);
    } else {
      write_draft_input(input.input_params.embedding.batch_rows(),
                        std::move(*draft_input),
                        std::move(*draft_input_sampling_state),
                        prior_input_failures);
    }
    stash_draft_input_topk_state(input.input_params.embedding.batch_rows(),
                                 draft_output.mtp_topk_state,
                                 draft_prepared.sampling_params);
  }
  clear_all_output_embeddings(output);

  transfer_retained_inputs(output, draft_output);
  finalize_output_on_stream(
      output, *compute_stream_, this->enable_schedule_overlap());

  if (!this->publishes_step_output()) {
    return std::nullopt;
  }
  return output;
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::prepare_prefill_inputs(
    const TargetInput& input,
    TargetInput& prefill_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  TargetInput host_input = input.clone();
  if constexpr (std::is_same_v<TargetInput, VlmForwardInput>) {
    // The draft forward is LLM-only; uploading the vision payload just to
    // drop it in make_llm_draft_input would waste the full image H2D.
    host_input.input_params.multimodal = VlmVisionInput();
  }
  prefill_input = host_input.to(device_, dtype_);
  prepare_draft_sampling(is_probabilistic_draft_,
                         prefill_input.sampling_params);
  clear_ready_events(prefill_input);
  auto& input_params = prefill_input.input_params;
  // The Qwen draft is a pure full-attention model; without this cleanup the
  // target's recurrent slot metadata makes MTP prefill enter a stateful path
  // it has neither a validity mask nor a recurrent cache for.
  if (!draft_impl_->has_request_state_cache()) {
    input_params.clear_linear_attention_state();
  }
  prefill_input.runtime.device_tensors_ready = false;
  prefill_input.token_ids_host = torch::empty({input.token_ids_host.numel()},
                                              pinned_cpu_options(torch::kInt));
  // Packed inputs can materialize row geometry during input.to(); use that
  // geometry with the original host token source.
  fill_mtp_prefill_token_ids(input.token_ids_host,
                             input_params,
                             input.input_params.embedding.extra_token_ids,
                             prefill_input.token_ids_host);
  prefill_input.token_ids = safe_to(prefill_input.token_ids_host,
                                    prefill_input.positions.options(),
                                    /*non_blocking=*/true);
  prefill_input.runtime.device_tensors_ready = true;
  record_metadata_ready_event(*prepare_stream_, prefill_input);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::release_algorithm_state(
    const std::vector<int32_t>& released) {
  flush_pending_target_context();
  if (draft_input_state_store_ != nullptr) {
    draft_input_state_store_->clear(released);
  }
}

template <typename TargetInput>
DraftInput MTPWorkerImpl<TargetInput>::assemble_draft_input(
    ForwardOutput& output,
    const SamplingParameters& sampling_params,
    bool require_confidence) {
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  SampleOutput& sample = output.sample_output;
  maybe_broadcast_spec_tokens(
      sample.next_tokens,
      sampling_params.sampling_mode() == SamplingMode::GREEDY);

  // Confidence is distinct from q and is selected before vocabulary remapping.
  const torch::Tensor selected_probs =
      require_confidence
          ? select_input_confidence(sample,
                                    output.draft_input,
                                    output.logits,
                                    sampling_params.logits_indices())
          : torch::Tensor();
  const bool probabilistic = is_probabilistic_draft_;
  draft_vocab_.remap(sample);
  return xllm::finish_draft_input(sample,
                                  sample.embeddings,
                                  sampling_params,
                                  probabilistic,
                                  std::move(selected_probs),
                                  this->draft_input_spec());
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::write_draft_input(
    const BatchRows& rows,
    DraftInput draft_input,
    DraftInputSamplingState sampling,
    const std::vector<size_t>& failed_rows) {
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  draft_input_state_store_->write_draft_input(
      rows,
      std::move(draft_input),
      std::move(sampling.histories),
      std::move(sampling.json_object_states),
      failed_rows);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::stash_draft_input_topk_state(
    const BatchRows& rows,
    const MtpTopkStatePtr& raw_topk_state,
    const SamplingParameters& sampling_params) {
  // Only DSA top-k reuse consumes this snapshot. PD prefill exporters never
  // store a seed batch, so publishing would CHECK-fail on the empty slot.
  if (is_pd_prefill_exporter() || !layer::is_mtp_dsa_topk_reuse_enabled(
                                      draft_impl_->context_.get_model_args())) {
    return;
  }
  draft_input_state_store_->publish_topk_state(
      rows,
      select_mtp_topk_state_for_next_step(raw_topk_state, sampling_params));
}

template <typename TargetInput>
DraftTokenHistories
MTPWorkerImpl<TargetInput>::restore_draft_input_sampling_history(
    TargetInput& input) {
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  return draft_input_state_store_->restore(
      input.input_params.embedding.batch_rows(),
      input.sampling_params,
      input.json_object_states,
      device_);
}

template <typename TargetInput>
TargetInput MTPWorkerImpl<TargetInput>::build_draft_extend_input(
    const TargetInput& validate_input,
    const SampleOutput& accepted_output,
    SamplingParameters sampling_params) {
  // The draft-0 forward reads the device-side accepted output alone; the
  // store-side sampling update and the JSON mismatch scan run in the settle,
  // never gating the launch.
  const MtpDraftExtendRows rows = build_mtp_draft_extend_rows(
      accepted_output.next_tokens, accepted_output.embeddings);
  // Snapshot on the stream that will next rewrite the target's stable
  // metadata; FIFO ordering prevents the snapshot copy from racing a
  // following prepare.
  c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
  prepare_stream_->wait_stream(*compute_stream_);
  TargetInput draft_extend_input = build_mtp_draft_extend_input(
      validate_input, rows, std::move(sampling_params));
  record_metadata_ready_event(*prepare_stream_, draft_extend_input);
  return draft_extend_input;
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::publish_round_result(
    PendingExtendRound& round) {
  // The settle's event wait (when the JSON scan did not cover it) already
  // guarantees the host tokens are readable here.
  const BatchRows rows = round.rows();
  DraftInput draft_input = assemble_draft_input(
      round.output, round.sampling_params, /*require_confidence=*/false);
  DraftInputSamplingState sampling =
      draft_input_state_store_->prepare(rows,
                                        round.sampling_params,
                                        round.json_states,
                                        round.accepted_tokens_host,
                                        device_);
  write_draft_input(
      rows, std::move(draft_input), std::move(sampling), round.failed_rows);
  stash_draft_input_topk_state(
      rows, round.output.mtp_topk_state, round.prepared_input.sampling_params);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::import_carried_draft_input(
    TargetInput& input) {
  if constexpr (std::is_same_v<TargetInput, LlmForwardInput>) {
    auto& request_state = input.request_state;
    if (!request_state.draft_input.has_value()) {
      return;
    }
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    target_state_store_->write_target_bootstrap_rows(
        request_state.draft_input_row_idxes,
        input.input_params.embedding.batch_rows(),
        input.token_ids_host);
    draft_input_state_store_->import_pd_draft_input(
        request_state.draft_input_row_idxes,
        input.input_params.embedding.batch_rows(),
        *request_state.draft_input,
        this->draft_input_spec(),
        device_);
    request_state.draft_input.reset();
    request_state.draft_input_row_idxes.clear();
  }
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::supports_explicit_spec_verify_replay_update()
    const {
  return target_verify_capabilities_.supports_explicit_replay_update;
}

template <typename TargetInput>
std::optional<ForwardOutput> MTPWorkerImpl<TargetInput>::step_decode(
    const TargetInput& raw_input) {
  TargetInput input = raw_input.clone();
  if (use_chunked_prefill_spec_verify_path()) {
    stabilize_decode_host_tensors(input);
  }
  // The preceding extend is already queued on compute_stream_. Only the
  // target's accepted-token handoff must be materialized for scheduler state.
  flush_pending_target_context();
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();

  import_carried_draft_input(input);

  const std::vector<TargetDecodeStateStore::DecodeRow> last_states =
      target_state_store_->read_decode_states(
          input.input_params.embedding.batch_rows());
  if (!input.input_params.meta.is_graph_warmup) {
    check_mtp_decode_states(last_states,
                            input.input_params.embedding.request_ids,
                            input.token_ids_host,
                            this->enable_schedule_overlap());
  }
  update_decode_step_input(input, last_states);
  const DraftTokenHistories sampling_histories =
      restore_draft_input_sampling_history(input);

  std::optional<DraftInput> draft_input;
  MtpTopkStatePtr mtp_topk_state;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // Read the top-k snapshot before take_draft_input detaches the batches:
    // reuse requires the producing batch still attached.
    if (layer::is_mtp_dsa_topk_reuse_enabled(
            draft_impl_->context_.get_model_args())) {
      mtp_topk_state = draft_input_state_store_->read_topk_state(
          input.input_params.embedding.batch_rows());
    }
    std::vector<uint8_t> failed_rows;
    draft_input = draft_input_state_store_->take_draft_input(
        input.input_params.embedding.batch_rows(), &failed_rows);
    if (!failed_rows.empty()) {
      // Seed with rows that already carry a JSON error for this input so a
      // persistently failed request does not get a duplicate error entry.
      std::vector<size_t> prior_failed_rows =
          failed_draft_input_rows(input, {});
      mark_invalid_draft_input_rows(
          input, failed_rows, &prior_failed_rows, input.json_object_errors);
    }
    if (!draft_input.has_value() && input.input_params.meta.is_graph_warmup) {
      draft_input = xllm::make_warmup_draft_input(
          input.input_params.meta.num_sequences,
          this->draft_input_spec().hidden_size,
          torch::dtype(draft_impl_->dtype()).device(device_),
          is_probabilistic_draft_);
    }
  }
  CHECK(draft_input.has_value())
      << "MTP decode requires a matching prefill/extend draft input";
  // The draft_input was already validated where it was produced or imported
  // (assemble_draft_input / PD import / warmup), so no re-check here.
  check_draft_input_embedding(draft_input->hidden_states, "decode");

  const bool has_json_object_states = !input.json_object_states.empty();
  detail::JsonDraftValidationScratch json_scratch;
  std::vector<uint8_t> json_invalid_suffix;
  bool halt_json_draft = false;
  if (has_json_object_states) {
    json_invalid_suffix.assign(input.json_object_states.size(), 0);
    json_scratch.states_after.reserve(num_speculative_tokens);
    json_scratch.invalid_draft_step_major.reserve(
        input.json_object_states.size() * num_speculative_tokens);
  }

  // Only sampling_params and json_object_states are read from this before it
  // is reassigned to next_step_input each round, so copy just those two and
  // skip a per-step deep copy of the whole ModelParamsView.
  TargetInput current_draft_input;
  current_draft_input.sampling_params = input.sampling_params;
  current_draft_input.json_object_states = input.json_object_states;
  prepare_draft_sampling(is_probabilistic_draft_,
                         current_draft_input.sampling_params);
  TargetInput validate_input;
  TargetInput next_step_input;
  std::vector<ForwardOutput> draft_outputs;
  draft_outputs.reserve(num_speculative_tokens);
  Timer draft_timer;

  for (int32_t draft_idx = 0; draft_idx < num_speculative_tokens; ++draft_idx) {
    const bool is_final_draft = draft_idx == num_speculative_tokens - 1;
    const bool static_graph_tasks_prepared =
        is_final_draft &&
        prepare_static_mtp_graph_tasks_before_final_draft(input, last_states);
    ForwardOutput draft_output;
    if (draft_idx == 0) {
      // Already in target token space (sampled by prefill or the preceding
      // extend); never process d2t twice.
      draft_output.draft_input = std::move(draft_input);
      const DraftInput& carried = *draft_output.draft_input;
      draft_output.sample_output.next_tokens = carried.token_ids;
      draft_output.sample_output.embeddings = carried.hidden_states;
      // The seed's sparse logits are the column-0 q the rejection sampler
      // needs; undefined for greedy point-mass seeds.
      draft_output.sample_output.draft_logits = carried.draft_logits;
    } else {
      if (mtp_topk_state != nullptr) {
        current_draft_input.input_params.mtp_topk_state = mtp_topk_state;
      }
      TargetInput prepared;
      draft_output =
          dispatch_draft_worker(current_draft_input, prepared).value();
      current_draft_input.sampling_params = std::move(prepared.sampling_params);
    }
    draft_outputs.emplace_back(std::move(draft_output));

    if (is_final_draft) {
      prepare_validate_inputs_impl(
          input, validate_input, static_graph_tasks_prepared);
    } else {
      // The first recursive proposal runs at root, not root+1: the draft
      // input already processed the root at source position root-1.
      const std::vector<uint8_t> vocabulary_invalid_rows = prepare_draft_inputs(
          input, next_step_input, draft_idx, sampling_histories);
      mark_invalid_draft_input_rows(input,
                                    vocabulary_invalid_rows,
                                    /*failed_rows=*/nullptr,
                                    input.json_object_errors);
    }

    if (has_json_object_states) {
      const torch::Tensor& next_tokens =
          draft_outputs.back().sample_output.next_tokens;
      std::vector<int32_t> draft_token_ids =
          tensor_to_vector<int32_t>(next_tokens.flatten());
      // A rejected seed (draft_idx 0) only marks the row invalid for the
      // verifier; only a rejected recursive proposal leaves nothing valid to
      // extend, so only that halts further drafts.
      const bool seed_step = draft_idx == 0;
      const bool step_halted =
          detail::append_json_draft_step(current_draft_input.json_object_states,
                                         json_invalid_suffix,
                                         draft_token_ids,
                                         json_scratch) &&
          !seed_step;
      halt_json_draft = halt_json_draft || step_halted;
    }
    if (is_final_draft) {
      continue;
    }

    const SampleOutput& last_output = draft_outputs.back().sample_output;
    check_draft_input_embedding(last_output.embeddings, "decode");
    std::vector<JsonObjectGrammarState> json_states =
        std::move(current_draft_input.json_object_states);
    current_draft_input = std::move(next_step_input);
    current_draft_input.json_object_states = std::move(json_states);
    set_token_ids_device_tensor(current_draft_input,
                                last_output.next_tokens,
                                current_draft_input.token_ids.options(),
                                *compute_stream_);
    current_draft_input.input_params.embedding.input_embedding =
        last_output.embeddings;
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    if (has_json_object_states) {
      if (halt_json_draft) {
        // Illegal draft under the current mask: stop further draft forwards
        // and pad remaining speculative slots with -1 so validate rejects them.
        prepare_validate_inputs_impl(input, validate_input);
        while (draft_outputs.size() <
               static_cast<size_t>(num_speculative_tokens)) {
          ForwardOutput rejected_output = draft_outputs.back();
          CHECK(rejected_output.sample_output.next_tokens.defined());
          rejected_output.sample_output.next_tokens = torch::full_like(
              rejected_output.sample_output.next_tokens, /*fill_value=*/-1);
          draft_outputs.push_back(std::move(rejected_output));
          detail::append_json_draft_step(
              current_draft_input.json_object_states,
              json_invalid_suffix,
              std::vector<int32_t>(
                  current_draft_input.json_object_states.size(), -1),
              json_scratch);
        }
        break;
      }
      apply_draft_grammar_bitmask(current_draft_input.sampling_params,
                                  current_draft_input.json_object_states,
                                  device_);
    }
    record_current_metadata_ready_event(current_draft_input, *compute_stream_);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());
  return run_validate(input,
                      draft_outputs,
                      validate_input,
                      num_speculative_tokens,
                      has_json_object_states ? &json_scratch : nullptr);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::fill_validate_input_from_draft_outputs(
    const TargetInput& input,
    std::span<const ForwardOutput> draft_outputs,
    TargetInput& validate_input,
    int32_t num_val_tokens,
    detail::JsonDraftValidationScratch* json_scratch,
    Stream& compute_stream) {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_GT(num_val_tokens, 0);
  CHECK(validate_input.token_ids.defined())
      << "validate token_ids must be prepared before draft token fill";
  CHECK_EQ(validate_input.token_ids.dim(), 1)
      << "validate token_ids must be flat";
  CHECK_EQ(validate_input.token_ids.numel(),
           static_cast<int64_t>(num_sequences) * num_val_tokens)
      << "MTP validation requires fixed-width rows";

  const torch::TensorOptions token_options = validate_input.token_ids.options();
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  wait_metadata_ready_event(validate_input, compute_stream);

  if (json_scratch != nullptr) {
    CHECK_EQ(input.json_object_states.size(),
             static_cast<size_t>(num_sequences))
        << "JSON grammar state rows must match validation sequences";
    const int32_t num_speculative_tokens = num_val_tokens - 1;
    CHECK_EQ(json_scratch->states_after.size(),
             static_cast<size_t>(num_speculative_tokens));
    validate_input.json_object_states = detail::build_json_validation_states(
        input.json_object_states,
        std::move(*json_scratch),
        validate_input.json_object_invalid_draft);
    validate_input.sampling_params.filter_bitmask() =
        build_json_object_filter_bitmask(validate_input.json_object_states,
                                         device_,
                                         JsonObjectMaskBuildPhase::TARGET);
  }

  validate_input.runtime.device_tensors_ready = false;
  auto& fused_draft_tokens =
      validate_input.input_params.graph.spec_verify_draft_token_sources;
  fused_draft_tokens.clear();
#if defined(USE_NPU)
  const bool use_fused_verify_token_update =
      validate_input.input_params.graph.spec_verify_source_addresses_stable &&
      validate_input.input_params.graph.input_tokens_override.defined() &&
      supports_explicit_spec_verify_replay_update() &&
      !ModelConfig::is_python_model_impl(context_.get_model_impl()) &&
      kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
          num_val_tokens, options_.block_size());
  if (use_fused_verify_token_update) {
    fused_draft_tokens.reserve(draft_outputs.size());
    for (const ForwardOutput& draft_output : draft_outputs) {
      const torch::Tensor& next_tokens = draft_output.sample_output.next_tokens;
      CHECK(next_tokens.defined() && next_tokens.numel() == num_sequences &&
            next_tokens.scalar_type() == torch::kInt64 &&
            next_tokens.device() == validate_input.token_ids.device() &&
            next_tokens.is_contiguous())
          << "fused speculative verify input update requires one contiguous "
             "int64 token per sequence and draft step";
      fused_draft_tokens.emplace_back(next_tokens.flatten());
    }
    validate_input.runtime.device_tensors_ready = true;
    return;
  }
#endif
  // Uniform widths: one stacked slice-copy instead of one device kernel
  // per draft step; launch-bound NPU decode is measurably faster with one
  // op than num_speculative_tokens ops writing the same bytes.
  const int32_t num_draft_tokens = num_val_tokens - 1;
  torch::Tensor validate_token_rows =
      validate_input.token_ids.view({num_sequences, num_val_tokens});
  if (num_draft_tokens > 0) {
    // Capacity-reused pack buffer; copy_ converts dtype/device inline.
    torch::Tensor packed_drafts =
        ensure_capacity(packed_drafts_buffer_,
                        {num_sequences, num_draft_tokens},
                        token_options)
            .narrow(0, 0, num_sequences)
            .narrow(1, 0, num_draft_tokens);
    for (int32_t i = 0; i < num_draft_tokens; ++i) {
      CHECK(static_cast<size_t>(i) < draft_outputs.size())
          << "draft_outputs index out of range for step " << i;
      const torch::Tensor& next_tokens =
          draft_outputs[static_cast<size_t>(i)].sample_output.next_tokens;
      CHECK(next_tokens.defined())
          << "draft next_tokens must be defined for validate token fill";
      packed_drafts.select(1, i).copy_(next_tokens.flatten());
    }
    validate_token_rows.slice(/*dim=*/1, /*start=*/1, /*end=*/num_val_tokens)
        .copy_(packed_drafts, /*non_blocking=*/true);
  }
  validate_input.runtime.device_tensors_ready = true;
  record_metadata_ready_event(compute_stream, validate_input);
}

template <typename TargetInput>
std::optional<ForwardOutput> MTPWorkerImpl<TargetInput>::run_validate(
    const TargetInput& input,
    std::vector<ForwardOutput>& draft_outputs,
    TargetInput& validate_input,
    int32_t num_speculative_tokens,
    detail::JsonDraftValidationScratch* json_scratch) {
  Timer target_timer;
  TargetInput target_prepared;
  fill_validate_input_from_draft_outputs(input,
                                         draft_outputs,
                                         validate_input,
                                         num_speculative_tokens + 1,
                                         json_scratch,
                                         *compute_stream_);
  // Under DP, every rank must run the collectives in lockstep.
  ForwardOutput target_output = detail::run_worker_no_sync(*impl_,
                                                           validate_input,
                                                           *compute_stream_,
                                                           *compute_stream_,
                                                           &target_prepared)
                                    .value();
  target_output.json_object_errors.insert(
      target_output.json_object_errors.end(),
      input.json_object_errors.begin(),
      input.json_object_errors.end());
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              target_timer.elapsed_seconds());

  const int32_t batch_size =
      static_cast<int32_t>(input.input_params.meta.num_sequences);
  const int32_t max_val_tokens = num_speculative_tokens + 1;
  const int64_t padded_total =
      static_cast<int64_t>(batch_size) * max_val_tokens;

  // Uniform widths keep the logits flat: view in place to [padded_total,
  // vocab] (validate() applies its own layout view on top).
  target_output.logits =
      target_output.logits.view({padded_total, target_output.logits.size(-1)});

  SampleOutput val_output;
  {
    AUTO_COUNTER(speculative_execution_latency_seconds_validation);
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // Lift the JSON grammar masks (installed at full validate width) onto the
    // request params: the mask is the one per-token verify field because its
    // per-sequence grammar state advances with each speculative token.
    SamplingParameters verify_params = input.sampling_params;
    if (verify_params.min_token_positions().defined()) {
      // Overlap can leave the scheduler's generated count behind accepted
      // tokens; use the corrected target positions. KV lengths count text
      // and visual tokens uniformly, unlike mRoPE coordinates.
      torch::Tensor first_positions;
      if constexpr (std::is_same_v<TargetInput, VlmForwardInput>) {
        const torch::Tensor& lengths =
            validate_input.input_params.attention.device.kv_seq_lens;
        const int64_t width = num_speculative_tokens + 1;
        if (lengths.numel() == batch_size) {
          first_positions = lengths - width;
        } else {
          first_positions =
              lengths.view({batch_size, width}).select(/*dim=*/1, /*index=*/0) -
              1;
        }
      } else {
        first_positions = validate_input.positions
                              .view({batch_size, num_speculative_tokens + 1})
                              .select(/*dim=*/1, /*index=*/0);
      }
      verify_params.min_tokens_remaining() =
          verify_params.min_token_positions() - first_positions;
    }
    if (validate_input.sampling_params.filter_bitmask().defined()) {
      verify_params.filter_bitmask() =
          validate_input.sampling_params.filter_bitmask();
    }
    val_output = validate(std::move(verify_params),
                          draft_outputs,
                          target_output,
                          num_speculative_tokens,
                          validate_input.json_object_invalid_draft);
  }

  torch::Tensor accepted_tokens_host =
      this->target_context_handoff_.acquire_host_buffer(val_output.next_tokens);
  StreamEventPtr target_context_ready_event;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

    // Catch-all for cross-rank RNG divergence: unify accepted tokens before
    // deriving any device-resident state used by the next draft iteration.
    maybe_broadcast_spec_tokens(
        val_output.next_tokens,
        input.sampling_params.sampling_mode() == SamplingMode::GREEDY);

    accepted_tokens_host.copy_(val_output.next_tokens,
                               /*non_blocking=*/true);
    // The event covers both the token consensus and the D2H copy.
    target_context_ready_event = compute_stream_->record_event();
  }
  if (target_context_ready_event == nullptr) {
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize MTP target context, ret=" << ret;
  }
  // The JSON mismatch scan reads the host handoff, so it must not gate the
  // draft-0 launch: launch_draft_extend computes the failed rows in the
  // settle, after the launch. Non-grammar batches have no host dependency.
  std::vector<size_t> failed_sequence_rows;
  if (input.json_object_states.empty()) {
    failed_sequence_rows =
        failed_draft_input_rows(input, target_output.json_object_errors);
  }
  PendingExtendRound pending =
      launch_draft_extend(input,
                          validate_input,
                          val_output,
                          accepted_tokens_host,
                          target_context_ready_event,
                          std::move(failed_sequence_rows),
                          target_output.json_object_errors);
  ForwardOutput extended = pending.output;
  // The settle stays inline: the seed's publication into the store is part of
  // this step's observable contract; the launch/settle split still keeps the
  // host-token wait out of the launch.
  append_json_errors(target_output.json_object_errors,
                     settle_round(std::move(pending)));
  transfer_retained_inputs(target_output, extended);
  // Expose the staged host snapshot so metrics and serialization reuse it
  // instead of issuing a second D2H; the device copy feeds the next step's
  // input update.
  if (enable_schedule_overlap()) {
    target_output.next_tokens_host = accepted_tokens_host;
  }
  this->target_context_handoff_.stage(input.input_params.embedding.batch_rows(),
                                      target_context_ready_event,
                                      accepted_tokens_host);
  record_output_ready_event(target_output, *compute_stream_);

  // Keep prepared inputs alive until the target-context event completes.
  for (ForwardOutput& draft_output : draft_outputs) {
    transfer_retained_inputs(target_output, draft_output);
  }

  if (!this->enable_schedule_overlap()) {
    finalize_output_on_stream(target_output, *compute_stream_, false);
    flush_pending_target_context();
    target_output.ready_event.reset();
    val_output.next_tokens = std::move(accepted_tokens_host);
  }

  if (!this->publishes_step_output()) {
    return std::nullopt;
  }
  return finalize_verify_output(std::move(target_output),
                                std::move(val_output),
                                input.json_object_states);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::update_decode_step_input(
    TargetInput& input,
    const std::vector<TargetDecodeStateStore::DecodeRow>& last_states) const {
  this->rewrite_decode_rows_from_target_state(
      input,
      last_states,
      {.positions_decoupled_from_kv_len =
           positions_are_decoupled_from_kv_length(),
       .chunked_prefill_verify = use_chunked_prefill_spec_verify_path(),
       .logical_block_size = logical_block_size(),
       .validate_width = options_.num_speculative_tokens() + 1,
       // Under overlap without cached state the recorded state token is the
       // only in-vocabulary substitute for the fake token.
       .fake_context_falls_back_to_state_token = true});
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::prepare_validate_inputs_impl(
    const TargetInput& input,
    TargetInput& validate_input,
    bool static_graph_tasks_prepared) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  validate_input = input.clone();
  clear_ready_events(validate_input);
  validate_input.runtime.device_tensors_ready = false;
  validate_input.skip_sampling_for_logits_only = true;
  validate_input.input_params.embedding.input_embedding = torch::Tensor();
  auto& input_params = validate_input.input_params;
  torch::TensorOptions token_options = validate_input.token_ids.options();

  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  const int32_t logical_block_size = this->logical_block_size();
#if defined(USE_NPU)
  const bool use_explicit_spec_verify_replay_update =
      should_use_explicit_spec_verify_replay_update(input);
#else
  const bool use_explicit_spec_verify_replay_update = false;
#endif
  const bool use_chunked_spec_verify = use_chunked_prefill_spec_verify_path();
  spec_verify_layout::ValidateInputPlan plan;
  plan.uniform_width = num_val_tokens;
  plan.logical_block_size = logical_block_size;
  plan.positions_decoupled = positions_are_decoupled_from_kv_length();
  plan.attention_mode =
      use_chunked_spec_verify
          ? spec_verify_layout::ValidateAttentionMode::CHUNKED_PREFILL
          : spec_verify_layout::ValidateAttentionMode::DECODE;
  plan.deferred_tokens = use_explicit_spec_verify_replay_update;
  spec_verify_layout::ValidateInputLayout layout =
      spec_verify_layout::build_validate_input_layout(input, plan);
  const int32_t total_num_val_tokens = layout.total_tokens;
#if defined(USE_NPU)
  std::vector<int32_t> expanded_linear_state_ids;
  // The Qwen fused verifier (uniform validate width) consumes S
  // tokens per logical recurrent state: attention rows expand, but the state
  // indices stay sequence-scoped.
  if (num_val_tokens > 1 &&
      target_verify_capabilities_.rewrites_validate_linear_state_ids) {
    if (!target_verify_capabilities_.requires_uniform_validate_width) {
      expanded_linear_state_ids =
          spec_verify_layout::expand_validate_linear_state_ids(input_params,
                                                               num_val_tokens);
    } else {
      expanded_linear_state_ids = input_params.embedding.linear_state_ids;
    }
  }
#endif
  spec_verify_layout::apply_validate_input_layout(validate_input, plan, layout);

  auto& validate_sampling_params = validate_input.sampling_params;
  bool sampling_params_ready = false;
#if defined(USE_NPU)
  // A single greedy sequence's repeat_interleave only expands [0]/[false]
  // into fixed validation controls, yet it lands on the final-draft ->
  // target dependency chain; reuse stable controls after warmup.
  sampling_params_ready =
      npu_validate_input_state_.try_prepare_stable_greedy_sampling(
          validate_sampling_params,
          total_num_val_tokens,
          use_explicit_spec_verify_replay_update,
          device_);
#endif
  if (!sampling_params_ready) {
    expand_sampling_params_for_verify(validate_sampling_params,
                                      num_val_tokens,
                                      total_num_val_tokens,
                                      device_);
  }

  scale_speculative_parallel_token_counts(input_params, num_val_tokens - 1);

  std::vector<int32_t> accepted_prefix_lengths;
  if (use_chunked_prefill_spec_verify_path()) {
    accepted_prefix_lengths = this->apply_chunked_prefill_spec_verify_tail(
        input,
        input_params,
        num_sequences,
        token_options,
        /*publish_device_tensor=*/!use_explicit_spec_verify_replay_update);
  }

#if defined(USE_NPU)
  const MtpNpuValidateInputState::Config npu_config{
      .device = device_,
      .verify_block_table_width =
          use_explicit_spec_verify_replay_update
              ? spec_verify_block_table_width(
                    input_params.attention.host.block_tables)
              : 0,
      .max_seqs_per_batch = options_.max_seqs_per_batch(),
      // Same gate as the chunked tail that populates the q/kv seq lens this
      // graph input consumes; the two must agree or the builder CHECKs.
      .needs_expanded_verify_graph_input =
          use_chunked_prefill_spec_verify_path()};
  npu_validate_input_state_.prepare_attention(validate_input,
                                              layout,
                                              plan,
                                              accepted_prefix_lengths,
                                              expanded_linear_state_ids,
                                              static_graph_tasks_prepared,
                                              npu_config);
#else
  input_params.attention.rebuild_device_buffer(device_);
#endif
  // This metadata is independent of the in-flight final draft. Keep it on the
  // auxiliary stream and hand it to the compute stream with a device event.
  validate_input.runtime.device_tensors_ready = true;
  record_metadata_ready_event(*prepare_stream_, validate_input);
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::
    prepare_static_mtp_graph_tasks_before_final_draft(
        const TargetInput& input,
        const std::vector<TargetDecodeStateStore::DecodeRow>& last_states) {
#if defined(USE_NPU)
  if (!should_use_explicit_spec_verify_replay_update(input) ||
      input.input_params.embedding.linear_state_ids.size() != 1 ||
      last_states.size() != 1) {
    return false;
  }
  const auto& block_tables = input.input_params.attention.host.block_tables;
  if (!block_tables.defined() || block_tables.dim() != 2 ||
      block_tables.size(0) != 1) {
    return false;
  }
  const int64_t verify_block_table_width =
      spec_verify_block_table_width(block_tables);
  const auto& kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  if (kv_seq_lens.size() != 1) {
    return false;
  }
  const int64_t spec_width = options_.num_speculative_tokens() + 1;
  const int64_t base_kv_seq_len = kv_seq_lens.front();
  const int64_t spec_verify_max_kv_seq_len = base_kv_seq_len + spec_width - 1;
  const SpecVerifyGraphTaskSignal signal{
      .linear_state_id = input.input_params.embedding.linear_state_ids.front(),
      // Reuse the state snapshot that prepared this decode step; invalid
      // states fall back to the same length 1 as the store's lookup.
      .num_accepted_tokens = last_states.front().accepted_prefix_length(),
      .spec_width = spec_width,
      .block_table_width = verify_block_table_width,
      .base_kv_seq_len = base_kv_seq_len,
      .max_kv_seq_len = spec_verify_max_kv_seq_len,
  };
  auto* llm_target = dynamic_cast<LLMWorkerImpl*>(impl_.get());
  if (llm_target == nullptr) {
    return false;
  }
  return llm_target->prepare_static_mtp_graph_tasks(signal, *compute_stream_);
#else
  (void)input;
  (void)last_states;
  return false;
#endif
}

template <typename TargetInput>
std::vector<uint8_t> MTPWorkerImpl<TargetInput>::prepare_draft_inputs(
    const TargetInput& input,
    TargetInput& draft_input,
    int32_t position_offset,
    const DraftTokenHistories& sampling_histories) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  draft_input = input.clone_for_draft_step();
  prepare_draft_proposal_sampling(is_probabilistic_draft_,
                                  draft_input.sampling_params);
  clear_ready_events(draft_input);
  draft_input.runtime.device_tensors_ready = false;
  auto& input_params = draft_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  const int32_t num_sequences = input_params.meta.num_sequences;
  if (draft_impl_->has_request_state_cache()) {
    input_params.attention.host.kpool_query_lens.assign(num_sequences, 1);
    input_params.is_spec_verify = false;
    // The first draft already restored its request slot. Subsequent drafts
    // append to that live state instead of replaying a prefix restore.
    input_params.embedding.linear_state_read_ids =
        input_params.embedding.linear_state_ids;
    input_params.embedding.linear_state_read_indices =
        input_params.embedding.linear_state_indices;
  } else {
    input_params.clear_linear_attention_state();
    input_params.num_accepted_tokens = torch::Tensor();
    input_params.num_accepted_tokens_host.clear();
  }
  // Metadata-only rows: token ids arrive later from the previous draft
  // output.
  const int32_t logical_block_size = this->logical_block_size();
  const DecodeBatchContext batch_ctx = make_decode_batch_context(input);
  DecodeBuildBuffers buf;
  prepare_decode_buffers(buf,
                         batch_ctx,
                         static_cast<size_t>(num_sequences),
                         /*emit_kv_len=*/true,
                         /*emit_block_tables=*/false,
                         /*emit_token=*/false);
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    DecodeTokenInput token_input;
    token_input.req_id = seq_id;
    token_input.position_offset = position_offset;
    append_decode_token(batch_ctx, token_input, logical_block_size, buf);
  }

  torch::TensorOptions position_options = input.positions.options();
  set_positions_tensor(
      draft_input, make_pinned_cpu_tensor(buf.out_positions), position_options);
  input_params.attention.host.q_seq_lens.assign(num_sequences, 1);
  update_kv_lengths_and_slots(input_params, buf);
  input_params.attention.rebuild_device_buffer(device_);
#if defined(USE_NPU)
  // Later draft steps share the B cache variant.
  draft_impl_->prepare_dp_ep_padding_on_stream(input_params, *prepare_stream_);
#endif
  // token_ids is intentionally filled later from the previous draft output.
  draft_input.runtime.device_tensors_ready = false;

  // Positions/KV metadata do not depend on the in-flight draft result.
  record_metadata_ready_event(*prepare_stream_, draft_input);
  const std::vector<uint8_t> vocabulary_invalid_rows =
      compute_vocabulary_invalid_rows(draft_input.sampling_params,
                                      sampling_histories,
                                      input.json_object_states,
                                      /*sampling_invalid_rows=*/{});
  return vocabulary_invalid_rows;
}

template <typename TargetInput>
SampleOutput MTPWorkerImpl<TargetInput>::validate(
    const SamplingParameters& sampling_params,
    std::span<const ForwardOutput> draft_outputs,
    const ForwardOutput& target_output,
    int32_t num_speculative_tokens,
    const std::vector<uint8_t>& invalid_draft) {
  CHECK(!draft_outputs.empty())
      << "MTP validate requires at least one draft step";
  // An all-greedy batch under probabilistic mode carries no draft logits.
  // The draft logits carry the DRAFT model's own vocabulary; size the buffer
  // from their last dim, never from the target logits.
  const torch::Tensor& first_step_logits =
      draft_outputs.front().sample_output.draft_logits;
  const torch::Tensor& first_tokens =
      draft_outputs.front().sample_output.next_tokens;
  const bool is_probabilistic_draft = first_step_logits.defined();
  DraftProposal proposal = DraftProposal::preallocate(
      first_tokens.size(0),
      static_cast<int64_t>(num_speculative_tokens),
      is_probabilistic_draft ? first_step_logits.size(-1) : 0,
      /*is_probabilistic_draft=*/is_probabilistic_draft,
      first_tokens.options(),
      is_probabilistic_draft ? first_step_logits.options()
                             : first_tokens.options());
  int64_t step = 0;
  for (const auto& draft_output : draft_outputs) {
    proposal.tokens_at(step).copy_(draft_output.sample_output.next_tokens);
    if (is_probabilistic_draft) {
      proposal.fill_column(step, draft_output.sample_output.draft_logits);
    }
    ++step;
  }

  return validate(sampling_params,
                  proposal,
                  target_output,
                  num_speculative_tokens,
                  invalid_draft);
}

template <typename TargetInput>
SampleOutput MTPWorkerImpl<TargetInput>::validate(
    const SamplingParameters& sampling_params,
    const DraftProposal& draft_proposal,
    const ForwardOutput& target_output,
    int32_t num_speculative_tokens,
    const std::vector<uint8_t>& invalid_draft) {
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const int32_t batch_size =
      static_cast<int32_t>(draft_proposal.token_ids().size(0));
  // Grammar-invalid draft positions carry the -1 sentinel directly in the
  // proposal token ids; the rejection sampler rejects negative ids.
  torch::Tensor validation_token_ids;
  if (!invalid_draft.empty()) {
    CHECK_EQ(invalid_draft.size(),
             static_cast<size_t>(batch_size * (num_val_tokens - 1)))
        << "MTP invalid draft mask shape mismatch";
    // The NPU masked_fill kernel requires a bool mask (uint8 is CUDA-only).
    // The producer guarantees 0/1 bytes, so the same-storage dtype view is a
    // free reinterpretation with no conversion kernel.
    torch::Tensor invalid_mask =
        async_h2d_tensor(invalid_draft, draft_proposal.token_ids().device())
            .view(torch::kBool)
            .view({batch_size, num_val_tokens - 1});
    validation_token_ids =
        draft_proposal.token_ids().masked_fill(invalid_mask, /*sentinel=*/-1);
  }
  SampleOutput sample_output = spec_verify::run_rejection_sampling(
      sampling_params,
      DraftProposal(validation_token_ids.defined() ? validation_token_ids
                                                   : draft_proposal.token_ids(),
                    draft_proposal.logits()),
      // Flat [batch*val, vocab] already matches the rejection sampler's
      // expected layout; the 3-D view would be flattened right back.
      target_output.logits,
      target_output.sample_output.embeddings);

  return sample_output;
}

template class MTPWorkerImpl<LlmForwardInput>;
template class MTPWorkerImpl<VlmForwardInput>;

}  // namespace xllm
