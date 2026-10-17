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

#include <limits>

#include "runtime/vlm_forward_params.h"

#if defined(USE_NPU)
#include <acl/acl.h>
#include <c10/core/DeviceType.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_set>

#include "common/metrics.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/sampling/draft_sampling.h"
#include "core/framework/speculative/mtp_utils.h"
#include "core/kv_cache/block/block_utils.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#endif
#include "core/framework/speculative/adaptive_speculative_controller.h"
#include "core/framework/speculative/draft_input_sampling.h"
#include "core/framework/speculative/pd_speculative_handoff.h"
#include "core/framework/speculative/speculative_profile_registry.h"
#include "core/layers/common/dsa_topk_share_plan.h"
#include "core/runtime/adaptive_pruning_helpers.h"
#include "core/runtime/speculative_output.h"
#if defined(USE_NPU)
#include "core/runtime/mtp_json_draft_token_handoff.h"
#endif
#include "core/runtime/mtp_verify_input_layout.h"
#include "core/runtime/spec_verify.h"
#include "core/runtime/speculative_input_builder.h"
#include "core/runtime/task_execution_pipeline.h"
#include "models/model_registry.h"
#include "runtime/llm_worker_impl.h"
#include "util/slice.h"
#include "util/tensor_helper.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
namespace {
void broadcast_tokens_in_group(torch::Tensor& tokens,
                               ProcessGroup* process_group,
                               int32_t root_rank = 0) {
  if (process_group == nullptr || process_group->world_size() <= 1 ||
      !tokens.defined()) {
    return;
  }
  tokens = tokens.contiguous();
  process_group->broadcast(tokens, root_rank);
}

// Orthogonal CP x TP publishes distinct tp_group_ and cp_group_; speculative
// tokens must reach consensus across both axes. Backends with a single group
// (cp_group_ == tp_group_ or no CP) keep their existing single-group behavior.
bool uses_orthogonal_cp_consensus(const ParallelArgs& parallel_args) {
  return parallel_args.cp_size() > 1 && parallel_args.tp_group_ != nullptr &&
         parallel_args.cp_group_ != nullptr &&
         parallel_args.cp_group_ != parallel_args.tp_group_;
}

constexpr int64_t kMaxSpecVerifyGraphUpdateBlockTableWidth = (1 << 15) - 1;

// Speculative state is replicated across every model rank within one DP
// replica. With orthogonal CP x TP, tp_group_ covers only one CP shard, while
// cp_group_ connects the same TP rank across CP shards. Broadcast along both
// axes so every replica caches and consumes the same sampled token, without
// crossing into another DP replica through the world group.
void broadcast_spec_tokens(torch::Tensor& tokens,
                           const ParallelArgs& parallel_args) {
  // DeepSeek-V4 TORCH publishes orthogonal TP and CP groups. Other backends
  // retain their existing single-group speculative broadcast behavior.
  ProcessGroup* tp_group = parallel_args.tp_group_ != nullptr
                               ? parallel_args.tp_group_
                               : parallel_args.process_group_;
  if (!uses_orthogonal_cp_consensus(parallel_args)) {
    broadcast_tokens_in_group(tokens, tp_group);
    return;
  }

  // Orthogonal CP x TP guarantees cp_group_ != tp_group_, so both axes need a
  // broadcast to reach the same sampled token across replicas.
  broadcast_tokens_in_group(tokens, parallel_args.tp_group_);
  broadcast_tokens_in_group(tokens, parallel_args.cp_group_);
}

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

// Maps failed sample-sequence ids to row indexes of the current batch.
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

// Unions the sample-sequence ids flagged by input and output grammar errors.
// Both failure paths (draft_input failures and JSON replay mismatches) start
// from this shared collection so the error shape and CHECK policy cannot drift.
std::unordered_set<std::string> failed_sequence_ids_from_errors(
    const std::vector<JsonObjectOutputError>& input_errors,
    const std::vector<JsonObjectOutputError>& output_errors,
    bool require_non_empty_id) {
  std::unordered_set<std::string> failed_ids;
  failed_ids.reserve(input_errors.size() + output_errors.size());
  for (const auto* errors : {&input_errors, &output_errors}) {
    const std::string_view phase = errors == &input_errors ? "input" : "output";
    for (const JsonObjectOutputError& error : *errors) {
      if (require_non_empty_id) {
        CHECK(!error.sample_sequence_id.empty())
            << "MTP JSON " << phase << " error requires a sampled sequence id";
      }
      failed_ids.emplace(error.sample_sequence_id);
    }
  }
  return failed_ids;
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
      failed_sequence_ids_from_errors(input.json_object_errors,
                                      output_errors,
                                      /*require_non_empty_id=*/false));
}

// Replays the JSON grammar check on the host-accepted tokens of the static
// validate tail: unions input and target grammar failures with accepted-token
// replay mismatches, appends a fresh output error for each new mismatch, and
// maps the failed sample-sequence ids to row indexes of the current batch.
template <typename TargetInput>
std::vector<size_t> json_failed_sequence_rows(
    const TargetInput& input,
    const torch::Tensor& accepted_tokens_host,
    const StreamEventPtr& accepted_tokens_ready,
    ForwardOutput& target_output) {
  CHECK_EQ(accepted_tokens_host.dim(), 2)
      << "MTP JSON accepted output must be [sequence, token]";
  CHECK_EQ(input.sample_sequence_ids.size(), input.json_object_states.size())
      << "MTP JSON sequence ids must match grammar state rows";
  CHECK_EQ(input.input_params.embedding.embedding_ids.size(),
           input.json_object_states.size())
      << "MTP JSON embedding ids must match grammar state rows";
  CHECK(accepted_tokens_ready == nullptr ||
        accepted_tokens_ready->synchronize())
      << "failed to wait for accepted MTP tokens before JSON replay";

  const torch::Tensor accepted_tokens =
      to_cpu_contiguous(accepted_tokens_host, torch::kLong);
  const std::vector<detail::JsonAcceptedTokenMismatch> mismatches =
      detail::find_json_accepted_token_mismatches(
          input.json_object_states,
          accepted_tokens.const_data_ptr<int64_t>(),
          static_cast<size_t>(accepted_tokens.size(0)),
          static_cast<size_t>(accepted_tokens.size(1)));
  std::unordered_set<std::string> failed_sequence_ids =
      failed_sequence_ids_from_errors(input.json_object_errors,
                                      target_output.json_object_errors,
                                      /*require_non_empty_id=*/true);

  for (const detail::JsonAcceptedTokenMismatch& mismatch : mismatches) {
    const std::string& sample_sequence_id =
        input.sample_sequence_ids[mismatch.sequence_index];
    CHECK(!sample_sequence_id.empty())
        << "MTP JSON mismatch requires a sampled sequence id";
    LOG(ERROR) << "MTP JSON accepted output replay mismatch: sequence_id="
               << sample_sequence_id
               << ", token_offset=" << mismatch.token_offset
               << ", token_id=" << mismatch.token_id
               << ", committed_tokens=" << mismatch.committed_tokens
               << ", state_fingerprint=" << mismatch.state_fingerprint;
    if (failed_sequence_ids.emplace(sample_sequence_id).second) {
      target_output.json_object_errors.push_back(
          {sample_sequence_id,
           "accepted MTP output violates json_object grammar, token_id=" +
               std::to_string(mismatch.token_id)});
    }
  }
  std::vector<size_t> failed_sequence_rows =
      rows_for_failed_sequence_ids(input, failed_sequence_ids);
  CHECK_EQ(failed_sequence_rows.size(), failed_sequence_ids.size())
      << "MTP JSON errors must reference sampled rows in the current batch";
  return failed_sequence_rows;
}

std::vector<uint8_t> filter_vocabulary_failures(
    std::vector<uint8_t> invalid_rows,
    const std::vector<JsonObjectGrammarState>& json_states) {
  if (invalid_rows.empty() || json_states.empty()) {
    return invalid_rows;
  }
  CHECK_EQ(invalid_rows.size(), json_states.size());
  // JSON validation rejects the remapper's placeholder before acceptance.
  // The target can still sample a legal token outside the hot vocabulary,
  // including EOS after a complete object. Committed-state errors remain
  // request failures; only these unverified draft proposals are recoverable.
  bool any_failed = false;
  for (size_t row = 0; row < invalid_rows.size(); ++row) {
    const auto& state = json_states[row];
    invalid_rows[row] =
        invalid_rows[row] && (!state.initialized() || !state.is_valid());
    any_failed = any_failed || invalid_rows[row] != 0;
  }
  if (!any_failed) {
    invalid_rows.clear();
  }
  return invalid_rows;
}

template <typename TargetInput>
void clear_mla_prefixcache_workspace(TargetInput& input) {
  auto& attention = input.input_params.attention.device;
  attention.history_compressed_kv = torch::Tensor();
  attention.history_k_rope = torch::Tensor();
}

template <typename TargetInput>
std::optional<ForwardOutput> run_worker_no_sync_impl(
    WorkerImpl& worker,
    const TargetInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    TargetInput& processed_input) {
  worker.prepare_work_before_execute_on_stream(
      input,
      processed_input,
      prepare_stream,
      /*record_ready_event=*/&prepare_stream != &compute_stream);
  if constexpr (std::is_same_v<TargetInput, LlmForwardInput>) {
    if (auto* llm_worker = dynamic_cast<LLMWorkerImpl*>(&worker);
        llm_worker != nullptr) {
      return llm_worker->execute_no_sync_on_stream(
          processed_input, compute_stream, /*record_ready_event=*/false);
    }
  }
  return worker.execute_no_sync_on_stream(processed_input, compute_stream);
}

template <typename TargetInput>
void stabilize_decode_host_tensors(TargetInput& input) {
  stabilize_host_views(input);
  for (torch::Tensor& block_table : input.input_params.multi_block_tables) {
    block_table = clone_cpu_contiguous_detached_tensor(block_table);
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
    const std::vector<TargetDecodeStateStore::DecodeState>& states,
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
    const TargetDecodeStateStore::DecodeState& state = states[i];
    const int32_t token_id = token_ids[i];
    CHECK(state.valid) << "MTP decode missing target state, request_id="
                       << request_ids[i];
    CHECK_EQ(state.request_id, request_ids[i])
        << "MTP decode target state request mismatch";
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
  CHECK(replacements.defined())
      << "speculative replacement tokens must be defined";
  CHECK(input.token_ids_host.defined())
      << "token_ids_host must be defined before speculative token update";
  CHECK(input.token_ids_host.device().is_cpu())
      << "token_ids_host must stay on CPU";
  CHECK_EQ(input.token_ids_host.scalar_type(), torch::kInt)
      << "token_ids_host must be int32";

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

// Chunked-prefill verify normalization: clamps the decode position to the
// block-table capacity (leaving room for one validate row) and resolves the
// scheduler's KV-length lag or lead against that position.
struct ChunkedPrefillRowClamp {
  int32_t logical_block_size;
  int32_t validate_width;
};

void normalize_chunked_prefill_row(const ChunkedPrefillRowClamp& clamp,
                                   const torch::Tensor& block_tables,
                                   int32_t seq_id,
                                   int32_t& current_position,
                                   int32_t& current_kv_len,
                                   int32_t& expected_kv_len) {
  if (block_tables.defined() && block_tables.dim() == 2 &&
      seq_id < block_tables.size(0)) {
    const int32_t allocated_kv_len =
        static_cast<int32_t>(block_tables.size(1)) * clamp.logical_block_size;
    const int32_t max_valid_position = allocated_kv_len - clamp.validate_width;
    if (current_position > max_valid_position) {
      CHECK_GT(allocated_kv_len, 0)
          << "decode context has empty block table, seq_id=" << seq_id;
      CHECK_GE(max_valid_position, 0)
          << "decode context block table is too small for validation, "
          << "seq_id=" << seq_id << ", allocated_kv_len=" << allocated_kv_len
          << ", validate_width=" << clamp.validate_width;
      CHECK_LE(current_position - max_valid_position, clamp.validate_width)
          << "decode context position exceeds allocated blocks, seq_id="
          << seq_id << ", current_position=" << current_position
          << ", current_kv_len=" << current_kv_len
          << ", allocated_kv_len=" << allocated_kv_len;
      current_position = max_valid_position;
      expected_kv_len = current_position + 1;
      current_kv_len = std::min(current_kv_len, expected_kv_len);
    }
  }
  if (current_kv_len < expected_kv_len) {
    // Qwen3.5/MiMo MTP can receive a scheduler KV length that has not yet
    // caught up with the speculative placeholder resolved into
    // current_position. Normalize only the lag explainable by the current
    // speculative step.
    CHECK_LE(expected_kv_len - current_kv_len, clamp.validate_width)
        << "decode context kv_len lag is too large, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;
    current_kv_len = expected_kv_len;
  } else if (current_kv_len > expected_kv_len) {
    // The first decode step can carry the prompt KV length while the decode
    // position is still initialized to zero. Align the position to the KV
    // context before building the MTP draft input.
    current_position = current_kv_len - 1;
    expected_kv_len = current_kv_len;
  }
}

}  // namespace

// Same-stream dispatch unless the draft is an embedded Eagle3 head, which
// needs the prepare stream for its MLA workspace and post-forward sync.
template <typename TargetInput>
std::optional<ForwardOutput>
MTPWorkerImpl<TargetInput>::dispatch_worker_on_compute(
    WorkerImpl& worker,
    const TargetInput& input,
    TargetInput& processed_input) {
  if (uses_embedded_eagle3_draft()) {
    return run_worker_no_sync(worker, input, processed_input);
  }
  // The colocated drafter is always an LLM worker: convert VLM target inputs
  // at this boundary so the draft forward stays LlmForwardInput-typed.
  if constexpr (std::is_same_v<TargetInput, VlmForwardInput>) {
    LlmForwardInput draft_input = make_llm_draft_input(input);
    LlmForwardInput processed;
    return run_worker_no_sync_impl(
        worker, draft_input, *compute_stream_, *compute_stream_, processed);
  } else {
    return run_worker_no_sync_impl(
        worker, input, *compute_stream_, *compute_stream_, processed_input);
  }
}

using adaptive_pruning::apply_pruned_prefix_lengths;
using adaptive_pruning::has_selected_probs_by_step;
using adaptive_pruning::selected_probs_by_step;
using adaptive_pruning::truncate_draft_outputs;

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
                                 worker_type,
                                 /*enable_adaptive_speculative_decode=*/true) {}

template <typename TargetInput>
MTPWorkerImpl<TargetInput>::MTPWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    const runtime::Options& target_options,
    const runtime::Options& draft_options,
    WorkerType worker_type,
    bool enable_adaptive_speculative_decode)
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
          }) {
  if (enable_adaptive_speculative_decode) {
    adaptive_spec_controller_ =
        std::make_unique<AdaptiveSpeculativeController>(options);
  }
}

template <typename TargetInput>
MTPWorkerImpl<TargetInput>::~MTPWorkerImpl() {
  drain_pending_execution();
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::drain_pending_execution() {
  if (execution_drained_) {
    return;
  }
  folly::Promise<folly::Unit> barrier;
  auto future = barrier.getFuture();
  threadpool_.schedule([barrier = std::move(barrier)]() mutable {
    barrier.setValue(folly::unit);
  });
  std::move(future).get();
  // Destruction may run on an RPC thread with a different current device.
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  CHECK_EQ(compute_stream_->synchronize(), 0);
  CHECK_EQ(prepare_stream_->synchronize(), 0);
  execution_drained_ = true;
}

template <typename TargetInput>
::xllm::Status MTPWorkerImpl<TargetInput>::create_task_pipeline(
    std::unique_ptr<TaskExecutionPipeline>& output) {
  if (impl_ == nullptr || draft_impl_ == nullptr ||
      options_.enable_adaptive_speculative_decode() ||
      options_.num_speculative_tokens() <= 0 ||
      draft_impl_->context_.get_model_args().model_type() !=
          "glm_moe_dsa_mtp") {
    return ::xllm::Status(
        StatusCode::INVALID_ARGUMENT,
        "MTP task pipeline requires fixed-width GLM DSA Target/Draft models.");
  }
  SpeculativeTaskCapacity capacity;
  ::xllm::Status status = impl_->task_capacity(options_, capacity.common);
  if (!status.ok()) {
    return status;
  }
  const auto& args = draft_impl_->context_.get_model_args();
  capacity.num_speculative_tokens = options_.num_speculative_tokens();
  const int64_t draft_input_hidden_width = this->draft_input_spec().hidden_size;
  CHECK_GT(draft_input_hidden_width, 0)
      << "MTP draft input hidden width must be positive";
  CHECK_LE(draft_input_hidden_width, std::numeric_limits<int32_t>::max())
      << "MTP draft input hidden width exceeds int32 range";
  capacity.draft_input_hidden_size =
      static_cast<uint32_t>(draft_input_hidden_width);
  capacity.index_topk = args.index_topk();
  capacity.reuse_topk = args.index_share_for_mtp_iteration();
  if (this->get_optimization_config().enable_spec_token_broadcast) {
    capacity.common.sampling_group = parallel_args_.tp_group_ != nullptr
                                         ? parallel_args_.tp_group_
                                         : parallel_args_.process_group_;
  }
  capacity.draft_sampling_mode = draft_sampling_mode_;
  capacity.export_pd_draft_input =
      options_.instance_role() == InstanceRole::PREFILL;
  status = TaskExecutionPipeline::create(threadpool_,
                                         impl_->task_model(),
                                         draft_impl_->task_model(),
                                         capacity,
                                         output);
  if (!status.ok()) {
    return status;
  }
  LOG(INFO) << "MTP task execution pipeline: slots="
            << capacity.common.slot_count
            << ", speculative_tokens=" << capacity.num_speculative_tokens
            << ", pinned_bytes=" << output->pinned_bytes()
            << ", device_bytes=" << output->device_bytes();
  return ::xllm::Status();
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
    draft_is_embedded_eagle3_ = xllm::uses_embedded_eagle3_draft(
        options_.speculative_algorithm(), context_.get_model_args());
    // The uniform-expanded verify path has no adaptive-controller host point.
    if (target_verify_capabilities_.requires_uniform_validate_width) {
      adaptive_spec_controller_.reset();
    }
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
#if defined(USE_NPU)
        if (::xllm::KernelConfig::get_instance().npu_kernel_backend() !=
            "TORCH") {
          if (share_lm_head) {
            auto head = impl_->get_npu_lm_head();
            draft_impl_->set_npu_lm_head(head);
          }
          auto word_embedding = impl_->get_npu_word_embedding();
          if (impl_->has_restored_npu_word_embedding()) {
            draft_impl_->set_restored_npu_word_embedding(word_embedding);
          } else {
            draft_impl_->set_npu_word_embedding(word_embedding);
          }
        } else {
          if (share_lm_head) {
            auto head = impl_->get_lm_head();
            draft_impl_->set_lm_head(head);
          }
          auto word_embedding = impl_->get_word_embedding();
          draft_impl_->set_word_embedding(word_embedding);
        }
#else
        if (share_lm_head) {
          auto head = impl_->get_lm_head();
          draft_impl_->set_lm_head(head);
        }
        auto word_embedding = impl_->get_word_embedding();
        draft_impl_->set_word_embedding(word_embedding);
#endif
      }
    }
  }
#if defined(USE_NPU)
  if (result && supports_explicit_spec_verify_replay_update()) {
    CHECK_EQ(::xllm::KernelConfig::get_instance().npu_kernel_backend(), "TORCH")
        << "Qwen3.5 MTP only supports NPU Torch backend";
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
  if (draft_is_embedded_eagle3_) {
    return this->build_draft_kv_cache_shape(target_kv_cache_shape);
  }
  if (options_.enable_mtp_draft_body_tp1()) {
    return this->build_draft_kv_cache_shape(target_kv_cache_shape,
                                            /*draft_world_size=*/1);
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
bool MTPWorkerImpl<TargetInput>::uses_embedded_eagle3_draft() const {
  return draft_is_embedded_eagle3_;
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::synchronize_embedded_eagle3_forward() {
#if defined(USE_NPU)
  if (uses_embedded_eagle3_draft()) {
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize embedded Eagle3 forward, ret="
                     << ret;
  }
#endif
}

template <typename TargetInput>
std::optional<ForwardOutput> MTPWorkerImpl<TargetInput>::run_worker_no_sync(
    WorkerImpl& worker,
    const TargetInput& input,
    TargetInput& processed_input) {
  std::optional<ForwardOutput> output = run_worker_no_sync_impl(
      worker, input, *prepare_stream_, *compute_stream_, processed_input);
  synchronize_embedded_eagle3_forward();
  if (uses_embedded_eagle3_draft()) {
    clear_mla_prefixcache_workspace(processed_input);
  }
  return output;
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::supports_explicit_spec_verify_replay_update()
    const {
  return target_verify_capabilities_.supports_explicit_replay_update;
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::requires_uniform_validate_width() const {
  // Currently only Qwen3.5's GDN spec-verify kernel requires uniform width;
  // this happens to coincide with the UNIFORM_EXPANDED_VERIFY mode used by
  // supports_explicit_spec_verify_replay_update, but the two are semantically
  // distinct capabilities (graph-update capability vs. per-seq varlen kernel
  // support).
  return target_verify_capabilities_.requires_uniform_validate_width;
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
      !supports_explicit_spec_verify_replay_update() || impl_ == nullptr ||
      options_.num_speculative_tokens() <= 0 || options_.block_size() <= 0 ||
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
  CHECK_GT(options_.block_size(), 0);
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
  return target_verify_capabilities_.uses_chunked_prefill;
}

template <typename TargetInput>
int32_t MTPWorkerImpl<TargetInput>::logical_block_size() const {
  CHECK_GT(options_.block_size(), 0);
  const int32_t kv_split_size = parallel_args_.kv_split_size_effective();
  CHECK_GT(kv_split_size, 0);
  return options_.block_size() * kv_split_size;
}

template <typename TargetInput>
TargetInput MTPWorkerImpl<TargetInput>::update_input_by_last_step_output(
    TargetInput& inputs) {
  return std::move(inputs);
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
        run_worker_no_sync(*impl_, input, target_prepared).value();
    ForwardOutput draft_output =
        run_worker_no_sync(*draft_impl_, input, draft_prepared).value();
    transfer_retained_inputs(output, draft_output);
    clear_all_output_embeddings(output);
    finalize_output_on_stream(
        output, *compute_stream_, this->enable_schedule_overlap());
    return output;
  }

  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  // Recursive drafts run at indices 1..K-1; index 0 reuses the caller's input.
  std::vector<TargetInput> prepared(num_speculative_tokens - 1);
  std::vector<ForwardOutput> draft_outputs;
  draft_outputs.reserve(num_speculative_tokens - 1);
  // Idle DP ranks must enter the same model collectives as active peers:
  // K-1 recursive drafts, target verification, then one target-conditioned
  // extend.
  for (int32_t draft_idx = 1; draft_idx < num_speculative_tokens; ++draft_idx) {
    draft_outputs.emplace_back(
        run_worker_no_sync(*draft_impl_, input, prepared[draft_idx - 1])
            .value());
  }

  TargetInput expanded = input.clone();
  const int32_t width = num_speculative_tokens + 1;
  scale_speculative_parallel_token_counts(expanded.input_params,
                                          /*multiplier=*/width);
  this->sync_dp_global_token_nums_for_idle_rank(expanded.input_params);
  TargetInput target_prepared;
  ForwardOutput output =
      run_worker_no_sync(*impl_, expanded, target_prepared).value();

  expanded.input_params.clear_linear_attention_state();
  expanded.input_params.is_spec_verify = false;
  expanded.input_params.embedding.input_embedding =
      output.sample_output.embeddings;
  if (::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel() ||
      use_chunked_prefill_spec_verify_path()) {
    expanded.input_params.meta.batch_forward_type =
        BatchForwardType::CHUNKED_PREFILL;
  }
  prepare_draft_sampling(expanded.sampling_params);
  TargetInput extend_prepared;
  ForwardOutput extended =
      run_worker_no_sync(*draft_impl_, expanded, extend_prepared).value();
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

  // run the target model to get first token and hidden states
  ForwardOutput output =
      run_worker_no_sync(*impl_, input, target_prepared).value();
  std::vector<size_t> prior_input_failures =
      failed_draft_input_rows(input, output.json_object_errors);
  mark_invalid_draft_input_rows(
      input,
      draft_input_state_store_->failed_draft_input_mask(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids),
      &prior_input_failures,
      output.json_object_errors);
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              target_timer.elapsed_seconds());

  // MTP path that depends on hidden states.
  TargetInput prefill_input;
  prepare_prefill_inputs(input, prefill_input);
  std::optional<DraftInputSamplingState> draft_input_sampling_state;
  torch::Tensor root_tokens_host;
  std::vector<uint8_t> vocabulary_invalid_rows;

  // prepare input for draft model
  auto& embeddings = output.sample_output.embeddings;
  check_draft_input_embedding(embeddings, "prefill");

  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // Target prefill populates the MTP decode cache. Under orthogonal CP x
    // TP each CP shard samples independently unless this token is synchronized
    // across both axes; caching divergent tokens makes the first decode input
    // disagree with the non-driver CP shard's DecodeState.
    maybe_broadcast_spec_tokens(output.sample_output.next_tokens,
                                input.sampling_params.all_greedy_sample);
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
    if (input.sampling_params.selected_token_idxes.defined() &&
        output.sample_output.next_tokens.defined()) {
      draft_input_sampling_state = draft_input_state_store_->prepare(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids,
          input.sampling_params,
          input.json_object_states,
          root_tokens_host.view({-1, 1}),
          device_);
      prefill_input.sampling_params =
          draft_input_sampling_state->sampling_params;
      prepare_draft_sampling(prefill_input.sampling_params);
      prefill_input.json_object_states =
          draft_input_sampling_state->json_object_states;
      vocabulary_invalid_rows =
          compute_vocabulary_invalid_rows(prefill_input.sampling_params,
                                          draft_input_sampling_state->histories,
                                          prefill_input.json_object_states);
    }
    if (embeddings.defined() || output.sample_output.next_tokens.defined()) {
      record_current_metadata_ready_event(prefill_input, *compute_stream_);
    }
  }
  // Generate draft KV and the first proposal for the following decode round.
  Timer draft_timer;
  ForwardOutput draft_output =
      run_worker_no_sync(*draft_impl_, prefill_input, draft_prepared).value();
  std::optional<DraftInput> draft_input;
  if (draft_input_sampling_state.has_value()) {
    draft_input =
        assemble_draft_input(draft_output,
                             draft_prepared.sampling_params,
                             adaptive_enabled() || options_.instance_role() ==
                                                       InstanceRole::PREFILL);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());

  if (input.sampling_params.selected_token_idxes.defined()) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    target_state_store_->write_prefill_target_context(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        root_tokens_host);
    output.sample_output.next_tokens = root_tokens_host;
    CHECK(draft_input_sampling_state.has_value());
    mark_invalid_draft_input_rows(input,
                                  draft_input_sampling_state->invalid_rows,
                                  &prior_input_failures,
                                  output.json_object_errors,
                                  vocabulary_invalid_rows);
    // PREFILL instances export the draft_input through the PD handoff and skip
    // write_draft_input entirely; decode instances store it for the next
    // round. Moving the only copy avoids a [batch, vocab] clone.
    if (options_.instance_role() == InstanceRole::PREFILL) {
      output.sample_output.draft_input = std::move(*draft_input);
    } else {
      write_draft_input(input,
                        std::move(*draft_input),
                        std::move(*draft_input_sampling_state),
                        prior_input_failures);
    }
    stash_draft_input_topk_state(
        input, draft_output.mtp_topk_state, draft_prepared.sampling_params);
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
  prefill_input = input.to(device_, dtype_);
  prepare_draft_sampling(prefill_input.sampling_params);
  clear_ready_events(prefill_input);
  auto& input_params = prefill_input.input_params;
  // The Qwen draft is a pure full-attention model; without this cleanup the
  // target's recurrent slot metadata makes MTP prefill enter a stateful path
  // it has neither a validity mask nor a recurrent cache for.
  // Full-attention KPool drafts still own per-request compression tails.
  if (!draft_impl_->has_request_state_cache()) {
    input_params.clear_linear_attention_state();
  }
  prefill_input.runtime.device_tensors_ready = false;
  prefill_input.token_ids_host = torch::empty({input.token_ids_host.numel()},
                                              pinned_cpu_options(torch::kInt));
  // Packed inputs can materialize row geometry during input.to(); use that
  // geometry with the original host token source.
  specBuilder::fill_mtp_prefill_token_ids(
      input.token_ids_host,
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
void MTPWorkerImpl<TargetInput>::prepare_draft_sampling(
    SamplingParameters& sampling_params) const {
  // Qwen3.5 derives adaptive probabilities from logits instead.
  configure_draft_sampling(
      sampling_params,
      draft_sampling_mode_,
      /*require_probs_for_greedy=*/
      adaptive_enabled() && !supports_explicit_spec_verify_replay_update());
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::release_algorithm_state(
    const std::vector<int32_t>& released) {
  flush_pending_target_context();
  if (draft_input_state_store_ != nullptr) {
    // Single access point: clears the store slots and reclaims any device
    // projection the released rows kept alive, in one call.
    draft_input_state_store_->clear(released);
  }
}

template <typename TargetInput>
std::vector<uint8_t>
MTPWorkerImpl<TargetInput>::compute_vocabulary_invalid_rows(
    SamplingParameters& sampling_params,
    const DraftTokenHistories& histories,
    const std::vector<JsonObjectGrammarState>& json_states) {
  return filter_vocabulary_failures(
      prepare_draft_vocabulary_sampling(sampling_params, histories),
      json_states);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::maybe_broadcast_spec_tokens(
    torch::Tensor& tokens,
    bool all_greedy_sample) {
  // Orthogonal CP needs cross-replica consensus unconditionally; otherwise
  // broadcast only probabilistic sampling (never all-greedy).
  if (uses_orthogonal_cp_consensus(parallel_args_) ||
      (this->get_optimization_config().enable_spec_token_broadcast &&
       !all_greedy_sample)) {
    broadcast_spec_tokens(tokens, parallel_args_);
  }
}

template <typename TargetInput>
DraftInput MTPWorkerImpl<TargetInput>::assemble_draft_input(
    ForwardOutput& output,
    const SamplingParameters& sampling_params,
    bool require_confidence) {
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  SampleOutput& sample = output.sample_output;
  CHECK(sample.next_tokens.defined());
  CHECK(sample.embeddings.defined());
  maybe_broadcast_spec_tokens(sample.next_tokens,
                              sampling_params.all_greedy_sample);

  // Confidence is distinct from q and is selected before vocabulary remapping.
  // Local fixed-width decoding does not need a full-vocabulary reduction.
  const torch::Tensor selected_probs =
      require_confidence
          ? select_input_confidence(
                sample, output.logits, sampling_params.sample_idxes)
          : torch::Tensor();
  // Keep selected confidence above, but never confuse an all-greedy
  // sampler's selected probability with the proposal distribution.
  const bool probabilistic =
      draft_sampling_mode_ == DraftSamplingMode::PROBABILISTIC;
  // The remapper (EAGLE3) expands q to the target vocabulary; it drops
  // proposals that cannot use q before that allocation. All paths get the
  // single authoritative drop from finalize_mixed_proposal_q afterwards.
  remap_draft_sample_output(sample);
  return xllm::finish_draft_input(sample,
                                  sample.embeddings,
                                  sampling_params,
                                  probabilistic,
                                  std::move(selected_probs),
                                  context_.get_model_args().vocab_size(),
                                  this->draft_input_spec());
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::mark_invalid_draft_input_rows(
    const TargetInput& input,
    const std::vector<uint8_t>& invalid_rows,
    std::vector<size_t>* failed_rows,
    std::vector<JsonObjectOutputError>& errors,
    const std::vector<uint8_t>& additional_invalid_rows) const {
  if (invalid_rows.empty() && additional_invalid_rows.empty()) {
    return;
  }
  const size_t batch_size = input.input_params.embedding.embedding_ids.size();
  CHECK(invalid_rows.empty() || invalid_rows.size() == batch_size);
  CHECK(additional_invalid_rows.empty() ||
        additional_invalid_rows.size() == batch_size);
  std::unordered_set<size_t> already_failed;
  if (failed_rows != nullptr) {
    already_failed.insert(failed_rows->begin(), failed_rows->end());
  }
  for (size_t row = 0; row < batch_size; ++row) {
    const bool invalid =
        (!invalid_rows.empty() && invalid_rows[row] != 0) ||
        (!additional_invalid_rows.empty() && additional_invalid_rows[row] != 0);
    if (!invalid || !already_failed.insert(row).second) {
      continue;
    }
    CHECK_LT(row, input.sample_sequence_ids.size());
    if (failed_rows != nullptr) {
      failed_rows->emplace_back(row);
    }
    errors.push_back(
        {input.sample_sequence_ids[row],
         "MTP draft input cannot advance committed sampling state"});
  }
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::write_draft_input(
    const TargetInput& input,
    DraftInput draft_input,
    DraftInputSamplingState sampling,
    const std::vector<size_t>& failed_rows) {
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  draft_input_state_store_->write_draft_input(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      std::move(draft_input),
      std::move(sampling.histories),
      std::move(sampling.json_object_states),
      failed_rows);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::stash_draft_input_topk_state(
    const TargetInput& input,
    const MtpTopkStatePtr& raw_topk_state,
    const SamplingParameters& sampling_params) {
  // Only DSA top-k reuse consumes this snapshot; skip the batch-sized
  // embedding-id / request-id copies for models without it.
  if (!layer::is_mtp_dsa_topk_reuse_enabled(
          draft_impl_->context_.get_model_args())) {
    return;
  }
  draft_input_state_store_->publish_topk_state(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      specBuilder::select_mtp_topk_state_for_next_step(raw_topk_state,
                                                       sampling_params));
}

template <typename TargetInput>
DraftTokenHistories
MTPWorkerImpl<TargetInput>::restore_draft_input_sampling_history(
    TargetInput& input) {
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  return draft_input_state_store_->restore(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      input.sampling_params,
      input.json_object_states,
      device_);
}

template <typename TargetInput>
ForwardOutput MTPWorkerImpl<TargetInput>::run_draft_input_extend(
    const TargetInput& input,
    const TargetInput& validate_input,
    const SampleOutput& accepted_output,
    const torch::Tensor& accepted_tokens_host,
    const StreamEventPtr& accepted_tokens_ready,
    const std::vector<int32_t>& per_seq_val_tokens,
    std::vector<size_t>& failed_rows,
    ForwardOutput& target_output) {
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  // Callers reach here through run_validate, which re-records
  // validate_input's metadata event on compute_stream itself; FIFO ordering
  // already orders this work after it.
  const bool needs_host_tokens =
      needs_committed_tokens(input.sampling_params, input.json_object_states);
  if (needs_host_tokens) {
    CHECK(accepted_tokens_host.defined() &&
          accepted_tokens_host.device().is_cpu());
    CHECK(accepted_tokens_ready == nullptr ||
          accepted_tokens_ready->synchronize())
        << "failed to wait for accepted tokens before draft_input sampling";
  }
  // The verifier already owns a host handoff for target bookkeeping. Reuse it
  // for penalties/grammar instead of transferring the same device tokens again.
  DraftInputSamplingState sampling = draft_input_state_store_->prepare(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      input.sampling_params,
      input.json_object_states,
      accepted_tokens_host,
      device_);
  const specBuilder::MtpDraftExtendRows rows =
      specBuilder::build_mtp_draft_extend_rows(accepted_output.next_tokens,
                                               accepted_output.embeddings,
                                               per_seq_val_tokens,
                                               uses_embedded_eagle3_draft());

  TargetInput draft_extend_input;
  {
    // Snapshot on the stream that will next rewrite the target's stable
    // metadata. FIFO ordering prevents even the snapshot copy from racing a
    // following prepare, while the draft consumes independent storage.
    c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
    prepare_stream_->wait_stream(*compute_stream_);
    draft_extend_input = specBuilder::build_mtp_draft_extend_input(
        validate_input, rows, sampling.sampling_params);
    record_metadata_ready_event(*prepare_stream_, draft_extend_input);
  }
  prepare_draft_sampling(draft_extend_input.sampling_params);
  draft_extend_input.json_object_states = sampling.json_object_states;
  const std::vector<uint8_t> vocabulary_invalid_rows =
      compute_vocabulary_invalid_rows(draft_extend_input.sampling_params,
                                      sampling.histories,
                                      draft_extend_input.json_object_states);
  mark_invalid_draft_input_rows(input,
                                sampling.invalid_rows,
                                &failed_rows,
                                target_output.json_object_errors,
                                vocabulary_invalid_rows);
  record_current_metadata_ready_event(draft_extend_input, *compute_stream_);

  TargetInput prepared;
  Timer timer;
  ForwardOutput output =
      run_worker_no_sync(*draft_impl_, draft_extend_input, prepared).value();
  DraftInput draft_input = assemble_draft_input(
      output, prepared.sampling_params, adaptive_enabled());
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  write_draft_input(
      input, std::move(draft_input), std::move(sampling), failed_rows);
  stash_draft_input_topk_state(
      input, output.mtp_topk_state, prepared.sampling_params);

  return output;
}

template <typename TargetInput>
DraftInput MTPWorkerImpl<TargetInput>::build_warmup_draft_input(
    int64_t batch_size) const {
  return xllm::make_warmup_draft_input(
      batch_size,
      this->draft_input_spec().hidden_size,
      torch::dtype(draft_impl_->dtype()).device(device_),
      /*probabilistic=*/false);
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
    DraftInput imported = import_pd_draft_input(
        *request_state.draft_input, this->draft_input_spec(), device_);
    CHECK_EQ(imported.token_ids.size(0),
             static_cast<int64_t>(request_state.draft_input_row_idxes.size()));
    target_state_store_->write_target_bootstrap_rows(
        request_state.draft_input_row_idxes,
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        input.token_ids_host);
    draft_input_state_store_->import_carried_draft_input(
        request_state.draft_input_row_idxes,
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        std::move(imported));
    request_state.draft_input.reset();
    request_state.draft_input_row_idxes.clear();
  }
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
  CHECK_GT(num_speculative_tokens, 0);
  CHECK(target_state_store_ != nullptr);
  CHECK(draft_input_state_store_ != nullptr);

  import_carried_draft_input(input);

  const std::vector<TargetDecodeStateStore::DecodeState> last_states =
      target_state_store_->read_decode_states(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids);
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
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // take_draft_input resolves the failure mask in the same pass.
    std::vector<uint8_t> failed_rows;
    draft_input = draft_input_state_store_->take_draft_input(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        &failed_rows);
    if (!failed_rows.empty()) {
      std::vector<size_t> prior_failed_rows =
          failed_draft_input_rows(input, {});
      mark_invalid_draft_input_rows(
          input, failed_rows, &prior_failed_rows, input.json_object_errors);
    }
    if (!draft_input.has_value() && input.input_params.meta.is_graph_warmup) {
      draft_input =
          build_warmup_draft_input(input.input_params.meta.num_sequences);
    }
  }
  CHECK(draft_input.has_value())
      << "MTP decode requires a matching prefill/extend draft input";
  // The draft_input was already validated where it was produced or imported
  // (assemble_draft_input / PD import / warmup), so no re-check here.
  check_draft_input_embedding(draft_input->hidden_states, "decode");

  const bool use_adaptive_speculative_decode =
      adaptive_enabled() &&
      SpeculativeProfileRegistry::get_instance()
          .has_validate_time_predictor() &&
      input.json_object_states.empty();
  const bool has_json_object_states = !input.json_object_states.empty();
  detail::JsonDraftValidationScratch json_scratch;
  std::vector<uint8_t> json_invalid_suffix;
  if (has_json_object_states) {
    json_invalid_suffix.assign(input.json_object_states.size(), 0);
    json_scratch.states_after.reserve(num_speculative_tokens);
    json_scratch.invalid_draft_step_major.reserve(
        input.json_object_states.size() * num_speculative_tokens);
  }

  // Only sampling_params and json_object_states are read from this before it
  // is reassigned to next_step_input each round, so copy just those two and
  // skip a per-step deep copy of the whole ModelInputParams.
  TargetInput current_draft_input;
  current_draft_input.sampling_params = input.sampling_params;
  current_draft_input.json_object_states = input.json_object_states;
  prepare_draft_sampling(current_draft_input.sampling_params);
  TargetInput validate_input;
  TargetInput next_step_input;
  std::vector<TargetInput> draft_prepared(num_speculative_tokens);
  std::vector<ForwardOutput> draft_outputs;
  draft_outputs.reserve(num_speculative_tokens);
  const bool reuse_mtp_topk_state = layer::is_mtp_dsa_topk_reuse_enabled(
      draft_impl_->context_.get_model_args());
  MtpTopkStatePtr mtp_topk_state =
      reuse_mtp_topk_state ? draft_input_state_store_->read_topk_state(
                                 input.input_params.embedding.embedding_ids,
                                 input.input_params.embedding.request_ids)
                           : nullptr;
  Timer draft_timer;

  for (int32_t draft_idx = 0; draft_idx < num_speculative_tokens; ++draft_idx) {
    const bool is_final_draft = draft_idx == num_speculative_tokens - 1;
    const bool static_graph_tasks_prepared =
        is_final_draft &&
        prepare_static_mtp_graph_tasks_before_final_draft(input, last_states);
    ForwardOutput draft_output;
    if (draft_idx == 0) {
      // This proposal was sampled by prefill or the preceding
      // target-conditioned extend. It is already in target token space; never
      // process d2t twice.
      draft_output.sample_output.draft_input = std::move(draft_input);
      const DraftInput& carried = *draft_output.sample_output.draft_input;
      draft_output.sample_output.next_tokens = carried.token_ids;
      draft_output.sample_output.embeddings = carried.hidden_states;
      draft_output.sample_output.probs = carried.probs;
    } else {
      if (reuse_mtp_topk_state) {
        current_draft_input.input_params.mtp_topk_state = mtp_topk_state;
      }
      std::optional<ForwardOutput> output = dispatch_worker_on_compute(
          *draft_impl_, current_draft_input, draft_prepared[draft_idx]);
      CHECK(output.has_value());
      draft_output = std::move(*output);
      draft_output.sample_output.draft_input =
          assemble_draft_input(draft_output,
                               draft_prepared[draft_idx].sampling_params,
                               use_adaptive_speculative_decode);
      {
        c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
        if (reuse_mtp_topk_state) {
          mtp_topk_state = specBuilder::select_mtp_topk_state_for_next_step(
              draft_output.mtp_topk_state,
              draft_prepared[draft_idx].sampling_params);
        }
      }
    }
    draft_outputs.emplace_back(std::move(draft_output));

    if (is_final_draft) {
      prepare_validate_inputs_impl(input,
                                   validate_input,
                                   /*per_seq_val_tokens=*/{},
                                   static_graph_tasks_prepared);
    } else {
      // The draft_input already processed the root at source position root-1.
      // The first recursive proposal therefore runs at root, not root+1.
      prepare_draft_inputs(input, next_step_input, draft_idx);
    }

    if (has_json_object_states) {
      const torch::Tensor& next_tokens =
          draft_outputs.back().sample_output.next_tokens;
      std::vector<int32_t> draft_token_ids;
#if defined(USE_NPU)
      if (json_draft_token_handoff_ == nullptr) {
        json_draft_token_handoff_ =
            std::make_unique<detail::NpuJsonDraftTokenHandoff>(
                static_cast<int64_t>(options_.max_seqs_per_batch()),
                num_speculative_tokens);
      }
      draft_token_ids = json_draft_token_handoff_->read_tokens(
          next_tokens, *compute_stream_, draft_idx);
#else
      draft_token_ids = detail::read_json_draft_tokens_host(next_tokens);
#endif
      (void)detail::append_json_draft_step(
          current_draft_input.json_object_states,
          json_invalid_suffix,
          draft_token_ids,
          json_scratch);
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
      // Invalid suffixes remain rejected by the verifier, but every DP rank
      // must still execute the same number of draft model collectives.
      apply_draft_grammar_bitmask(current_draft_input.sampling_params,
                                  current_draft_input.json_object_states,
                                  device_);
    }
    const std::vector<uint8_t> vocabulary_invalid_rows =
        compute_vocabulary_invalid_rows(current_draft_input.sampling_params,
                                        sampling_histories,
                                        current_draft_input.json_object_states);
    if (!vocabulary_invalid_rows.empty()) {
      mark_invalid_draft_input_rows(input,
                                    vocabulary_invalid_rows,
                                    /*failed_rows=*/nullptr,
                                    input.json_object_errors);
    }
    record_current_metadata_ready_event(current_draft_input, *compute_stream_);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());
  if (use_adaptive_speculative_decode) {
    return run_adaptive_validate(input,
                                 draft_outputs,
                                 validate_input,
                                 num_speculative_tokens,
                                 last_states);
  }
  const std::vector<int32_t> per_seq_val_tokens(
      static_cast<size_t>(input.input_params.meta.num_sequences),
      num_speculative_tokens + 1);
  return run_validate(input,
                      draft_outputs,
                      validate_input,
                      num_speculative_tokens,
                      per_seq_val_tokens,
                      /*pruned_prefix_lengths=*/nullptr,
                      has_json_object_states ? &json_scratch : nullptr);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::fill_validate_input_from_draft_outputs(
    const TargetInput& input,
    std::span<const ForwardOutput> draft_outputs,
    TargetInput& validate_input,
    const std::vector<int32_t>& per_seq_val_tokens,
    detail::JsonDraftValidationScratch* json_scratch,
    Stream& compute_stream) {
  CHECK(!per_seq_val_tokens.empty()) << "per_seq_val_tokens must not be empty";
  const int32_t num_sequences = static_cast<int32_t>(per_seq_val_tokens.size());
  const int32_t max_val_tokens =
      *std::max_element(per_seq_val_tokens.begin(), per_seq_val_tokens.end());
  CHECK(validate_input.token_ids.defined())
      << "validate token_ids must be prepared before draft token fill";
  CHECK_EQ(validate_input.token_ids.dim(), 1)
      << "validate token_ids must be flat";

  const torch::TensorOptions token_options = validate_input.token_ids.options();
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  wait_metadata_ready_event(validate_input, compute_stream);

  if (json_scratch != nullptr) {
    CHECK_EQ(input.json_object_states.size(),
             static_cast<size_t>(num_sequences))
        << "JSON grammar state rows must match validation sequences";
    CHECK_EQ(validate_input.token_ids.numel(),
             static_cast<int64_t>(num_sequences) * max_val_tokens)
        << "adaptive MTP validation is not supported for JSON grammar rows";
    const int32_t num_speculative_tokens = max_val_tokens - 1;
    CHECK_EQ(json_scratch->states_after.size(),
             static_cast<size_t>(num_speculative_tokens));
    validate_input.json_object_states = detail::build_json_validation_states(
        input.json_object_states,
        std::move(*json_scratch),
        validate_input.json_object_invalid_draft);
    validate_input.sampling_params.filter_bitmask =
        build_json_object_filter_bitmask(validate_input.json_object_states,
                                         device_,
                                         JsonObjectMaskBuildPhase::TARGET);
    validate_input.sampling_params.filter_mask = torch::Tensor();
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
      kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
          max_val_tokens, options_.block_size());
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
  const int32_t total_val_tokens =
      static_cast<int32_t>(validate_input.token_ids.numel());
  const bool is_uniform = (total_val_tokens == num_sequences * max_val_tokens);

  if (is_uniform) {
    // Fast path: all seqs share the same val_tokens. Stack the draft-token
    // columns once into a contiguous [batch, num_draft_tokens] tensor and
    // issue a single strided slice-copy, instead of one device kernel per
    // draft step. This is the static (non-adaptive) MTP baseline and runs
    // every decode step; launch-bound NPU decode is measurably faster with
    // one op than num_speculative_tokens ops writing the same bytes.
    const int32_t num_draft_tokens = max_val_tokens - 1;
    const bool step_major_validate_layout = uses_embedded_eagle3_draft();
    torch::Tensor validate_token_rows =
        step_major_validate_layout
            ? validate_input.token_ids.view({max_val_tokens, num_sequences})
            : validate_input.token_ids.view({num_sequences, max_val_tokens});
    if (num_draft_tokens > 0) {
      std::vector<torch::Tensor> draft_token_columns;
      draft_token_columns.reserve(static_cast<size_t>(num_draft_tokens));
      for (int32_t i = 0; i < num_draft_tokens; ++i) {
        CHECK(static_cast<size_t>(i) < draft_outputs.size())
            << "draft_outputs index out of range for step " << i;
        const torch::Tensor& next_tokens =
            draft_outputs[static_cast<size_t>(i)].sample_output.next_tokens;
        CHECK(next_tokens.defined())
            << "draft next_tokens must be defined for validate token fill";
        draft_token_columns.push_back(safe_to(
            next_tokens.flatten(), token_options, /*non_blocking=*/true));
      }
      torch::Tensor packed_drafts =
          torch::stack(draft_token_columns, /*dim=*/1);
      if (step_major_validate_layout) {
        validate_token_rows
            .slice(/*dim=*/0, /*start=*/1, /*end=*/max_val_tokens)
            .copy_(packed_drafts.transpose(/*dim0=*/0, /*dim1=*/1),
                   /*non_blocking=*/true);
      } else {
        validate_token_rows
            .slice(/*dim=*/1, /*start=*/1, /*end=*/max_val_tokens)
            .copy_(packed_drafts, /*non_blocking=*/true);
      }
    }
  } else {
    // Slow path: per-seq variable-length, group by draft step.
    std::vector<int64_t> dst_idx_vec;
    std::vector<int64_t> src_idx_vec;
    std::vector<int64_t> step_vec;
    int32_t offset = 0;
    int32_t max_draft_step = -1;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      const int32_t seq_val_tokens =
          per_seq_val_tokens[static_cast<size_t>(seq_id)];
      const int32_t seq_draft_tokens = seq_val_tokens - 1;
      for (int32_t draft_idx = 0; draft_idx < seq_draft_tokens; ++draft_idx) {
        dst_idx_vec.push_back(offset + draft_idx + 1);
        src_idx_vec.push_back(seq_id);
        step_vec.push_back(draft_idx);
        max_draft_step = std::max(max_draft_step, draft_idx);
      }
      offset += seq_val_tokens;
    }

    if (!dst_idx_vec.empty()) {
      // Move all indices to device once (instead of a per-step H2D copy) and
      // select each step's entries on-device via a boolean mask. Stage through
      // pinned host memory: a pageable source silently demotes the copy to a
      // blocking memcpy, which would stall prepare once per decode step.
      const torch::Device& index_device = validate_input.token_ids.device();
      torch::Tensor dst_idx_all = async_h2d_tensor(dst_idx_vec, index_device);
      torch::Tensor src_idx_all = async_h2d_tensor(src_idx_vec, index_device);
      torch::Tensor step_all = async_h2d_tensor(step_vec, index_device);
      for (int32_t step = 0; step <= max_draft_step; ++step) {
        CHECK(static_cast<size_t>(step) < draft_outputs.size())
            << "draft_outputs index out of range for step " << step;
        const torch::Tensor& next_tokens =
            draft_outputs[static_cast<size_t>(step)].sample_output.next_tokens;
        CHECK(next_tokens.defined())
            << "draft next_tokens must be defined for validate token fill";
        torch::Tensor flat_tokens = safe_to(
            next_tokens.flatten(), token_options, /*non_blocking=*/true);

        torch::Tensor step_mask = step_all.eq(step);
        torch::Tensor step_dst = dst_idx_all.masked_select(step_mask);
        if (step_dst.numel() == 0) {
          continue;
        }
        torch::Tensor step_src = src_idx_all.masked_select(step_mask);
        torch::Tensor gathered = flat_tokens.index_select(/*dim=*/0, step_src);
        validate_input.token_ids.index_copy_(/*dim=*/0, step_dst, gathered);
      }
    }
  }
  validate_input.runtime.device_tensors_ready = true;
  record_metadata_ready_event(compute_stream, validate_input);
}

template <typename TargetInput>
std::optional<ForwardOutput> MTPWorkerImpl<TargetInput>::run_adaptive_validate(
    const TargetInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    TargetInput& validate_input,
    int32_t num_speculative_tokens,
    const std::vector<TargetDecodeStateStore::DecodeState>& last_states) {
  const int32_t batch_size = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(batch_size));
  // Single static-tail exit point: pruned_prefix_lengths=nullptr makes
  // run_validate take the async handoff tail with no pruned post-processing.
  const std::vector<int32_t> uniform_val_tokens(static_cast<size_t>(batch_size),
                                                num_speculative_tokens + 1);
  const adaptive_pruning::MtpValidatePlan plan =
      adaptive_pruning::plan_mtp_validate(
          last_states,
          num_speculative_tokens,
          supports_explicit_spec_verify_replay_update(),
          requires_uniform_validate_width(),
          [&]() {
            // All ranks use deterministic inputs. Measured draft latency is
            // rank-local, so only validate cost drives the controller.
            if (has_selected_probs_by_step(draft_outputs)) {
              return adaptive_spec_controller_->select_pruned_prefix_lengths(
                  selected_probs_by_step(draft_outputs),
                  /*full_draft_time_ms=*/0.0,
                  adaptive_pruning::controller_kv_lens(input, batch_size));
            }
            LOG_FIRST_N(WARNING, 1)
                << "Adaptive speculative pruning disabled: draft exposed "
                   "neither probs nor logits. Falling back to full width.";
            return std::vector<int32_t>(static_cast<size_t>(batch_size),
                                        num_speculative_tokens);
          });
  if (plan.use_static_tail) {
    return run_validate(input,
                        draft_outputs,
                        validate_input,
                        num_speculative_tokens,
                        uniform_val_tokens,
                        /*pruned_prefix_lengths=*/nullptr,
                        /*json_scratch=*/nullptr);
  }
  const std::span<const ForwardOutput> pruned_draft_outputs =
      truncate_draft_outputs(draft_outputs, plan.effective_speculative_tokens);
  prepare_validate_inputs_impl(input, validate_input, plan.per_seq_val_tokens);
  return run_validate(input,
                      pruned_draft_outputs,
                      validate_input,
                      plan.effective_speculative_tokens,
                      plan.per_seq_val_tokens,
                      &plan.prefix_lengths,
                      /*json_scratch=*/nullptr);
}

// Run target model validate with per-seq variable-length support.
// When all seqs have the same val_tokens (uniform case), uses zero-copy view.
// When variable-length, pads logits to [batch * max_val_tokens, vocab] for
// RejectionSampler compatibility, with -inf at padding positions.
template <typename TargetInput>
std::optional<ForwardOutput> MTPWorkerImpl<TargetInput>::run_validate(
    const TargetInput& input,
    std::span<const ForwardOutput> draft_outputs,
    TargetInput& validate_input,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>& per_seq_val_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths,
    detail::JsonDraftValidationScratch* json_scratch) {
  Timer target_timer;
  TargetInput target_prepared;
  fill_validate_input_from_draft_outputs(input,
                                         draft_outputs,
                                         validate_input,
                                         per_seq_val_tokens,
                                         json_scratch,
                                         *compute_stream_);
  // Under DP, publish this rank's true validate token count to all DP peers so
  // DpEpPadding computes matching MoE all-to-all pads. per_seq_val_tokens is
  // uniform (N+1) on the static path and varlen on the adaptive path; both
  // funnel through here, so every DP rank runs the collective in lockstep.
  // No-op when the DP group spans a single rank.
  int32_t local_total_val_tokens = 0;
  for (int32_t v : per_seq_val_tokens) {
    local_total_val_tokens += v;
  }
  this->sync_dp_global_token_nums_after_prune(validate_input.input_params,
                                              local_total_val_tokens);
  ForwardOutput target_output =
      dispatch_worker_on_compute(*impl_, validate_input, target_prepared)
          .value();
  target_output.json_object_errors.insert(
      target_output.json_object_errors.end(),
      input.json_object_errors.begin(),
      input.json_object_errors.end());
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              target_timer.elapsed_seconds());

  const int32_t batch_size = static_cast<int32_t>(per_seq_val_tokens.size());
  const int32_t max_val_tokens = num_speculative_tokens + 1;
  const int32_t total_tokens =
      static_cast<int32_t>(target_output.logits.size(0));
  const int64_t padded_total =
      static_cast<int64_t>(batch_size) * max_val_tokens;

  // Uniform logits are already flat: view in place to [padded_total, vocab]
  // (validate() applies its own layout view on top). Only the variable-length
  // path needs the padded copy.
  std::optional<ForwardOutput> padded_target_output_slow;
  const bool needs_padding =
      (total_tokens != static_cast<int32_t>(padded_total));
  if (needs_padding) {
    padded_target_output_slow.emplace(target_output);
    adaptive_pruning::scatter_varlen_target_output_to_dense(
        *padded_target_output_slow,
        per_seq_val_tokens,
        batch_size,
        max_val_tokens);
  } else {
    target_output.logits = target_output.logits.view(
        {padded_total, target_output.logits.size(-1)});
  }
  const ForwardOutput& target_output_for_validate =
      needs_padding ? *padded_target_output_slow : target_output;

  // verify the proposals with target and update the batch
  SampleOutput val_output;
  {
    AUTO_COUNTER(speculative_execution_latency_seconds_validation);
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    val_output = validate(input.sampling_params,
                          draft_outputs,
                          target_output_for_validate,
                          num_speculative_tokens,
                          pruned_prefix_lengths,
                          validate_input.json_object_invalid_draft);
  }

  if (pruned_prefix_lengths != nullptr || uses_embedded_eagle3_draft()) {
    // Adaptive pruning path: per-seq validate width is variable, which is
    // incompatible with the async handoff's fixed-width base-state derivation.
    // Use the synchronous tail: unify tokens, then write target context inline.
    {
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      maybe_broadcast_spec_tokens(val_output.next_tokens,
                                  input.sampling_params.all_greedy_sample);
    }

    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize MTP compute stream, ret=" << ret;
    target_output.retained_inputs.clear();
    val_output.next_tokens = val_output.next_tokens.to(torch::kCPU);
    target_state_store_->write_target_context(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        val_output.next_tokens);
    std::vector<size_t> failed_rows =
        failed_draft_input_rows(input, target_output.json_object_errors);
    ForwardOutput extended = run_draft_input_extend(input,
                                                    validate_input,
                                                    val_output,
                                                    val_output.next_tokens,
                                                    nullptr,
                                                    per_seq_val_tokens,
                                                    failed_rows,
                                                    target_output);
    transfer_retained_inputs(target_output, extended);
    finalize_output_on_stream(
        target_output, *compute_stream_, this->enable_schedule_overlap());
    if (!this->publishes_step_output()) {
      return std::nullopt;
    }
    return finalize_verify_output(std::move(target_output),
                                  std::move(val_output),
                                  pruned_prefix_lengths,
                                  input.json_object_states);
  }

  torch::Tensor validate_positions = validate_input.positions;
  CHECK_EQ(validate_positions.numel(), batch_size * max_val_tokens)
      << "validate positions must contain one row per speculative token";
  const torch::Tensor& validate_kv_seq_lens =
      validate_input.input_params.attention.device.kv_seq_lens;
  CHECK_GE(validate_kv_seq_lens.numel(), batch_size)
      << "validate KV lengths must be sequence-scoped";
  const int64_t configured_capacity =
      static_cast<int64_t>(options_.max_seqs_per_batch() * max_val_tokens);
  torch::Tensor accepted_tokens_host =
      target_context_handoff_.acquire_host_buffer(val_output.next_tokens,
                                                  configured_capacity);
  StreamEventPtr target_context_ready_event;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

    // Catch-all for cross-rank RNG divergence: unify accepted tokens before
    // deriving any device-resident state used by the next draft iteration.
    maybe_broadcast_spec_tokens(val_output.next_tokens,
                                input.sampling_params.all_greedy_sample);

    accepted_tokens_host.copy_(val_output.next_tokens,
                               /*non_blocking=*/true);
    // The event covers both the token consensus and the D2H copy.
    target_context_ready_event = compute_stream_->record_event();
  }
  if (target_context_ready_event == nullptr) {
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize MTP target context, ret=" << ret;
  }
  std::vector<size_t> failed_sequence_rows;
  if (!input.json_object_states.empty()) {
    failed_sequence_rows = json_failed_sequence_rows(
        input, accepted_tokens_host, target_context_ready_event, target_output);
  } else {
    failed_sequence_rows =
        failed_draft_input_rows(input, target_output.json_object_errors);
  }
  ForwardOutput extended = run_draft_input_extend(input,
                                                  validate_input,
                                                  val_output,
                                                  accepted_tokens_host,
                                                  target_context_ready_event,
                                                  per_seq_val_tokens,
                                                  failed_sequence_rows,
                                                  target_output);
  transfer_retained_inputs(target_output, extended);
  // Overlap mode hands the verify tokens back still on device (the device copy
  // feeds the next step's input update); expose the staged host snapshot so
  // metrics and serialization reuse it instead of issuing a second D2H.
  if (enable_schedule_overlap()) {
    target_output.next_tokens_host = accepted_tokens_host;
  }
  target_context_handoff_.stage(
      input, target_context_ready_event, accepted_tokens_host);
  record_output_ready_event(target_output, *compute_stream_);

  // Target validation consumes all draft outputs on the same compute stream;
  // keep their prepared inputs alive on target_output until the target-context
  // event completes. Copy (not move): draft_outputs live beyond this loop.
  for (const ForwardOutput& draft_output : draft_outputs) {
    copy_retained_inputs(target_output, draft_output);
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
                                /*pruned_prefix_lengths=*/nullptr,
                                input.json_object_states);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::flush_pending_target_context() {
  if (!target_context_handoff_.has_pending()) {
    return;
  }
  CHECK(target_state_store_ != nullptr)
      << "target_state_store_ must be initialized before target state write";
  // Constrained/plain acceptance metrics are aggregated once by
  // WorkerService::record_llm_speculative_metrics from the verify layouts
  // (VerifyRowLayout::json_constrained); this flush only persists the
  // staged target context.
  target_context_handoff_.flush(*target_state_store_,
                                options_.num_speculative_tokens());
}

template <typename TargetInput>
bool MTPWorkerImpl<TargetInput>::adaptive_enabled() const {
  return adaptive_spec_controller_ != nullptr &&
         adaptive_spec_controller_->enabled();
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::update_decode_step_input(
    TargetInput& input,
    const std::vector<TargetDecodeStateStore::DecodeState>& last_states) const {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "decode context state count mismatch";
  const bool enable_cache_correction = this->enable_schedule_overlap();
  const bool positions_decoupled = positions_are_decoupled_from_kv_length();
  const bool chunked_prefill_verify = use_chunked_prefill_spec_verify_path();

  std::vector<int32_t> token_ids_vec;
  std::vector<int32_t> kv_seq_lens_vec;
  token_ids_vec.reserve(num_sequences);
  // NPU stores one row entry; other backends append a cumulative tail.
  kv_seq_lens_vec.reserve(num_sequences + 1);

  const torch::Tensor& token_ids_cpu = input.token_ids_host;
  const torch::Tensor& positions_cpu = input.positions_host;
  Slice<int32_t> input_token_ids = tensor_slice(token_ids_cpu);
  Slice<int32_t> input_positions = tensor_slice(positions_cpu);
  std::vector<int32_t> positions_vec;
  positions_vec.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size())
        << "decode context token seq_id out of range, seq_id=" << seq_id;
    CHECK_LT(static_cast<size_t>(seq_id), input_positions.size())
        << "decode context position seq_id out of range, seq_id=" << seq_id;
    const TargetDecodeStateStore::DecodeState& state = last_states[seq_id];
    const int32_t input_token_id = input_token_ids[seq_id];
    const bool input_is_fake_token = input_token_id < 0;
    const bool use_cache_correction =
        enable_cache_correction && input_is_fake_token && state.valid;
    const bool use_fake_context =
        enable_cache_correction && input_is_fake_token && !state.valid;
    const int32_t position_offset =
        use_cache_correction ? state.position_offset : 0;
    int32_t current_position = input_positions[seq_id] + position_offset;
    int32_t current_kv_len = specBuilder::calc_kv_len(
        input.input_params.attention.host.kv_seq_lens, seq_id, position_offset);
    int32_t expected_kv_len =
        positions_decoupled ? current_kv_len : current_position + 1;
    if (chunked_prefill_verify) {
      normalize_chunked_prefill_row(
          {/*logical_block_size=*/logical_block_size(),
           /*validate_width=*/options_.num_speculative_tokens() + 1},
          input.input_params.attention.host.block_tables,
          seq_id,
          current_position,
          current_kv_len,
          expected_kv_len);
    }

    CHECK_EQ(expected_kv_len, current_kv_len)
        << "decode context position/kv_len mismatch, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;

    token_ids_vec.emplace_back((use_cache_correction || use_fake_context)
                                   ? state.token_id
                                   : input_token_id);
    positions_vec.emplace_back(current_position);
    specBuilder::append_seq_len_by_layout(kv_seq_lens_vec, current_kv_len);
  }

  input.token_ids_host = make_pinned_cpu_tensor(token_ids_vec);
  input.positions_host = make_pinned_cpu_tensor(positions_vec);
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  input.runtime.device_tensors_ready = false;
}

// Per-sequence widths use the same validate preparation as fixed-width MTP:
// an empty per_seq_val_tokens selects the configured fixed width.
template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::prepare_validate_inputs_impl(
    const TargetInput& input,
    TargetInput& validate_input,
    const std::vector<int32_t>& per_seq_val_tokens,
    bool static_graph_tasks_prepared) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  validate_input = input.clone();
  clear_ready_events(validate_input);
  validate_input.runtime.device_tensors_ready = false;
  validate_input.skip_sampling_for_logits_only = true;
  validate_input.skip_process_logits = true;
  validate_input.input_params.embedding.input_embedding = torch::Tensor();
  auto& input_params = validate_input.input_params;
  torch::TensorOptions token_options = validate_input.token_ids.options();

  const int32_t num_sequences = input_params.meta.num_sequences;
  CHECK(per_seq_val_tokens.empty() ||
        per_seq_val_tokens.size() == static_cast<size_t>(num_sequences))
      << "per-seq validate widths must be empty or match the batch";
  const bool variable_widths = !per_seq_val_tokens.empty();
  const int32_t num_val_tokens =
      variable_widths ? 0 : options_.num_speculative_tokens() + 1;
  const int32_t logical_block_size = this->logical_block_size();
  const bool use_atb_spec_kernel =
      ::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel() ||
      use_chunked_prefill_spec_verify_path();
#if defined(USE_NPU)
  const bool use_explicit_spec_verify_replay_update =
      !variable_widths && should_use_explicit_spec_verify_replay_update(input);
#else
  const bool use_explicit_spec_verify_replay_update = false;
#endif
  mtp_verify::ValidateInputPlan plan;
  plan.widths = per_seq_val_tokens;
  plan.uniform_width = num_val_tokens;
  plan.logical_block_size = logical_block_size;
  plan.positions_decoupled = positions_are_decoupled_from_kv_length();
  plan.attention_mode = use_atb_spec_kernel
                            ? mtp_verify::ValidateAttentionMode::CHUNKED_PREFILL
                            : mtp_verify::ValidateAttentionMode::DECODE;
  plan.row_order = !variable_widths && uses_embedded_eagle3_draft()
                       ? mtp_verify::ValidateRowOrder::STEP_MAJOR
                       : mtp_verify::ValidateRowOrder::SEQUENCE_MAJOR;
  plan.deferred_tokens = use_explicit_spec_verify_replay_update;
  mtp_verify::ValidateInputLayout layout =
      mtp_verify::build_validate_input_layout(input, plan);
  const int32_t total_num_val_tokens = layout.total_tokens;
#if defined(USE_NPU)
  std::vector<int32_t> expanded_linear_state_ids;
  if ((variable_widths || num_val_tokens > 1) &&
      ModelConfig::is_python_model_impl(context_.get_model_impl())) {
    expanded_linear_state_ids =
        mtp_verify::expand_validate_linear_state_ids(input_params,
                                                     per_seq_val_tokens,
                                                     num_val_tokens,
                                                     num_sequences,
                                                     total_num_val_tokens);
  }
#endif
  mtp_verify::apply_validate_input_layout(validate_input, plan, layout);

  auto& validate_sampling_params = validate_input.sampling_params;
  if (variable_widths) {
    this->update_sampling_params(
        validate_sampling_params, per_seq_val_tokens, total_num_val_tokens);
#if defined(USE_NPU)
    // this->update_sampling_params() uses repeat_interleave on device
    // tensors. For a single greedy sequence that work only expands [0] and
    // [false] into fixed validation controls, yet it lands on the
    // final-draft -> target dependency chain. Reuse stable controls after
    // warmup and retain the generic builder for sampling, penalties,
    // multi-sequence batches and other backends.
  } else if (!npu_validate_input_state_.try_prepare_stable_greedy_sampling(
                 validate_sampling_params,
                 total_num_val_tokens,
                 use_explicit_spec_verify_replay_update,
                 device_)) {
    this->update_sampling_params(
        validate_sampling_params, num_val_tokens, total_num_val_tokens);
  }
#else
  } else {
    this->update_sampling_params(
        validate_sampling_params, num_val_tokens, total_num_val_tokens);
  }
#endif

  if (variable_widths) {
    for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
      token_num = total_num_val_tokens;
    }
  } else {
    scale_speculative_parallel_token_counts(input_params, num_val_tokens);
  }

  std::vector<int32_t> accepted_prefix_lengths;
  if (use_chunked_prefill_spec_verify_path()) {
    accepted_prefix_lengths = this->apply_chunked_prefill_spec_verify_tail(
        input, input_params, num_sequences);
    specBuilder::publish_accepted_prefix_lengths(
        input_params,
        accepted_prefix_lengths,
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
      .supports_explicit_replay_update =
          supports_explicit_spec_verify_replay_update()};
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
        const std::vector<TargetDecodeStateStore::DecodeState>& last_states) {
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
      .num_accepted_tokens =
          TargetDecodeStateStore::accepted_prefix_length(last_states.front()),
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
template <typename Params>
std::vector<int32_t>
MTPWorkerImpl<TargetInput>::apply_chunked_prefill_spec_verify_tail(
    const TargetInput& input,
    Params& input_params,
    int32_t num_sequences) {
  input_params.embedding.input_embedding = torch::Tensor();
  input_params.is_spec_verify = true;
  if (!input_params.attention.host.q_seq_lens.empty()) {
    input_params.attention.host.q_cu_seq_lens =
        specBuilder::build_cu_seq_lens(input_params.attention.host.q_seq_lens);
  }
  if (target_state_store_ == nullptr) {
    return std::vector<int32_t>(num_sequences, 1);
  }
  return target_state_store_->resolve_accepted_prefix_lengths(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      num_sequences);
}

template <typename TargetInput>
void MTPWorkerImpl<TargetInput>::prepare_draft_inputs(const TargetInput& input,
                                                      TargetInput& draft_input,
                                                      int32_t position_offset) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  draft_input = input.clone();
  prepare_draft_sampling(draft_input.sampling_params);
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
    for (auto& op : input_params.linear_state_cache_ops) {
      op.reset_requested = false;
      op.restore_requested = false;
      op.restore_src_slot_id = -1;
    }
  }
  const int32_t logical_block_size = this->logical_block_size();
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  specBuilder::DecodeBuildBuffers buf;
  buf.out_positions.reserve(num_sequences);
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_new_cache_slots.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    specBuilder::RowSpec row;
    row.seq_id = seq_id;
    row.position_offset = position_offset;
    row.append_token = false;
    specBuilder::append_decode_row(row_ctx, row, logical_block_size, buf);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "draft kv slots/positions mismatch";

  torch::TensorOptions position_options = input.positions.options();
  set_positions_tensor(
      draft_input, make_pinned_cpu_tensor(buf.out_positions), position_options);
  specBuilder::update_input_params(
      input_params,
      buf,
      input_params.meta.q_max_seq_len,
      std::move(input_params.attention.host.q_seq_lens),
      std::move(input_params.attention.host.q_cu_seq_lens),
      buf.meta.kv_max_seq_len,
      std::move(buf.out_kv_seq_lens));
  if (supports_explicit_spec_verify_replay_update()) {
    specBuilder::rebuild_q_cu_seq_lens(input_params);
  }
  input_params.attention.rebuild_device_buffer(device_);
#if defined(USE_NPU)
  // Later draft steps share the B cache variant.
  draft_impl_->prepare_dp_ep_padding_on_stream(input_params, *prepare_stream_);
#endif
  // token_ids is intentionally filled later from the previous draft output.
  draft_input.runtime.device_tensors_ready = false;

  // Positions/KV metadata do not depend on the in-flight draft result. Prepare
  // them concurrently; token ids and embeddings are filled on compute_stream.
  record_metadata_ready_event(*prepare_stream_, draft_input);
}

template <typename TargetInput>
SampleOutput MTPWorkerImpl<TargetInput>::validate(
    const SamplingParameters& sampling_params,
    std::span<const ForwardOutput> draft_outputs,
    const ForwardOutput& target_output,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths,
    const std::vector<uint8_t>& invalid_draft) {
  const bool probs_required = draft_probs_required(
      draft_sampling_mode_, sampling_params.all_greedy_sample);
  CHECK(!draft_outputs.empty()) << "MTP validation requires draft outputs";

  const int32_t batch_size = static_cast<int32_t>(
      draft_outputs.front().sample_output.next_tokens.size(0));
  const int32_t num_draft_steps = static_cast<int32_t>(draft_outputs.size());
  const torch::Device& proposal_device =
      draft_outputs.front().sample_output.next_tokens.device();

  // Capacity-based grow-and-reuse (mirrors spec_verify_control_block_table_):
  // a decode batch that shrinks then grows must not thrash reallocation, and
  // the narrow view handed to the proposal keeps the exact shape the
  // rejection sampler validates against.
  auto ensure_capacity = [&](torch::Tensor& buffer,
                             c10::IntArrayRef capacity_sizes,
                             torch::ScalarType dtype) {
    bool fits = buffer.defined() && buffer.device() == proposal_device &&
                buffer.scalar_type() == dtype;
    for (size_t dim = 0; fits && dim < capacity_sizes.size(); ++dim) {
      fits = buffer.size(static_cast<int64_t>(dim)) >= capacity_sizes[dim];
    }
    if (!fits) {
      buffer = torch::empty(
          capacity_sizes,
          torch::TensorOptions().dtype(dtype).device(proposal_device));
    }
    return buffer;
  };

  torch::Tensor token_ids =
      ensure_capacity(validate_token_ids_buffer_,
                      {batch_size, num_draft_steps},
                      torch::kLong)
          .narrow(/*dim=*/0, /*start=*/0, batch_size)
          .narrow(/*dim=*/1, /*start=*/0, num_draft_steps);
  std::optional<torch::Tensor> draft_probs;
  if (probs_required) {
    const torch::Tensor& first_probs =
        draft_outputs.front().sample_output.probs;
    CHECK(first_probs.defined())
        << "probabilistic MTP validation requires each proposal distribution";
    draft_probs =
        ensure_capacity(validate_draft_probs_buffer_,
                        {batch_size, num_draft_steps, first_probs.size(-1)},
                        first_probs.scalar_type())
            .narrow(/*dim=*/0, /*start=*/0, batch_size)
            .narrow(/*dim=*/1, /*start=*/0, num_draft_steps)
            .narrow(/*dim=*/2, /*start=*/0, first_probs.size(-1));
  }
  for (int32_t step = 0; step < num_draft_steps; ++step) {
    const SampleOutput& sample = draft_outputs[step].sample_output;
    CHECK(!probs_required || sample.probs.defined())
        << "probabilistic MTP validation requires each proposal distribution";
    token_ids.select(/*dim=*/1, step)
        .copy_(sample.next_tokens.to(proposal_device, torch::kLong),
               /*non_blocking=*/true);
    if (probs_required) {
      draft_probs->select(/*dim=*/1, step)
          .copy_(sample.probs, /*non_blocking=*/true);
    }
  }
  return validate(sampling_params,
                  DraftProposal(std::move(token_ids), std::move(draft_probs)),
                  target_output,
                  num_speculative_tokens,
                  pruned_prefix_lengths,
                  invalid_draft);
}

template <typename TargetInput>
SampleOutput MTPWorkerImpl<TargetInput>::validate(
    const SamplingParameters& sampling_params,
    const DraftProposal& draft_proposal,
    const ForwardOutput& target_output,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths,
    const std::vector<uint8_t>& invalid_draft) {
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const int32_t batch_size =
      static_cast<int32_t>(draft_proposal.token_ids().size(0));
  const int32_t vocab_size = target_output.logits.size(/*dim=*/-1);
  const bool step_major_validate_layout = uses_embedded_eagle3_draft();
  torch::Tensor target_logits;
  torch::Tensor target_embeddings;
  if (step_major_validate_layout) {
    target_logits =
        target_output.logits.view({num_val_tokens, batch_size, vocab_size})
            .permute({1, 0, 2})
            .contiguous();
    target_embeddings = target_output.sample_output.embeddings;
    if (target_embeddings.defined()) {
      target_embeddings =
          target_embeddings
              .view({num_val_tokens, batch_size, target_embeddings.size(-1)})
              .permute({1, 0, 2})
              .contiguous();
    }
  } else {
    target_logits =
        target_output.logits.view({batch_size, num_val_tokens, vocab_size});
  }

  // Grammar-invalid draft positions carry the -1 sentinel directly in the
  // proposal token ids; the rejection sampler rejects negative ids.
  torch::Tensor validation_token_ids;
  if (!invalid_draft.empty()) {
    CHECK_EQ(invalid_draft.size(),
             static_cast<size_t>(batch_size * (num_val_tokens - 1)))
        << "MTP invalid draft mask shape mismatch";
    // uint8 and bool share the 1-byte layout; masked_fill accepts a uint8
    // mask, so the H2D result is used directly without a conversion kernel.
    torch::Tensor invalid_mask =
        async_h2d_tensor(invalid_draft, draft_proposal.token_ids().device())
            .view({batch_size, num_val_tokens - 1});
    validation_token_ids =
        draft_proposal.token_ids().masked_fill(invalid_mask, /*sentinel=*/-1);
  }
  SampleOutput sample_output = spec_verify::run_rejection_sampling(
      {.do_sample = sampling_params.do_sample,
       .all_random_sample = sampling_params.all_random_sample,
       .all_greedy_sample = sampling_params.all_greedy_sample},
      DraftProposal(validation_token_ids.defined() ? validation_token_ids
                                                   : draft_proposal.token_ids(),
                    draft_proposal.draft_probs()),
      target_logits,
      target_output);
  // Only the step-major layout needs the permuted embeddings copy.
  if (step_major_validate_layout) {
    sample_output.embeddings = target_embeddings;
  }

  if (pruned_prefix_lengths != nullptr) {
    const torch::Tensor pruning_masks =
        adaptive_pruning::build_pruned_prefix_masks(
            *pruned_prefix_lengths,
            num_speculative_tokens,
            sample_output.next_tokens.device());
    apply_pruned_prefix_lengths(
        sample_output, num_speculative_tokens, pruning_masks);
  }

  return sample_output;
}

template class MTPWorkerImpl<LlmForwardInput>;
template class MTPWorkerImpl<VlmForwardInput>;

}  // namespace xllm
