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

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>

#include "core/framework/speculative/target_decode_state_store.h"
#include "framework/sampling/draft_sampling_mode.h"
#include "runtime/speculative_input_builder.h"
#include "runtime/speculative_worker_impl.h"

namespace xllm {

class LLMWorkerImpl;

// Base for draft-model speculative workers (MTP, DFlash/DSpark); Suffix has no
// draft model and derives from SpeculativeWorkerImpl directly.
template <typename TargetInput>
class DraftModelSpecWorkerImpl : public SpeculativeWorkerImpl<TargetInput> {
 public:
  ~DraftModelSpecWorkerImpl() override;

  void drain_pending_execution() override;

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;
  bool task_models_loaded() const override;

  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

#if defined(USE_NPU) || defined(USE_MLU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

  // Derived step_* paths build their own draft inputs.
  TargetInput update_input_by_last_step_output(TargetInput& inputs) override;

 protected:
  std::unique_ptr<TargetDecodeStateStore> target_state_store_;

  // Shared chunked-prefill spec-verify tail: flags the input as spec-verify,
  // derives q_cu_seq_lens from q_seq_lens, and resolves accepted prefix
  // lengths (defaulting to 1 when the store is unavailable).
  template <typename Params>
  std::vector<int32_t> apply_chunked_prefill_spec_verify_tail(
      const TargetInput& input,
      Params& input_params,
      int32_t num_sequences) {
    input_params.embedding.input_embedding = torch::Tensor();
    input_params.is_spec_verify = true;
    if (!input_params.attention.host.q_seq_lens.empty()) {
      input_params.attention.host.q_cu_seq_lens =
          specBuilder::build_cu_seq_lens(
              input_params.attention.host.q_seq_lens);
    }
    if (target_state_store_ == nullptr) {
      return std::vector<int32_t>(num_sequences, 1);
    }
    return target_state_store_->resolve_accepted_prefix_lengths(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        num_sequences);
  }

  using TargetModelParams =
      typename SpeculativeWorkerImpl<TargetInput>::TargetModelParams;
  using SpeculativeWorkerImpl<TargetInput>::compute_stream_;
  using SpeculativeWorkerImpl<TargetInput>::context_;
  using SpeculativeWorkerImpl<TargetInput>::device_;
  using SpeculativeWorkerImpl<TargetInput>::dtype_;
  using SpeculativeWorkerImpl<TargetInput>::hierarchy_kv_cache_transfer_;
  using SpeculativeWorkerImpl<TargetInput>::impl_;
  using SpeculativeWorkerImpl<TargetInput>::kv_cache_transfer_;
  using SpeculativeWorkerImpl<TargetInput>::options_;
  using SpeculativeWorkerImpl<TargetInput>::parallel_args_;
  using SpeculativeWorkerImpl<TargetInput>::set_hierarchy_kv_cache_transfer;

  DraftModelSpecWorkerImpl(
      const ParallelArgs& parallel_args,
      const torch::Device& device,
      const runtime::Options& options,
      const runtime::Options& target_options,
      WorkerType worker_type,
      const std::function<std::unique_ptr<LLMWorkerImpl>()>& draft_factory);

  virtual KVCacheShape draft_kv_cache_shape(
      const KVCacheShape& target_kv_cache_shape) const;
  // Reuse grouped pool counts; otherwise use the draft's geometry and DP-local
  // TP size, borrowing only the target's block count.
  KVCacheShape build_draft_kv_cache_shape(
      const KVCacheShape& target_kv_cache_shape,
      int64_t draft_world_size = -1) const;

  // Called once per successful allocate_kv_cache* with the target pool's
  // block count, after both target and draft pools are allocated. Creates
  // the shared target decode state store; derived workers override, call
  // the base, and create algorithm-specific stores.
  virtual void on_kv_pools_allocated(int64_t num_blocks);

  void release_request_state(
      const std::vector<int32_t>& embedding_ids) override;

  // Model contract of the seed this drafter carries across rounds. Only
  // meaningful for draft-input-carrying drafters (see SpeculativeConfig::
  // produces_draft_input).
  DraftInputModelSpec draft_input_spec() const;

  // Algorithm-owned per-request state. Runs before the shared target decode
  // state store is cleared: a pending-context flush may re-populate the store,
  // so the shared clear must come after it (see release_request_state).
  virtual void release_algorithm_state(
      const std::vector<int32_t>& /*embedding_ids*/) {}

  void prepare_hierarchy_kv_cache_transfers();
  void finalize_hierarchy_kv_cache_transfers();
  using AllocateFn = std::function<bool(WorkerImpl&, const KVCacheShape&)>;
  bool allocate_pools(const KVCacheShape& kv_cache_shape,
                      const AllocateFn& allocate);

  // Target-side cache budget after reserving storage for the colocated draft.
  // DeepSeek-V4's fixed SWA pools require both geometries to participate.
  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity_with_draft(
      const runtime::Options& target_options,
      const runtime::Options& draft_options);

  // Per-algorithm policy for rewrite_decode_rows_from_target_state.
  struct DecodeRowPolicy {
    // Positions track kv_len instead of position+1 (MTP decoupled targets).
    bool positions_decoupled_from_kv_len = false;
    // Chunked-prefill verify: clamp the position to block-table capacity and
    // resolve the scheduler's KV-length lag/lead against it.
    bool chunked_prefill_verify = false;
    int32_t logical_block_size = 0;
    int32_t validate_width = 0;
    // Without schedule overlap a fake token is a contract violation; with
    // overlap but no cached state, MTP still substitutes the recorded state
    // token while DFlash keeps the raw input id.
    bool fake_context_falls_back_to_state_token = false;
  };

  // Rewrites the decode rows from the target's committed per-request state:
  // fake-token correction under schedule overlap, position offsets, and the
  // position/kv_len invariant check. Single shared implementation so the
  // MTP and DFlash/DSpark decode legs cannot drift apart.
  void rewrite_decode_rows_from_target_state(
      TargetInput& input,
      const std::vector<TargetDecodeStateStore::DecodeState>& last_states,
      const DecodeRowPolicy& policy) const;

  // Draft model worker created by the ctor's factory.
  std::unique_ptr<LLMWorkerImpl> draft_impl_;

  // Resolved once after the draft model loads; the spec is invariant for the
  // worker's lifetime, so the per-draft-step lookup must not re-resolve it.
  mutable DraftInputModelSpec draft_input_spec_{};
  mutable bool draft_input_spec_valid_ = false;

  DraftSamplingMode draft_sampling_mode_ = DraftSamplingMode::GREEDY;
};

}  // namespace xllm
