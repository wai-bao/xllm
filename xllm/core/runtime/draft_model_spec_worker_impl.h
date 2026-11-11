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
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/speculative/draft_sampling_history.h"
#include "core/framework/speculative/draft_vocab_mapping.h"
#include "core/framework/speculative/mtp_json_object_state.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/framework/speculative/target_decode_state_store.h"
#include "core/platform/stream.h"
#include "core/runtime/target_context_handoff.h"
#include "runtime/speculative_worker_impl.h"

namespace xllm {

class LLMWorkerImpl;
class LlmForwardInput;
class ModelLoader;
class VlmForwardInput;

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

  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

  // Drains a staged target context into the decode state store. No-op when
  // nothing is pending.
  void flush_pending_target_context();

  std::optional<ForwardOutput> step(const TargetInput& input) override;

#if defined(USE_NPU) || defined(USE_MLU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

 protected:
  // Derived step_* paths build their own draft inputs.
  bool prepare_model_decode_continuation(
      TargetInput& inputs,
      const ForwardOutput& prior_output) override;
  std::unique_ptr<TargetDecodeStateStore> target_state_store_;
  // Owns the pinned accepted-token mirror and its ready event across the
  // verify tail and the next step entry; flushed before the next target state
  // write.
  TargetContextHandoff target_context_handoff_;

  // Shared chunked-prefill spec-verify tail: flags the input as spec-verify
  // and resolves accepted prefix lengths from the target decode state store.
  // q_cu_seq_lens stays owned by the validate-input layout builder that ran
  // just before; rebuilding it here would double-write the field.
  template <typename Params>
  std::vector<int32_t> apply_chunked_prefill_spec_verify_tail(
      const TargetInput& input,
      Params& input_params,
      int32_t num_sequences,
      const torch::TensorOptions& token_options,
      bool publish_device_tensor) {
    input_params.embedding.input_embedding = torch::Tensor();
    input_params.is_spec_verify = true;
    // on_kv_pools_allocated constructs the store before any spec verify.
    CHECK(target_state_store_ != nullptr)
        << "chunked spec verify requires the target decode state store";
    std::vector<int32_t> accepted_prefix_lengths =
        target_state_store_->resolve_accepted_prefix_lengths(
            input.input_params.embedding.batch_rows(), num_sequences);
    publish_accepted_prefix_lengths(input_params,
                                    accepted_prefix_lengths,
                                    token_options,
                                    publish_device_tensor);
    return accepted_prefix_lengths;
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

  // Draft-variant conversions (eagle3/dspark vocab offsets) read from the
  // loader before the draft worker initializes.
  torch::Tensor load_draft_vocab_mapping(ModelLoader& loader);

  // Draft-side KV layout: shared static storage reuses the target arena,
  // otherwise the draft's own layer geometry drives the create options.
  KVCacheCreateOptions build_draft_kv_cache_layout(
      const KVCacheShape& shape) const;
  int64_t draft_num_layers() const;

  // Called once per successful allocate_kv_cache* with the target pool's
  // block count, after both pools are allocated. Creates the shared target
  // decode state store; derived workers override, call the base, and create
  // algorithm-specific stores.
  virtual void on_kv_pools_allocated(int64_t num_blocks);

  void release_request_state(
      const std::vector<int32_t>& embedding_ids) override;

  // Model contract of the seed this drafter carries across rounds. Only
  // meaningful for draft-input-carrying drafters (see SpeculativeConfig::
  // requires_bootstrap_hidden).
  DraftInputModelSpec draft_input_spec() const;

  // Cross-round boundary state for a draft forward launched from the
  // previous verify tail: the launch never waits on the accepted-token D2H,
  // so the settle-side consumers snapshot everything they need here. The
  // worker owns at most one pending round (ScheduleOverlap's FIFO execution
  // serializes steps).
  struct PendingExtendRound {
    // Owned copies: BatchRows itself is a borrowed view into the producing
    // step's input, which does not outlive the step.
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    SamplingParameters sampling_params;
    std::vector<JsonObjectGrammarState> json_states;
    std::vector<std::string> sample_sequence_ids;
    std::vector<JsonObjectOutputError> prior_errors;
    torch::Tensor accepted_tokens_host;
    StreamEventPtr accepted_ready;
    ForwardOutput output;
    TargetInput prepared_input;
    std::vector<size_t> failed_rows;

    BatchRows rows() const { return {embedding_ids, request_ids}; }
  };

  // Launches the next round's draft forward from device-side accepted state,
  // ordered behind rejection sampling on compute_stream_ and never waiting on
  // the accepted-token D2H. Vocabulary-rejected rows surface as errors on the
  // producing step's output.
  PendingExtendRound launch_draft_extend(
      const TargetInput& input,
      const TargetInput& validate_input,
      const SampleOutput& accepted_output,
      torch::Tensor accepted_tokens_host,
      const StreamEventPtr& accepted_ready,
      std::vector<size_t> failed_rows,
      std::vector<JsonObjectOutputError>& errors);

  // Finishes a pending round: waits accepted_ready only when
  // needs_committed_tokens(), runs the JSON mismatch scan, and publishes the
  // algorithm's round result through publish_round_result. Returns
  // request-level errors for the producing step's output.
  std::vector<JsonObjectOutputError> settle_round(PendingExtendRound round);

  // Single draft-dispatch seam for the launch/settle path: converts VLM
  // target inputs to the LLM draft worker's input type and mirrors prepared
  // sampling params back to the caller.
  std::optional<ForwardOutput> dispatch_draft_worker(
      const TargetInput& input,
      TargetInput& processed_input);

  // Algorithm hook: builds the draft forward's input from the device-side
  // accepted state. Stream contract: the hook runs under the base launch's
  // compute-stream guard; an implementation that snapshots the target's
  // stable metadata must do so on prepare_stream_ (waiting compute_stream_
  // first) and record the metadata-ready event there, so the launch's
  // re-arm before the draft forward orders the snapshot against a following
  // prepare.
  virtual TargetInput build_draft_extend_input(
      const TargetInput& /*validate_input*/,
      const SampleOutput& /*accepted_output*/,
      SamplingParameters /*sampling_params*/) {
    LOG(FATAL) << "build_draft_extend_input is not implemented for this "
               << "drafter";
    __builtin_unreachable();
  }

  // Runs prepare_draft_vocabulary_sampling, folds the JSON grammar
  // acceptance into the invalid-row mask, and unions it with the sampling
  // state's own invalid rows. The proposal path clears the filter/penalty
  // fields first, so the whole chain is gated on the fields
  // prepare_draft_vocabulary_sampling consumes. The virtual seam runs
  // unconditionally: overrides may force failures regardless of the params.
  std::vector<uint8_t> compute_vocabulary_invalid_rows(
      SamplingParameters& sampling_params,
      const DraftTokenHistories& histories,
      const std::vector<JsonObjectGrammarState>& json_states,
      const std::vector<uint8_t>& sampling_invalid_rows);

  virtual std::vector<uint8_t> prepare_draft_vocabulary_sampling(
      SamplingParameters& /*sampling_params*/,
      const DraftTokenHistories& /*histories*/) {
    return {};
  }

  // Algorithm hook: consumes the settled round (draft output plus the
  // host-committed tokens) and publishes whatever the next round consumes.
  virtual void publish_round_result(PendingExtendRound& /*round*/) {
    LOG(FATAL) << "publish_round_result is not implemented for this drafter";
  }

  // `invalid_rows` may be empty or batch-sized; rows set in it and not yet
  // present in `*failed_rows` are appended there and to `errors`.
  void mark_invalid_draft_input_rows(
      const TargetInput& input,
      const std::vector<uint8_t>& invalid_rows,
      std::vector<size_t>* failed_rows,
      std::vector<JsonObjectOutputError>& errors) const;

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
  // position/kv_len invariant check. Shared by the MTP and DFlash/DSpark
  // decode legs.
  void rewrite_decode_rows_from_target_state(
      TargetInput& input,
      const std::vector<TargetDecodeStateStore::DecodeRow>& last_states,
      const DecodeRowPolicy& policy) const;

  // Draft model worker created by the ctor's factory.
  std::unique_ptr<LLMWorkerImpl> draft_impl_;

  // Resolved once after the draft model loads; the spec is invariant for the
  // worker's lifetime, so the per-draft-step lookup must not re-resolve it.
  mutable DraftInputModelSpec draft_input_spec_{};
  mutable bool draft_input_spec_valid_ = false;

  DraftVocabMapping draft_vocab_;

  bool is_probabilistic_draft_ = false;
};

}  // namespace xllm
