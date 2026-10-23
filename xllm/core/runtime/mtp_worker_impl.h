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

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/framework/speculative/draft_input_state_store.h"
#include "core/framework/speculative/mtp_json_object_state.h"
#include "core/framework/speculative/target_decode_state_store.h"
#include "core/platform/stream.h"
#include "core/runtime/mtp_target_context_handoff.h"
#include "core/runtime/spec_verify.h"
#include "runtime/draft_model_spec_worker_impl.h"
#if defined(USE_NPU)
#include "core/runtime/mtp_npu_validate_input_state.h"
#endif

namespace xllm {

#if defined(USE_NPU)
namespace detail {
class NpuJsonDraftTokenHandoff;
}  // namespace detail
#endif

// MTP (Multi-Token Prediction) speculative worker.
// Uses a draft model to generate proposals, then validates with target model.
// Eagle3WorkerImpl inherits from this class.
template <typename TargetInput>
class MTPWorkerImpl : public DraftModelSpecWorkerImpl<TargetInput> {
 public:
  MTPWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options,
                WorkerType worker_type);

  ~MTPWorkerImpl() override;

 protected:
  // For derived classes (e.g. Eagle3WorkerImpl) that need custom options for
  // target and draft models. `options` is passed to WorkerImpl (preserves
  // enable_schedule_overlap etc.), `target_options` / `draft_options` are used
  // to create the respective workers.
  MTPWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options,
                const runtime::Options& target_options,
                const runtime::Options& draft_options,
                WorkerType worker_type);

 public:
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  ::xllm::Status create_task_pipeline(
      std::unique_ptr<TaskExecutionPipeline>& output) override;

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override;

  // step_prefill/step_decode build their draft inputs from the store; the
  // caller reassigns its input with the return value, so the defensive
  // clone() in DraftModelSpecWorkerImpl would only be discarded.
  TargetInput update_input_by_last_step_output(TargetInput& inputs) override;
  void release_algorithm_state(
      const std::vector<int32_t>& embedding_ids) override;

 protected:
  using typename DraftModelSpecWorkerImpl<TargetInput>::TargetModelParams;
  using DraftModelSpecWorkerImpl<TargetInput>::compute_stream_;
  using DraftModelSpecWorkerImpl<TargetInput>::context_;
  using DraftModelSpecWorkerImpl<TargetInput>::device_;
  using DraftModelSpecWorkerImpl<TargetInput>::draft_impl_;
  using DraftModelSpecWorkerImpl<TargetInput>::draft_sampling_mode_;
  using DraftModelSpecWorkerImpl<TargetInput>::dtype_;
  using DraftModelSpecWorkerImpl<TargetInput>::target_state_store_;
  using DraftModelSpecWorkerImpl<TargetInput>::impl_;
  using DraftModelSpecWorkerImpl<TargetInput>::options_;
  using DraftModelSpecWorkerImpl<TargetInput>::parallel_args_;
  using DraftModelSpecWorkerImpl<TargetInput>::prepare_stream_;
  using DraftModelSpecWorkerImpl<TargetInput>::threadpool_;
  using SpeculativeWorkerImpl<TargetInput>::maybe_broadcast_spec_tokens;
  using DraftModelSpecWorkerImpl<TargetInput>::build_draft_kv_cache_shape;
  using DraftModelSpecWorkerImpl<TargetInput>::enable_schedule_overlap;
  using DraftModelSpecWorkerImpl<
      TargetInput>::estimate_kv_cache_capacity_with_draft;
  using DraftModelSpecWorkerImpl<TargetInput>::get_optimization_config;
  using DraftModelSpecWorkerImpl<TargetInput>::update_sampling_params;

  // MTP composite: leaves own model-specific NPU input preparation.
  bool owns_npu_parallel_input_prepare() const override;

  std::optional<ForwardOutput> step_prefill(const TargetInput& input) override;
  std::optional<ForwardOutput> step_decode(const TargetInput& inputs) override;
  std::optional<ForwardOutput> step_empty(const TargetInput& inputs) override;

  void fill_validate_input_from_draft_outputs(
      const TargetInput& input,
      std::span<const ForwardOutput> draft_outputs,
      TargetInput& validate_input,
      const std::vector<int32_t>& per_seq_val_tokens,
      detail::JsonDraftValidationScratch* json_scratch,
      Stream& compute_stream);
  // Adaptive pruning path: compute per-seq prefix lengths, truncate draft
  // outputs, and run variable-length validate.
  std::optional<ForwardOutput> run_validate(
      const TargetInput& input,
      std::span<const ForwardOutput> draft_outputs,
      TargetInput& validate_input,
      int32_t num_speculative_tokens,
      const std::vector<int32_t>& per_seq_val_tokens,
      detail::JsonDraftValidationScratch* json_scratch);

  virtual SampleOutput validate(
      const SamplingParameters& sampling_params,
      std::span<const ForwardOutput> draft_outputs,
      const ForwardOutput& target_output,
      int32_t num_speculative_tokens,
      // No default: Google Style bans default args on virtuals — they resolve
      // statically from the declared base type, so an override changing the
      // default would silently diverge when called through a base reference.
      // Callers must pass nullptr explicitly for the static path.
      const std::vector<uint8_t>& invalid_draft);

  // Map draft token IDs and the sparse draft logits to the target
  // vocabulary. Implementations that cannot use the logits drop them here;
  // finish_draft_input derives the point-mass marker for every path.
  virtual void remap_draft_sample_output(SampleOutput& /*sample_output*/) {}

  virtual std::vector<uint8_t> prepare_draft_vocabulary_sampling(
      SamplingParameters& /*sampling_params*/,
      const DraftTokenHistories& /*histories*/) {
    return {};
  }
  // Runs prepare_draft_vocabulary_sampling and then folds the JSON grammar
  // acceptance into the returned invalid-row mask. Every draft seam that
  // produces a proposal composes these two steps in the same order; consumers
  // must not skip one of them.
  std::vector<uint8_t> compute_vocabulary_invalid_rows(
      SamplingParameters& sampling_params,
      const DraftTokenHistories& histories,
      const std::vector<JsonObjectGrammarState>& json_states);

  virtual void check_draft_input_embedding(const torch::Tensor& /*embedding*/,
                                           const std::string& /*phase*/) const {
  }
  virtual bool share_target_lm_head_with_draft() const { return true; }

  SampleOutput validate(const SamplingParameters& sampling_params,
                        const DraftProposal& draft_proposal,
                        const ForwardOutput& target_output,
                        int32_t num_speculative_tokens,
                        const std::vector<uint8_t>& invalid_draft = {});

  KVCacheShape draft_kv_cache_shape(
      const KVCacheShape& target_kv_cache_shape) const override;

  // prepare inputs for draft model at Prefill phase.
  void prepare_prefill_inputs(const TargetInput& inputs,
                              TargetInput& prefill_inputs);
  // Drains a PD-carried draft input from the batch embedding into the
  // per-request state cache (one bootstrap + draft-input row per carried row),
  // then clears the carried payload so the normal decode path can proceed.
  void import_carried_draft_input(TargetInput& input);
  DraftInput assemble_draft_input(ForwardOutput& output,
                                  const SamplingParameters& sampling_params,
                                  bool require_confidence);
  void write_draft_input(const TargetInput& input,
                         DraftInput draft_input,
                         DraftInputSamplingState sampling,
                         const std::vector<size_t>& failed_rows);
  // Snapshot the batch identity and reusable DSA top-k state for the next step.
  void stash_draft_input_topk_state(const TargetInput& input,
                                    const MtpTopkStatePtr& raw_topk_state,
                                    const SamplingParameters& sampling_params);
  DraftTokenHistories restore_draft_input_sampling_history(TargetInput& input);
  // `invalid_rows` may be empty or batch-sized; rows set in it and not yet
  // present in `*failed_rows` are appended there and to `errors`.
  void mark_invalid_draft_input_rows(
      const TargetInput& input,
      const std::vector<uint8_t>& invalid_rows,
      std::vector<size_t>* failed_rows,
      std::vector<JsonObjectOutputError>& errors) const;
  ForwardOutput run_draft_input_extend(
      const TargetInput& input,
      const TargetInput& validate_input,
      const SampleOutput& accepted_output,
      const torch::Tensor& accepted_tokens_host,
      const StreamEventPtr& accepted_tokens_ready,
      const std::vector<int32_t>& per_seq_val_tokens,
      std::vector<size_t>& failed_rows,
      ForwardOutput& target_output);
  bool supports_explicit_spec_verify_replay_update() const;
  bool should_use_explicit_spec_verify_replay_update(
      const TargetInput& input) const;
  // Returns true when the target model's spec-verify kernel requires the
  // validate width (val_tokens) to be identical across every sequence in the
  // batch. Currently Qwen3.5 GDN's FusedRecurrentGatedDeltaRule spec-verify
  // path has this constraint; other paths accept per-seq variable widths.
  // Kept separate from supports_explicit_spec_verify_replay_update() so the
  // two capabilities can diverge for future targets.
  bool requires_uniform_validate_width() const;
  // A DCP block-table entry covers one logical KV page per KV split. All
  // speculative cache-slot and expanded-attention metadata must use this
  // logical size rather than the per-rank physical allocator block size.
  int32_t logical_block_size() const;
  int64_t spec_verify_block_table_width(
      const torch::Tensor& block_tables) const;
  // Returns true when validation must use chunked-prefill to avoid the
  // FlashInfer batch-decode read-before-write race on the bonus token.
  bool use_chunked_prefill_spec_verify_path() const;
  bool uses_embedded_eagle3_draft() const;
  // Single source for the PD role: PREFILL instances export the round's seed
  // through the sample output (eager path and task-pipeline capacity both
  // read this) instead of storing it for the next decode round.
  bool is_pd_prefill_exporter() const {
    return options_.instance_role() == InstanceRole::PREFILL;
  }
  // Dispatches on the compute stream unless the draft is an embedded Eagle3
  // head, which needs the prepare stream for its MLA workspace and a
  // post-forward sync.
  std::optional<ForwardOutput> dispatch_worker_on_compute(
      WorkerImpl& worker,
      const TargetInput& input,
      TargetInput& processed_input);
  // Multiaxis RoPE positions can include a prompt-dependent offset and do not
  // identify the corresponding KV cache length.
  bool positions_are_decoupled_from_kv_length() const;
  void synchronize_embedded_eagle3_forward();
  // Single draft-dispatch seam: converts VLM target inputs to the LLM draft
  // worker's input type and mirrors prepared sampling params back to the
  // caller. Every draft_impl_ forward must go through it.
  std::optional<ForwardOutput> dispatch_draft_worker(
      const TargetInput& input,
      TargetInput& processed_input);
  std::optional<ForwardOutput> run_worker_no_sync(WorkerImpl& worker,
                                                  const TargetInput& input,
                                                  TargetInput& processed_input);
  // Prepare target validate input from cached target context. An empty
  // per_seq_val_tokens selects the configured fixed validate width.
  void prepare_validate_inputs_impl(
      const TargetInput& input,
      TargetInput& validate_input,
      const std::vector<int32_t>& per_seq_val_tokens = {},
      bool static_graph_tasks_prepared = false);
  bool prepare_static_mtp_graph_tasks_before_final_draft(
      const TargetInput& input,
      const std::vector<TargetDecodeStateStore::DecodeState>& last_states);

  // prepare inputs for draft model at Decode phase.
  void prepare_draft_inputs(const TargetInput& inputs,
                            TargetInput& draft_inputs,
                            int32_t position_offset);
  void update_decode_step_input(
      TargetInput& input,
      const std::vector<TargetDecodeStateStore::DecodeState>& last_states)
      const;

  // Both target/draft pools plus the seed/target state stores come up here.
  void on_kv_pools_allocated(int64_t num_blocks) override;

  void flush_pending_target_context();

 protected:
  // Staging buffer for the packed [batch, num_draft_tokens] draft-token block
  // in fill_validate_input_from_draft_outputs; capacity-reused so the uniform
  // fast path does not allocate every decode round.
  torch::Tensor packed_drafts_buffer_;
  // Shared target/draft decode state for model-based speculative decoding;
  // owned here (not the base) because the base serves workers without it.
  MtpTargetContextHandoff target_context_handoff_;

  // Single access point for proposal state, including the draft-input
  // lifecycle, sampling history, device projection, and cross-round DSA top-k
  // snapshot.
  std::unique_ptr<DraftInputStateStore> draft_input_state_store_;

  // Resolved once at model load; decode only consumes the closed contract.
  spec_verify::TargetSpecVerifyCapabilities target_verify_capabilities_;
  // True when the draft is an embedded Eagle3 head of the loaded target; this
  // single axis drives the separate draft KV shape, probability-based
  // validation, and step-major validate-layout decisions.
  bool draft_is_embedded_eagle3_ = false;

#if defined(USE_NPU)
  MtpNpuValidateInputState npu_validate_input_state_;
  std::unique_ptr<detail::NpuJsonDraftTokenHandoff> json_draft_token_handoff_;
#endif
};
}  // namespace xllm
