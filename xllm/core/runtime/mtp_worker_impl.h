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
#include <type_traits>
#include <vector>

#include "core/framework/speculative/draft_input_state_store.h"
#include "core/framework/speculative/mtp_json_object_state.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/framework/speculative/target_decode_state_store.h"
#include "core/platform/stream.h"
#include "runtime/draft_model_spec_worker_impl.h"
#if defined(USE_NPU)
#include "core/runtime/mtp_npu_validate_input_state.h"
#endif

namespace xllm {

// MTP (Multi-Token Prediction) speculative worker: drafts proposals with a
// draft model, then validates them with the target model.
template <typename TargetInput>
class MTPWorkerImpl : public DraftModelSpecWorkerImpl<TargetInput> {
 public:
  MTPWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options,
                WorkerType worker_type);

  ~MTPWorkerImpl() override;

 protected:
  // For derived classes (e.g. Eagle3WorkerImpl) that need custom options:
  // `options` is passed to WorkerImpl (preserves enable_schedule_overlap
  // etc.), `target_options` / `draft_options` create the respective workers.
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

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override;

  void release_algorithm_state(
      const std::vector<int32_t>& embedding_ids) override;

 protected:
  using typename DraftModelSpecWorkerImpl<TargetInput>::TargetModelParams;
  using DraftModelSpecWorkerImpl<TargetInput>::compute_stream_;
  using DraftModelSpecWorkerImpl<TargetInput>::context_;
  using DraftModelSpecWorkerImpl<TargetInput>::device_;
  using DraftModelSpecWorkerImpl<TargetInput>::draft_impl_;
  using DraftModelSpecWorkerImpl<TargetInput>::draft_vocab_;
  using DraftModelSpecWorkerImpl<TargetInput>::is_probabilistic_draft_;
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
  using typename DraftModelSpecWorkerImpl<TargetInput>::PendingExtendRound;
  using DraftModelSpecWorkerImpl<TargetInput>::launch_draft_extend;
  using DraftModelSpecWorkerImpl<TargetInput>::settle_round;
  using DraftModelSpecWorkerImpl<TargetInput>::dispatch_draft_worker;
  using DraftModelSpecWorkerImpl<TargetInput>::compute_vocabulary_invalid_rows;
  using DraftModelSpecWorkerImpl<TargetInput>::flush_pending_target_context;
  using DraftModelSpecWorkerImpl<TargetInput>::mark_invalid_draft_input_rows;
  using DraftModelSpecWorkerImpl<
      TargetInput>::prepare_draft_vocabulary_sampling;

  // MTP composite: leaves own model-specific NPU input preparation.
  bool owns_npu_parallel_input_prepare() const override;

  std::optional<ForwardOutput> step_prefill(const TargetInput& input) override;
  std::optional<ForwardOutput> step_decode(const TargetInput& inputs) override;
  std::optional<ForwardOutput> step_empty(const TargetInput& inputs) override;

  void fill_validate_input_from_draft_outputs(
      const TargetInput& input,
      std::span<const ForwardOutput> draft_outputs,
      TargetInput& validate_input,
      int32_t num_val_tokens,
      detail::JsonDraftValidationScratch* json_scratch,
      Stream& compute_stream);
  std::optional<ForwardOutput> run_validate(
      const TargetInput& input,
      std::vector<ForwardOutput>& draft_outputs,
      TargetInput& validate_input,
      int32_t num_speculative_tokens,
      detail::JsonDraftValidationScratch* json_scratch);

  SampleOutput validate(const SamplingParameters& sampling_params,
                        std::span<const ForwardOutput> draft_outputs,
                        const ForwardOutput& target_output,
                        int32_t num_speculative_tokens,
                        const std::vector<uint8_t>& invalid_draft);

  // MTP's extend input: flatten the accepted block into draft rows; Eagle3's
  // step-major layout transposes first.
  TargetInput build_draft_extend_input(
      const TargetInput& validate_input,
      const SampleOutput& accepted_output,
      SamplingParameters sampling_params) override;

  // MTP's round result: publish the seed + sampling state into the draft
  // input store and stash the DSA top-k snapshot.
  void publish_round_result(PendingExtendRound& round) override;

  virtual void check_draft_input_embedding(const torch::Tensor& /*embedding*/,
                                           const std::string& /*phase*/) const {
  }
  virtual bool share_target_lm_head_with_draft() const { return true; }

  SampleOutput validate(const SamplingParameters& sampling_params,
                        const DraftProposal& draft_proposal,
                        const ForwardOutput& target_output,
                        int32_t num_speculative_tokens,
                        const std::vector<uint8_t>& invalid_draft);

  KVCacheShape draft_kv_cache_shape(
      const KVCacheShape& target_kv_cache_shape) const override;

  void prepare_prefill_inputs(const TargetInput& inputs,
                              TargetInput& prefill_inputs);
  // Drains a PD-carried draft input into the per-request state cache, then
  // clears the carried payload so the normal decode path can proceed.
  void import_carried_draft_input(TargetInput& input);
  DraftInput assemble_draft_input(ForwardOutput& output,
                                  const SamplingParameters& sampling_params,
                                  bool require_confidence);
  void write_draft_input(const BatchRows& rows,
                         DraftInput draft_input,
                         DraftInputSamplingState sampling,
                         const std::vector<size_t>& failed_rows);
  void stash_draft_input_topk_state(const BatchRows& rows,
                                    const MtpTopkStatePtr& raw_topk_state,
                                    const SamplingParameters& sampling_params);
  DraftTokenHistories restore_draft_input_sampling_history(TargetInput& input);

  bool supports_explicit_spec_verify_replay_update() const;
  bool should_use_explicit_spec_verify_replay_update(
      const TargetInput& input) const;
  // A DCP block-table entry covers one logical KV page per KV split. All
  // speculative cache-slot and expanded-attention metadata must use this
  // logical size rather than the per-rank physical allocator block size.
  int32_t logical_block_size() const;
  int64_t spec_verify_block_table_width(
      const torch::Tensor& block_tables) const;
  // Returns true when validation must use chunked-prefill to avoid the
  // FlashInfer batch-decode read-before-write race on the bonus token.
  bool use_chunked_prefill_spec_verify_path() const;
  // PREFILL instances export the round's seed through the sample output
  // instead of storing it for the next decode round.
  bool is_pd_prefill_exporter() const {
    return options_.instance_role() == InstanceRole::PREFILL;
  }
  // Multiaxis RoPE positions can include a prompt-dependent offset and do not
  // identify the corresponding KV cache length.
  bool positions_are_decoupled_from_kv_length() const;
  void prepare_validate_inputs_impl(const TargetInput& input,
                                    TargetInput& validate_input,
                                    bool static_graph_tasks_prepared = false);
  bool prepare_static_mtp_graph_tasks_before_final_draft(
      const TargetInput& input,
      const std::vector<TargetDecodeStateStore::DecodeRow>& last_states);

  std::vector<uint8_t> prepare_draft_inputs(
      const TargetInput& inputs,
      TargetInput& draft_inputs,
      int32_t position_offset,
      const DraftTokenHistories& sampling_histories);
  void update_decode_step_input(
      TargetInput& input,
      const std::vector<TargetDecodeStateStore::DecodeRow>& last_states) const;

  // Both target/draft pools plus the seed/target state stores come up here.
  void on_kv_pools_allocated(int64_t num_blocks) override;

  // Staging buffer for the packed [batch, num_draft_tokens] draft-token block;
  // capacity-reused so the uniform fast path does not allocate every decode
  // round.
  torch::Tensor packed_drafts_buffer_;

  // Single access point for proposal state, including the draft-input
  // lifecycle, sampling history, device projection, and cross-round DSA top-k
  // snapshot.
  std::unique_ptr<DraftInputStateStore> draft_input_state_store_;

  // Resolved once at model load; decode only consumes the closed contract.
  spec_verify::TargetSpecVerifyCapabilities target_verify_capabilities_;

#if defined(USE_NPU)
  MtpNpuValidateInputState npu_validate_input_state_;
#endif
};
}  // namespace xllm
