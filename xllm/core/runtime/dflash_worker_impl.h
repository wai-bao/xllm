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
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/framework/speculative/draft_proposal.h"
#include "core/kv_cache/transfer/kv_cache_transfer.h"
#include "framework/model/model_args.h"
#include "runtime/draft_model_spec_worker_impl.h"
#include "runtime/llm_worker_impl.h"

namespace xllm {

namespace dflash_detail {

inline void invalidate_draft_model_geometry(LlmModelParams& input_params) {
  // Attention metadata is model-owned: DeepSeek-V4 bakes DSA group layout and
  // sparse tiling values such as ori_win_left into opaque tensors. A draft
  // input copied from the target must rebuild those tensors for draft geometry.
  input_params.attn_metadata.reset();
  // Block drafters own no recurrent layers. Target slots and validity masks
  // must not reach the draft metadata builder, including during graph warmup.
  input_params.clear_linear_attention_state();
}

enum class DraftRowLayout : uint8_t {
  SEQUENCE,
  BLOCK_ROWS,
  SPARSE_INDICES,
};

}  // namespace dflash_detail

class DFlashWorkerImpl : public DraftModelSpecWorkerImpl<LlmForwardInput> {
 public:
  DFlashWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options);

  ~DFlashWorkerImpl() override;

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override;

 protected:
  std::optional<ForwardOutput> step_prefill(
      const LlmForwardInput& input) override;

  std::optional<ForwardOutput> step_decode(
      const LlmForwardInput& input) override;

  std::optional<ForwardOutput> step_empty(
      const LlmForwardInput& input) override;

  // Draft produces all speculative tokens of a block in one forward, so its
  // output is a single [batch, num_speculative_tokens] block rather than the
  // per-step outputs an autoregressive drafter (e.g. MTP) yields; the
  // proposal carries the draft forward's retained no-sync handles.
  //
  // virtual: DSpark overrides the draft sampling (parallel block sample ->
  // one forward + sequential Markov-head sampling loop).
  virtual DraftProposal run_decode_draft(const LlmForwardInput& input,
                                         LlmForwardInput& validate_input);

  // Shared with subclasses (DSpark): build the N/N+1-wide draft query block and
  // the target validate input. A DSpark override of run_decode_draft calls both
  // before its draft forward.
  void prepare_query_inputs(const LlmForwardInput& input,
                            LlmForwardInput& query_input);
  void prepare_validate_inputs(const LlmForwardInput& input,
                               LlmForwardInput& validate_input);

  int32_t draft_query_width_;
  dflash_detail::DraftRowLayout draft_row_layout_ =
      dflash_detail::DraftRowLayout::SEQUENCE;
  bool draft_needs_dummy_rows_ = false;

 private:
  void fill_validate_input_from_draft_outputs(
      const DraftProposal& draft_proposal,
      LlmForwardInput& validate_input,
      Stream& compute_stream);

  std::optional<ForwardOutput> run_validate(const LlmForwardInput& input,
                                            const DraftProposal& draft_proposal,
                                            LlmForwardInput& validate_input);

  // Mirrors sampled tokens to rank 0 under schedule-overlap so every rank
  // commits the same prefix. No-op for a single rank. Distinct from the
  // base's two-argument maybe_broadcast_spec_tokens, which additionally
  // covers orthogonal CP consensus.
  void broadcast_committed_tokens(torch::Tensor& tokens);

  // Per-row bases uploaded on prepare_stream_ for the device row builders.
  struct RowBases {
    torch::Tensor positions;
    torch::Tensor kv_lens;
    torch::Tensor block_tables;
  };
  RowBases upload_row_bases(const LlmForwardInput& input);

  std::vector<std::string> write_context_kv(
      const LlmForwardInput& input,
      const torch::Tensor& context_hidden,
      const torch::Tensor& positions_device,
      const torch::Tensor& new_cache_slots_device);

  // Device-side context-KV scatter behind rejection sampling; never waits on
  // the host. The store-side bookkeeping rides the pinned handoff instead.
  // Returns the request ids whose PD KV push failed (empty when no push).
  // `bases`/`bases_ready` come from the caller's upload_row_bases so the
  // scatter and the next-query launch share one upload per verify round.
  std::vector<std::string> scatter_context_kv(
      const LlmForwardInput& input,
      const SampleOutput& validate_output,
      const RowBases& bases,
      const StreamEventPtr& bases_ready);

 protected:
  // The query forward launched at the tail of the previous run_validate,
  // consumed by the next step's run_decode_draft. Owned copies, at most one
  // outstanding (same ownership rule as MTP's pending round).
  struct PendingQueryRound {
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    ForwardOutput output;
    // Keeps the no-sync query input's tensors alive until the stream work
    // that reads them completes (CALLER_MANAGED completion).
    LlmForwardInput prepared_input;
    torch::Tensor anchors;  // last accepted token per sequence, device
  };
  std::optional<PendingQueryRound> pending_query_;

  // Launches the next round's draft query from the device-side accepted
  // block (build_next_query_rows_device) behind the context-KV scatter.
  // Only the SEQUENCE layout under schedule overlap qualifies.
  // Per-head query forward flags: DFlash samples in the query pass, DSpark
  // needs logits only, DFlash2 also returns selected hidden states.
  virtual bool query_forward_logits_only() const { return false; }
  virtual bool query_forward_returns_selected_hidden() const { return false; }

  void launch_next_query(const LlmForwardInput& input,
                         const SampleOutput& val_output,
                         const RowBases& bases,
                         const StreamEventPtr& bases_ready);
  // Broadcast + sampled-output → proposal tail shared by both draft paths.
  DraftProposal proposal_from_sampled_output(ForwardOutput& draft_output);
  bool take_pending_query(const LlmForwardInput& input, PendingQueryRound& out);

  // Shared by the DSpark/DFlash2 sampling sites.
  ProcessGroup* sampling_process_group_ = nullptr;

  // Block drafts (DFlash2/DSpark) condition their first step on the decode
  // batch's first tokens.
  torch::Tensor anchor_token_ids(const LlmForwardInput& input) const;
  int32_t mask_token_id_ = -1;
  int64_t expected_context_hidden_size_ = 0;
  // Whether the target model owns recurrent (linear attention) layers;
  // fixed once the target finishes loading.
  bool target_is_hybrid_recurrent_ = false;
};

}  // namespace xllm
