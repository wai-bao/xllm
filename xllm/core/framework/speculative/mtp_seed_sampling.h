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

#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/mtp_sampling_history.h"

namespace xllm {

// Canonical form for per-row failure masks across the speculative path: an
// empty vector means no row failed, a non-empty one holds one flag per row.
// Returns {} when no flag is set.
inline std::vector<uint8_t> normalize_row_failure_mask(
    std::vector<uint8_t> mask) {
  const bool any_failed =
      std::any_of(mask.begin(), mask.end(), [](uint8_t flag) { return flag; });
  if (!any_failed) {
    mask.clear();
  }
  return mask;
}

struct MtpSeedSamplingState {
  SamplingParameters sampling_params;
  std::vector<JsonObjectGrammarState> json_object_states;
  // An empty vector means no row failed; otherwise it holds B flags and a
  // failed row must not publish its seed.
  std::vector<uint8_t> invalid_rows;
  MtpTokenHistories histories;
};

// The confidence source laddered through by select_mtp_seed_confidence:
// the carried seed's selected probs, then the sampler's probs, then a
// per-row softmax over logits+next_tokens.
enum class MtpConfidenceSource {
  NONE = 0,
  SELECTED_PROBS = 1,
  PROBS = 2,
  LOGITS = 3,
};

// Reports which rung of the confidence ladder select_mtp_seed_confidence
// would take for this sample output, without reading any tensor data. NONE
// means no source is available and select_mtp_seed_confidence would
// CHECK-fail.
MtpConfidenceSource mtp_seed_confidence_source(
    const SampleOutput& sample_output,
    const torch::Tensor& logits);

// Extracts the pre-sampling in-vocabulary confidence without allocating a full
// softmax and without modifying the proposal distribution q. When a 1D selected
// probability is already available, this only normalizes the probability dtype
// and reads neither tokens, logits nor sample indices.
torch::Tensor select_mtp_seed_confidence(const SampleOutput& sample_output,
                                         const torch::Tensor& logits,
                                         const torch::Tensor& sample_idxes);

// Appends only the tokens committed by this step; whether an existing root is
// already in the history is the caller's decision. The unconstrained path does
// not read committed_tokens device data, and the constrained path accepts only
// the [B, W] layout with a -1 suffix. histories is the request-level CPU
// authority; batch is reused in execution-stream order.
MtpSeedSamplingState prepare_mtp_seed_sampling(
    SamplingParameters params,
    std::vector<JsonObjectGrammarState> states,
    const torch::Tensor& committed_tokens,
    const torch::Device& device,
    MtpTokenHistories histories = {},
    MtpHistoryBatch* batch = nullptr);

// Overwrites params with a freshly built DRAFT-phase grammar bitmask and clears
// the possibly stale full filter mask, so downstream sampling follows the
// bitmask alone. prepare_mtp_seed_sampling and restore_draft_sampling share
// this single implementation.
void apply_draft_grammar_bitmask(
    SamplingParameters& params,
    const std::vector<JsonObjectGrammarState>& json_states,
    const torch::Device& device);

// Finalizes the proposal q of a freshly sampled draft seed in one call:
// drops q for non-probabilistic and all-greedy seeds (marking the latter as
// point-mass), materializes point masses for mixed greedy/random batches,
// and leaves all-random q untouched. Must run after any draft-to-target
// vocabulary remapping, which consumes raw sampler q. Returns the
// point_mass_q marker for the resulting MtpDraftSeed.
bool finalize_mixed_mtp_proposal_q(SampleOutput& sample,
                                   const SamplingParameters& params,
                                   bool probabilistic);

}  // namespace xllm
