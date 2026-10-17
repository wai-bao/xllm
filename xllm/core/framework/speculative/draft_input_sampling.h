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
#include <span>
#include <string>
#include <vector>

#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_input.h"
#include "core/framework/speculative/draft_sampling_history.h"

namespace xllm {

struct DraftInputSamplingState {
  SamplingParameters sampling_params;
  std::vector<JsonObjectGrammarState> json_object_states;
  // An empty vector means no row failed; otherwise it holds B flags and a
  // failed row must not publish its seed.
  std::vector<uint8_t> invalid_rows;
  DraftTokenHistories histories;
};

// Reports whether select_input_confidence has an available source for this
// sample output, without reading any tensor data: the carried draft input's
// selected probs, then the sampler's probs, then a per-row softmax over
// logits+next_tokens. False means no source is available and
// select_input_confidence would CHECK-fail.
bool has_input_confidence(const SampleOutput& sample_output,
                          const torch::Tensor& logits);

// Extracts the pre-sampling in-vocabulary confidence without allocating a full
// softmax and without modifying the proposal distribution q. When a 1D selected
// probability is already available, this only normalizes the probability dtype
// and reads neither tokens, logits nor sample indices.
torch::Tensor select_input_confidence(const SampleOutput& sample_output,
                                      const torch::Tensor& logits,
                                      const torch::Tensor& sample_idxes);

// Appends only the tokens committed by this step; whether an existing root is
// already in the history is the caller's decision. The unconstrained path does
// not read committed_tokens device data, and the constrained path accepts only
// the [B, W] layout with a -1 suffix. histories is the request-level CPU
// authority; batch is reused in execution-stream order.
// Whether prepare_draft_input_sampling dereferences the committed-tokens
// device tensor: penalty history or grammar state requires the host copy. The
// event-sync callers must gate on the same predicate to avoid racing the
// pinned read.
inline bool needs_committed_tokens(
    const SamplingParameters& params,
    std::span<const JsonObjectGrammarState> states) {
  return has_penalty_params(params) ||
         std::any_of(states.begin(), states.end(), [](const auto& state) {
           return state.initialized();
         });
}

DraftInputSamplingState prepare_draft_input_sampling(
    SamplingParameters params,
    std::vector<JsonObjectGrammarState> states,
    const torch::Tensor& committed_tokens,
    const torch::Device& device,
    DraftHistoryBatch& batch,
    DraftTokenHistories histories = {});

// Overwrites params with a freshly built DRAFT-phase grammar bitmask and clears
// the possibly stale full filter mask, so downstream sampling follows the
// bitmask alone. prepare_draft_input_sampling and import_sampling_histories
// share this single implementation.
void apply_draft_grammar_bitmask(
    SamplingParameters& params,
    const std::vector<JsonObjectGrammarState>& json_states,
    const torch::Device& device);

// Sampler returns softmax for every row in a mixed greedy/random batch. The
// greedy rows were selected deterministically, so replace their q with point
// masses in place after any draft-to-target vocabulary remapping.
void normalize_mixed_proposal_q(const torch::Tensor& token_ids,
                                torch::Tensor& probs,
                                const torch::Tensor& do_sample);

// Finalizes the proposal q of a freshly sampled draft input in one call:
// drops q for non-probabilistic and all-greedy seeds (marking the latter as
// point-mass), materializes point masses for mixed greedy/random batches,
// and leaves all-random q untouched. Must run after any draft-to-target
// vocabulary remapping, which consumes raw sampler q. Returns the
// point_mass_q marker for the resulting DraftInput.
bool finalize_mixed_proposal_q(SampleOutput& sample,
                               const SamplingParameters& params,
                               bool probabilistic);

// Shared finish tail of a freshly sampled draft input: finalizes the proposal
// q (see finalize_mixed_proposal_q), assembles the DraftInput from the
// sample output plus the already-remapped draft hidden states, and checks it
// against both the draft model contract and the sampling-mode contract. The
// eager worker and the task pipeline route through this so the seed assembly
// contract cannot drift between them.
DraftInput finish_draft_input(SampleOutput& sample,
                              torch::Tensor hidden_states,
                              const SamplingParameters& params,
                              bool probabilistic,
                              torch::Tensor selected_probs,
                              int64_t vocab_size,
                              const DraftInputModelSpec& spec);

}  // namespace xllm
