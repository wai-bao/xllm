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
#include <span>
#include <vector>

#include "core/framework/constrained/json_object_grammar.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_input.h"
#include "core/framework/speculative/draft_sampling_history.h"
#include "core/util/tensor_helper.h"

namespace xllm {

struct DraftInputSamplingState {
  SamplingParameters sampling_params;
  std::vector<JsonObjectGrammarState> json_object_states;
  // An empty vector means no row failed; otherwise it holds B flags and a
  // failed row must not publish its seed.
  std::vector<uint8_t> invalid_rows;
  DraftTokenHistories histories;
};

// Pre-sampling in-vocabulary confidence without a full softmax materialization
// or modifying q. The carried seed must come from the same forward as
// sample_output.
torch::Tensor select_input_confidence(
    const SampleOutput& sample_output,
    const std::optional<DraftInput>& draft_input,
    const torch::Tensor& logits,
    const torch::Tensor& sample_idxes);

// Whether any row has an initialized grammar state and thus needs its
// committed tokens appended.
inline bool needs_committed_tokens_for_grammar(
    std::span<const JsonObjectGrammarState> states) {
  return std::any_of(states.begin(), states.end(), [](const auto& state) {
    return state.initialized();
  });
}

// Whether prepare_draft_input_sampling dereferences the committed-tokens
// device tensor. Event-sync callers must gate on the same predicate.
inline bool needs_committed_tokens(
    const SamplingParameters& params,
    std::span<const JsonObjectGrammarState> states) {
  return has_penalty_params(params) ||
         needs_committed_tokens_for_grammar(states);
}

// Appends only the tokens committed by this step to the request-level CPU
// histories; batch is the reused device projection.
DraftInputSamplingState prepare_draft_input_sampling(
    SamplingParameters params,
    std::vector<JsonObjectGrammarState> states,
    const torch::Tensor& committed_tokens,
    const torch::Device& device,
    DraftHistoryBatch& batch,
    DraftTokenHistories histories = {});

// Replaces params' filter mask with a freshly built DRAFT-phase grammar
// bitmask so downstream sampling follows the bitmask alone.
void apply_draft_grammar_bitmask(
    SamplingParameters& params,
    const std::vector<JsonObjectGrammarState>& json_states,
    const torch::Device& device);

// Folds JSON grammar acceptance into a vocabulary invalid-row mask. The
// target can still sample a legal token outside the hot vocabulary;
// committed-state errors remain request failures and are dropped from the
// mask.
std::vector<uint8_t> filter_vocabulary_failures(
    std::vector<uint8_t> invalid_rows,
    const std::vector<JsonObjectGrammarState>& json_states);

// Invalid-row masks are either empty (no row failed) or hold one flag per
// row. Merges two masks under that convention; an empty operand means "no
// failures" and never widens the other side.
std::vector<uint8_t> merge_invalid_rows(const std::vector<uint8_t>& lhs,
                                        const std::vector<uint8_t>& rhs);

// Finish tail of a freshly sampled draft input: derives the point-mass
// marker, assembles the DraftInput, and checks the model/sampling-mode
// contracts.
DraftInput finish_draft_input(SampleOutput& sample,
                              torch::Tensor hidden_states,
                              const SamplingParameters& params,
                              bool probabilistic,
                              torch::Tensor selected_probs,
                              const DraftInputModelSpec& spec);

}  // namespace xllm
