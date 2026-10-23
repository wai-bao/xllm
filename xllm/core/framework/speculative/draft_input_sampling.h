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
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/framework/sampling/json_object_grammar.h"
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

// Row tensors draft sampling repeats to match the expanded draft token count:
// kSamplingRowTensors plus the draft-only members (min_p, seed_offsets, seeds,
// logits_bias). The first kSamplingRowTensors.size() entries mirror
// kSamplingRowTensors; the static_assert below keeps the two in lockstep.
inline constexpr std::array<torch::Tensor SamplingParameters::*, 14>
    kSamplingRepeatedRows = {&SamplingParameters::frequency_penalties,
                             &SamplingParameters::presence_penalties,
                             &SamplingParameters::repetition_penalties,
                             &SamplingParameters::temperatures,
                             &SamplingParameters::top_p,
                             &SamplingParameters::top_k,
                             &SamplingParameters::unique_token_ids,
                             &SamplingParameters::unique_token_counts,
                             &SamplingParameters::unique_token_ids_lens,
                             &SamplingParameters::do_sample,
                             &SamplingParameters::min_p,
                             &SamplingParameters::seed_offsets,
                             &SamplingParameters::seeds,
                             &SamplingParameters::logits_bias};

static_assert(
    []() consteval {
      for (size_t i = 0; i < kSamplingRowTensors.size(); ++i) {
        if (kSamplingRepeatedRows[i] != kSamplingRowTensors[i]) {
          return false;
        }
      }
      return true;
    }(),
    "kSamplingRepeatedRows must start with the kSamplingRowTensors members");

// Repeats every kSamplingRepeatedRows member by the same repeats (scalar count
// or per-row tensor). Repeats follows repeat_interleave semantics; a new
// kSamplingRowTensors member must also be added to kSamplingRepeatedRows; the
// static_assert above enforces this.
template <typename Repeats>
inline void repeat_sampling_rows(SamplingParameters& sampling_params,
                                 const Repeats& repeats) {
  for (const auto member : kSamplingRepeatedRows) {
    repeat_interleave_if_defined(sampling_params.*member, repeats);
  }
}

// Reports whether select_input_confidence has an available source for this
// sample output, without reading any tensor data: the carried draft input's
// selected probs, then the sampler's probs, then a per-row softmax over
// logits+next_tokens. False means no source is available and
// select_input_confidence would CHECK-fail. The carried seed must come from
// the same forward as sample_output.
bool has_input_confidence(const SampleOutput& sample_output,
                          const std::optional<DraftInput>& draft_input,
                          const torch::Tensor& logits);

// Extracts the pre-sampling in-vocabulary confidence without allocating a full
// softmax and without modifying the proposal distribution q. When a 1D selected
// probability is already available, this only normalizes the probability dtype
// and reads neither tokens, logits nor sample indices. The carried seed must
// come from the same forward as sample_output.
torch::Tensor select_input_confidence(
    const SampleOutput& sample_output,
    const std::optional<DraftInput>& draft_input,
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

// Folds the JSON grammar acceptance into a vocabulary invalid-row mask.
// JSON validation rejects the remapper's placeholder before acceptance. The
// target can still sample a legal token outside the hot vocabulary, including
// EOS after a complete object. Committed-state errors remain request
// failures; only these unverified draft proposals are recoverable, so rows
// whose grammar state is initialized and valid are dropped from the mask.
std::vector<uint8_t> filter_vocabulary_failures(
    std::vector<uint8_t> invalid_rows,
    const std::vector<JsonObjectGrammarState>& json_states);

// Shared finish tail of a freshly sampled draft input: derives the
// point-mass marker for all-greedy probabilistic batches, assembles the
// DraftInput from the
// sample output plus the already-remapped draft hidden states, and checks it
// against both the draft model contract and the sampling-mode contract. The
// eager worker and the task pipeline route through this so the seed assembly
// contract cannot drift between them.
DraftInput finish_draft_input(SampleOutput& sample,
                              torch::Tensor hidden_states,
                              const SamplingParameters& params,
                              bool probabilistic,
                              torch::Tensor selected_probs,
                              const DraftInputModelSpec& spec);

}  // namespace xllm
