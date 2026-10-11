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

#include "core/framework/sampling/draft_proposal.h"

#include <glog/logging.h>

#include "core/framework/sampling/sampling_params.h"

namespace xllm {

DraftProposal DraftProposal::from_steps(
    std::span<const SampleOutput* const> steps,
    bool probabilistic,
    torch::Tensor stacked_tokens) {
  CHECK(!steps.empty());
  std::vector<torch::Tensor> tokens;
  std::vector<torch::Tensor> logits;
  std::vector<torch::Tensor> indices;
  // from_steps consumes shared compact support (Eagle3). Per-position
  // support is supplied directly to the DraftProposal constructor (DFlash2).
  const auto& vocab_ids = steps.front()->vocab_ids;
  // Reused token storage skips the stack; collecting the handles anyway
  // costs only refcount bumps and keeps the loop branch-free.
  tokens.reserve(steps.size());
  if (probabilistic) {
    logits.reserve(steps.size());
    if (vocab_ids.defined()) {
      indices.reserve(steps.size());
    }
  }
  for (const auto* step : steps) {
    tokens.push_back(step->next_tokens);
    if (probabilistic) {
      CHECK(step->logits.defined());
      CHECK_EQ(step->vocab_ids.defined(), vocab_ids.defined());
      logits.push_back(step->logits);
      if (vocab_ids.defined()) {
        CHECK(step->sampled_indices.defined());
        indices.push_back(step->sampled_indices);
      }
    }
  }
  return DraftProposal(
      stacked_tokens.defined() ? std::move(stacked_tokens)
                               : torch::stack(tokens, 1).to(torch::kLong),
      probabilistic ? std::make_optional(torch::stack(logits, 1))
                    : std::nullopt,
      probabilistic ? vocab_ids : torch::Tensor(),
      indices.empty() ? torch::Tensor() : torch::stack(indices, 1));
}

}  // namespace xllm
