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

#include "core/framework/speculative/draft_proposal.h"

#include <glog/logging.h>

#include <utility>

namespace xllm {

DraftProposal::DraftProposal() = default;
DraftProposal::DraftProposal(DraftProposal&&) noexcept = default;
DraftProposal& DraftProposal::operator=(DraftProposal&&) noexcept = default;
DraftProposal::~DraftProposal() = default;

DraftProposal::DraftProposal(torch::Tensor token_ids)
    : token_ids_(std::move(token_ids)) {
  validate();
}

DraftProposal::DraftProposal(torch::Tensor token_ids, torch::Tensor logits)
    : token_ids_(std::move(token_ids)), q_(DraftQ{{}, {}, std::move(logits)}) {
  validate();
}

DraftProposal::DraftProposal(torch::Tensor token_ids,
                             torch::Tensor candidate_ids,
                             torch::Tensor positions,
                             torch::Tensor logits)
    : token_ids_(std::move(token_ids)),
      q_(DraftQ{std::move(candidate_ids),
                std::move(positions),
                std::move(logits)}) {
  validate();
  CHECK_EQ(q_.candidate_ids.sizes(), q_.logits.sizes());
  CHECK_EQ(q_.positions.sizes(),
           torch::IntArrayRef({token_ids_.size(0), token_ids_.size(1)}));
}

DraftProposal DraftProposal::from_step_logits(
    torch::Tensor token_ids,
    const std::vector<torch::Tensor>& pages) {
  if (pages.empty()) {
    return DraftProposal(std::move(token_ids));
  }
  return DraftProposal(std::move(token_ids), torch::stack(pages, /*dim=*/1));
}

void DraftProposal::validate() const {
  CHECK(token_ids_.defined());
  CHECK_EQ(token_ids_.dim(), 2);
  if (!q_.logits.defined()) {
    return;
  }
  CHECK_EQ(q_.logits.dim(), 3);
  CHECK_EQ(q_.logits.size(0), token_ids_.size(0));
  CHECK_EQ(q_.logits.size(1), token_ids_.size(1));
}

}  // namespace xllm
