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
#include <torch/torch.h>

#include <limits>
#include <utility>

#include "core/util/tensor_helper.h"

namespace xllm {
namespace {

// log_q <- log((exp(log_q) - exp(log_p))+) in log space, shared by the
// residual-recovery paths.
void subtract_log_mass(torch::Tensor& log_q, const torch::Tensor& log_p) {
  log_q.sub_(log_p)
      .nan_to_num_(/*nan=*/0.0)
      .clamp_max_(0)
      .exp_()
      .neg_()
      .log1p_();
}

}  // namespace

// Sparse candidate-set q: per-step [batch, steps, top_k] candidate logits
// keyed by candidate ids (unique per step, target vocabulary), plus the
// sampled candidate column per step.
struct DraftProposal::Distribution {
  torch::Tensor candidate_ids;
  torch::Tensor positions;
  torch::Tensor logits;

  torch::Tensor logsumexp() const {
    return torch::logsumexp(logits.to(torch::kFloat32), /*dim=*/-1);
  }

  void apply_acceptance(torch::Tensor& log_p, const torch::Tensor& lse) const {
    log_p.sub_((index_select_2d(logits, /*dim=*/-1, positions))
                   .to(torch::kFloat32)
                   .sub(lse));
  }

  void apply_residual(torch::Tensor& log_p,
                      const torch::Tensor& steps,
                      const torch::Tensor& rejected_tokens,
                      const torch::Tensor& lse) const {
    const torch::Tensor rows = steps.expand({log_p.size(0), logits.size(-1)});
    const torch::Tensor ids = index_select_2d(candidate_ids, /*dim=*/1, rows);
    torch::Tensor log_q =
        (index_select_2d(logits, /*dim=*/1, rows)).to(torch::kFloat32);
    log_q.sub_(lse.gather(/*dim=*/1, steps))
        .masked_fill_(rejected_tokens.lt(0),
                      -std::numeric_limits<float>::infinity());
    const torch::Tensor support_log_p = log_p.gather(/*dim=*/-1, ids);
    subtract_log_mass(log_q, support_log_p);
    log_p.scatter_(/*dim=*/-1, ids, log_q.add_(support_log_p));
  }
};

DraftProposal::DraftProposal() = default;
DraftProposal::DraftProposal(DraftProposal&&) noexcept = default;
DraftProposal& DraftProposal::operator=(DraftProposal&&) noexcept = default;
DraftProposal::~DraftProposal() = default;

DraftProposal::DraftProposal(torch::Tensor token_ids)
    : token_ids_(std::move(token_ids)) {
  validate();
}

DraftProposal::DraftProposal(torch::Tensor token_ids, torch::Tensor logits)
    : token_ids_(std::move(token_ids)), logits_(std::move(logits)) {
  validate();
}

DraftProposal::DraftProposal(torch::Tensor token_ids,
                             torch::Tensor candidate_ids,
                             torch::Tensor positions,
                             torch::Tensor logits)
    : token_ids_(std::move(token_ids)),
      distribution_(new Distribution{std::move(candidate_ids),
                                     std::move(positions),
                                     std::move(logits)}) {
  validate();
  CHECK_EQ(distribution_->candidate_ids.dim(), 3);
  CHECK_EQ(distribution_->candidate_ids.sizes(), distribution_->logits.sizes());
  CHECK_EQ(distribution_->positions.sizes(),
           torch::IntArrayRef({distribution_->candidate_ids.size(0),
                               distribution_->candidate_ids.size(1)}));
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
  if (!logits_.defined()) {
    return;
  }
  CHECK_EQ(logits_.dim(), 3);
  CHECK_EQ(logits_.size(0), token_ids_.size(0));
  CHECK_EQ(logits_.size(1), token_ids_.size(1));
}

torch::Tensor DraftProposal::logsumexp() const {
  if (distribution_ != nullptr) {
    return distribution_->logsumexp();
  }
  return logits_.defined()
             ? torch::logsumexp(logits_.to(torch::kFloat32), /*dim=*/-1)
             : torch::Tensor();
}

void DraftProposal::apply_acceptance(torch::Tensor& log_p,
                                     const torch::Tensor& token_ids,
                                     const torch::Tensor& lse) const {
  if (distribution_ != nullptr) {
    distribution_->apply_acceptance(log_p, lse);
    return;
  }
  if (logits_.defined()) {
    log_p.sub_((index_select_2d(logits_, /*dim=*/-1, token_ids))
                   .to(torch::kFloat32)
                   .sub(lse));
  }
}

void DraftProposal::apply_residual(torch::Tensor& log_p,
                                   const torch::Tensor& steps,
                                   const torch::Tensor& rejected_tokens,
                                   const torch::Tensor& lse) const {
  if (distribution_ != nullptr) {
    distribution_->apply_residual(log_p, steps, rejected_tokens, lse);
    return;
  }
  if (!logits_.defined()) {
    // A point mass concentrates q on the drafted token: the residual keeps
    // every p mass except the rejected token itself.
    const torch::Tensor ids = rejected_tokens.clamp_min(0);
    const torch::Tensor remaining = log_p.gather(/*dim=*/-1, ids);
    remaining.masked_fill_(rejected_tokens.ge(0),
                           -std::numeric_limits<float>::infinity());
    log_p.scatter_(/*dim=*/-1, ids, remaining);
    return;
  }
  const torch::Tensor rows = steps.expand({log_p.size(0), logits_.size(-1)});
  torch::Tensor log_q = (index_select_2d(logits_, /*dim=*/1, rows))
                            .to(torch::kFloat32)
                            .sub_(lse.gather(/*dim=*/1, steps));
  log_q.masked_fill_(rejected_tokens.lt(0),
                     -std::numeric_limits<float>::infinity());
  subtract_log_mass(log_q, log_p);
  log_p.add_(log_q);
}

}  // namespace xllm
