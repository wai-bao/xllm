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

#include <torch/types.h>

#include <memory>
#include <vector>

namespace xllm {

// Completed draft tokens and their distribution q, owned through verification.
// Verification tokens may contain -1 sentinels; q retains the original draft
// distribution. q is either a point mass on the drafted tokens (no logits), a
// dense [batch, steps, vocab] distribution, or a sparse candidate-set
// distribution (see the sparse constructor).
class DraftProposal final {
 public:
  DraftProposal();
  explicit DraftProposal(torch::Tensor token_ids);
  DraftProposal(torch::Tensor token_ids, torch::Tensor logits);
  // Stacks per-step [batch, vocab] distributions into dense q [batch, steps,
  // vocab]; empty pages mean a point mass.
  static DraftProposal from_step_logits(
      torch::Tensor token_ids,
      const std::vector<torch::Tensor>& pages);
  // Sparse candidate-set q: candidate ids are unique within each step and
  // index the target vocabulary; positions [batch, steps] records the
  // candidate column drafted at each step, so q gathers never search.
  DraftProposal(torch::Tensor token_ids,
                torch::Tensor candidate_ids,
                torch::Tensor positions,
                torch::Tensor logits);

  DraftProposal(const DraftProposal&) = delete;
  DraftProposal& operator=(const DraftProposal&) = delete;
  DraftProposal(DraftProposal&&) noexcept;
  DraftProposal& operator=(DraftProposal&&) noexcept;
  ~DraftProposal();

  const torch::Tensor& token_ids() const { return token_ids_; }
  // logsumexp(q) per [batch, step]; undefined for a point mass.
  torch::Tensor logsumexp() const;
  // log_p <- log(p/q) at the given gather indices; negative sentinels are
  // rejected by the verifier, not here.
  void apply_acceptance(torch::Tensor& log_p,
                        const torch::Tensor& token_ids,
                        const torch::Tensor& lse) const;
  // log_p <- log((p - q)+) at the selected steps. Negative rejected tokens
  // disable q, recovering directly from p.
  void apply_residual(torch::Tensor& log_p,
                      const torch::Tensor& steps,
                      const torch::Tensor& rejected_tokens,
                      const torch::Tensor& lse) const;

  // Lifetime owner for no-sync draft inputs that must outlive the verify
  // step's stream sync; never read by the rejection sampler.
  std::vector<std::shared_ptr<const void>>& keep_alive() { return keep_alive_; }

 private:
  // Sparse candidate-set q encoding; defined in the .cpp.
  struct Distribution;

  void validate() const;

  torch::Tensor token_ids_;
  // Dense q; undefined means a point mass on the drafted tokens.
  torch::Tensor logits_;
  std::unique_ptr<Distribution> distribution_;
  std::vector<std::shared_ptr<const void>> keep_alive_;
};

}  // namespace xllm
