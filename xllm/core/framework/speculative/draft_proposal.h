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

// The draft distribution q. Undefined candidate ids mean dense [batch,
// steps, vocab] logits; otherwise a sparse candidate set of [batch, steps,
// top_k] logits keyed by candidate ids, with positions naming the drafted
// candidate column per step.
struct DraftQ {
  torch::Tensor candidate_ids;
  torch::Tensor positions;
  torch::Tensor logits;
};

// Completed draft tokens and their distribution q, owned through
// verification. Undefined q logits mean a point mass on the drafted tokens;
// -1 sentinels are force-rejected by the verifier.
class DraftProposal final {
 public:
  DraftProposal();
  explicit DraftProposal(torch::Tensor token_ids);
  DraftProposal(torch::Tensor token_ids, torch::Tensor logits);
  DraftProposal(torch::Tensor token_ids,
                torch::Tensor candidate_ids,
                torch::Tensor positions,
                torch::Tensor logits);
  // Empty pages mean a point mass.
  static DraftProposal from_step_logits(
      torch::Tensor token_ids,
      const std::vector<torch::Tensor>& pages);

  DraftProposal(const DraftProposal&) = delete;
  DraftProposal& operator=(const DraftProposal&) = delete;
  DraftProposal(DraftProposal&&) noexcept;
  DraftProposal& operator=(DraftProposal&&) noexcept;
  ~DraftProposal();

  const torch::Tensor& token_ids() const { return token_ids_; }
  const DraftQ& q() const { return q_; }

  // Lifetime owner for no-sync draft inputs that must outlive the verify
  // step's stream sync; never read by the rejection sampler.
  std::vector<std::shared_ptr<const void>>& keep_alive() { return keep_alive_; }

 private:
  void validate() const;

  torch::Tensor token_ids_;
  DraftQ q_;
  std::vector<std::shared_ptr<const void>> keep_alive_;
};

}  // namespace xllm
