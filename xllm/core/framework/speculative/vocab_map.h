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

#include <cstdint>
#include <vector>

#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_sampling_history.h"

namespace xllm {

// Owns one immutable model vocabulary, its CPU inverse and device mask indices.
// Construct once when loading the drafter; a default map leaves sampling alone.
class VocabMap final {
 public:
  VocabMap() = default;
  VocabMap(torch::Tensor draft_to_target, int64_t target_vocab_size);

  const torch::Tensor& draft_to_target() const { return draft_to_target_; }
  const torch::Tensor& target_to_draft() const { return target_to_draft_; }

  // Only remaps temporary draft-sampler parameters. Histories and model inputs
  // retain target IDs; input mask storage is never modified.
  // Packed masks take precedence and discard any shadowed dense mask.
  // Rows without a legal hot token receive a private placeholder and invalid
  // flags. The caller must reject those proposals through target constraints
  // or fail the request; placeholders are not valid output tokens.
  std::vector<uint8_t> remap(SamplingParameters& params,
                             DraftHistoryBatch& batch,
                             const DraftTokenHistories& histories = {}) const;

 private:
  std::vector<uint8_t> remap_filter_masks(SamplingParameters& params) const;

  torch::Tensor draft_to_target_;
  torch::Tensor target_to_draft_;
  torch::Tensor mask_word_indices_;
  torch::Tensor mask_bit_weights_;
  torch::Tensor packed_bit_weights_;
};

}  // namespace xllm
