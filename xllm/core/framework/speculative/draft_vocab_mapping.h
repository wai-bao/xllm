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

#include <cstdint>

namespace xllm {

class SampleOutput;

// Checkpoint d2t stores offsets: draft_to_target_ids[draft_id] = draft_id +
// d2t[draft_id]. An undefined mapping keeps the shared vocabulary unchanged.
class DraftVocabMapping final {
 public:
  DraftVocabMapping(torch::Tensor draft_token_id_offsets,
                    int64_t draft_vocab_size,
                    int64_t target_vocab_size,
                    const torch::Device& device);
  DraftVocabMapping() = default;

  bool is_identity() const { return !draft_to_target_ids_.defined(); }

  int64_t target_vocab_size() const { return target_vocab_size_; }

  torch::Tensor remap_ids(const torch::Tensor& ids) const {
    if (!draft_to_target_ids_.defined()) {
      return ids;
    }
    return draft_to_target_ids_
        .index_select(0, ids.view(-1).to(draft_to_target_ids_.scalar_type()))
        .view(ids.sizes())
        .to(ids.scalar_type());
  }

  void expand_to_target(const torch::Tensor& draft_logits,
                        torch::Tensor& target_logits) const;

  void remap(SampleOutput& sample_output) const;

  torch::Tensor draft_to_target_ids_;
  int64_t target_vocab_size_ = 0;
};

}  // namespace xllm
