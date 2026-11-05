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

#include "core/framework/model/model_output.h"

namespace xllm {

// Framework-level orchestration of the logits stage: hidden states →
// head projection with row selection. Mirrors vLLM's LogitsProcessor layer
// and SGLang's LogitsProcessor: models never call lm_head directly; the
// runtime drives one contract. Pre-logits transforms (final norms, head
// scales) stay at the end of model forward, per xLLM convention.
//
// Constraints, temperatures, and top-k/top-p remain in the sampling layer
// (sampling/logits_processor.h apply_* / process_logits) — this class stops
// at producing raw logits rows. That split matches vLLM (Sampler owns
// post-projection transforms) and keeps every stage a pure, testable
// function.
//
// The receiver is a template parameter so both CausalLM and RecModel satisfy
// the logits(hidden, idxes[, out_hidden]) contract without a shared base.
class LogitsProcessor {
 public:
  // hidden_states: [num_tokens, hidden_size]; selected_idxes: [num_rows] or
  // undefined for all rows; returns [num_rows, vocab_size] logits, aligned
  // with selected_idxes order.
  template <typename Model>
  torch::Tensor forward(Model& model,
                        const torch::Tensor& hidden_states,
                        const torch::Tensor& selected_idxes) {
    return model.logits(hidden_states, selected_idxes);
  }

  // Single-pass variant that also exports the selected hidden rows aligned
  // with the returned logits (the ConfidenceHead and CP speculative paths
  // consume the hidden without a second projection).
  template <typename Model>
  torch::Tensor forward(Model& model,
                        const torch::Tensor& hidden_states,
                        const torch::Tensor& selected_idxes,
                        torch::Tensor& out_hidden) {
    return model.logits(hidden_states, selected_idxes, out_hidden);
  }
};

}  // namespace xllm
