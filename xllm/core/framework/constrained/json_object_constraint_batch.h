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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/framework/constrained/json_object_grammar.h"

namespace xllm {

struct JsonObjectConstraintViolation final {
  size_t state_index;
  int64_t token_offset;
  int32_t token_id;
  size_t committed_tokens;
  uint64_t state_fingerprint;
};

// Borrows grammar states for accepted-token advancement and mask construction.
// The caller owns the states and any grammar objects they reference. Token
// tensors are read only during advance() and are never retained.
class JsonObjectConstraintBatch final {
 public:
  explicit JsonObjectConstraintBatch(
      std::vector<JsonObjectGrammarState>& states)
      : states_(states) {}

  // Tokens must already be ready on CPU, int64, and 1-D or 2-D. Each state maps
  // to one token row, or -1 when it has no newly accepted tokens. Uninitialized
  // states are skipped. An undefined tensor is allowed only when every
  // initialized state maps to -1. A negative token ends its row.
  std::vector<JsonObjectConstraintViolation> advance(
      const torch::Tensor& accepted_tokens,
      const std::vector<int32_t>& token_rows);

 private:
  std::vector<JsonObjectGrammarState>& states_;
};

}  // namespace xllm
