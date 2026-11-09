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

#include <cstdint>
#include <optional>
#include <string_view>

#include "core/framework/speculative/draft_input.h"

namespace xllm {

enum class SpeculativeHandoffKind : uint8_t {
  NONE,
  BOOTSTRAP_TOKEN,
  DRAFT_INPUT,
};

SpeculativeHandoffKind pd_speculative_handoff_kind(
    std::string_view algorithm,
    int32_t num_speculative_tokens);

// Validation half of the decode-side PD handoff, kept free of Sequence so
// the speculative layer never depends on the request layer. Returns nullptr
// when the handoff may be recorded; otherwise a static reason string the
// caller logs before rejecting the request. NONE is always valid.
const char* validate_pd_handoff(SpeculativeHandoffKind kind,
                                int32_t slot_id,
                                int64_t token_id,
                                const std::optional<DraftInput>& draft_input,
                                bool probabilistic,
                                int64_t vocab_size,
                                const DraftInputModelSpec* draft_spec);

}  // namespace xllm
