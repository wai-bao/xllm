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

// Validate decoded state before the scheduler mutates the receiving request.
const char* validate_pd_bootstrap_handoff(int32_t slot_id, int64_t token_id);

const char* validate_pd_draft_input_handoff(
    int32_t slot_id,
    int64_t token_id,
    const std::optional<DraftInput>& draft_input,
    bool probabilistic,
    int64_t vocab_size,
    const DraftInputModelSpec* draft_spec);

// Shared import prelude for every decode-side PD draft_input consumer (the
// eager worker and the task pipeline): re-checks the model contract the
// scheduler-side validation guaranteed, then moves the payload onto the
// consuming device. The returned draft_input owns fresh device storage, so
// the store can snapshot it without re-cloning. Callers gate on
// draft_input.has_value() themselves.
DraftInput import_pd_draft_input(const DraftInput& draft_input,
                                 const DraftInputModelSpec& spec,
                                 const torch::Device& device);

}  // namespace xllm
