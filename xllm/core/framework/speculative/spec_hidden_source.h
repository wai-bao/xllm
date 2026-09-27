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

namespace xllm {

enum class SpecHiddenRole : uint8_t { TARGET, DRAFT };

// The source of the hidden state consumed by a speculative drafter. Token-row
// selection is a separate runtime concern; these values never imply a copy.
enum class SpecHiddenSource : uint8_t {
  FINAL_OUTPUT,     // Borrow the model's primary hidden output (ordinary MTP).
  CAPTURED_LAYERS,  // Require the target's intermediate-layer aux output.
  MODEL_HOOK,       // Require a model-specific hidden carried in aux output.
  MODEL_PROVIDED,   // Prefer model-provided aux; otherwise use primary output.
};

}  // namespace xllm
