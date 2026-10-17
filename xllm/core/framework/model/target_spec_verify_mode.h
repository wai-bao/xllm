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

// Static target verification contract declared with each model registration.
// Runtime separately checks whether the current batch and device can execute
// the advertised mode.
enum class TargetSpecVerifyMode : int8_t {
  GENERIC,
  CAUSAL_CHUNKED_PREFILL,
  // The target's verify kernel requires the same width for every sequence.
  UNIFORM_EXPANDED_VERIFY,
  // Expanded replay is available through the Python executor only.
  PYTHON_EXPANDED_VERIFY,
};

}  // namespace xllm
