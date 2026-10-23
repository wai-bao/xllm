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

#include "models/model_registry.h"

// Platform-agnostic registry entries for the deepseek_v32 family. Every
// platform's deepseek_v32 headers include this file once; the registrations
// carry no platform-specific information, so they live here instead of being
// pasted into every platform copy.
// Include at global scope only: the REGISTER_* variables must not land
// inside another namespace.
namespace xllm::deepseek_v32_registry {

REGISTER_TARGET_SPEC_VERIFY_MODE(deepseek_v32, PYTHON_EXPANDED_VERIFY);

}  // namespace xllm::deepseek_v32_registry
