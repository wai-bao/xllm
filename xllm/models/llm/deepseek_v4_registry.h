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

// Platform-agnostic registry entries for the deepseek_v4 family. Every
// platform's deepseek_v4 headers include this file once; the registrations
// carry no platform-specific information, so they live here instead of being
// pasted into every platform copy.
// Include at global scope only: the nested namespace and REGISTER_* variables
// must not land inside another namespace.
namespace xllm::deepseek_v4_registry {

// Hyper-connections scale the draft embedding width beyond hidden_size.
inline const bool draft_hidden_width_registered = [] {
  ModelRegistry::register_draft_hidden_width(
      "deepseek_v4", [](const ModelArgs& args) {
        return args.hc_mult() * args.hidden_size();
      });
  return true;
}();

REGISTER_TARGET_SPEC_VERIFY_MODE(deepseek_v4, PYTHON_EXPANDED_VERIFY);

REGISTER_MTP_MODEL_PAIR(deepseek_v4, deepseek_v4_mtp);

}  // namespace xllm::deepseek_v4_registry
