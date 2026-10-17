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

#include "core/framework/speculative/capture_hidden_mode.h"

#include "core/framework/config/speculative_config.h"

namespace xllm {

// Resolve once per worker, not once per token. Ordinary MTP consumes the
// primary model output; native DeepSeek-V4 MTP uses a model-specific hook
// on both target and draft. EAGLE3/block-diffusion targets require captured
// layers. EAGLE3 drafts may expose recurrent hidden in aux or primary output.
CaptureHiddenMode resolve_capture_hidden_mode(std::string_view algorithm,
                                              CaptureHiddenRole role,
                                              bool model_hook) {
  if (model_hook) {
    return CaptureHiddenMode::MODEL_HOOK;
  }
  if (role == CaptureHiddenRole::DRAFT) {
    return SpeculativeConfig::is_eagle3_algorithm(algorithm)
               ? CaptureHiddenMode::MODEL_PROVIDED
               : CaptureHiddenMode::FINAL_OUTPUT;
  }
  return SpeculativeConfig::requires_aux_hidden_capture(algorithm)
             ? CaptureHiddenMode::CAPTURED_LAYERS
             : CaptureHiddenMode::FINAL_OUTPUT;
}

}  // namespace xllm
