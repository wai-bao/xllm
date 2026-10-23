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

#include <torch/torch.h>

#include <cstdint>

#include "core/framework/model/model_args.h"
#include "models/model_registry.h"

namespace xllm {

// Model geometry needed to validate a speculative draft input at handoff.
// Kept independent of the algorithm-specific wire payload.
struct DraftInputModelSpec {
  int64_t hidden_size = 0;
  torch::ScalarType hidden_dtype = torch::kFloat;
};

// Hidden width a DraftInput row must carry for this target model. The
// default is hidden_size(); targets whose draft embedding width differs
// (e.g. DeepSeek-V4 hyper-connections) register their own policy with
// ModelRegistry::register_draft_hidden_width.
inline int64_t draft_input_hidden_width(const ModelArgs& model_args) {
  return ModelRegistry::get_draft_hidden_width(model_args.model_type(),
                                               model_args);
}

}  // namespace xllm
