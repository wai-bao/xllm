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
#include "core/util/utils.h"

namespace xllm {

// Model geometry needed to validate a speculative draft input at handoff.
// Kept independent of the algorithm-specific wire payload.
struct DraftInputModelSpec {
  int64_t hidden_size = 0;
  torch::ScalarType hidden_dtype = torch::kFloat;
};

// Hidden width a DraftInput row must carry for this target model.
inline int64_t draft_input_hidden_width(const ModelArgs& model_args) {
  if (util::is_deepseek_v4_model_type(model_args.model_type())) {
    return model_args.hc_mult() * model_args.hidden_size();
  }
  return model_args.hidden_size();
}

}  // namespace xllm
