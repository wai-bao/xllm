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
#include <span>
#include <vector>

#include "core/framework/batch/decode_input_builder.h"

namespace xllm {

namespace spec_verify_layout {

enum class ValidateAttentionMode : uint8_t {
  DECODE,
  CHUNKED_PREFILL,
};

struct ValidateInputPlan {
  int32_t uniform_width = 0;
  int32_t logical_block_size = 0;
  bool positions_decoupled = false;
  ValidateAttentionMode attention_mode = ValidateAttentionMode::DECODE;
  // True when token/position tensors are materialized lazily by the
  // explicit-replay update instead of up front.
  bool deferred_tokens = false;
};

struct ValidateInputLayout {
  DecodeBuildBuffers rows;
  // Chunked-prefill verify emits one per-sequence row of q/kv metadata while
  // the expanded token rows stay metadata-free; these carry that row.
  std::vector<int32_t> chunked_kv_seq_lens;
  int32_t chunked_kv_max_seq_len = 0;
  int32_t total_tokens = 0;
};

template <typename TargetInput>
ValidateInputLayout build_validate_input_layout(const TargetInput& input,
                                                const ValidateInputPlan& plan);

template <typename TargetInput>
void apply_validate_input_layout(TargetInput& validate_input,
                                 const ValidateInputPlan& plan,
                                 ValidateInputLayout& layout);

#if defined(USE_NPU)
template <typename Params>
std::vector<int32_t> expand_validate_linear_state_ids(
    const Params& input_params,
    int32_t uniform_width);
#endif

}  // namespace spec_verify_layout

}  // namespace xllm
