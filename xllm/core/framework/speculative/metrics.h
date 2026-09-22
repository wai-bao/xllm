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
#include <vector>

#include <torch/torch.h>

#include "framework/sampling/sampling_params.h"

namespace xllm {

struct SpeculativeOutputStats {
  std::vector<int64_t> accepted_per_position;
  std::vector<SpeculativeTokenStats> sequence_stats;
  int64_t committed_tokens = 0;
};

SpeculativeOutputStats calculate_speculative_output_stats(
    const torch::Tensor& tokens,
    int64_t num_speculative_tokens);

std::vector<SpeculativeTokenStats> calculate_mtp_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens);

std::vector<SpeculativeTokenStats> calculate_block_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens);

}  // namespace xllm
