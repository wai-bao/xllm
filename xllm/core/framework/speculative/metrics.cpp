/* Copyright 2025-2026 The xLLM Authors.

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

#include "metrics.h"

#include <glog/logging.h>

#include <cstddef>

namespace xllm {
namespace {

std::vector<SpeculativeTokenStats> calculate_contiguous_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens,
    int32_t accepted_token_offset) {
  CHECK(tokens.defined()) << "speculative output tokens are undefined";
  CHECK_EQ(tokens.dim(), 2) << "speculative output tokens should be 2D";
  const int64_t batch_size = tokens.size(0);
  const int64_t token_width = tokens.size(1);
  CHECK_EQ(proposed_tokens.size(), static_cast<size_t>(batch_size))
      << "proposed token count batch mismatch";

  torch::Tensor int_tokens = tokens.to(torch::kInt64).contiguous();
  const int64_t* data = int_tokens.const_data_ptr<int64_t>();
  std::vector<SpeculativeTokenStats> sequence_stats(
      static_cast<size_t>(batch_size));
  for (int64_t row = 0; row < batch_size; ++row) {
    const int64_t proposed = proposed_tokens[static_cast<size_t>(row)];
    CHECK_GE(proposed, 0) << "proposed token count should not be negative";
    CHECK_LE(proposed + accepted_token_offset, token_width)
        << "proposed token count exceeds output width";

    SpeculativeTokenStats& stats = sequence_stats[static_cast<size_t>(row)];
    stats.proposed_tokens = proposed;
    const int64_t* row_ptr = data + row * token_width;
    for (int64_t column = accepted_token_offset;
         column < proposed + accepted_token_offset;
         ++column) {
      if (row_ptr[column] < 0) {
        break;
      }
      ++stats.accepted_tokens;
    }
  }
  return sequence_stats;
}

}  // namespace

std::vector<SpeculativeTokenStats> calculate_mtp_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens) {
  return calculate_contiguous_token_stats(
      tokens, proposed_tokens, /*accepted_token_offset=*/1);
}

std::vector<SpeculativeTokenStats> calculate_block_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens) {
  // The contiguous output includes one target-model token: the replacement
  // at the first rejection, or the bonus after all drafts are accepted.
  // Count one fewer valid token, capped by each row's actual proposal width.
  return calculate_contiguous_token_stats(
      tokens, proposed_tokens, /*accepted_token_offset=*/1);
}

SpeculativeOutputStats calculate_speculative_output_stats(
    const torch::Tensor& tokens,
    int64_t num_speculative_tokens) {
  torch::Tensor int_tokens = tokens.to(torch::kInt64).contiguous();
  const int64_t* data = int_tokens.const_data_ptr<int64_t>();
  const int64_t batch_size = int_tokens.size(0);
  const int64_t token_width = int_tokens.size(1);
  CHECK_LE(token_width, num_speculative_tokens + 1)
      << "next_tokens width exceeds num_speculative_tokens + 1.";
  SpeculativeOutputStats stats;
  stats.accepted_per_position.resize(
      static_cast<size_t>(num_speculative_tokens));
  stats.sequence_stats.resize(static_cast<size_t>(batch_size));
  for (int64_t row = 0; row < batch_size; ++row) {
    const int64_t* row_ptr = data + row * token_width;
    SpeculativeTokenStats& sequence_stats =
        stats.sequence_stats[static_cast<size_t>(row)];
    sequence_stats.proposed_tokens = token_width - 1;
    for (int64_t column = 0; column < token_width; ++column) {
      if (row_ptr[column] < 0) {
        continue;
      }
      ++stats.committed_tokens;
      if (column > 0) {
        ++stats.accepted_per_position[static_cast<size_t>(column - 1)];
        ++sequence_stats.accepted_tokens;
      }
    }
  }
  return stats;
}

}  // namespace xllm
