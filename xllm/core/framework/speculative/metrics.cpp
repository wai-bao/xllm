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

#include "metrics.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>

namespace xllm {

SpeculativeOutputStats calculate_speculative_metrics(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens,
    int64_t max_speculative_tokens) {
  CHECK(tokens.defined()) << "speculative output tokens are undefined";
  CHECK_EQ(tokens.dim(), 2) << "speculative output tokens should be 2D";
  CHECK_GE(max_speculative_tokens, 0)
      << "max speculative token count should not be negative";

  const int64_t batch_size = tokens.size(0);
  const int64_t token_width = tokens.size(1);
  CHECK_EQ(proposed_tokens.size(), static_cast<size_t>(batch_size))
      << "proposed token count batch mismatch";

  torch::Tensor int_tokens = tokens.to(torch::kInt64).contiguous();
  const int64_t* data = int_tokens.const_data_ptr<int64_t>();

  SpeculativeOutputStats output;
  output.accepted_per_position.resize(
      static_cast<size_t>(max_speculative_tokens));
  output.sequence_stats.resize(static_cast<size_t>(batch_size));

  for (int64_t row = 0; row < batch_size; ++row) {
    const int64_t proposed = proposed_tokens[static_cast<size_t>(row)];
    CHECK_GE(proposed, 0) << "proposed token count should not be negative";
    CHECK_LE(proposed, max_speculative_tokens)
        << "proposed token count exceeds configured maximum";
    CHECK_LE(proposed + 1, token_width)
        << "proposed token count exceeds output width";

    SpeculativeTokenStats& sequence =
        output.sequence_stats[static_cast<size_t>(row)];
    sequence.proposed_tokens = proposed;
    output.proposed_tokens += proposed;

    const int64_t* row_ptr = data + row * token_width;
    for (int64_t column = 0; column <= proposed; ++column) {
      if (row_ptr[column] < 0) {
        break;
      }
      ++output.committed_tokens;
      if (column == 0) {
        continue;
      }
      ++sequence.accepted_tokens;
      ++output.accepted_tokens;
      ++output.accepted_per_position[static_cast<size_t>(column - 1)];
    }
  }
  return output;
}

SpeculativeOutputStats calculate_speculative_output_stats(
    const torch::Tensor& tokens,
    int64_t num_speculative_tokens) {
  CHECK(tokens.defined()) << "speculative output tokens are undefined";
  CHECK_EQ(tokens.dim(), 2) << "speculative output tokens should be 2D";
  CHECK_LE(tokens.size(1), num_speculative_tokens + 1)
      << "next_tokens width exceeds num_speculative_tokens + 1.";
  std::vector<int32_t> proposed_tokens(
      static_cast<size_t>(tokens.size(0)),
      static_cast<int32_t>(std::max<int64_t>(tokens.size(1) - 1, 0)));
  return calculate_speculative_metrics(
      tokens, proposed_tokens, num_speculative_tokens);
}

std::vector<SpeculativeTokenStats> calculate_mtp_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens) {
  const int64_t max_speculative_tokens =
      proposed_tokens.empty()
          ? 0
          : *std::max_element(proposed_tokens.begin(), proposed_tokens.end());
  return calculate_speculative_metrics(
             tokens, proposed_tokens, max_speculative_tokens)
      .sequence_stats;
}

std::vector<SpeculativeTokenStats> calculate_block_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens) {
  return calculate_mtp_speculative_token_stats(tokens, proposed_tokens);
}

}  // namespace xllm
