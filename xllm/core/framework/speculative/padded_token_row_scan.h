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
#include <limits>

namespace xllm::detail {

// Reports the leading valid token prefix and whether the remainder is -1
// padding. Callers choose their own policy for malformed tails.
struct PaddedTokenRowScan final {
  int64_t length = 0;
  bool over_int32_range = false;
  bool tail_all_padding = true;
};

inline PaddedTokenRowScan scan_padded_token_row(const int64_t* data,
                                                int64_t width) {
  PaddedTokenRowScan scan;
  int64_t column = 0;
  for (; column < width; ++column) {
    const int64_t token = data[column];
    if (token < 0) {
      break;
    }
    if (token > std::numeric_limits<int32_t>::max()) {
      scan.over_int32_range = true;
      break;
    }
  }
  scan.length = column;
  for (int64_t tail = column; tail < width; ++tail) {
    if (data[tail] != -1) {
      scan.tail_all_padding = false;
      break;
    }
  }
  return scan;
}

}  // namespace xllm::detail
