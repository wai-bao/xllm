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

#include <glog/logging.h>

#include <cstdint>
#include <type_traits>

namespace xllm::util {

template <typename T>
inline std::enable_if_t<std::is_integral_v<T>, T> ceil_div(T value, T divisor) {
  CHECK_GT(divisor, 0) << "divisor must be positive.";
  return value / divisor + static_cast<T>(value % divisor != 0);
}

// Token-bitmask wire layout shared by the grammar generator
// (json_object_grammar), the vocab remapper (vocab_map) and the consumers
// (apply_token_bitmask_wrapper): 32-bit little-endian words, bit i of word
// (i / 32) marks token i. Keeping the indexing here means a layout change
// (e.g. 64-bit words) touches one place instead of every producer/consumer.
constexpr int32_t kBitmaskBitsPerWord = 32;

constexpr int32_t bitmask_word_index(int64_t token_id) {
  return static_cast<int32_t>(token_id / kBitmaskBitsPerWord);
}

constexpr uint32_t bitmask_bit_mask(int64_t token_id) {
  return uint32_t{1} << (static_cast<uint32_t>(token_id) %
                         static_cast<uint32_t>(kBitmaskBitsPerWord));
}

constexpr int32_t bitmask_word_count(int64_t vocab_size) {
  return static_cast<int32_t>(vocab_size / kBitmaskBitsPerWord +
                              (vocab_size % kBitmaskBitsPerWord != 0));
}

}  // namespace xllm::util
