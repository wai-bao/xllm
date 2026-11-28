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

namespace xllm::npu {

inline bool is_acl_graph_compatibility_batch_size(uint64_t batch_size,
                                                  uint64_t max_batch_size) {
  if (batch_size == max_batch_size) {
    return true;
  }
  if (batch_size <= 16) {
    return batch_size != 0 && (batch_size & (batch_size - 1)) == 0;
  }
  return batch_size >= 32 && batch_size % 16 == 0;
}

constexpr uint32_t acl_graph_max_global_batch_size(
    uint32_t configured_max_batch_size,
    uint32_t local_batch_size_limit,
    uint32_t dp_size) {
  const uint64_t graph_capacity =
      static_cast<uint64_t>(local_batch_size_limit) * dp_size;
  return graph_capacity < configured_max_batch_size
             ? static_cast<uint32_t>(graph_capacity)
             : configured_max_batch_size;
}

bool is_acl_graph_decode_capture_allowed(uint32_t batch_size,
                                         uint32_t max_batch_size,
                                         uint32_t dp_size,
                                         bool is_graph_warmup,
                                         int32_t decode_query_len);

// A nonzero DP-local graph limit enables the no-padding compatibility filter.
std::vector<int32_t> get_acl_graph_mtp_warmup_batch_sizes(
    uint32_t max_global_batch_size,
    uint32_t dp_size,
    int32_t decode_query_len,
    bool enable_no_padding,
    uint32_t max_graph_batch_size);

}  // namespace xllm::npu
