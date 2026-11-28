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

#include "core/runtime/acl_graph_bucket_policy.h"

#include <algorithm>

#include "core/runtime/decode_graph_bucket.h"

namespace xllm::npu {

bool is_acl_graph_decode_capture_allowed(uint32_t batch_size,
                                         uint32_t max_batch_size,
                                         uint32_t dp_size,
                                         bool is_graph_warmup,
                                         int32_t decode_query_len) {
  if (batch_size == 0 || dp_size == 0) {
    return false;
  }
  const uint32_t max_full_local_batch_size = max_batch_size / dp_size;
  const uint32_t max_local_batch_size =
      max_full_local_batch_size + (max_batch_size % dp_size != 0);
  if (batch_size > max_local_batch_size) {
    return false;
  }
  if (is_graph_warmup) {
    return true;
  }
  const uint64_t first_global_batch_size = std::max<uint64_t>(
      dp_size, static_cast<uint64_t>(batch_size - 1) * dp_size + 1);
  const uint64_t last_global_batch_size = std::min<uint64_t>(
      max_batch_size, static_cast<uint64_t>(batch_size) * dp_size);
  for (uint64_t global_batch_size = first_global_batch_size;
       global_batch_size <= last_global_batch_size;
       ++global_batch_size) {
    if (is_acl_graph_compatibility_batch_size(global_batch_size,
                                              max_batch_size)) {
      return true;
    }
  }

  // No-padding MTP warmup also captures compatibility buckets in token rows.
  if (decode_query_len <= 1 || batch_size > max_full_local_batch_size) {
    return false;
  }
  const uint64_t token_batch_size =
      static_cast<uint64_t>(batch_size) * decode_query_len;
  const uint64_t max_token_batch_size =
      static_cast<uint64_t>(max_full_local_batch_size) * decode_query_len;
  return is_acl_graph_compatibility_batch_size(token_batch_size,
                                               max_token_batch_size);
}

std::vector<int32_t> get_acl_graph_mtp_warmup_batch_sizes(
    uint32_t max_global_batch_size,
    uint32_t dp_size,
    int32_t decode_query_len,
    bool enable_no_padding,
    uint32_t max_graph_batch_size) {
  std::vector<int32_t> batch_sizes;
  if (decode_query_len <= 1 || dp_size == 0) {
    return batch_sizes;
  }

  const uint32_t max_local_batch_size = max_global_batch_size / dp_size;
  const uint32_t max_graph_local_batch_size =
      max_graph_batch_size > 0
          ? std::min(max_local_batch_size, max_graph_batch_size)
          : max_local_batch_size;
  const uint64_t max_graph_token_count =
      static_cast<uint64_t>(max_graph_local_batch_size) * decode_query_len;
  const bool filter_compatibility =
      max_graph_batch_size > 0 && enable_no_padding;
  int64_t last_token_bucket = 0;
  for (uint32_t local_batch_size = 1;
       local_batch_size <= max_graph_local_batch_size;
       ++local_batch_size) {
    const int64_t token_bucket = runtime::get_decode_graph_token_bucket(
        static_cast<int64_t>(local_batch_size) * decode_query_len,
        enable_no_padding);
    if (filter_compatibility &&
        !is_acl_graph_compatibility_batch_size(
            static_cast<uint64_t>(token_bucket), max_graph_token_count)) {
      continue;
    }
    const int32_t global_batch_size =
        static_cast<int32_t>(local_batch_size * dp_size);
    if (token_bucket != last_token_bucket) {
      batch_sizes.emplace_back(global_batch_size);
      last_token_bucket = token_bucket;
    } else {
      batch_sizes.back() = global_batch_size;
    }
  }
  return batch_sizes;
}

}  // namespace xllm::npu
