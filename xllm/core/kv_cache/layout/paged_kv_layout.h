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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/util/seq_lens.h"
#include "core/util/utils.h"

namespace xllm {

// Write to this slot is dropped by the kernel instead of landing in block 0.
inline constexpr int32_t kPadSlotId = -1;

inline int32_t kv_page_count(int32_t kv_seq_len, int32_t block_size) {
  return util::ceil_div(std::max(kv_seq_len, 1), block_size);
}

inline int32_t paged_kv_last_page_len(int32_t kv_seq_len, int32_t block_size) {
  return (std::max(kv_seq_len, 1) - 1) % block_size + 1;
}

inline int32_t append_paged_kv_indptr(int32_t kv_seq_len,
                                      int64_t max_blocks_per_seq,
                                      int32_t block_size,
                                      std::vector<int32_t>& paged_kv_indptr,
                                      std::vector<int32_t>& last_page_len_out) {
  const int32_t page_count = kv_page_count(kv_seq_len, block_size);
  CHECK_LE(page_count, max_blocks_per_seq)
      << "KV length exceeds block-table capacity, kv_seq_len=" << kv_seq_len
      << ", page_count=" << page_count
      << ", max_blocks_per_seq=" << max_blocks_per_seq;
  seq_lens::append_indptr(paged_kv_indptr, page_count);
  last_page_len_out.emplace_back(
      paged_kv_last_page_len(kv_seq_len, block_size));
  return page_count;
}

template <bool Ring, typename BlockIds>
inline int32_t slot_for_position(const BlockIds& block_ids,
                                 int64_t pos,
                                 int64_t block_size,
                                 int64_t block_table_width) {
  // Kept here instead of at each call site: a zero block size would divide by
  // zero, and a negative position indexes the block table before its start.
  CHECK_GT(block_size, 0) << "block size must be positive, block_size="
                          << block_size;
  CHECK_GE(pos, 0) << "position must be non-negative, position=" << pos;
  if (block_table_width <= 0) {
    return kPadSlotId;
  }
  int64_t block_idx = pos / block_size;
  if (Ring) {
    block_idx %= block_table_width;
  } else if (block_idx >= block_table_width) {
    return kPadSlotId;
  }
  const int32_t block_id = block_ids[block_idx];
  if (block_id < 0) {
    return kPadSlotId;
  }
  return static_cast<int32_t>(static_cast<int64_t>(block_id) * block_size +
                              pos % block_size);
}

}  // namespace xllm
