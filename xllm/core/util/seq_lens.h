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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "util/slice.h"

// Sequence-length layouts shared by batch builders, metadata builders and graph
// executors. Two cumulative forms coexist and must not be confused:
//   query_start_loc: n+1 entries with a leading zero, the FA/FlashInfer form.
//   cu_num_tokens:   n entries, no leading zero, the per-token offsets form.
namespace xllm::seq_lens {

inline int32_t max_or_zero(const Slice<int32_t>& lens) {
  return lens.empty() ? 0 : *std::max_element(lens.begin(), lens.end());
}

template <typename T = int32_t>
void fill_query_start_loc(std::vector<T>& dst, const Slice<int32_t>& lens) {
  dst.assign(lens.size() + 1, T{0});
  std::partial_sum(lens.begin(), lens.end(), dst.begin() + 1, std::plus<T>());
}

inline std::vector<int32_t> query_start_loc(const Slice<int32_t>& lens) {
  std::vector<int32_t> cumulative_lens;
  fill_query_start_loc(cumulative_lens, lens);
  return cumulative_lens;
}

inline std::vector<int32_t> cu_num_tokens(const Slice<int32_t>& lens) {
  std::vector<int32_t> offsets(lens.size());
  std::partial_sum(lens.begin(), lens.end(), offsets.begin());
  return offsets;
}

inline void append_indptr(std::vector<int32_t>& cumulative_lens,
                          int32_t length) {
  if (cumulative_lens.empty()) {
    cumulative_lens.emplace_back(0);
  }
  cumulative_lens.emplace_back(cumulative_lens.back() + length);
}

// Padded rows keep a decode-width query, but own no KV pages: a non-zero KV
// length would make attention read the zero-filled dummy block-table rows.
inline void pad_uniform_decode_rows(std::vector<int32_t>& q_seq_lens,
                                    std::vector<int32_t>& kv_seq_lens,
                                    size_t num_tokens_padded,
                                    int32_t uniform_decode_query_len) {
  if (q_seq_lens.empty() && kv_seq_lens.empty()) {
    return;
  }
  q_seq_lens.resize(q_seq_lens.size() + num_tokens_padded,
                    uniform_decode_query_len);
  kv_seq_lens.resize(kv_seq_lens.size() + num_tokens_padded, /*value=*/0);
}

// Device counterparts. `dst` keeps the caller's persistent buffer; only rows
// [0, num_sequences] are rewritten so graph capture sees stable addresses.

inline void fill_query_start_loc(const torch::Tensor& seq_lens,
                                 int64_t num_sequences,
                                 torch::Tensor& dst) {
  torch::Tensor cumsum_dst =
      dst.slice(/*dim=*/0, /*start=*/1, /*end=*/num_sequences + 1);
  // `dtype` accumulates straight into int32, so the per-step narrowing copy
  // into a temporary int32 input is not needed.
  torch::cumsum_out(
      cumsum_dst,
      seq_lens.slice(/*dim=*/0, /*start=*/0, /*end=*/num_sequences),
      /*dim=*/0,
      /*dtype=*/torch::kInt32);
}

inline torch::Tensor make_query_start_loc(const torch::Tensor& seq_lens,
                                          int64_t num_sequences) {
  if (!seq_lens.defined()) {
    return {};
  }
  torch::Tensor cumulative = torch::empty(
      {num_sequences + 1}, seq_lens.options().dtype(torch::kInt32));
  cumulative[0].fill_(/*value=*/0);
  fill_query_start_loc(seq_lens, num_sequences, cumulative);
  return cumulative;
}

// Graph padding rows repeat the last filled cumulative entry, so padded
// queries see a zero-length extent and never read a padding block-table slot.
inline void pad_query_start_loc(torch::Tensor& cumulative,
                                int64_t filled_size,
                                int64_t total_size) {
  if (total_size <= filled_size) {
    return;
  }
  cumulative.slice(/*dim=*/0, /*start=*/filled_size, /*end=*/total_size)
      .copy_(cumulative
                 .slice(/*dim=*/0,
                        /*start=*/filled_size - 1,
                        /*end=*/filled_size)
                 .expand({total_size - filled_size}),
             /*non_blocking=*/true);
}

}  // namespace xllm::seq_lens
