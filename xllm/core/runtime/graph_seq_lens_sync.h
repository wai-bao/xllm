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
#include <torch/torch.h>

#include <cstdint>

#include "core/common/global_flags.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model/rec_model_params.h"
#include "core/util/seq_lens.h"

namespace xllm {

// Refresh the graph's persistent cumulative seq-lens buffers from this step's
// per-sequence lengths. `params` is a ModelParamsView or a RecModelParams.
template <typename Params>
inline void sync_graph_seq_lens(const Params& params,
                                int64_t actual_batch_size,
                                int64_t actual_num_tokens,
                                int64_t padded_num_tokens,
                                torch::Tensor& q_cu_seq_lens,
                                torch::Tensor& kv_cu_seq_lens) {
  VLOG(kGraphExecutorLogVerboseLevel)
      << "cumsum q_seq_lens into persistent buffer: src shape="
      << params.attention.device.q_seq_lens.sizes()
      << ", rows=" << actual_batch_size;
  seq_lens::fill_query_start_loc(
      params.attention.device.q_seq_lens, actual_batch_size, q_cu_seq_lens);

  VLOG(kGraphExecutorLogVerboseLevel)
      << "cumsum kv_seq_lens into persistent buffer: src shape="
      << params.attention.device.kv_seq_lens.sizes()
      << ", rows=" << actual_batch_size;
  seq_lens::fill_query_start_loc(
      params.attention.device.kv_seq_lens, actual_batch_size, kv_cu_seq_lens);

  if (params.meta.batch_forward_type.is_decode() &&
      padded_num_tokens > actual_num_tokens) {
    seq_lens::pad_query_start_loc(
        kv_cu_seq_lens, actual_batch_size + 1, padded_num_tokens + 1);
  }
}

}  // namespace xllm
