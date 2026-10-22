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

#include "core/framework/speculative/spec_input_builder.h"

#include <glog/logging.h>

#include "core/framework/batch/decode_row_builder.h"
#include "framework/sampling/sampling_params.h"

namespace xllm::specBuilder {

MtpTopkStatePtr select_mtp_topk_state_for_next_step(
    const MtpTopkStatePtr& state,
    const SamplingParameters& sampling_params) {
  if (state == nullptr) {
    return state;
  }
  const int64_t num_seqs = sampling_params.num_sample_seqs();
  if (num_seqs == 0 || state->num_rows() == num_seqs) {
    return state;
  }
  const torch::Tensor& selected_idxes = sampling_params.selected_token_idxes;
  if (selected_idxes.device().is_cpu()) {
    CHECK_GT(state->num_rows(), selected_idxes.max().item<int64_t>())
        << "MTP selected top-k index exceeds top-k rows.";
  }
  torch::Tensor index =
      selected_idxes.to(torch::dtype(torch::kLong).device(state->device()))
          .contiguous();
  return state->index_select_rows(index);
}

void materialize_empty_block_draft_rows(LlmForwardInput& input,
                                        int32_t query_width,
                                        int32_t mask_token_id) {
  CHECK_GT(query_width, 0);
  auto& params = input.input_params;
  for (int32_t& count : params.parallel.dp_global_token_nums) {
    if (count == 0) {
      count = query_width;
    }
  }
  if (params.meta.num_sequences != 0) {
    return;
  }
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  input.token_ids_host = torch::full({query_width}, mask_token_id, options);
  input.positions_host = torch::arange(query_width, options);
  input.token_ids = input.token_ids_host;
  input.positions = input.positions_host;
  input.runtime.device_tensors_ready = false;
  input.runtime.input_host_buffer_has_layout = false;
  params.meta.num_sequences = 1;
  params.meta.actual_num_sequences = 0;
  params.meta.q_max_seq_len = query_width;
  params.meta.kv_max_seq_len = 1;
  params.attention.host.q_seq_lens.clear();
  input_builder::append_seq_len_by_layout(params.attention.host.q_seq_lens,
                                          query_width);
  params.attention.host.q_cu_seq_lens = {0, query_width};
  params.attention.host.kv_seq_lens.clear();
  input_builder::append_seq_len_by_layout(params.attention.host.kv_seq_lens, 1);
  params.attention.host.new_cache_slots.assign(query_width, -1);
  params.attention.host.block_tables = torch::zeros({1, 1}, options);
  params.attention.device.paged_kv_indptr = torch::tensor({0, 1}, options);
  params.attention.device.paged_kv_indices = torch::tensor({0}, options);
  params.attention.device.paged_kv_last_page_len = torch::tensor({1}, options);
}

}  // namespace xllm::specBuilder
