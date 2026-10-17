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

#include "core/runtime/mtp_verify_input_layout.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "core/framework/model/llm_model_params.h"
#include "core/framework/model/vlm_model_params.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/vlm_forward_params.h"
#include "util/tensor_helper.h"

namespace xllm::mtp_verify {

template <typename TargetInput>
ValidateInputLayout build_validate_input_layout(const TargetInput& input,
                                                const ValidateInputPlan& plan) {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  const bool per_sequence = !plan.widths.empty();
  if (per_sequence) {
    CHECK_EQ(plan.widths.size(), static_cast<size_t>(num_sequences));
  } else {
    CHECK(plan.widths.empty());
    CHECK(plan.uniform_width > 0 || num_sequences == 0);
  }
  const bool use_atb_spec_kernel =
      plan.attention_mode == ValidateAttentionMode::CHUNKED_PREFILL;
  const bool step_major = plan.row_order == ValidateRowOrder::STEP_MAJOR;
  const auto width_at = [&](int32_t seq_id) {
    return per_sequence ? plan.widths[seq_id] : plan.uniform_width;
  };
  ValidateInputLayout layout;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t width = width_at(seq_id);
    CHECK_GT(width, 0);
    layout.total_tokens += width;
    layout.max_width = std::max(layout.max_width, width);
  }
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  const Slice<int32_t> token_ids = tensor_slice(input.token_ids_host);
  const Slice<int32_t> positions = tensor_slice(input.positions_host);
  const Slice<int32_t> kv_seq_lens =
      input.input_params.attention.host.kv_seq_lens;
  auto& rows = layout.rows;
  rows.out_token_ids.reserve(layout.total_tokens);
  rows.out_positions.reserve(layout.total_tokens);
  rows.out_new_cache_slots.reserve(layout.total_tokens);
  if (!use_atb_spec_kernel || step_major) {
    rows.out_kv_seq_lens.reserve(layout.total_tokens);
    rows.out_q_seq_lens.reserve(layout.total_tokens);
    rows.out_q_cu_seq_lens.reserve(layout.total_tokens);
    rows.out_block_tables.reserve(static_cast<size_t>(layout.total_tokens) *
                                  row_ctx.block_table_stride);
  }
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t start_position = positions[seq_id];
    const int32_t kv_len =
        specBuilder::calc_kv_len(kv_seq_lens, seq_id, /*offset=*/0);
    if (!plan.positions_decoupled) {
      CHECK_EQ(start_position + 1, kv_len)
          << "validate position/kv_len mismatch, seq_id=" << seq_id
          << ", start_position=" << start_position << ", kv_len=" << kv_len;
    }
    if (use_atb_spec_kernel) {
      specBuilder::update_kv_seq_lens_and_max(layout.atb_kv_seq_lens,
                                              kv_len + width_at(seq_id) - 1,
                                              layout.atb_kv_max_seq_len);
      specBuilder::append_q_seq_len(
          layout.atb_q_seq_lens, layout.atb_q_cu_seq_lens, width_at(seq_id));
    }
  }
  const auto append_row = [&](int32_t seq_id, int32_t val_idx) {
    specBuilder::RowSpec row;
    row.seq_id = seq_id;
    row.token_id = val_idx == 0 ? token_ids[seq_id] : -val_idx;
    row.position_offset = val_idx;
    row.append_kv_len = step_major || !use_atb_spec_kernel;
    row.append_q_len_one = step_major || !use_atb_spec_kernel;
    row.append_block_table = step_major || !use_atb_spec_kernel;
    specBuilder::append_decode_row(row_ctx, row, plan.logical_block_size, rows);
  };
  if (step_major) {
    for (int32_t val_idx = 0; val_idx < layout.max_width; ++val_idx) {
      for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
        if (val_idx < width_at(seq_id)) {
          append_row(seq_id, val_idx);
        }
      }
    }
  } else {
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      for (int32_t val_idx = 0; val_idx < width_at(seq_id); ++val_idx) {
        append_row(seq_id, val_idx);
      }
    }
  }
  CHECK_EQ(rows.out_new_cache_slots.size(), rows.out_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(rows.out_positions.size(), rows.out_token_ids.size())
      << "validate positions/tokens mismatch";
  return layout;
}

template <typename TargetInput>
void apply_validate_input_layout(TargetInput& validate_input,
                                 const ValidateInputPlan& plan,
                                 ValidateInputLayout& layout) {
  auto& input_params = validate_input.input_params;
  auto& rows = layout.rows;
  if (!plan.deferred_tokens) {
    specBuilder::set_token_position_tensors(validate_input,
                                            rows.out_token_ids,
                                            rows.out_positions,
                                            validate_input.token_ids.options(),
                                            validate_input.positions.options());
  }
  if (plan.attention_mode == ValidateAttentionMode::CHUNKED_PREFILL) {
    input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
    specBuilder::update_input_params(input_params,
                                     rows,
                                     layout.max_width,
                                     std::move(layout.atb_q_seq_lens),
                                     std::move(layout.atb_q_cu_seq_lens),
                                     layout.atb_kv_max_seq_len,
                                     std::move(layout.atb_kv_seq_lens));
  } else {
    input_params.meta.num_sequences = layout.total_tokens;
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    specBuilder::update_input_params(input_params,
                                     rows,
                                     1,
                                     std::move(rows.out_q_seq_lens),
                                     std::move(rows.out_q_cu_seq_lens),
                                     rows.meta.kv_max_seq_len,
                                     std::move(rows.out_kv_seq_lens),
                                     /*update_block_tables=*/true);
  }
  if (!validate_input.json_object_states.empty()) {
    // Exact per-position masks are installed after draft tokens are known.
    validate_input.sampling_params.filter_mask = torch::Tensor();
    validate_input.sampling_params.filter_bitmask = torch::Tensor();
  }
}

#if defined(USE_NPU)
template <typename Params>
std::vector<int32_t> expand_validate_linear_state_ids(
    const Params& input_params,
    std::span<const int32_t> widths,
    int32_t uniform_width,
    int32_t num_sequences,
    int32_t total_tokens) {
  const auto& ids = input_params.embedding.linear_state_ids;
  if (ids.empty()) {
    return {};
  }
  CHECK_EQ(ids.size(), static_cast<size_t>(num_sequences));
  CHECK(widths.empty() || widths.size() == ids.size());
  std::vector<int32_t> expanded_ids;
  expanded_ids.reserve(total_tokens);
  for (size_t seq_id = 0; seq_id < ids.size(); ++seq_id) {
    const int32_t width = widths.empty() ? uniform_width : widths[seq_id];
    expanded_ids.insert(expanded_ids.end(), width, ids[seq_id]);
  }
  return expanded_ids;
}
template std::vector<int32_t> expand_validate_linear_state_ids(
    const LlmModelParams&,
    std::span<const int32_t>,
    int32_t,
    int32_t,
    int32_t);
template std::vector<int32_t> expand_validate_linear_state_ids(
    const VlmModelParams&,
    std::span<const int32_t>,
    int32_t,
    int32_t,
    int32_t);
#endif

template ValidateInputLayout build_validate_input_layout(
    const LlmForwardInput&,
    const ValidateInputPlan&);
template ValidateInputLayout build_validate_input_layout(
    const VlmForwardInput&,
    const ValidateInputPlan&);
template void apply_validate_input_layout(LlmForwardInput&,
                                          const ValidateInputPlan&,
                                          ValidateInputLayout&);
template void apply_validate_input_layout(VlmForwardInput&,
                                          const ValidateInputPlan&,
                                          ValidateInputLayout&);

}  // namespace xllm::mtp_verify
