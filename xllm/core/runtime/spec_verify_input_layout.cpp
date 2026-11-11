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

#include "core/runtime/spec_verify_input_layout.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "core/framework/model/llm_model_params.h"
#include "core/framework/model/vlm_model_params.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/tensor_helper.h"

namespace xllm::spec_verify_layout {

template <typename TargetInput>
ValidateInputLayout build_validate_input_layout(const TargetInput& input,
                                                const ValidateInputPlan& plan) {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK(plan.uniform_width > 0 || num_sequences == 0);
  const bool use_atb_spec_kernel =
      plan.attention_mode == ValidateAttentionMode::CHUNKED_PREFILL;
  if (!plan.positions_decoupled) {
    const Slice<int32_t> positions = tensor_slice(input.positions_host);
    const Slice<int32_t> kv_seq_lens =
        input.input_params.attention.host.kv_seq_lens;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      CHECK_EQ(positions[seq_id] + 1, kv_seq_lens[seq_id])
          << "validate position/kv_len mismatch, seq_id=" << seq_id
          << ", start_position=" << positions[seq_id]
          << ", kv_len=" << kv_seq_lens[seq_id];
    }
  }
  // Both layouts share the decode-input expansion: chunked verify is the
  // per-request mode (one q/kv row per sequence), the ATB decode kernel is
  // the flat causal mode. Placeholder ids use the -val_idx convention.
  const DecodeExpansionOptions decode_options =
      make_verify_layout(plan.uniform_width - 1,
                         plan.logical_block_size,
                         /*chunked=*/use_atb_spec_kernel);
  ValidateInputLayout layout;
  layout.rows =
      expand_decode_inputs(make_decode_batch_context(input), decode_options);
  layout.total_tokens = num_sequences * plan.uniform_width;
  if (use_atb_spec_kernel) {
    layout.chunked_kv_seq_lens = std::move(layout.rows.out_kv_seq_lens);
    for (const int32_t kv_len : layout.chunked_kv_seq_lens) {
      layout.chunked_kv_max_seq_len =
          std::max(layout.chunked_kv_max_seq_len, kv_len);
    }
  }
  return layout;
}

template <typename TargetInput>
void apply_validate_input_layout(TargetInput& validate_input,
                                 const ValidateInputPlan& plan,
                                 ValidateInputLayout& layout) {
  auto& input_params = validate_input.input_params;
  auto& rows = layout.rows;
  if (!plan.deferred_tokens) {
    set_token_position_tensors(
        validate_input, rows.out_token_ids, rows.out_positions);
  }
  if (plan.attention_mode == ValidateAttentionMode::CHUNKED_PREFILL) {
    input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
    // Per-request mode already emitted one metadata row per sequence; move
    // the per-sequence kv lengths into the metadata the kernel expects.
    rows.out_kv_seq_lens = std::move(layout.chunked_kv_seq_lens);
    build_decode_metadata(
        input_params,
        rows,
        DecodeInputOptions{.num_speculative_tokens = plan.uniform_width - 1,
                           .per_request = true,
                           .forward_type = BatchForwardType::CHUNKED_PREFILL,
                           .leading_zero_cu = true});
  } else {
    input_params.meta.num_sequences = layout.total_tokens;
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    build_decode_metadata(input_params, rows, DecodeInputOptions{});
  }
  if (!validate_input.json_object_states.empty()) {
    // Exact per-position masks are installed after draft tokens are known.
    validate_input.sampling_params.filter_bitmask() = torch::Tensor();
  }
}

#if defined(USE_NPU)
template <typename Params>
std::vector<int32_t> expand_validate_linear_state_ids(
    const Params& input_params,
    int32_t uniform_width) {
  const auto& ids = input_params.embedding.linear_state_ids;
  if (ids.empty()) {
    return {};
  }
  std::vector<int32_t> expanded_ids;
  expanded_ids.reserve(ids.size() * static_cast<size_t>(uniform_width));
  for (const int32_t id : ids) {
    expanded_ids.insert(expanded_ids.end(), uniform_width, id);
  }
  return expanded_ids;
}
template std::vector<int32_t> expand_validate_linear_state_ids(
    const LlmModelParams&,
    int32_t);
template std::vector<int32_t> expand_validate_linear_state_ids(
    const VlmModelParams&,
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

}  // namespace xllm::spec_verify_layout
