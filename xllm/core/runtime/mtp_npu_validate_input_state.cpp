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

#include "core/runtime/mtp_npu_validate_input_state.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstring>
#include <utility>

#include "core/framework/model/llm_model_params.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/mtp_verify_input_layout.h"
#include "core/runtime/speculative_input_builder.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/integer_math.h"
#include "core/util/tensor_helper.h"

namespace xllm {
namespace {

// The stable-greedy path only applies when a single-row batch is purely
// greedy with no sampling parameters that could diverge between draft and
// validate rounds.
bool is_pure_single_row_greedy(const SamplingParameters& sampling_params) {
  // do_sample is deliberately not consulted: SamplingParameters::init only
  // defines it for MIXED batches, and a single-row batch can never be MIXED,
  // so requiring it would make this fast path unreachable.
  return sampling_params.sampling_mode == SamplingMode::GREEDY &&
         sampling_params.num_sample_seqs() == 1 &&
         !sampling_params.has_sampling_fields();
}

}  // namespace

bool MtpNpuValidateInputState::try_prepare_stable_greedy_sampling(
    SamplingParameters& sampling_params,
    int32_t total_num_val_tokens,
    bool use_explicit_spec_verify_replay_update,
    const torch::Device& device) {
  if (!use_explicit_spec_verify_replay_update ||
      !is_pure_single_row_greedy(sampling_params)) {
    return false;
  }
  if (!greedy_indices_.defined() ||
      greedy_indices_.numel() != total_num_val_tokens) {
    greedy_indices_ = arange_indices(total_num_val_tokens, device);
    greedy_do_sample_ =
        torch::zeros({total_num_val_tokens},
                     torch::TensorOptions().dtype(torch::kBool).device(device));
  }
  sampling_params.selected_token_idxes = greedy_indices_;

  sampling_params.do_sample = greedy_do_sample_;
  sampling_params.sampling_mode = SamplingMode::GREEDY;
  return true;
}

template <typename TargetInput>
void MtpNpuValidateInputState::prepare_explicit_replay_attention(
    TargetInput& validate_input,
    mtp_verify::ValidateInputLayout& layout,
    const mtp_verify::ValidateInputPlan& plan,
    std::vector<int32_t>& accepted_prefix_lengths,
    std::vector<int32_t>& expanded_linear_state_ids,
    bool static_graph_tasks_prepared,
    const Config& config) {
  auto& input_params = validate_input.input_params;
  specBuilder::DecodeBuildBuffers& buf = layout.rows;
  const int32_t total_num_val_tokens = layout.total_tokens;
  const int32_t num_val_tokens = plan.uniform_width;
  const int32_t logical_block_size = plan.logical_block_size;
  CHECK_GT(num_val_tokens, 0);
  const int32_t num_sequences = total_num_val_tokens / num_val_tokens;
  CHECK_EQ(total_num_val_tokens % num_val_tokens, 0);
  CHECK(
      specBuilder::prepare_expanded_spec_verify_graph_host_input(input_params))
      << "explicit spec-verify replay requires q/kv seq lens";

  auto& attention = input_params.attention;
  CHECK(attention.host.block_tables.defined());
  CHECK_EQ(attention.host.block_tables.dim(), 2);
  CHECK_EQ(attention.host.block_tables.size(0), num_sequences);
  CHECK_EQ(attention.host.block_tables.scalar_type(), torch::kInt32);
  const int64_t active_block_table_width = attention.host.block_tables.size(1);
  const int64_t verify_block_table_width = config.verify_block_table_width;
  if (active_block_table_width != verify_block_table_width) {
    // Not served from a retained member buffer: the padded table is handed
    // to attention.host and may outlive this round under schedule overlap,
    // so the next round must not be able to overwrite it in place.
    attention.host.block_tables = torch::constant_pad_nd(
        attention.host.block_tables,
        {0, verify_block_table_width - active_block_table_width},
        /*value=*/0);
  }

  const bool initialize_stable_buffer = !attention_host_buffer_.defined();
  attention.attention_host_buffer = attention_host_buffer_;
  attention.attention_device_buffer = attention_device_buffer_;
  attention.attention_buffer_bytes = 0;
  attention.attention_buffer_capacity = attention_buffer_capacity_;
  attention.attention_buffer_owner = attention_buffer_owner_;

  const int64_t expanded_block_rows =
      static_cast<int64_t>(input_params.graph.expanded_kv_seq_lens_vec.size());
  CHECK_EQ(expanded_block_rows, total_num_val_tokens);
  const torch::Tensor dense_block_source =
      attention.host.block_tables.contiguous();
  // Scratch source for the packed H2D copy below; never handed out, and the
  // memcpy loop rewrites every element, so the retained storage skips both
  // the per-step allocation and its zero-fill.
  expanded_block_tables_dense_.resize(
      static_cast<size_t>(expanded_block_rows * verify_block_table_width));
  const int32_t* block_row = dense_block_source.data_ptr<int32_t>();
  for (int64_t row = 0; row < expanded_block_rows; ++row) {
    const int64_t sequence_index = row / num_val_tokens;
    std::memcpy(
        expanded_block_tables_dense_.data() + row * verify_block_table_width,
        block_row + sequence_index * verify_block_table_width,
        static_cast<size_t>(verify_block_table_width) * sizeof(int32_t));
  }
  torch::Tensor expanded_block_tables_flat;
  std::vector<LlmAttentionInput::PackedIntInput> extra_int_inputs;
  extra_int_inputs.push_back({&buf.out_token_ids,
                              &validate_input.token_ids_host,
                              &validate_input.token_ids});
  extra_int_inputs.push_back({&buf.out_positions,
                              &validate_input.positions_host,
                              &validate_input.positions});
  if (!expanded_linear_state_ids.empty()) {
    extra_int_inputs.push_back({&expanded_linear_state_ids,
                                nullptr,
                                &input_params.embedding.linear_state_indices});
  }
  if (!input_params.num_accepted_tokens_host.empty()) {
    extra_int_inputs.push_back(
        {&accepted_prefix_lengths, nullptr, &input_params.num_accepted_tokens});
  }
  if (!input_params.graph.expanded_kv_seq_lens_vec.empty()) {
    extra_int_inputs.push_back({&input_params.graph.expanded_kv_seq_lens_vec,
                                nullptr,
                                &input_params.graph.expanded_kv_seq_lens});
  }
  extra_int_inputs.push_back(
      {&expanded_block_tables_dense_, nullptr, &expanded_block_tables_flat});
  attention.rebuild_device_buffer(
      config.device,
      extra_int_inputs,
      initialize_stable_buffer
          ? LlmAttentionInput::BufferReusePolicy::GROWABLE
          : LlmAttentionInput::BufferReusePolicy::FIXED_CAPACITY);
  if (initialize_stable_buffer && num_sequences > 0) {
    // Source views are part of the explicit replay contract. Reserve enough
    // packed storage for any configured decode batch before the first graph
    // captures their addresses, so a later larger batch cannot relocate the
    // buffer and invalidate an already captured variant.
    const uint64_t max_sequences =
        static_cast<uint64_t>(config.max_seqs_per_batch);
    const uint64_t batch_scale = xllm::util::ceil_div(
        max_sequences, static_cast<uint64_t>(num_sequences));
    const uint64_t reserve_capacity =
        attention.attention_buffer_bytes * batch_scale;
    if (reserve_capacity > attention.attention_buffer_capacity) {
      attention.reserve_device_buffer_capacity(reserve_capacity, config.device);
      attention.rebuild_device_buffer(
          config.device,
          extra_int_inputs,
          LlmAttentionInput::BufferReusePolicy::FIXED_CAPACITY);
    }
  }
  attention_host_buffer_ = attention.attention_host_buffer;
  attention_device_buffer_ = attention.attention_device_buffer;
  attention_buffer_capacity_ = attention.attention_buffer_capacity;
  input_params.graph.expanded_block_tables = expanded_block_tables_flat.view(
      {expanded_block_rows, verify_block_table_width});
  layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      ModelParamsView(input_params),
      input_params.graph.expanded_kv_seq_lens,
      input_params.graph.expanded_block_tables,
      input_params.graph.expanded_kv_seq_lens_vec,
      logical_block_size);
  input_params.graph.input_tokens_override = validate_input.token_ids;
  input_params.graph.spec_verify_source_addresses_stable = true;
  input_params.graph.spec_verify_static_graph_tasks_prepared =
      static_graph_tasks_prepared;
}

template <typename TargetInput>
void MtpNpuValidateInputState::prepare_attention(
    TargetInput& validate_input,
    mtp_verify::ValidateInputLayout& layout,
    const mtp_verify::ValidateInputPlan& plan,
    std::vector<int32_t>& accepted_prefix_lengths,
    std::vector<int32_t>& expanded_linear_state_ids,
    bool static_graph_tasks_prepared,
    const Config& config) {
  if (plan.deferred_tokens) {
    prepare_explicit_replay_attention(validate_input,
                                      layout,
                                      plan,
                                      accepted_prefix_lengths,
                                      expanded_linear_state_ids,
                                      static_graph_tasks_prepared,
                                      config);
    return;
  }
  auto& input_params = validate_input.input_params;
  const int32_t logical_block_size = plan.logical_block_size;
  std::vector<LlmAttentionInput::PackedIntInput> extra_int_inputs;
  if (!expanded_linear_state_ids.empty()) {
    extra_int_inputs.push_back({&expanded_linear_state_ids,
                                nullptr,
                                &input_params.embedding.linear_state_indices});
  }
  input_params.attention.rebuild_device_buffer(config.device, extra_int_inputs);
  if (config.supports_explicit_replay_update) {
    specBuilder::build_expanded_spec_verify_graph_input(
        input_params, config.device, logical_block_size);
  }
  input_params.graph.spec_verify_static_graph_tasks_prepared = false;
}

template void MtpNpuValidateInputState::prepare_attention(
    LlmForwardInput&,
    mtp_verify::ValidateInputLayout&,
    const mtp_verify::ValidateInputPlan&,
    std::vector<int32_t>&,
    std::vector<int32_t>&,
    bool,
    const MtpNpuValidateInputState::Config&);
template void MtpNpuValidateInputState::prepare_attention(
    VlmForwardInput&,
    mtp_verify::ValidateInputLayout&,
    const mtp_verify::ValidateInputPlan&,
    std::vector<int32_t>&,
    std::vector<int32_t>&,
    bool,
    const MtpNpuValidateInputState::Config&);

}  // namespace xllm
