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

#include "core/runtime/forward_params.h"

#include <type_traits>

#include "core/framework/config/execution_config.h"
#include "core/runtime/vlm_forward_params.h"

namespace xllm {

namespace detail {
namespace {

torch::Tensor build_host_buffer(ForwardInputBufferPlan& plan) {
  const uint64_t total_bytes = plan.prepare_layout();
  if (total_bytes == 0) {
    return torch::Tensor();
  }
  auto buffer = torch::empty({static_cast<int64_t>(total_bytes)},
                             torch::TensorOptions()
                                 .dtype(/*dtype=*/torch::kUInt8)
                                 .device(/*device=*/torch::kCPU)
                                 .pinned_memory(/*pinned_memory=*/true));
  auto* base = static_cast<char*>(buffer.data_ptr());
  for (const auto& entry : plan.entries) {
    const uint64_t bytes = static_cast<uint64_t>(
        entry.host_tensor.numel() * entry.host_tensor.element_size());
    if (bytes == 0) {
      continue;
    }
    std::memcpy(base + entry.offset, entry.host_tensor.data_ptr(), bytes);
    if (entry.aligned_bytes > bytes) {
      std::memset(base + entry.offset + bytes,
                  /*ch=*/0,
                  static_cast<size_t>(entry.aligned_bytes - bytes));
    }
  }
  return buffer;
}

template <typename Input>
bool add_forward_tensors_to_plan(const Input& source,
                                 Input& target,
                                 ForwardInputBufferPlan& plan) {
  const auto& source_params = source.input_params;
  auto& target_params = target.input_params;
  const auto add = [&plan](const torch::Tensor& tensor, torch::Tensor& output) {
    output = torch::Tensor();
    return plan.add(tensor, &output);
  };
  return add(source_params.embedding.input_embedding,
             target_params.embedding.input_embedding) &&
         add(source_params.cache_state_.linear_state_indices,
             target_params.cache_state_.linear_state_indices) &&
         add(source.speculative_.mtp_bootstrap_embeddings_,
             target.speculative_.mtp_bootstrap_embeddings_) &&
         add(source_params.graph.attn_mask, target_params.graph.attn_mask) &&
         add(source_params.graph.tiling_data, target_params.graph.tiling_data);
}

template <typename Input>
void copy_non_runtime_metadata(const Input& source, Input& inputs) {
  inputs.transfer_kv_infos = source.transfer_kv_infos;
  inputs.input_host_sample_count = source.input_host_sample_count;
  inputs.skip_sampling_for_logits_only = source.skip_sampling_for_logits_only;
  inputs.return_selected_hidden = source.return_selected_hidden;
  inputs.request_ids = source.request_ids;
  inputs.sample_sequence_ids = source.sample_sequence_ids;
  inputs.sample_prior_output_rows = source.sample_prior_output_rows;
  inputs.json_object_states = source.json_object_states;
  inputs.json_object_state_snapshots = source.json_object_state_snapshots;
}

template <typename Input>
void copy_metadata(const Input& source, Input& inputs) {
  copy_non_runtime_metadata(source, inputs);
  inputs.runtime.kv_slot_layout = source.runtime.kv_slot_layout;
  inputs.runtime.metadata_ready_event = source.runtime.metadata_ready_event;
  inputs.runtime.retained_device_tensors =
      source.runtime.retained_device_tensors;
}

torch::Tensor cpu_view(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.device().is_cpu() ? tensor
                                                      : torch::Tensor();
}

template <typename Input>
void set_host_views(const Input& source, Input& inputs) {
  inputs.token_ids_host = source.token_ids_host.defined()
                              ? source.token_ids_host
                              : cpu_view(source.token_ids);
  inputs.positions_host = source.positions_host.defined()
                              ? source.positions_host
                              : cpu_view(source.positions);
}

template <typename Input>
Input clone_input(const Input& source) {
  Input inputs;
  inputs.runtime = source.runtime;
  copy_non_runtime_metadata(source, inputs);
  inputs.token_ids = source.token_ids;
  inputs.positions = source.positions;
  inputs.token_ids_host = source.token_ids_host;
  inputs.positions_host = source.positions_host;
  inputs.input_params = source.input_params.clone();
  inputs.speculative_ = source.speculative_;
  inputs.sampling_params = source.sampling_params;
  inputs.json_object_errors = source.json_object_errors;
  if constexpr (std::is_same_v<Input, VlmForwardInput>) {
    inputs.mm_data_ = source.mm_data_;
  }
  return inputs;
}

template <typename Input>
bool stage_contiguous_input(const Input& source,
                            const torch::Device& device,
                            Input& inputs) {
  copy_metadata(source, inputs);
  set_host_views(source, inputs);
  if ((source.token_ids.defined() && !inputs.token_ids_host.defined()) ||
      (source.positions.defined() && !inputs.positions_host.defined())) {
    return false;
  }

  const auto& source_params = source.input_params;
  if constexpr (std::is_same_v<Input, VlmForwardInput>) {
    if (source.mm_data_.valid() ||
        !source_params.embedding.deep_stacks_.empty()) {
      return false;
    }
  }

  inputs.input_params = source_params.clone();
  inputs.speculative_ = source.speculative_;
  inputs.sampling_params = source.sampling_params;

  torch::Tensor positions_for_device =
      normalize_positions_for_device(inputs.positions_host);
  ForwardInputBufferPlan plan;
  if (!plan.add(inputs.token_ids_host, &inputs.token_ids) ||
      !plan.add(positions_for_device, &inputs.positions) ||
      !add_attention_to_plan(
          source_params.attention, inputs.input_params.attention, plan) ||
      !add_forward_tensors_to_plan(source, inputs, plan) ||
      !add_sampling_to_plan(
          source.sampling_params, inputs.sampling_params, plan)) {
    return false;
  }

  if (auto host_buffer = build_host_buffer(plan); host_buffer.defined()) {
    inputs.runtime.input_host_buffer = std::move(host_buffer);
    inputs.runtime.device_input_buffer = safe_to(
        inputs.runtime.input_host_buffer,
        torch::TensorOptions().dtype(/*dtype=*/torch::kUInt8).device(device),
        /*non_blocking=*/true);
    plan.bind_device_views(inputs.runtime.device_input_buffer, device);
  }

  inputs.runtime.device_tensors_ready = true;
  inputs.runtime.input_host_buffer_has_layout = false;
  return true;
}

void clear_shared_full_cache(LlmAttentionInput& attention) {
  attention.host.shared_full_block_table = torch::Tensor();
  attention.host.shared_full_slot_mapping = torch::Tensor();
  attention.device.shared_full_block_table = torch::Tensor();
  attention.device.shared_full_slot_mapping = torch::Tensor();
}

}  // namespace
}  // namespace detail

SpeculativeInput SpeculativeInput::to(const torch::Device& device) const {
  SpeculativeInput out = *this;
  out.mtp_bootstrap_embeddings_ =
      safe_to(mtp_bootstrap_embeddings_, device, /*non_blocking=*/true);
  return out;
}

LlmForwardInput LlmForwardInput::clone() const {
  return detail::clone_input(*this);
}

LlmForwardInput LlmForwardInput::to(const torch::Device& device,
                                    torch::ScalarType dtype) const {
  if (runtime.device_tensors_ready) {
    return clone();
  }

  const bool use_contiguous_input_buffer =
      ::xllm::ExecutionConfig::get_instance().use_contiguous_input_buffer() &&
      detail::supports_contiguous_forward_input_buffer(device);
  if (runtime.input_host_buffer_has_layout) {
    LlmForwardInput buffer_inputs;
    if (detail::unpack_from_input_host_buffer(
            *this, device, dtype, buffer_inputs, use_contiguous_input_buffer)) {
      if (buffer_inputs.runtime.device_tensors_ready) {
        return buffer_inputs;
      }
      return buffer_inputs.to(device, dtype);
    }
  }

  if (use_contiguous_input_buffer) {
    LlmForwardInput contiguous_inputs;
    if (detail::stage_contiguous_input(*this, device, contiguous_inputs)) {
      return contiguous_inputs;
    }
  }

  LlmForwardInput inputs;
  detail::set_host_views(*this, inputs);
  const torch::Tensor& source_token_ids =
      inputs.token_ids_host.defined() ? inputs.token_ids_host : token_ids;
  const torch::Tensor& source_positions =
      inputs.positions_host.defined() ? inputs.positions_host : positions;
  inputs.token_ids = safe_to(source_token_ids, device, /*non_blocking=*/true);
  inputs.positions = detail::normalize_positions_for_device(
      safe_to(source_positions, device, /*non_blocking=*/true));
  inputs.input_params = input_params.to(device);
  inputs.speculative_ = speculative_.to(device);
  inputs.sampling_params = sampling_params.to(device, dtype);
  detail::copy_metadata(*this, inputs);
  inputs.runtime.input_host_buffer = runtime.input_host_buffer;
  inputs.runtime.device_input_buffer = runtime.device_input_buffer;
  inputs.runtime.input_host_buffer_has_layout =
      runtime.input_host_buffer_has_layout;
  inputs.runtime.device_tensors_ready = true;
  return inputs;
}

LlmForwardInput make_llm_draft_input(const LlmForwardInput& target,
                                     bool shared_full_cache) {
  LlmForwardInput draft = target.clone();
  draft.input_params.cache_state_.clear_linear_attention_state();
  if (!shared_full_cache) {
    draft.input_params.attention.host.shared_full_block_table = torch::Tensor();
    draft.input_params.attention.host.shared_full_slot_mapping =
        torch::Tensor();
    draft.input_params.attention.device.shared_full_block_table =
        torch::Tensor();
    draft.input_params.attention.device.shared_full_slot_mapping =
        torch::Tensor();
  }
  return draft;
}

VlmForwardInput VlmForwardInput::clone() const {
  return detail::clone_input(*this);
}

VlmForwardInput VlmForwardInput::to(const torch::Device& device,
                                    torch::ScalarType dtype) const {
  if (runtime.device_tensors_ready) {
    VlmForwardInput inputs = clone();
    inputs.runtime.input_host_buffer_has_layout = false;
    return inputs;
  }

  const bool use_contiguous_input_buffer =
      ::xllm::ExecutionConfig::get_instance().use_contiguous_input_buffer() &&
      detail::supports_contiguous_forward_input_buffer(device);
  if (runtime.input_host_buffer_has_layout) {
    VlmForwardInput buffer_inputs;
    if (detail::unpack_from_input_host_buffer(
            *this, device, dtype, buffer_inputs, use_contiguous_input_buffer)) {
      if (buffer_inputs.runtime.device_tensors_ready) {
        buffer_inputs.runtime.input_host_buffer_has_layout = false;
        return buffer_inputs;
      }
      return buffer_inputs.to(device, dtype);
    }
  }

  if (use_contiguous_input_buffer) {
    VlmForwardInput contiguous_inputs;
    if (detail::stage_contiguous_input(*this, device, contiguous_inputs)) {
      return contiguous_inputs;
    }
  }

  VlmForwardInput inputs;
  detail::set_host_views(*this, inputs);
  const torch::Tensor& source_token_ids =
      inputs.token_ids_host.defined() ? inputs.token_ids_host : token_ids;
  const torch::Tensor& source_positions =
      inputs.positions_host.defined() ? inputs.positions_host : positions;
  inputs.token_ids = safe_to(source_token_ids, device, /*non_blocking=*/true);
  inputs.positions = detail::normalize_positions_for_device(
      safe_to(source_positions, device, /*non_blocking=*/true));
  inputs.input_params = input_params.to(device);
  inputs.speculative_ = speculative_.to(device);
  inputs.mm_data_ = mm_data_.copy_to(device);
  inputs.sampling_params = sampling_params.to(device, dtype);
  detail::copy_metadata(*this, inputs);
  inputs.runtime.input_host_buffer = runtime.input_host_buffer;
  inputs.runtime.device_input_buffer = runtime.device_input_buffer;
  inputs.runtime.device_tensors_ready = true;
  return inputs;
}

LlmForwardInput make_llm_draft_input(const VlmForwardInput& target,
                                     bool shared_full_cache) {
  CHECK(!target.runtime.input_host_buffer_has_layout ||
        target.runtime.device_tensors_ready)
      << "Cannot create an LLM draft input from an unmaterialized packed VLM "
         "input.";
  LlmForwardInput draft;
  draft.token_ids = target.token_ids;
  draft.positions = target.positions;
  draft.token_ids_host = target.token_ids_host;
  draft.positions_host = target.positions_host;
  auto& params = draft.input_params;
  const auto& source = target.input_params;
  params.meta = source.meta;
  detail::copy_attention_host_input(source.attention.host,
                                    params.attention.host);
  detail::copy_attention_device_input(source.attention.device,
                                      params.attention.device);
  if (!shared_full_cache) {
    detail::clear_shared_full_cache(params.attention);
  }
  params.attention.attention_host_buffer =
      source.attention.attention_host_buffer;
  params.attention.attention_device_buffer =
      source.attention.attention_device_buffer;
  params.attention.attention_buffer_bytes =
      source.attention.attention_buffer_bytes;
  params.attention.attention_buffer_capacity =
      source.attention.attention_buffer_capacity;
  params.attention.attention_buffer_owner =
      source.attention.attention_buffer_owner;
  params.embedding.input_embedding = source.embedding.input_embedding;
  draft.speculative_ = target.speculative_;
  draft.request_ids = target.request_ids;
  params.parallel = source.parallel;
  params.expert = source.expert;
  params.graph = source.graph;
  params.multi_block_tables = source.multi_block_tables;
  params.prefill_without_cache = source.prefill_without_cache;
  params.num_accepted_tokens = source.num_accepted_tokens;
  params.mtp_topk_state = source.mtp_topk_state;
  params.num_accepted_tokens_host = source.num_accepted_tokens_host;
  params.attn_metadata = source.attn_metadata;
  params.enable_graph = source.enable_graph;
  draft.sampling_params = target.sampling_params;
  draft.sample_sequence_ids = target.sample_sequence_ids;
  draft.sample_prior_output_rows = target.sample_prior_output_rows;
  draft.json_object_states = target.json_object_states;
  draft.json_object_state_snapshots = target.json_object_state_snapshots;
  draft.skip_sampling_for_logits_only = target.skip_sampling_for_logits_only;
  draft.return_selected_hidden = target.return_selected_hidden;
  draft.transfer_kv_infos = target.transfer_kv_infos;
  draft.runtime = target.runtime;
  draft.runtime.input_host_buffer_has_layout = false;
  return draft;
}

}  // namespace xllm
