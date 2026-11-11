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

#include "core/framework/speculative/draft_extend_input.h"

#include <torch/torch.h>

#include "common/macros.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/tensor_helper.h"

namespace xllm {

namespace {

void clear_selected_embeddings(ForwardOutput& output) {
  output.sample_output.selected_embeddings = torch::Tensor();
}

}  // namespace

void output_spec_hidden_states(
    SampleOutput& sample_output,
    const ModelOutput& model_output,
    const torch::Tensor& logits_indices,
    const torch::Tensor& gathered_sample_hidden_states,
    bool is_target_prefill,
    bool cp_enabled) {
  const torch::Tensor& target_hidden_states =
      get_hidden_states_for_drafter(model_output);
  CHECK(target_hidden_states.defined())
      << "speculative hidden states are undefined";
  torch::Tensor sample_hidden_states;
  if (logits_indices.defined()) {
    // The lm_head selection contains primary rows, not auxiliary rows.
    const bool has_aux = model_output.aux_hidden_states.defined();
    if (cp_enabled && !is_target_prefill && !has_aux) {
      CHECK(gathered_sample_hidden_states.defined())
          << "gathered selected hidden is required on the CP decode path";
    }
    if (gathered_sample_hidden_states.defined() && !has_aux) {
      sample_hidden_states = gathered_sample_hidden_states;
    } else {
      sample_hidden_states = target_hidden_states.index_select(
          /*dim=*/0,
          logits_indices.to(torch::dtype(torch::kLong)
                                .device(target_hidden_states.device())));
    }
    CHECK_EQ(sample_hidden_states.dim(), 2);
    CHECK_EQ(sample_hidden_states.size(0), logits_indices.numel())
        << "selected speculative hidden must match logits_indices rows";
  }
  if (is_target_prefill) {
    sample_output.embeddings = target_hidden_states;
    sample_output.selected_embeddings = sample_hidden_states;
  } else {
    sample_output.embeddings = sample_hidden_states;
  }
}

void clear_all_output_embeddings(ForwardOutput& output) {
  output.sample_output.embeddings = torch::Tensor();
  clear_selected_embeddings(output);
}
void prepare_first_draft_inputs(ForwardOutput& output) {
  const torch::Tensor& sample_hidden_states =
      output.sample_output.selected_embeddings;
  CHECK(sample_hidden_states.defined())
      << "target worker must populate "
         "selected_embeddings on the speculative "
         "prefill path";
  output.sample_output.embeddings = sample_hidden_states.detach();
  clear_selected_embeddings(output);
}

namespace {

template <size_t N>
void clone_packed_metadata(const torch::Tensor& buffer,
                           uint64_t active_bytes,
                           std::array<torch::Tensor*, N>& tensors) {
  if (!buffer.defined() || buffer.scalar_type() != torch::kUInt8 ||
      buffer.dim() != 1 || !buffer.is_contiguous() || active_bytes == 0 ||
      active_bytes > static_cast<uint64_t>(buffer.numel())) {
    return;
  }
  const uintptr_t base = reinterpret_cast<uintptr_t>(buffer.const_data_ptr());
  std::array<bool, N> packed{};
  std::array<uint64_t, N> offsets{};
  std::array<uint64_t, N> ends{};
  uint64_t begin = active_bytes;
  uint64_t end = 0;
  uint64_t alignment = 1;
  size_t count = 0;
  for (size_t i = 0; i < N; ++i) {
    const torch::Tensor& tensor = *tensors[i];
    if (!tensor.defined() || tensor.numel() == 0 ||
        tensor.device() != buffer.device() || !tensor.is_contiguous() ||
        tensor.is_quantized()) {
      continue;
    }
    const uintptr_t address =
        reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
    if (address < base) {
      continue;
    }
    const uint64_t offset = address - base;
    const uint64_t bytes = tensor.nbytes();
    if (offset > active_bytes || bytes > active_bytes - offset ||
        offset % tensor.element_size() != 0) {
      continue;
    }
    bool overlaps = false;
    for (size_t j = 0; j < i; ++j) {
      if (packed[j] && offset < ends[j] && offsets[j] < offset + bytes) {
        overlaps = true;
        break;
      }
    }
    if (overlaps) {
      continue;
    }
    packed[i] = true;
    offsets[i] = offset;
    ends[i] = offset + bytes;
    begin = std::min(begin, offset);
    end = std::max(end, offset + bytes);
    alignment =
        std::max(alignment, static_cast<uint64_t>(tensor.element_size()));
    ++count;
  }
  if (count < 2) {
    return;
  }
  // Keep dtype views aligned while excluding unused prefix/capacity. Each
  // resulting view owns the new storage even after the packed owner is reset.
  begin -= begin % alignment;
  const torch::Tensor snapshot =
      buffer
          .narrow(
              0, static_cast<int64_t>(begin), static_cast<int64_t>(end - begin))
          .detach()
          .clone();
  for (size_t i = 0; i < N; ++i) {
    if (!packed[i]) {
      continue;
    }
    torch::Tensor& tensor = *tensors[i];
    tensor = snapshot
                 .narrow(0,
                         static_cast<int64_t>(offsets[i] - begin),
                         static_cast<int64_t>(ends[i] - offsets[i]))
                 .view(tensor.scalar_type())
                 .view(tensor.sizes());
    tensors[i] = nullptr;
  }
}

}  // namespace

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
  const torch::Tensor& selected_idxes = sampling_params.logits_indices();
  if (selected_idxes.device().is_cpu()) {
    CHECK_GT(state->num_rows(), selected_idxes.max().item<int64_t>())
        << "MTP selected top-k index exceeds top-k rows.";
  }
  torch::Tensor index =
      selected_idxes.to(torch::dtype(torch::kLong).device(state->device()))
          .contiguous();
  return state->index_select_rows(index);
}

template <typename Params>
void fill_mtp_prefill_token_ids(const torch::Tensor& token_ids_host,
                                const Params& params,
                                const std::vector<int32_t>& extra_token_ids,
                                torch::Tensor& output) {
  const Slice<int32_t> tokens = tensor_slice(token_ids_host);
  const Slice<int32_t> result = tensor_slice(output);
  CHECK_EQ(result.size(), tokens.size());
  const int32_t num_sequences = params.meta.num_sequences;
  CHECK_GE(num_sequences, 0);
  CHECK_EQ(extra_token_ids.size(), static_cast<size_t>(num_sequences));
  int32_t* destination = output.data_ptr<int32_t>();
  size_t offset = 0;
  for (int32_t sequence = 0; sequence < num_sequences; ++sequence) {
    const int32_t query_length =
        params.attention.host.q_seq_lens[static_cast<size_t>(sequence)];
    CHECK_GT(query_length, 0);
    CHECK_LE(static_cast<size_t>(query_length), tokens.size() - offset);
    const size_t length = static_cast<size_t>(query_length);
    std::copy(tokens.data() + offset + 1,
              tokens.data() + offset + length,
              destination + offset);
    destination[offset + length - 1] = extra_token_ids[sequence];
    offset += length;
  }
  CHECK_EQ(offset, tokens.size());
}

MtpDraftExtendRows build_mtp_draft_extend_rows(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& target_hidden_states) {
  CHECK(accepted_tokens.defined());
  CHECK(target_hidden_states.defined());
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(target_hidden_states.dim(), 3);
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t width = accepted_tokens.size(1);
  CHECK_EQ(target_hidden_states.size(0), batch_size);
  CHECK_EQ(target_hidden_states.size(1), width);
  CHECK_GT(batch_size, 0);
  CHECK_GT(width, 0);

  torch::Tensor last_rows =
      accepted_tokens.ge(0).sum(/*dim=*/1).to(torch::kLong);
  last_rows.sub_(1).clamp_min_(0);
  const torch::Tensor safe_tokens = accepted_tokens.clamp_min(0);
  const torch::Tensor sequence_idxes =
      torch::arange(batch_size, accepted_tokens.options().dtype(torch::kLong));
  MtpDraftExtendRows rows;
  rows.token_ids = safe_tokens.flatten();
  rows.hidden_states = target_hidden_states.flatten(0, 1);
  rows.logits_indices = sequence_idxes * width + last_rows;
  return rows;
}

template <typename TargetInput>
TargetInput build_mtp_draft_extend_input(const TargetInput& validate_input,
                                         const MtpDraftExtendRows& rows,
                                         SamplingParameters sampling_params) {
  TargetInput input;
  input.positions = validate_input.positions;
  input.positions_host = validate_input.positions_host;
  input.runtime.kv_slot_layout = validate_input.runtime.kv_slot_layout;
  input.sample_sequence_ids = validate_input.sample_sequence_ids;
  // LlmModelParams/VlmModelParams delete copy-assignment by design, so the
  // carried field set is written as this explicit whitelist; the draft
  // worker rebuilds MLA prefix-cache workspace for its own cache.
  auto& params = input.input_params;
  params.meta = validate_input.input_params.meta;
  params.attention = validate_input.input_params.attention;
  params.parallel = validate_input.input_params.parallel;
  params.multi_block_tables = validate_input.input_params.multi_block_tables;
  input.input_params.embedding.embedding_ids =
      validate_input.input_params.embedding.embedding_ids;
  input.input_params.embedding.request_ids =
      validate_input.input_params.embedding.request_ids;
  params.attention.device.history_compressed_kv = torch::Tensor();
  params.attention.device.history_k_rope = torch::Tensor();
  clone_mtp_draft_metadata(input);

  input.token_ids = safe_to(rows.token_ids, validate_input.token_ids.options());
  params.embedding.input_embedding = rows.hidden_states;
  input.sampling_params = std::move(sampling_params);
  input.sampling_params.logits_indices() =
      rows.logits_indices.to(input.token_ids.device());
  input.runtime.device_tensors_ready = true;
  return input;
}

template <typename TargetInput>
void clone_mtp_draft_metadata(TargetInput& input) {
  const auto clone_tensor = [](torch::Tensor& tensor) {
    if (tensor.defined()) {
      tensor = clone_contiguous_detached_tensor(tensor);
    }
  };
  clone_tensor(input.positions);
  clone_tensor(input.positions_host);
  auto& attention = input.input_params.attention;
  // Block tables are the largest host snapshot per decode round; the memcpy
  // clone avoids launching an ATen parallel copy team on every TP rank.
  if (attention.host.block_tables.defined()) {
    attention.host.block_tables =
        clone_host_block_tables(attention.host.block_tables);
  }
  for (torch::Tensor& block_table : input.input_params.multi_block_tables) {
    if (block_table.defined()) {
      block_table = clone_host_block_tables(block_table);
    }
  }
  std::array tensors = {
      &attention.device.q_seq_lens,
      &attention.device.kv_seq_lens,
      &attention.device.q_cu_seq_lens,
      &attention.device.new_cache_slots,
      &attention.device.block_tables,
      &attention.device.paged_kv_indptr,
      &attention.device.paged_kv_indices,
      &attention.device.paged_kv_last_page_len,
      &attention.device.new_cache_slot_offsets,
      &attention.device.kv_cache_start_offsets,
      &attention.device.kv_cache_tokens_nums,
      &attention.device.history_compressed_kv,
      &attention.device.history_k_rope,
      &attention.device.ring_cur_seqlen,
      &attention.device.ring_cache_seqlen,
  };
  clone_packed_metadata(attention.attention_device_buffer,
                        attention.attention_buffer_bytes,
                        tensors);
  for (torch::Tensor* tensor : tensors) {
    if (tensor != nullptr) {
      clone_tensor(*tensor);
    }
  }

  // Every view holds storage independent from the target, so neither its
  // reusable buffers nor borrowed pointers are exposed.
  attention.host.graph_q_seq_lens_data = nullptr;
  attention.host.graph_kv_seq_lens_data = nullptr;
  attention.attention_host_buffer = torch::Tensor();
  attention.attention_device_buffer = torch::Tensor();
  attention.attention_buffer_bytes = 0;
  attention.attention_buffer_capacity = 0;
  attention.attention_buffer_owner = std::make_shared<int>(0);
}
template LlmForwardInput build_mtp_draft_extend_input(const LlmForwardInput&,
                                                      const MtpDraftExtendRows&,
                                                      SamplingParameters);
template VlmForwardInput build_mtp_draft_extend_input(const VlmForwardInput&,
                                                      const MtpDraftExtendRows&,
                                                      SamplingParameters);
template void fill_mtp_prefill_token_ids(const torch::Tensor&,
                                         const LlmModelParams&,
                                         const std::vector<int32_t>&,
                                         torch::Tensor&);
template void fill_mtp_prefill_token_ids(const torch::Tensor&,
                                         const VlmModelParams&,
                                         const std::vector<int32_t>&,
                                         torch::Tensor&);

}  // namespace xllm
