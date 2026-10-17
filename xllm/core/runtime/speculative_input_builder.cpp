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

#include "core/runtime/speculative_input_builder.h"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>

#if defined(USE_NPU)
#include "core/layers/common/expanded_decode_metadata_builder.h"
#endif
#include "framework/model/model_input_params.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/vlm_forward_params.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm::specBuilder {

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
template <typename Input>
Slice<int32_t> get_token_ids(const Input& input) {
  return tensor_slice(input.token_ids_host);
}
template <typename Input>
Slice<int32_t> get_positions(const Input& input) {
  return tensor_slice(input.positions_host);
}
template <typename Input>
Slice<int32_t> get_kv_seq_lens(const Input& input) {
  return input.input_params.attention.host.kv_seq_lens;
}

// Resolves a row token from either input token_ids[seq_id] or row.token_id.
int32_t resolve_row_token_id(const DecodeRowContext& ctx, const RowSpec& row) {
  if (!row.use_input_token) {
    return row.token_id;
  }
  CHECK_LT(static_cast<size_t>(row.seq_id), ctx.token_ids.size())
      << "seq_id out of range for token_ids, seq_id=" << row.seq_id
      << ", token_ids_size=" << ctx.token_ids.size();
  return ctx.token_ids[row.seq_id];
}

Slice<int32_t> get_block_table_slice(const DecodeRowContext& ctx,
                                     int32_t seq_id) {
  CHECK_GE(seq_id, 0) << "invalid seq_id=" << seq_id;
  CHECK_LT(seq_id, ctx.num_sequences)
      << "seq_id out of range for block tables, seq_id=" << seq_id
      << ", num_sequences=" << ctx.num_sequences;
  CHECK_GT(ctx.block_table_stride, 0)
      << "invalid block table row stride=" << ctx.block_table_stride;

  const size_t row_offset =
      static_cast<size_t>(seq_id) * static_cast<size_t>(ctx.block_table_stride);
  CHECK_LE(row_offset + static_cast<size_t>(ctx.block_table_stride),
           ctx.block_tables.size())
      << "block table row out of range, seq_id=" << seq_id
      << ", row_offset=" << row_offset
      << ", row_stride=" << ctx.block_table_stride
      << ", block_tables_size=" << ctx.block_tables.size();
  return {ctx.block_tables.data() + row_offset,
          static_cast<size_t>(ctx.block_table_stride)};
}

torch::Tensor create_flat_2d_tensor(const std::vector<int32_t>& values,
                                    int32_t rows,
                                    int32_t stride) {
  if (rows == 0) {
    return torch::Tensor();
  }
  CHECK_GT(rows, 0) << "invalid rows=" << rows;
  CHECK_GT(stride, 0) << "invalid stride=" << stride;
  CHECK_EQ(values.size(), static_cast<size_t>(rows) * stride)
      << "flat 2D tensor size mismatch, rows=" << rows << ", stride=" << stride
      << ", values_size=" << values.size();
  auto tensor = torch::empty({rows, stride}, pinned_cpu_options(torch::kInt));
  std::copy(values.begin(), values.end(), tensor.data_ptr<int32_t>());
  return tensor;
}

void fill_multi_block_table_slices(DecodeRowContext& ctx) {
  ctx.model_managed_multiblock = !ctx.multi_block_tables_owner.empty();
  ctx.multi_block_tables.resize(ctx.multi_block_tables_owner.size());
  for (size_t m = 0; m < ctx.multi_block_tables_owner.size(); ++m) {
    const torch::Tensor& manager_table = ctx.multi_block_tables_owner[m];
    CHECK(manager_table.defined())
        << "multi_block_tables[" << m << "] is undefined";
    CHECK_EQ(manager_table.dim(), 2)
        << "multi_block_tables[" << m << "] must be 2D, got "
        << manager_table.sizes();
    CHECK_LE(ctx.num_sequences, manager_table.size(0))
        << "num_sequences exceeds multi_block_tables[" << m
        << "] rows, num_sequences=" << ctx.num_sequences
        << ", rows=" << manager_table.size(0);
    // A Slice borrows contiguous row storage, so each row must be unit-strided.
    // Indexing per row would build a throwaway TensorImpl per sequence.
    CHECK_EQ(manager_table.stride(1), 1)
        << "multi_block_tables[" << m << "] rows must be contiguous";
    const int64_t row_stride = manager_table.stride(0);
    const size_t row_length = static_cast<size_t>(manager_table.size(1));
    int32_t* base = manager_table.data_ptr<int32_t>();
    ctx.multi_block_tables[m].reserve(static_cast<size_t>(ctx.num_sequences));
    for (int32_t seq_id = 0; seq_id < ctx.num_sequences; ++seq_id) {
      ctx.multi_block_tables[m].emplace_back(base + seq_id * row_stride,
                                             row_length);
    }
  }
}

}  // namespace

MtpTopkStatePtr select_mtp_topk_state_for_next_step(
    const MtpTopkStatePtr& state,
    const SamplingParameters& sampling_params) {
  if (state == nullptr || !sampling_params.selected_token_idxes.defined()) {
    return state;
  }
  const torch::Tensor& selected_idxes = sampling_params.selected_token_idxes;
  if (selected_idxes.numel() == 0 ||
      state->num_rows() == selected_idxes.numel()) {
    return state;
  }
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
    const int32_t query_length = params.get_q_seq_len(sequence);
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
    const torch::Tensor& target_hidden_states,
    const std::vector<int32_t>& per_seq_val_tokens,
    bool step_major_layout) {
  CHECK(accepted_tokens.defined());
  CHECK(target_hidden_states.defined());
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(target_hidden_states.dim(), 3);
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t width = accepted_tokens.size(1);
  CHECK_EQ(target_hidden_states.size(0), batch_size);
  CHECK_EQ(target_hidden_states.size(1), width);
  CHECK_EQ(per_seq_val_tokens.size(), static_cast<size_t>(batch_size));
  CHECK_GT(batch_size, 0);
  CHECK_GT(width, 0);

  bool uniform_width = true;
  for (int32_t query_len : per_seq_val_tokens) {
    CHECK_GT(query_len, 0);
    CHECK_LE(query_len, width);
    uniform_width = uniform_width && query_len == width;
  }

  torch::Tensor last_rows =
      accepted_tokens.ge(0).sum(/*dim=*/1).to(torch::kLong);
  last_rows.sub_(1).clamp_min_(0);
  const torch::Tensor safe_tokens = accepted_tokens.clamp_min(0);
  const torch::Tensor sequence_idxes =
      uniform_width
          ? torch::arange(batch_size,
                          accepted_tokens.options().dtype(torch::kLong))
          : torch::Tensor();
  MtpDraftExtendRows rows;
  if (step_major_layout) {
    CHECK(uniform_width) << "step-major MTP extend requires uniform queries";
    rows.token_ids = safe_tokens.transpose(0, 1).contiguous().flatten();
    rows.hidden_states =
        target_hidden_states.transpose(0, 1).contiguous().flatten(0, 1);
    rows.selected_token_idxes = last_rows * batch_size + sequence_idxes;
    return rows;
  }
  if (uniform_width) {
    rows.token_ids = safe_tokens.flatten();
    rows.hidden_states = target_hidden_states.flatten(0, 1);
    rows.selected_token_idxes = sequence_idxes * width + last_rows;
    return rows;
  }

  std::vector<torch::Tensor> token_rows;
  std::vector<torch::Tensor> hidden_rows;
  std::vector<int64_t> offsets;
  token_rows.reserve(batch_size);
  hidden_rows.reserve(batch_size);
  offsets.reserve(batch_size);
  int64_t offset = 0;
  for (int64_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    const int64_t query_len = per_seq_val_tokens[seq_id];
    token_rows.emplace_back(safe_tokens[seq_id].narrow(0, 0, query_len));
    hidden_rows.emplace_back(
        target_hidden_states[seq_id].narrow(0, 0, query_len));
    offsets.emplace_back(offset);
    offset += query_len;
  }
  rows.token_ids = torch::cat(token_rows);
  rows.hidden_states = torch::cat(hidden_rows);
  rows.selected_token_idxes =
      async_h2d_tensor(offsets, accepted_tokens.options().dtype(torch::kLong)) +
      last_rows;
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
  auto& params = input.input_params;
  params.meta = validate_input.input_params.meta;
  params.attention = validate_input.input_params.attention;
  params.parallel = validate_input.input_params.parallel;
  params.multi_block_tables = validate_input.input_params.multi_block_tables;
  input.input_params.embedding.embedding_ids =
      validate_input.input_params.embedding.embedding_ids;
  input.input_params.embedding.request_ids =
      validate_input.input_params.embedding.request_ids;
  // The draft worker rebuilds MLA prefix-cache workspace for its own cache.
  params.attention.device.history_compressed_kv = torch::Tensor();
  params.attention.device.history_k_rope = torch::Tensor();
  clone_mtp_draft_metadata(input);

  input.token_ids = safe_to(rows.token_ids, validate_input.token_ids.options());
  params.embedding.input_embedding = rows.hidden_states;
  input.sampling_params = std::move(sampling_params);
  const torch::Tensor sample_idxes = arange_indices(
      rows.selected_token_idxes.numel(), input.token_ids.device());
  input.sampling_params.sample_idxes = sample_idxes;
  input.sampling_params.selected_token_idxes =
      rows.selected_token_idxes.to(sample_idxes.options());
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
  for (torch::Tensor& block_table : input.input_params.multi_block_tables) {
    clone_tensor(block_table);
  }

  auto& attention = input.input_params.attention;
  clone_tensor(attention.host.block_tables);
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
      &attention.device.in_prefix_slots,
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

int32_t calc_slot_id(int32_t position,
                     const Slice<int32_t>& block_table_slice,
                     int32_t block_size) {
  CHECK_GT(block_size, 0) << "invalid block_size=" << block_size;
  CHECK_GE(position, 0) << "invalid position=" << position;
  const int32_t block_idx = position / block_size;
  CHECK_LT(static_cast<size_t>(block_idx), block_table_slice.size())
      << "block table index out of range, block_idx=" << block_idx
      << ", block_table_size=" << block_table_slice.size()
      << ", position=" << position << ", block_size=" << block_size;
  const int32_t block_id = block_table_slice[block_idx];
  CHECK_GE(block_id, 0) << "invalid block_id=" << block_id;
  const int32_t block_offset = position % block_size;
  return block_id * block_size + block_offset;
}

int32_t calc_ring_slot_id(int32_t position,
                          const Slice<int32_t>& block_table_slice,
                          int32_t block_size) {
  CHECK_GT(block_size, 0) << "invalid block_size=" << block_size;
  CHECK_GE(position, 0) << "invalid position=" << position;
  CHECK(!block_table_slice.empty()) << "circular block table must not be empty";
  const int32_t block_idx =
      (position / block_size) % static_cast<int32_t>(block_table_slice.size());
  const int32_t block_id = block_table_slice[block_idx];
  CHECK_GE(block_id, 0) << "invalid circular block_id=" << block_id;
  const int32_t block_offset = position % block_size;
  return block_id * block_size + block_offset;
}
template <typename Input>
std::vector<int32_t> build_grouped_prefill_swa_slots(const Input& input,
                                                     int32_t block_size) {
  DecodeRowContext ctx = make_decode_row_context(input);
  CHECK(ctx.model_managed_multiblock)
      << "grouped prefill SWA slots require multi_block_tables";
  CHECK(!ctx.multi_block_tables.empty())
      << "grouped prefill SWA slots require manager 0";
  CHECK_EQ(ctx.multi_block_tables.front().size(),
           static_cast<size_t>(ctx.num_sequences))
      << "grouped prefill SWA manager row count mismatch";

  std::vector<int32_t> slots;
  slots.reserve(ctx.positions.size());
  for (int32_t seq_id = 0; seq_id < ctx.num_sequences; ++seq_id) {
    const int32_t kv_len = calc_kv_len(ctx.kv_seq_lens, seq_id, /*offset=*/0);
    const int32_t q_len = input.input_params.get_q_seq_len(seq_id);
    CHECK_GE(q_len, 0) << "grouped prefill q length must be non-negative";
    CHECK_GE(kv_len, q_len) << "grouped prefill kv length must cover the query";
    const int32_t query_start = kv_len - q_len;
    const Slice<int32_t>& swa_block_table =
        ctx.multi_block_tables.front()[static_cast<size_t>(seq_id)];
    for (int32_t query_idx = 0; query_idx < q_len; ++query_idx) {
      slots.emplace_back(calc_ring_slot_id(
          query_start + query_idx, swa_block_table, block_size));
    }
  }
  CHECK_EQ(slots.size(), ctx.positions.size())
      << "grouped prefill SWA slot/position count mismatch";
  return slots;
}

int32_t calc_kv_len(const Slice<int32_t>& kv_seq_lens_slice,
                    int32_t seq_id,
                    int32_t offset) {
  CHECK_GE(seq_id, 0) << "invalid seq_id=" << seq_id;
#if defined(USE_NPU)
  CHECK_LT(static_cast<size_t>(seq_id), kv_seq_lens_slice.size())
      << "seq_id out of range, seq_id=" << seq_id
      << ", kv_seq_lens_size=" << kv_seq_lens_slice.size();
  return kv_seq_lens_slice[seq_id] + offset;
#else
  CHECK_LT(static_cast<size_t>(seq_id + 1), kv_seq_lens_slice.size())
      << "seq_id out of range for cumulative layout, seq_id=" << seq_id
      << ", kv_seq_lens_size=" << kv_seq_lens_slice.size();
  return kv_seq_lens_slice[seq_id + 1] - kv_seq_lens_slice[seq_id] + offset;
#endif
}

void append_seq_len_by_layout(std::vector<int32_t>& vec, int32_t len) {
#if defined(USE_NPU)
  vec.emplace_back(len);
#else
  // Cumulative seq-lens layout: [0, l0, l0+l1, ...].
  if (vec.empty()) {
    vec.emplace_back(0);
  }
  vec.emplace_back(vec.back() + len);
#endif
}

void append_q_seq_len(std::vector<int32_t>& q_seq_lens,
                      std::vector<int32_t>& q_cu_seq_lens,
                      int32_t len) {
  append_seq_len_by_layout(q_seq_lens, len);
  q_cu_seq_lens.emplace_back(
      (q_cu_seq_lens.empty() ? 0 : q_cu_seq_lens.back()) + len);
}

std::vector<int32_t> build_cu_seq_lens(const std::vector<int32_t>& seq_lens) {
  std::vector<int32_t> cu_seq_lens(seq_lens.size() + 1);
  cu_seq_lens[0] = 0;
  std::partial_sum(seq_lens.begin(), seq_lens.end(), cu_seq_lens.begin() + 1);
  return cu_seq_lens;
}

template <typename Params>
void publish_accepted_prefix_lengths(
    Params& input_params,
    const std::vector<int32_t>& accepted_prefix_lengths,
    const torch::TensorOptions& token_options,
    bool publish_device_tensor) {
  // num_accepted_tokens is a logical checkpoint index, not the physical
  // conv-state capacity. Clamping it commits a stale checkpoint (issue #2247).
  input_params.num_accepted_tokens_host.assign(accepted_prefix_lengths.begin(),
                                               accepted_prefix_lengths.end());
  if (publish_device_tensor) {
    input_params.num_accepted_tokens =
        torch::tensor(accepted_prefix_lengths, token_options);
  }
}

#if defined(USE_NPU)
template <typename Params>
bool prepare_expanded_spec_verify_graph_host_input(Params& input_params) {
  input_params.graph.use_expanded_decode_for_spec_verify_attention = false;
  input_params.graph.expanded_kv_seq_lens = torch::Tensor();
  input_params.graph.expanded_block_tables = torch::Tensor();
  input_params.graph.expanded_paged_kv_indptr = torch::Tensor();
  input_params.graph.expanded_paged_kv_indices = torch::Tensor();
  input_params.graph.expanded_paged_kv_last_page_len = torch::Tensor();
  input_params.graph.expanded_tiling_data = torch::Tensor();
  input_params.graph.expanded_kv_seq_lens_vec.clear();
  if (!input_params.is_spec_verify ||
      !input_params.meta.batch_forward_type.is_chunked_prefill()) {
    return false;
  }

  const std::vector<int32_t>& q_seq_lens =
      input_params.attention.host.q_seq_lens;
  const std::vector<int32_t>& kv_seq_lens =
      input_params.attention.host.kv_seq_lens;
  if (q_seq_lens.empty() || kv_seq_lens.empty()) {
    return false;
  }
  input_params.graph.expanded_kv_seq_lens_vec =
      layer::ExpandedDecodeMetadataBuilder::build_tokenwise_kv_seq_lens(
          q_seq_lens, kv_seq_lens);
  if (input_params.graph.expanded_kv_seq_lens_vec.empty()) {
    return false;
  }
  input_params.graph.use_expanded_decode_for_spec_verify_attention = true;
  return true;
}

template <typename Params>
void bind_expanded_spec_verify_graph_input(Params& input_params,
                                           const torch::Device& device,
                                           int32_t block_size) {
  if (!input_params.graph.use_expanded_decode_for_spec_verify_attention) {
    return;
  }
  CHECK(input_params.attention.device.block_tables.defined())
      << "spec verify block tables must be rebuilt before graph input";
  const std::vector<int32_t>& q_seq_lens =
      input_params.attention.host.q_seq_lens;
  CHECK_GE(input_params.attention.device.block_tables.size(0),
           static_cast<int64_t>(q_seq_lens.size()))
      << "spec verify block table rows are fewer than sequences";

  std::vector<torch::Tensor> expanded_block_rows;
  expanded_block_rows.reserve(
      input_params.graph.expanded_kv_seq_lens_vec.size());
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(q_seq_lens.size());
       ++seq_idx) {
    for (int32_t token_idx = 0;
         token_idx < q_seq_lens[static_cast<size_t>(seq_idx)];
         ++token_idx) {
      expanded_block_rows.emplace_back(
          input_params.attention.device.block_tables.select(/*dim=*/0,
                                                            seq_idx));
    }
  }
  CHECK_EQ(expanded_block_rows.size(),
           input_params.graph.expanded_kv_seq_lens_vec.size());
  if (!input_params.graph.expanded_kv_seq_lens.defined()) {
    input_params.graph.expanded_kv_seq_lens =
        async_h2d_tensor(input_params.graph.expanded_kv_seq_lens_vec, device);
  }
  // torch::stack already produces a contiguous tensor.
  torch::Tensor expanded_block_tables =
      torch::stack(expanded_block_rows, /*dim=*/0);
  layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      ModelInputParams(input_params),
      input_params.graph.expanded_kv_seq_lens,
      expanded_block_tables,
      input_params.graph.expanded_kv_seq_lens_vec,
      block_size);
}

template <typename Params>
void build_expanded_spec_verify_graph_input(Params& input_params,
                                            const torch::Device& device,
                                            int32_t block_size) {
  CHECK(prepare_expanded_spec_verify_graph_host_input(input_params))
      << "spec-verify chunked prefill requires q/kv seq lens";
  bind_expanded_spec_verify_graph_input(input_params, device, block_size);
}
#endif

void update_kv_seq_lens_and_max(std::vector<int32_t>& kv_seq_lens_vec,
                                int32_t kv_len,
                                int32_t& kv_max_seq_len) {
  if (kv_len > kv_max_seq_len) {
    kv_max_seq_len = kv_len;
  }
  append_seq_len_by_layout(kv_seq_lens_vec, kv_len);
}
template <typename Input>
DecodeRowContext make_decode_row_context(const Input& input) {
  DecodeRowContext ctx;
  ctx.num_sequences = input.input_params.meta.num_sequences;
  CHECK_GE(ctx.num_sequences, 0) << "invalid num_sequences";

  if (input.token_ids_host.defined()) {
    ctx.token_ids = get_token_ids(input);
  }
  CHECK(input.positions_host.defined())
      << "positions_host must be defined for decode row build";
  ctx.positions = get_positions(input);
  CHECK_GE(static_cast<int32_t>(ctx.positions.size()), ctx.num_sequences)
      << "positions size is smaller than num_sequences, positions_size="
      << ctx.positions.size() << ", num_sequences=" << ctx.num_sequences;

  ctx.kv_seq_lens = get_kv_seq_lens(input);
  if (!input.input_params.multi_block_tables.empty()) {
    ctx.multi_block_tables_owner.reserve(
        input.input_params.multi_block_tables.size());
    for (const torch::Tensor& block_table :
         input.input_params.multi_block_tables) {
      ctx.multi_block_tables_owner.emplace_back(to_cpu_contiguous(block_table));
    }
    fill_multi_block_table_slices(ctx);
    return ctx;
  }

  CHECK(input.input_params.attention.host.block_tables.defined())
      << "host block_tables must be defined for decode row build";
  ctx.block_tables_owner =
      input.input_params.attention.host.block_tables.contiguous();
  CHECK_EQ(ctx.block_tables_owner.dim(), 2)
      << "block_tables must be 2D, got " << ctx.block_tables_owner.sizes();
  CHECK_LE(ctx.num_sequences, ctx.block_tables_owner.size(0))
      << "num_sequences exceeds block table rows, num_sequences="
      << ctx.num_sequences
      << ", block_table_rows=" << ctx.block_tables_owner.size(0);
  ctx.block_table_stride = static_cast<int32_t>(ctx.block_tables_owner.size(1));
  CHECK_GT(ctx.block_table_stride, 0)
      << "invalid block table row stride=" << ctx.block_table_stride;
  ctx.block_tables = tensor_slice(ctx.block_tables_owner);
  return ctx;
}

void append_decode_row(const DecodeRowContext& ctx,
                       const RowSpec& row,
                       int32_t block_size,
                       DecodeBuildBuffers& buf) {
  CHECK_GE(row.seq_id, 0);
  CHECK_LT(row.seq_id, ctx.num_sequences);
  CHECK_LT(static_cast<size_t>(row.seq_id), ctx.positions.size());
  const int32_t model_position =
      ctx.positions[row.seq_id] + row.position_offset;
  const int32_t cache_position =
      calc_kv_len(ctx.kv_seq_lens, row.seq_id, row.position_offset) - 1;
  CHECK_GE(model_position, 0) << "invalid decode model position";
  CHECK_GE(cache_position, 0) << "invalid decode cache position";

  // All decode paths can toggle which fields are emitted, so one row builder
  // can serve draft/validate/first-decode/update-last-step scenarios.
  if (row.append_token) {
    buf.out_token_ids.emplace_back(resolve_row_token_id(ctx, row));
  }
  buf.out_positions.emplace_back(model_position);

  if (ctx.model_managed_multiblock) {
    CHECK(!ctx.multi_block_tables.empty())
        << "model-managed multiblock input requires an SWA manager";
    CHECK_LT(static_cast<size_t>(row.seq_id),
             ctx.multi_block_tables.front().size());
    buf.out_new_cache_slots.emplace_back(calc_ring_slot_id(
        cache_position,
        ctx.multi_block_tables.front()[static_cast<size_t>(row.seq_id)],
        block_size));
    if (row.append_block_table) {
      if (buf.out_multi_block_tables.size() < ctx.multi_block_tables.size()) {
        buf.out_multi_block_tables.resize(ctx.multi_block_tables.size());
      }
      for (size_t m = 0; m < ctx.multi_block_tables.size(); ++m) {
        CHECK_LT(static_cast<size_t>(row.seq_id),
                 ctx.multi_block_tables[m].size());
        const Slice<int32_t>& block_table_slice =
            ctx.multi_block_tables[m][row.seq_id];
        buf.out_multi_block_tables[m].emplace_back(block_table_slice.begin(),
                                                   block_table_slice.end());
      }
    }
  } else {
    const Slice<int32_t> block_table_slice =
        get_block_table_slice(ctx, row.seq_id);
    buf.out_new_cache_slots.emplace_back(
        calc_slot_id(cache_position, block_table_slice, block_size));
    if (row.append_block_table) {
      if (buf.out_block_table_stride == 0) {
        buf.out_block_table_stride =
            static_cast<int32_t>(block_table_slice.size());
      }
      CHECK_EQ(buf.out_block_table_stride,
               static_cast<int32_t>(block_table_slice.size()))
          << "block table stride mismatch";
      buf.out_block_tables.insert(buf.out_block_tables.end(),
                                  block_table_slice.begin(),
                                  block_table_slice.end());
      ++buf.out_block_table_rows;
    }
  }

  if (row.append_kv_len) {
    const int32_t kv_len_offset =
        row.kv_len_offset.value_or(row.position_offset);
    const int32_t kv_len =
        calc_kv_len(ctx.kv_seq_lens, row.seq_id, kv_len_offset);
    update_kv_seq_lens_and_max(
        buf.out_kv_seq_lens, kv_len, buf.meta.kv_max_seq_len);
  }
  if (row.append_q_len_one) {
    append_q_seq_len(buf.out_q_seq_lens, buf.out_q_cu_seq_lens, 1);
  }
}

TokenWithOffset resolve_token_with_position_offset(
    int32_t input_token_id,
    int32_t seq_id,
    const Slice<int64_t>& last_step_tokens,
    int32_t last_step_decode_num) {
  CHECK_GT(last_step_decode_num, 0)
      << "invalid last_step_decode_num=" << last_step_decode_num;
  if (input_token_id >= 0) {
    TokenWithOffset direct;
    direct.token_id = input_token_id;
    direct.position_offset = 0;
    return direct;
  }

  const int32_t placeholder_idx = -input_token_id - 1;
  CHECK_GE(placeholder_idx, 0)
      << "invalid placeholder token id=" << input_token_id
      << ", seq_id=" << seq_id;
  const int32_t base_idx = placeholder_idx * last_step_decode_num;
  CHECK_LE(base_idx + last_step_decode_num,
           static_cast<int32_t>(last_step_tokens.size()))
      << "last_step_tokens out of range, seq_id=" << seq_id
      << ", placeholder_idx=" << placeholder_idx
      << ", last_step_decode_num=" << last_step_decode_num
      << ", last_step_tokens_size=" << last_step_tokens.size();

  TokenWithOffset resolved;
  resolved.position_offset = -1;
  for (int32_t i = 0; i < last_step_decode_num; ++i) {
    const int64_t candidate = last_step_tokens[base_idx + i];
    if (candidate >= 0) {
      CHECK_LE(candidate,
               static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
          << "token id overflow, seq_id=" << seq_id
          << ", candidate=" << candidate;
      resolved.token_id = static_cast<int32_t>(candidate);
      resolved.position_offset += 1;
    }
  }
  return resolved;
}

void append_decode_row_from_last_step(const DecodeRowContext& ctx,
                                      int32_t seq_id,
                                      int32_t input_token_id,
                                      const Slice<int64_t>& last_step_tokens,
                                      int32_t last_step_decode_num,
                                      int32_t block_size,
                                      DecodeBuildBuffers& buf) {
  // Placeholder tokens (-1/-2/...) are resolved from last-step outputs first,
  // then appended via the same row builder used by all decode paths.
  const TokenWithOffset resolved = resolve_token_with_position_offset(
      input_token_id, seq_id, last_step_tokens, last_step_decode_num);

  RowSpec row;
  row.seq_id = seq_id;
  row.token_id = resolved.token_id;
  row.position_offset = resolved.position_offset;
  append_decode_row(ctx, row, block_size, buf);
}
template <typename Params>
void update_input_params(Params& input_params,
                         DecodeBuildBuffers& buf,
                         int32_t q_max_seq_len,
                         std::vector<int32_t> q_seq_lens_vec,
                         std::vector<int32_t> q_cu_seq_lens_vec,
                         int32_t kv_max_seq_len,
                         std::vector<int32_t> kv_seq_lens_vec,
                         bool update_block_tables) {
  CHECK_EQ(q_seq_lens_vec.empty(), q_cu_seq_lens_vec.empty())
      << "q_seq_lens and q_cu_seq_lens must be provided together";
  input_params.meta.q_max_seq_len = q_max_seq_len;
  input_params.attention.host.q_seq_lens = std::move(q_seq_lens_vec);
  input_params.attention.host.q_cu_seq_lens = std::move(q_cu_seq_lens_vec);
  input_params.meta.kv_max_seq_len = kv_max_seq_len;
  input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  input_params.attention.host.new_cache_slots =
      std::move(buf.out_new_cache_slots);
  if (update_block_tables) {
    if (!buf.out_multi_block_tables.empty()) {
      input_params.multi_block_tables.clear();
      input_params.multi_block_tables.reserve(
          buf.out_multi_block_tables.size());
      for (std::vector<std::vector<int32_t>>& manager_tables :
           buf.out_multi_block_tables) {
        util::pad_2d_vector(manager_tables, /*pad_value=*/-1);
        input_params.multi_block_tables.emplace_back(
            create_2d_tensor(manager_tables, torch::kInt));
      }
      input_params.attention.host.block_tables = torch::Tensor();
    } else {
      input_params.attention.host.block_tables =
          create_flat_2d_tensor(buf.out_block_tables,
                                buf.out_block_table_rows,
                                buf.out_block_table_stride);
      input_params.multi_block_tables.clear();
    }
  }
}

template <typename Input>
void set_token_position_tensors(Input& input,
                                const std::vector<int32_t>& token_ids,
                                const std::vector<int32_t>& positions,
                                const torch::TensorOptions& token_options,
                                const torch::TensorOptions& position_options) {
  input.runtime.device_tensors_ready = false;
  input.token_ids_host = make_pinned_cpu_tensor(token_ids);
  input.positions_host = make_pinned_cpu_tensor(positions);
  input.token_ids =
      safe_to(input.token_ids_host, token_options, /*non_blocking=*/true);
  input.positions =
      safe_to(input.positions_host, position_options, /*non_blocking=*/true);
  input.runtime.device_tensors_ready = true;
}

template DecodeRowContext make_decode_row_context(const LlmForwardInput&);
template std::vector<int32_t> build_grouped_prefill_swa_slots(
    const LlmForwardInput&,
    int32_t);
template void set_token_position_tensors(LlmForwardInput&,
                                         const std::vector<int32_t>&,
                                         const std::vector<int32_t>&,
                                         const torch::TensorOptions&,
                                         const torch::TensorOptions&);
template DecodeRowContext make_decode_row_context(const VlmForwardInput&);
template std::vector<int32_t> build_grouped_prefill_swa_slots(
    const VlmForwardInput&,
    int32_t);
template void set_token_position_tensors(VlmForwardInput&,
                                         const std::vector<int32_t>&,
                                         const std::vector<int32_t>&,
                                         const torch::TensorOptions&,
                                         const torch::TensorOptions&);
template void update_input_params(ModelInputParams&,
                                  DecodeBuildBuffers&,
                                  int32_t,
                                  std::vector<int32_t>,
                                  std::vector<int32_t>,
                                  int32_t,
                                  std::vector<int32_t>,
                                  bool);
template void update_input_params(LlmModelParams&,
                                  DecodeBuildBuffers&,
                                  int32_t,
                                  std::vector<int32_t>,
                                  std::vector<int32_t>,
                                  int32_t,
                                  std::vector<int32_t>,
                                  bool);
template void update_input_params(VlmModelParams&,
                                  DecodeBuildBuffers&,
                                  int32_t,
                                  std::vector<int32_t>,
                                  std::vector<int32_t>,
                                  int32_t,
                                  std::vector<int32_t>,
                                  bool);

// Explicit instantiations for the parameter-view templates shared by the
// LLM/VLM MTP workers and DFlash.
template void publish_accepted_prefix_lengths(LlmModelParams&,
                                              const std::vector<int32_t>&,
                                              const torch::TensorOptions&,
                                              bool);
template void publish_accepted_prefix_lengths(VlmModelParams&,
                                              const std::vector<int32_t>&,
                                              const torch::TensorOptions&,
                                              bool);
template void publish_accepted_prefix_lengths(ModelInputParams&,
                                              const std::vector<int32_t>&,
                                              const torch::TensorOptions&,
                                              bool);
template void build_expanded_spec_verify_graph_input(LlmModelParams&,
                                                     const torch::Device&,
                                                     int32_t);
template void build_expanded_spec_verify_graph_input(ModelInputParams&,
                                                     const torch::Device&,
                                                     int32_t);
template void build_expanded_spec_verify_graph_input(VlmModelParams&,
                                                     const torch::Device&,
                                                     int32_t);
template bool prepare_expanded_spec_verify_graph_host_input(LlmModelParams&);
template bool prepare_expanded_spec_verify_graph_host_input(VlmModelParams&);
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
  append_seq_len_by_layout(params.attention.host.q_seq_lens, query_width);
  params.attention.host.q_cu_seq_lens = {0, query_width};
  params.attention.host.kv_seq_lens.clear();
  append_seq_len_by_layout(params.attention.host.kv_seq_lens, 1);
  params.attention.host.new_cache_slots.assign(query_width, -1);
  params.attention.host.block_tables = torch::zeros({1, 1}, options);
  params.attention.device.paged_kv_indptr = torch::tensor({0, 1}, options);
  params.attention.device.paged_kv_indices = torch::tensor({0}, options);
  params.attention.device.paged_kv_last_page_len = torch::tensor({1}, options);
}

}  // namespace xllm::specBuilder
