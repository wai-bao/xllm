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

// decode_row_builder.cpp
#include "core/framework/batch/decode_row_builder.h"

#include <glog/logging.h>

#include <limits>

#include "core/framework/model/model_params_view.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/utils.h"

namespace xllm {
namespace input_builder {

namespace {

// Builds cumulative seq-lens layout: [0, l0, l0+l1, ...].
void push_cumsum(std::vector<int32_t>& vec, int32_t len) {
  if (vec.empty()) {
    vec.emplace_back(0);
  }
  vec.emplace_back(vec.back() + len);
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
    ctx.multi_block_tables[m].reserve(static_cast<size_t>(ctx.num_sequences));
    for (int32_t seq_id = 0; seq_id < ctx.num_sequences; ++seq_id) {
      torch::Tensor row = manager_table[seq_id];
      ctx.multi_block_tables[m].emplace_back(row.data_ptr<int32_t>(),
                                             static_cast<size_t>(row.numel()));
    }
  }
}

void append_paged_kv_row(const Slice<int32_t>& block_table,
                         int32_t kv_seq_len,
                         int32_t block_size,
                         DecodeBuildBuffers& buf) {
  CHECK_GT(block_size, 0) << "invalid block_size=" << block_size;
  const int32_t page_count = layer::kv_page_count(kv_seq_len, block_size);
  CHECK_LE(static_cast<size_t>(page_count), block_table.size())
      << "row KV length exceeds block-table capacity, kv_seq_len=" << kv_seq_len
      << ", page_count=" << page_count
      << ", block_table_size=" << block_table.size();
  if (buf.out_paged_kv_indptr.empty()) {
    buf.out_paged_kv_indptr.emplace_back(0);
  }
  buf.out_paged_kv_indices.insert(buf.out_paged_kv_indices.end(),
                                  block_table.begin(),
                                  block_table.begin() + page_count);
  buf.out_paged_kv_indptr.emplace_back(buf.out_paged_kv_indptr.back() +
                                       page_count);
  buf.out_paged_kv_last_page_len.emplace_back(
      layer::kv_last_page_len(kv_seq_len, block_size));
}

template <typename Input>
DecodeRowContext make_decode_row_context_impl(const Input& input) {
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

}  // namespace

template <typename Input>
std::vector<int32_t> build_grouped_prefill_swa_slots_impl(const Input& input,
                                                          int32_t block_size) {
  DecodeRowContext ctx = make_decode_row_context_impl(input);
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

DecodeRowContext make_decode_row_context(const LlmForwardInput& input) {
  return make_decode_row_context_impl(input);
}

DecodeRowContext make_decode_row_context(const VlmForwardInput& input) {
  return make_decode_row_context_impl(input);
}

std::vector<int32_t> build_grouped_prefill_swa_slots(
    const LlmForwardInput& input,
    int32_t block_size) {
  return build_grouped_prefill_swa_slots_impl(input, block_size);
}

std::vector<int32_t> build_grouped_prefill_swa_slots(
    const VlmForwardInput& input,
    int32_t block_size) {
  return build_grouped_prefill_swa_slots_impl(input, block_size);
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

  std::optional<int32_t> row_kv_seq_len;
  if (row.append_kv_len) {
    const int32_t kv_len_offset =
        row.kv_len_offset.value_or(row.position_offset);
    row_kv_seq_len = calc_kv_len(ctx.kv_seq_lens, row.seq_id, kv_len_offset);
    update_kv_seq_lens_and_max(
        buf.out_kv_seq_lens, *row_kv_seq_len, buf.meta.kv_max_seq_len);
  }

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
      if (row_kv_seq_len.has_value()) {
        append_paged_kv_row(
            block_table_slice, *row_kv_seq_len, block_size, buf);
      }
    }
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

void update_input_params(ModelParamsView& input_params,
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
          make_pinned_cpu_tensor(buf.out_block_tables)
              .view({buf.out_block_table_rows, buf.out_block_table_stride});
      input_params.multi_block_tables.clear();
      if (!buf.out_paged_kv_indptr.empty()) {
        CHECK_EQ(buf.out_paged_kv_indptr.size(),
                 static_cast<size_t>(buf.out_block_table_rows + 1));
        CHECK_EQ(buf.out_paged_kv_last_page_len.size(),
                 static_cast<size_t>(buf.out_block_table_rows));
        CHECK_EQ(buf.out_paged_kv_indptr.back(),
                 static_cast<int32_t>(buf.out_paged_kv_indices.size()));
        input_params.attention.device.paged_kv_indptr =
            make_pinned_cpu_tensor(buf.out_paged_kv_indptr);
        input_params.attention.device.paged_kv_indices =
            make_pinned_cpu_tensor(buf.out_paged_kv_indices);
        input_params.attention.device.paged_kv_last_page_len =
            make_pinned_cpu_tensor(buf.out_paged_kv_last_page_len);
      } else {
        input_params.attention.device.paged_kv_indptr = torch::Tensor();
        input_params.attention.device.paged_kv_indices = torch::Tensor();
        input_params.attention.device.paged_kv_last_page_len = torch::Tensor();
      }
    }
  }
}

void update_input_params(LlmModelParams& input_params,
                         DecodeBuildBuffers& buf,
                         int32_t q_max_seq_len,
                         std::vector<int32_t> q_seq_lens_vec,
                         std::vector<int32_t> q_cu_seq_lens_vec,
                         int32_t kv_max_seq_len,
                         std::vector<int32_t> kv_seq_lens_vec,
                         bool update_block_tables) {
  ModelParamsView input_params_view(input_params);
  update_input_params(input_params_view,
                      buf,
                      q_max_seq_len,
                      std::move(q_seq_lens_vec),
                      std::move(q_cu_seq_lens_vec),
                      kv_max_seq_len,
                      std::move(kv_seq_lens_vec),
                      update_block_tables);
}

void update_input_params(VlmModelParams& input_params,
                         DecodeBuildBuffers& buf,
                         int32_t q_max_seq_len,
                         std::vector<int32_t> q_seq_lens_vec,
                         std::vector<int32_t> q_cu_seq_lens_vec,
                         int32_t kv_max_seq_len,
                         std::vector<int32_t> kv_seq_lens_vec,
                         bool update_block_tables) {
  ModelParamsView input_params_view(input_params);
  update_input_params(input_params_view,
                      buf,
                      q_max_seq_len,
                      std::move(q_seq_lens_vec),
                      std::move(q_cu_seq_lens_vec),
                      kv_max_seq_len,
                      std::move(kv_seq_lens_vec),
                      update_block_tables);
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

template void set_token_position_tensors(LlmForwardInput&,
                                         const std::vector<int32_t>&,
                                         const std::vector<int32_t>&,
                                         const torch::TensorOptions&,
                                         const torch::TensorOptions&);
template void set_token_position_tensors(VlmForwardInput&,
                                         const std::vector<int32_t>&,
                                         const std::vector<int32_t>&,
                                         const torch::TensorOptions&,
                                         const torch::TensorOptions&);

ValidateRowResult build_validate_rows(const DecodeRowContext& ctx,
                                      const ValidateRowPlan& plan) {
  const int32_t num_sequences = ctx.num_sequences;
  const int32_t num_val_tokens = plan.num_val_tokens;
  const int32_t total = num_sequences * num_val_tokens;
  ValidateRowResult result;
  result.total_num_val_tokens = total;
  auto& buf = result.buf;
  buf.out_token_ids.reserve(total);
  buf.out_positions.reserve(total);
  buf.out_new_cache_slots.reserve(total);
  if (!plan.chunked_layout) {
    buf.out_kv_seq_lens.reserve(total);
    buf.out_q_seq_lens.reserve(total);
    buf.out_q_cu_seq_lens.reserve(total);
    buf.reserve_row_buffers(static_cast<size_t>(total), ctx.block_table_stride);
  }

  auto emit_row = [&](int32_t seq_id, int32_t val_idx) {
    RowSpec row;
    row.seq_id = seq_id;
    row.token_id = val_idx == 0 ? ctx.token_ids[seq_id] : -val_idx;
    row.position_offset = val_idx;
    row.append_kv_len = !plan.chunked_layout;
    row.append_q_len_one = !plan.chunked_layout;
    row.append_block_table = !plan.chunked_layout;
    append_decode_row(ctx, row, plan.block_size, buf);
  };

  if (plan.step_major) {
    // val-major order (embedded Eagle3 layout): all sequences' row k first.
    for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
      for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
        emit_row(seq_id, val_idx);
      }
    }
  } else {
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
        emit_row(seq_id, val_idx);
      }
    }
  }
  if (plan.chunked_layout) {
    result.chunked_q_seq_lens.reserve(num_sequences);
    result.chunked_q_cu_seq_lens.reserve(num_sequences + 1);
    result.chunked_kv_seq_lens.reserve(num_sequences + 1);
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      const int32_t kv_len = calc_kv_len(ctx.kv_seq_lens, seq_id, /*offset=*/0);
      const int32_t kv_len_after_validation = kv_len + num_val_tokens - 1;
      update_kv_seq_lens_and_max(result.chunked_kv_seq_lens,
                                 kv_len_after_validation,
                                 result.chunked_kv_max_seq_len);
      append_q_seq_len(result.chunked_q_seq_lens,
                       result.chunked_q_cu_seq_lens,
                       num_val_tokens);
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "validate positions/tokens mismatch";
  return result;
}

void commit_validate_rows(ModelParamsView& input_params,
                          ValidateRowResult& row_result,
                          const ValidateRowPlan& plan) {
  auto& buf = row_result.buf;
  if (!plan.chunked_layout) {
    input_params.meta.num_sequences = row_result.total_num_val_tokens;
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    update_input_params(input_params,
                        buf,
                        /*q_max_seq_len=*/1,
                        std::move(buf.out_q_seq_lens),
                        std::move(buf.out_q_cu_seq_lens),
                        buf.meta.kv_max_seq_len,
                        std::move(buf.out_kv_seq_lens),
                        /*update_block_tables=*/true);
  } else {
    input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
    update_input_params(input_params,
                        buf,
                        plan.num_val_tokens,
                        std::move(row_result.chunked_q_seq_lens),
                        std::move(row_result.chunked_q_cu_seq_lens),
                        row_result.chunked_kv_max_seq_len,
                        std::move(row_result.chunked_kv_seq_lens));
  }
}

void commit_validate_rows(LlmModelParams& input_params,
                          ValidateRowResult& row_result,
                          const ValidateRowPlan& plan) {
  ModelParamsView input_params_view(input_params);
  commit_validate_rows(input_params_view, row_result, plan);
}

void commit_validate_rows(VlmModelParams& input_params,
                          ValidateRowResult& row_result,
                          const ValidateRowPlan& plan) {
  ModelParamsView input_params_view(input_params);
  commit_validate_rows(input_params_view, row_result, plan);
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
  push_cumsum(vec, len);
#endif
}

void rebuild_q_cu_seq_lens(LlmModelParams& params) {
  ModelParamsView params_view(params);
  rebuild_q_cu_seq_lens(params_view);
}

void rebuild_q_cu_seq_lens(ModelParamsView& params) {
  auto& q_cu_seq_lens = params.attention.host.q_cu_seq_lens;
  q_cu_seq_lens.clear();
  q_cu_seq_lens.reserve(params.meta.num_sequences + 1);
  q_cu_seq_lens.emplace_back(0);
  for (int32_t i = 0; i < params.meta.num_sequences; ++i) {
    q_cu_seq_lens.emplace_back(q_cu_seq_lens.back() + params.get_q_seq_len(i));
  }
}

void append_q_seq_len(std::vector<int32_t>& q_seq_lens,
                      std::vector<int32_t>& q_cu_seq_lens,
                      int32_t len) {
  append_seq_len_by_layout(q_seq_lens, len);
  q_cu_seq_lens.emplace_back(
      (q_cu_seq_lens.empty() ? 0 : q_cu_seq_lens.back()) + len);
}

std::vector<int32_t> build_q_cu_seq_lens(
    const std::vector<int32_t>& q_seq_lens) {
  std::vector<int32_t> q_cu_seq_lens;
  if (q_seq_lens.empty()) {
    return q_cu_seq_lens;
  }
#if defined(USE_NPU)
  q_cu_seq_lens.reserve(q_seq_lens.size());
  int32_t cum_seq_len = 0;
  for (int32_t q_len : q_seq_lens) {
    cum_seq_len += q_len;
    q_cu_seq_lens.emplace_back(cum_seq_len);
  }
#else
  CHECK_EQ(q_seq_lens.front(), 0)
      << "q_seq_lens must be cumulative with leading zero";
  q_cu_seq_lens.assign(q_seq_lens.begin() + 1, q_seq_lens.end());
#endif
  return q_cu_seq_lens;
}

void update_kv_seq_lens_and_max(std::vector<int32_t>& kv_seq_lens_vec,
                                int32_t kv_len,
                                int32_t& kv_max_seq_len) {
  if (kv_len > kv_max_seq_len) {
    kv_max_seq_len = kv_len;
  }
  append_seq_len_by_layout(kv_seq_lens_vec, kv_len);
}

}  // namespace input_builder
}  // namespace xllm
