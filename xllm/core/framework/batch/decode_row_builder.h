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

// decode_row_builder.h
#pragma once

#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "core/framework/model/llm_model_params.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/tensor_helper.h"
#include "util/slice.h"

namespace xllm {

// Generic decode row building primitives shared by the scheduler-side batch
// builders, the worker-side speculative builders, and the rec builders. They
// sit below all of them: spec decode is decode with a wider row count, so the
// row layout lives in this leaf module rather than in any one consumer.
namespace input_builder {

struct DecodeBuildMeta {
  int32_t kv_max_seq_len = 0;
};

// Aggregated output vectors produced by row builders.
// Callers convert these vectors to tensors and write back to input params.
struct DecodeBuildBuffers {
  DecodeBuildMeta meta;
  std::vector<int32_t> out_token_ids;
  std::vector<int32_t> out_positions;
  std::vector<int32_t> out_kv_seq_lens;
  std::vector<int32_t> out_q_seq_lens;
  std::vector<int32_t> out_q_cu_seq_lens;
  std::vector<int32_t> out_new_cache_slots;
  std::vector<int32_t> out_block_tables;
  std::vector<int32_t> out_paged_kv_indptr;
  std::vector<int32_t> out_paged_kv_indices;
  std::vector<int32_t> out_paged_kv_last_page_len;
  std::vector<std::vector<std::vector<int32_t>>> out_multi_block_tables;
  int32_t out_block_table_rows = 0;
  int32_t out_block_table_stride = 0;

  void reserve_row_buffers(size_t num_rows, size_t block_table_stride) {
    out_block_tables.reserve(num_rows * block_table_stride);
    out_paged_kv_indices.reserve(num_rows * block_table_stride);
    out_paged_kv_indptr.reserve(num_rows + 1);
    out_paged_kv_last_page_len.reserve(num_rows);
  }
};

// CPU-side row context shared by decode row builders.
// It owns the contiguous block table tensor so row slices stay valid.
struct DecodeRowContext {
  torch::Tensor block_tables_owner;
  std::vector<torch::Tensor> multi_block_tables_owner;
  Slice<int32_t> token_ids;
  Slice<int32_t> positions;
  Slice<int32_t> kv_seq_lens;
  Slice<int32_t> block_tables;
  std::vector<std::vector<Slice<int32_t>>> multi_block_tables;
  int32_t num_sequences = 0;
  int32_t block_table_stride = 0;
  bool model_managed_multiblock = false;
};

// Declarative spec for one emitted decode row.
// A row can selectively append token/kv/q/block fields depending on caller
// needs.
struct RowSpec {
  int32_t seq_id = 0;
  // When true, token_id is ignored and token comes from input
  // token_ids[seq_id].
  bool use_input_token = false;
  int32_t token_id = 0;
  int32_t position_offset = 0;
  bool append_token = true;
  bool append_kv_len = true;
  // When set, decouples the visible KV extent from this row's position. This
  // lets block-parallel draft rows share the full block KV length.
  std::optional<int32_t> kv_len_offset;
  bool append_q_len_one = false;
  bool append_block_table = false;
};

// Resolved token and relative position offset for placeholder token handling.
struct TokenWithOffset {
  int32_t token_id = 0;
  int32_t position_offset = 0;
};

// Builds a reusable row context from LlmForwardInput/VlmForwardInput
// host-side fields.
DecodeRowContext make_decode_row_context(const LlmForwardInput& input);
DecodeRowContext make_decode_row_context(const VlmForwardInput& input);

// Builds the grouped-prefill SWA cache slots for a decode input with
// model-managed multi block tables: one slot per query position, mapped
// through the circular SWA block table.
std::vector<int32_t> build_grouped_prefill_swa_slots(
    const LlmForwardInput& input,
    int32_t block_size);
std::vector<int32_t> build_grouped_prefill_swa_slots(
    const VlmForwardInput& input,
    int32_t block_size);

// Appends one logical decode row into output buffers.
void append_decode_row(const DecodeRowContext& ctx,
                       const RowSpec& row,
                       int32_t block_size,
                       DecodeBuildBuffers& buf);

// Resolves direct/placeholder input token to a real token and position offset.
// Placeholder tokens use negative ids (-1, -2, ...) and are read from last-step
// outputs.
TokenWithOffset resolve_token_with_position_offset(
    int32_t input_token_id,
    int32_t seq_id,
    const Slice<int64_t>& last_step_tokens,
    int32_t last_step_decode_num);

// Resolves a token from last-step output and appends one decode row.
void append_decode_row_from_last_step(const DecodeRowContext& ctx,
                                      int32_t seq_id,
                                      int32_t input_token_id,
                                      const Slice<int64_t>& last_step_tokens,
                                      int32_t last_step_decode_num,
                                      int32_t block_size,
                                      DecodeBuildBuffers& buf);

// Computes one cache slot id from absolute position and block table mapping.
int32_t calc_slot_id(int32_t position,
                     const Slice<int32_t>& block_table_slice,
                     int32_t block_size);

// Circular (ring) variant of calc_slot_id for DSV4 SWA block tables.
int32_t calc_ring_slot_id(int32_t position,
                          const Slice<int32_t>& block_table_slice,
                          int32_t block_size);

// Computes sequence kv length with platform-specific seq-lens layout handling.
int32_t calc_kv_len(const Slice<int32_t>& kv_seq_lens_slice,
                    int32_t seq_id,
                    int32_t offset);

// Appends one q/kv length element using current backend layout policy.
void append_seq_len_by_layout(std::vector<int32_t>& vec, int32_t len);

// Rebuilds leading-zero query offsets from the backend-neutral length accessor.
void rebuild_q_cu_seq_lens(LlmModelParams& params);
void rebuild_q_cu_seq_lens(VlmModelParams& params);
void rebuild_q_cu_seq_lens(ModelParamsView& params);

// Appends one q length and the matching cumulative q length.
void append_q_seq_len(std::vector<int32_t>& q_seq_lens,
                      std::vector<int32_t>& q_cu_seq_lens,
                      int32_t len);

// Converts the backend-flavor q_seq_lens vector into q_cu_seq_lens: NPU
// per-sequence lengths become cumulative-with-leading-zero; other backends
// already store cumulative lengths with a leading zero and pass them through
// unchanged. Shared by the scheduler-side batch builders and the speculative
// validate-row builders.
std::vector<int32_t> build_q_cu_seq_lens(
    const std::vector<int32_t>& q_seq_lens);

// Appends kv_len into output vector and updates kv_max_seq_len.
void update_kv_seq_lens_and_max(std::vector<int32_t>& kv_seq_lens_vec,
                                int32_t kv_len,
                                int32_t& kv_max_seq_len);

// Updates common decode-side ModelParamsView fields from built buffers.
void update_input_params(ModelParamsView& input_params,
                         DecodeBuildBuffers& buf,
                         int32_t q_max_seq_len,
                         std::vector<int32_t> q_seq_lens_vec,
                         std::vector<int32_t> q_cu_seq_lens_vec,
                         int32_t kv_max_seq_len,
                         std::vector<int32_t> kv_seq_lens_vec,
                         bool update_block_tables = false);
template <typename Params>
void update_input_params(Params& input_params,
                         DecodeBuildBuffers& buf,
                         int32_t q_max_seq_len,
                         std::vector<int32_t> q_seq_lens_vec,
                         std::vector<int32_t> q_cu_seq_lens_vec,
                         int32_t kv_max_seq_len,
                         std::vector<int32_t> kv_seq_lens_vec,
                         bool update_block_tables = false);

// Builds the validate row expansion of a decode input: each sequence emits
// num_val_tokens rows (the committed token plus N continuation slots at
// positions +0..+N). chunked_layout selects the chunked-prefill
// flavor (one q_len = num_val_tokens row per sequence) versus the flat
// decode-row flavor. Returns the assembled buffers plus the chunked-side
// length vectors when chunked_layout is set.
struct ValidateRowResult {
  DecodeBuildBuffers buf;
  std::vector<int32_t> chunked_q_seq_lens;
  std::vector<int32_t> chunked_q_cu_seq_lens;
  std::vector<int32_t> chunked_kv_seq_lens;
  int32_t chunked_kv_max_seq_len = 0;
};

struct ValidateRowPlan {
  int32_t num_val_tokens = 1;
  int32_t block_size = 0;
  bool chunked_layout = false;
  // step-major emits rows in val_idx-major order instead of sequence-major.
  bool step_major = false;
};

ValidateRowResult build_validate_rows(const DecodeRowContext& ctx,
                                      const ValidateRowPlan& plan);

// Commits a validate-row result into input params: picks the flat-vs-chunked
// length vectors, sets num_sequences/q_max_seq_len/batch_forward_type from the
// plan, and writes the buffers back via update_input_params. Shared epilogue
// of every build_validate_rows caller.
template <typename Params>
void commit_validate_rows(Params& input_params,
                          ValidateRowResult& row_result,
                          const ValidateRowPlan& plan);
void commit_validate_rows(ModelParamsView& input_params,
                          ValidateRowResult& row_result,
                          const ValidateRowPlan& plan);

// Stages token_ids/positions into both the host and device tensors of
// LlmForwardInput/VlmForwardInput with async H2D copies, toggling
// device_tensors_ready around the write.
template <typename Input>
void set_token_position_tensors(Input& input,
                                const std::vector<int32_t>& token_ids,
                                const std::vector<int32_t>& positions,
                                const torch::TensorOptions& token_options,
                                const torch::TensorOptions& position_options);

}  // namespace input_builder
}  // namespace xllm
