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

// forward_input_builder.h
#pragma once

#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_input_data.h"
#include "core/framework/batch/sampling_input_builder.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/request/sequence.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/rec_forward_params.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/threadpool.h"

namespace xllm {

class ForwardInputBuilderTestPeer;
struct ModelArgs;

class ForwardInputBuilder final {
 public:
  ForwardInputBuilder(const BatchInputData& data,
                      const ModelArgs* args,
                      int32_t cp_size = 1,
                      ThreadPool* thread_pool = nullptr);

  explicit ForwardInputBuilder(
      const std::vector<Sequence*>& sequences,
      const std::vector<uint32_t>& allowed_max_tokens,
      const std::vector<torch::Tensor>& input_embeddings_vec,
      const std::vector<MMData>& mm_data_vec,
      // for beam-search
      std::vector<BlockTransferInfo>* swap_block_transfer_infos,
      const uint64_t batch_id,
      const ModelArgs* args,
      BatchForwardType batch_forward_type,
      int32_t cp_size = 1,
      ThreadPool* thread_pool = nullptr);

  LlmForwardInput build_forward_input(uint32_t num_decoding_tokens,
                                      uint32_t min_decoding_batch_size);
  RecForwardInput build_rec_forward_input(uint32_t num_decoding_tokens,
                                          uint32_t min_decoding_batch_size);
  VlmForwardInput build_vlm_forward_input(uint32_t num_decoding_tokens,
                                          uint32_t min_decoding_batch_size);

  std::vector<Block> take_linear_restore_src_blocks() {
    return std::move(state_.linear_restore_src_blocks);
  }

 private:
  friend class ForwardInputBuilderTestPeer;

  // Core building methods
  void process_sequences();
  void process_sequences_multithreaded();
  template <typename Input>
  Input state_to_forward_input();
  void padding_decode_batch_size(uint32_t num_decoding_tokens,
                                 uint32_t min_decoding_batch_size);

  static TransferKVInfo build_step_transfer_info(
      const TransferKVInfo& full_info,
      Sequence* sequence,
      uint32_t seq_len,
      uint32_t kv_split_size);

  template <typename Input>
  void process_swap_block_infos(Input& forward_input);

  // State management
  struct BuilderState {
    // Token and position data
    std::vector<int32_t> flatten_tokens_vec;
    std::vector<int32_t> flatten_positions_vec;
    std::vector<torch::Tensor> mrope_positions_vec;

    SamplingInputBuilder sampling;

    // Sequence-specific grammar state aligned with sampling rows.
    std::vector<JsonObjectGrammarState> json_object_states;
    std::vector<std::string> sample_sequence_ids;
    std::vector<int32_t> sample_prior_output_rows;

    // Sequence metadata
    BatchForwardType batch_forward_type;
    uint32_t max_seq_len = 0;
    uint32_t q_max_seq_len = 0;
    // Tracking token counts in KV cache for linear-attention state setup.
    std::vector<int32_t> kv_cache_tokens_nums;
#if defined(USE_NPU)
    std::vector<int32_t> seq_lens;
    std::vector<int32_t> q_seq_lens;
#elif defined(USE_MLU) || defined(USE_CUDA) || defined(USE_ILU) || \
    defined(USE_DCU) || defined(USE_MUSA)
    std::vector<int32_t> seq_lens = {0};    // cu_seq_lens
    std::vector<int32_t> q_seq_lens = {0};  // q_cu_seq_len
#endif

    // Cache and block data
    std::vector<int32_t> new_token_slot_ids;
    std::vector<std::vector<int32_t>> block_tables_vec;
    // Grouped cache block tables.
    // [manager_num][batch_size][block_ids]
    std::vector<std::vector<std::vector<int32_t>>> multi_block_tables;

    // beam search kernel input
    std::vector<float> acc_logprob_vec;

    // Additional data
    std::vector<int32_t> embedding_ids;
    std::vector<int32_t> linear_state_ids;
    std::vector<LinearStateCacheOp> linear_state_cache_ops;
    std::vector<Block> linear_restore_src_blocks;
    std::vector<std::string> request_ids;
    std::vector<int32_t> extra_token_ids;
    std::vector<int32_t> mtp_shifted_token_ids;
    std::vector<int32_t> eplb_decode_token_mask;
    std::vector<int32_t> mtp_bootstrap_row_idxes;
    std::vector<torch::Tensor> mtp_bootstrap_embeddings;
    std::vector<TransferKVInfo> transfer_kv_infos;
    std::vector<MMData> scheduled_mm_data_vec;

    // for continuous kvcache
    std::vector<int64_t> new_cache_slot_offsets;  //[n_tokens]
    std::vector<int64_t> kv_cache_start_offsets;  //[n_seq]

    // for flashinfer
    std::vector<int32_t> paged_kv_indptr = {0};
    std::vector<int32_t> paged_kv_indices;
    std::vector<int32_t> paged_kv_last_page_len;
  };

  // Helper methods for sequence processing
  void process_single_sequence(int32_t seq_index,
                               BuilderState* state_ptr = nullptr);
  void process_multi_modal_inputs(Sequence* sequence,
                                  uint32_t n_kv_cache_tokens,
                                  uint32_t q_seq_len,
                                  int32_t seq_index,
                                  BuilderState* state_ptr = nullptr);
  void extract_tokens_and_positions(Sequence* sequence,
                                    uint32_t n_kv_cache_tokens,
                                    uint32_t seq_len,
                                    BuilderState* state_ptr = nullptr);
  // Append this batch row's linear-state transport fields: the live slot id
  // (always, so rows stay aligned) plus a LinearStateCacheOp carrying the
  // resolved restore/save plan for linear-attention models.
  void append_linear_state_row(Sequence* sequence,
                               uint32_t n_kv_cache_tokens,
                               uint32_t seq_len,
                               BuilderState& state);
  torch::Tensor get_mrope_positions(Sequence* sequence,
                                    uint32_t start,
                                    uint32_t end);

  void handle_sampling_parameters(Sequence* sequence,
                                  BuilderState* state_ptr = nullptr);
  void setup_kv_cache_info(Sequence* sequence,
                           uint32_t n_kv_cache_tokens,
                           uint32_t seq_len,
                           uint32_t q_seq_len,
                           BuilderState* state_ptr = nullptr);

  // Input data
  const std::vector<Sequence*>& sequences_;
  const std::vector<uint32_t>& allowed_max_tokens_;
  const std::vector<torch::Tensor>& input_embeddings_vec_;
  std::vector<MMData> mm_data_vec_;
  const ModelArgs* args_;

  // Builder state
  BuilderState state_;

  // Configuration
  bool enable_json_object_output_ = true;
  bool use_mrope_ = false;
  uint32_t num_sequences_ = 0;
  bool need_unique_tokens_ = true;
  bool build_eplb_decode_token_mask_ = false;
  bool is_graph_warmup_ = false;
  int32_t cp_size_ = 1;

  // copy in and out cache contents
  std::vector<BlockTransferInfo>* swap_block_transfer_infos_ = nullptr;

  // thread pool for multithreaded processing, not owned
  ThreadPool* thread_pool_ = nullptr;
  uint64_t batch_id_;
};

// Generic decode row building primitives shared by the scheduler-side batch
// builders and the worker-side speculative builders. Batch owns them: spec
// decode is decode with a wider row count, so the row layout lives here.
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

// Packs a host int32 vector into a pinned CPU tensor for async H2D staging.
torch::Tensor make_cpu_int_tensor(const std::vector<int32_t>& values);

// Builds a reusable row context from LlmForwardInput host-side fields.
template <typename Input>
DecodeRowContext make_decode_row_context(const Input& input);

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

// Manager 0 in a grouped-cache input is the DSV4 SWA manager.
template <typename Input>
std::vector<int32_t> build_grouped_prefill_swa_slots(const Input& input,
                                                     int32_t block_size);

// Computes sequence kv length with platform-specific seq-lens layout handling.
int32_t calc_kv_len(const Slice<int32_t>& kv_seq_lens_slice,
                    int32_t seq_id,
                    int32_t offset);

// Appends one q/kv length element using current backend layout policy.
void append_seq_len_by_layout(std::vector<int32_t>& vec, int32_t len);

// Rebuilds leading-zero query offsets from the backend-neutral length accessor.
template <typename Params>
void rebuild_q_cu_seq_lens(Params& params) {
  auto& q_cu_seq_lens = params.attention.host.q_cu_seq_lens;
  q_cu_seq_lens.clear();
  q_cu_seq_lens.reserve(params.meta.num_sequences + 1);
  q_cu_seq_lens.emplace_back(0);
  for (int32_t i = 0; i < params.meta.num_sequences; ++i) {
    q_cu_seq_lens.emplace_back(q_cu_seq_lens.back() + params.get_q_seq_len(i));
  }
}

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
template <typename Params>
void update_input_params(Params& input_params,
                         DecodeBuildBuffers& buf,
                         int32_t q_max_seq_len,
                         std::vector<int32_t> q_seq_lens_vec,
                         std::vector<int32_t> q_cu_seq_lens_vec,
                         int32_t kv_max_seq_len,
                         std::vector<int32_t> kv_seq_lens_vec,
                         bool update_block_tables = false);

// Builds the validate (spec-verify) row expansion of a decode input: each
// sequence emits num_val_tokens rows (the committed token plus N speculative
// slots at positions +0..+N). chunked_layout selects the ATB chunked-prefill
// flavor (one q_len = num_val_tokens row per sequence) versus the flat
// decode-row flavor. Returns the assembled buffers plus the ATB-side length
// vectors when chunked_layout is set.
struct ValidateRowResult {
  DecodeBuildBuffers buf;
  std::vector<int32_t> atb_q_seq_lens;
  std::vector<int32_t> atb_q_cu_seq_lens;
  std::vector<int32_t> atb_kv_seq_lens;
  int32_t atb_kv_max_seq_len = 0;
  int32_t total_num_val_tokens = 0;
};

struct ValidateRowPlan {
  int32_t num_val_tokens = 1;
  int32_t block_size = 0;
  bool chunked_layout = false;
  // step-major emits rows in val_idx-major order (embedded Eagle3 layout).
  bool step_major = false;
};

// Stages token_ids/positions into both the host and device tensors of `input`
// with async H2D copies, toggling device_tensors_ready around the write.
template <typename Input>
void set_token_position_tensors(Input& input,
                                const std::vector<int32_t>& token_ids,
                                const std::vector<int32_t>& positions,
                                const torch::TensorOptions& token_options,
                                const torch::TensorOptions& position_options);

ValidateRowResult build_validate_rows(const DecodeRowContext& ctx,
                                      const Slice<int32_t>& token_ids,
                                      const Slice<int32_t>& kv_seq_lens,
                                      const ValidateRowPlan& plan);

}  // namespace input_builder
}  // namespace xllm
