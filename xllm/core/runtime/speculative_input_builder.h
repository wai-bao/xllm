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

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "core/framework/model/mtp_topk_state.h"
#include "util/slice.h"

// Runtime input construction and metadata ownership for speculative execution.

namespace xllm {

class ModelInputParams;
class LlmForwardInput;
struct SamplingParameters;

namespace specBuilder {

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
  std::vector<std::vector<std::vector<int32_t>>> out_multi_block_tables;
  int32_t out_block_table_rows = 0;
  int32_t out_block_table_stride = 0;
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

// Builds the leading-zero cumulative layout [0, l0, l0+l1, ...].
std::vector<int32_t> build_cu_seq_lens(const std::vector<int32_t>& seq_lens);

// Rebuilds leading-zero query offsets from the backend-neutral length accessor.
template <typename Params>
void rebuild_q_cu_seq_lens(Params& params) {
  std::vector<int32_t> seq_lens;
  seq_lens.reserve(params.meta.num_sequences);
  for (int32_t i = 0; i < params.meta.num_sequences; ++i) {
    seq_lens.push_back(params.get_q_seq_len(i));
  }
  params.attention.host.q_cu_seq_lens = build_cu_seq_lens(seq_lens);
}

// Appends one q length and the matching cumulative q length.
void append_q_seq_len(std::vector<int32_t>& q_seq_lens,
                      std::vector<int32_t>& q_cu_seq_lens,
                      int32_t len);

// Keep the logical accepted count for recurrent checkpoint selection. The
// device tensor is materialized only when replay does not supply it directly.
template <typename Params>
void publish_accepted_prefix_lengths(
    Params& input_params,
    const std::vector<int32_t>& accepted_prefix_lengths,
    const torch::TensorOptions& token_options,
    bool publish_device_tensor);

#if defined(USE_NPU)
// Build the host and device parts of expanded target-verification metadata.
// The host layout may be prepared before the packed replay inputs are bound.
template <typename Params>
bool prepare_expanded_spec_verify_graph_host_input(Params& input_params);
template <typename Params>
void bind_expanded_spec_verify_graph_input(Params& input_params,
                                           const torch::Device& device,
                                           int32_t block_size);
// Prepares the host layout (CHECKing that the required seq lens exist) and
// binds the device graph inputs. Callers decide when expanded metadata is
// needed: MTP also uses it in eager mode, while DFlash uses it in graph mode.
template <typename Params>
void build_expanded_spec_verify_graph_input(Params& input_params,
                                            const torch::Device& device,
                                            int32_t block_size);
#endif

// Appends kv_len into output vector and updates kv_max_seq_len.
void update_kv_seq_lens_and_max(std::vector<int32_t>& kv_seq_lens_vec,
                                int32_t kv_len,
                                int32_t& kv_max_seq_len);

// Updates common decode-side ModelInputParams fields from built buffers.
template <typename Params>
void update_input_params(Params& input_params,
                         DecodeBuildBuffers& buf,
                         int32_t q_max_seq_len,
                         std::vector<int32_t> q_seq_lens_vec,
                         std::vector<int32_t> q_cu_seq_lens_vec,
                         int32_t kv_max_seq_len,
                         std::vector<int32_t> kv_seq_lens_vec,
                         bool update_block_tables = false);

// Empty DP ranks participate in block-draft execution with a complete query
// block. Fill their execution counts on every rank and materialize an idle
// rank's host input without writing any real cache slots.
void materialize_empty_block_draft_rows(LlmForwardInput& input,
                                        int32_t query_width,
                                        int32_t mask_token_id);

// Stages token_ids/positions into both the host and device tensors of `input`
// with async H2D copies, toggling device_tensors_ready around the write.
template <typename Input>
void set_token_position_tensors(Input& input,
                                const std::vector<int32_t>& token_ids,
                                const std::vector<int32_t>& positions,
                                const torch::TensorOptions& token_options,
                                const torch::TensorOptions& position_options);

// Reuses the NPU MTP draft-step row-selection flow for every backend. Backend
// state implementations only interpret index_select_rows() for their payload.
MtpTopkStatePtr select_mtp_topk_state_for_next_step(
    const MtpTopkStatePtr& state,
    const SamplingParameters& sampling_params);

// Shift each prefill query left and append its target-sampled root token into
// caller-owned CPU storage. The runtime supplies pinned storage for H2D.
template <typename Params>
void fill_mtp_prefill_token_ids(const torch::Tensor& token_ids_host,
                                const Params& params,
                                const std::vector<int32_t>& extra_token_ids,
                                torch::Tensor& destination);

struct MtpDraftExtendRows {
  torch::Tensor token_ids;
  torch::Tensor hidden_states;
  torch::Tensor selected_token_idxes;
};

// Pair committed next tokens with their producing target rows. The returned
// layout matches the verifier metadata; only the last accepted row is sampled.
MtpDraftExtendRows build_mtp_draft_extend_rows(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& target_hidden_states,
    const std::vector<int32_t>& per_seq_val_tokens,
    bool step_major_layout);

// Construct a draft input from the verifier's row layout. Target execution,
// graph, transfer and recurrent-state controls never cross this boundary.
// The caller orders this snapshot after verification on the metadata stream.
template <typename TargetInput>
TargetInput build_mtp_draft_extend_input(const TargetInput& validate_input,
                                         const MtpDraftExtendRows& rows,
                                         SamplingParameters sampling_params);

// Copies the refresh metadata independently on the current stream; the caller
// is responsible for synchronizing with updates to the source buffers.
template <typename TargetInput>
void clone_mtp_draft_metadata(TargetInput& input);

}  // namespace specBuilder

}  // namespace xllm
