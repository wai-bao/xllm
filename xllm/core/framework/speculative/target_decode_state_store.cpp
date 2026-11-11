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

#include "core/framework/speculative/target_decode_state_store.h"

#include <glog/logging.h>

#include <utility>

#include "core/framework/speculative/padded_token_row_scan.h"
#include "core/util/tensor_helper.h"

namespace xllm {

namespace {

// Prefill/bootstrap rows may arrive without request identity; every other
// path always pairs. Anonymous rows resolve to an empty request id, which
// owned_state never matches (states are keyed by a non-empty request id).
const std::string& row_request_id(const BatchRows& rows, size_t row) {
  static const std::string kEmpty;
  return rows.request_ids.empty() ? kEmpty : rows.request_ids[row];
}

// Pulls a [batch] token row to the host and validates every entry as a
// storable int32 token id.
torch::Tensor host_prefill_tokens(const torch::Tensor& next_tokens,
                                  int64_t batch) {
  const torch::Tensor tokens = to_cpu_contiguous(next_tokens, torch::kInt64);
  const int64_t* data = tokens.const_data_ptr<int64_t>();
  for (int64_t row = 0; row < batch; ++row) {
    CHECK(detail::is_storable_token_id(data[row]))
        << "prefill target token should be valid";
  }
  return tokens;
}

}  // namespace

TargetDecodeStateStore::TargetDecodeStateStore(int32_t total_nums) {
  CHECK_GT(total_nums, 0) << "No target decode states to allocate";
  states_.resize(total_nums);
}

void TargetDecodeStateStore::write_prefill_target_context(
    const BatchRows& rows,
    const torch::Tensor& next_tokens) {
  CHECK(next_tokens.defined()) << "prefill target tokens are undefined";
  CHECK_EQ(next_tokens.dim(), 1) << "prefill target tokens should be [batch]";
  CHECK_EQ(next_tokens.size(0),
           static_cast<int64_t>(rows.embedding_ids.size()));
  const torch::Tensor tokens = host_prefill_tokens(
      next_tokens, static_cast<int64_t>(rows.embedding_ids.size()));
  const int64_t* tokens_data = tokens.const_data_ptr<int64_t>();
  for (int32_t row = 0; row < static_cast<int32_t>(rows.embedding_ids.size());
       ++row) {
    const int64_t token = tokens_data[row];
    reset_state(rows.embedding_ids[row],
                row_request_id(rows, row),
                static_cast<int32_t>(token),
                /*position_offset=*/0);
  }
}

void TargetDecodeStateStore::write_target_bootstrap_rows(
    const std::vector<int32_t>& rows,
    const BatchRows& batch_rows,
    const torch::Tensor& token_ids_host) {
  CHECK(token_ids_host.defined()) << "bootstrap target tokens are undefined";
  CHECK(token_ids_host.device().is_cpu())
      << "bootstrap target tokens must be on CPU";
  CHECK_EQ(token_ids_host.scalar_type(), torch::kInt)
      << "bootstrap target tokens must be int32";
  const Slice<int32_t> token_ids = tensor_slice(token_ids_host);
  for (int32_t row : rows) {
    CHECK_GE(row, 0) << "bootstrap row index should be valid";
    CHECK_LT(static_cast<int64_t>(row), token_ids_host.numel())
        << "bootstrap row index exceeds token ids";
    CHECK_GE(token_ids[static_cast<size_t>(row)], 0);
    reset_state(batch_rows.embedding_ids[row],
                row_request_id(batch_rows, row),
                token_ids[static_cast<size_t>(row)],
                /*position_offset=*/0);
  }
}

void TargetDecodeStateStore::write_target_context(
    const BatchRows& rows,
    const torch::Tensor& accepted_tokens) {
  CHECK(accepted_tokens.defined());
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(accepted_tokens.size(0),
           static_cast<int64_t>(rows.embedding_ids.size()));
  CHECK_GT(accepted_tokens.size(1), 0)
      << "Accepted token rows must not be empty";
  const torch::Tensor tokens = to_cpu_contiguous(accepted_tokens, torch::kLong);
  const int64_t* tokens_data = tokens.const_data_ptr<int64_t>();
  const int32_t width = static_cast<int32_t>(tokens.size(1));
  for (int32_t row = 0; row < static_cast<int32_t>(rows.embedding_ids.size());
       ++row) {
    const int64_t row_offset = static_cast<int64_t>(row) * width;
    const detail::PaddedTokenRowScan scan = detail::scan_padded_token_row(
        tokens_data + row_offset, static_cast<int64_t>(width));
    CHECK(!scan.over_int32_range)
        << "Accepted token row " << row
        << " contains a token id above the int32 range";
    CHECK(scan.tail_all_padding)
        << "Accepted token row " << row
        << " must have only -1 padding after the valid prefix";
    const int32_t accepted_length = static_cast<int32_t>(scan.length);
    CHECK_GT(accepted_length, 0) << "Accepted token row " << row << " is empty";
    const int32_t last_index = accepted_length - 1;
    const int32_t last_token_id =
        static_cast<int32_t>(tokens_data[row_offset + last_index]);
    reset_state(rows.embedding_ids[row],
                row_request_id(rows, row),
                last_token_id,
                last_index);
  }
}

std::vector<TargetDecodeStateStore::DecodeRow>
TargetDecodeStateStore::read_decode_states(const BatchRows& rows) const {
  CHECK(!rows.embedding_ids.empty());
  std::vector<DecodeRow> output;
  output.reserve(rows.embedding_ids.size());
  for (int32_t row = 0; row < static_cast<int32_t>(rows.embedding_ids.size());
       ++row) {
    const DecodeState* owned =
        owned_state(rows.embedding_ids[row], &row_request_id(rows, row));
    DecodeRow decoded;
    if (owned != nullptr && owned->valid) {
      decoded.valid = true;
      decoded.token_id = owned->token_id;
      decoded.position_offset = owned->position_offset;
    }
    output.emplace_back(decoded);
  }
  return output;
}

std::vector<int32_t> TargetDecodeStateStore::resolve_accepted_prefix_lengths(
    const BatchRows& rows,
    int32_t num_sequences) const {
  // The explicit default vector only covers the no-ids case; owned slots
  // default to 1 below.
  if (rows.embedding_ids.empty()) {
    return std::vector<int32_t>(num_sequences, 1);
  }
  std::vector<int32_t> lengths;
  lengths.reserve(rows.embedding_ids.size());
  for (int32_t row = 0; row < static_cast<int32_t>(rows.embedding_ids.size());
       ++row) {
    const DecodeState* owned =
        owned_state(rows.embedding_ids[row], &row_request_id(rows, row));
    lengths.emplace_back(
        owned == nullptr
            ? 1
            : DecodeRow{/*.valid=*/owned->valid,
                        /*.token_id=*/0,
                        /*.position_offset=*/owned->position_offset}
                  .accepted_prefix_length());
  }
  return lengths;
}

void TargetDecodeStateStore::clear(const std::vector<int32_t>& ids) {
  for (int32_t id : ids) {
    detail::checked_slot(states_, id) = DecodeState();
  }
}

// Ownership here is the slot's `valid` flag plus the request id: a recycled
// slot is invalidated by reset_state before it can be handed to a new request.
const TargetDecodeStateStore::DecodeState* TargetDecodeStateStore::owned_state(
    int32_t embedding_id,
    const std::string* request_id) const {
  const DecodeState& state = detail::checked_slot(states_, embedding_id);
  if (!state.valid ||
      (request_id != nullptr && state.request_id != *request_id)) {
    return nullptr;
  }
  return &state;
}

void TargetDecodeStateStore::reset_state(int32_t embedding_id,
                                         const std::string& request_id,
                                         int32_t token_id,
                                         int32_t position_offset) {
  DecodeState& state = detail::checked_slot(states_, embedding_id);
  // No `state = DecodeState()` first: DecodeState has exactly these four
  // fields and all four are assigned below, so the reset only served to free
  // the slot's request-id buffer that the assign right after re-allocates.
  state.valid = true;
  state.request_id = request_id;
  state.token_id = token_id;
  state.position_offset = position_offset;
}

}  // namespace xllm
