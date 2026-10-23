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

#include <limits>
#include <utility>

#include "core/framework/speculative/padded_token_row_scan.h"
#include "core/framework/speculative/slot_table.h"
#include "core/util/tensor_helper.h"

namespace xllm {

TargetDecodeStateStore::TargetDecodeStateStore(int32_t total_nums) {
  CHECK_GT(total_nums, 0) << "No target decode states to allocate";
  states_.resize(total_nums);
}

void TargetDecodeStateStore::write_prefill_target_context(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& next_tokens) {
  CHECK(next_tokens.defined()) << "prefill target tokens are undefined";
  CHECK_EQ(next_tokens.dim(), 1) << "prefill target tokens should be [batch]";
  CHECK_EQ(next_tokens.size(0), static_cast<int64_t>(ids.size()));
  CHECK(request_ids.empty() || request_ids.size() == ids.size());
  const torch::Tensor tokens = to_cpu_contiguous(next_tokens, torch::kLong);
  const int64_t* tokens_data = tokens.const_data_ptr<int64_t>();
  for (int32_t row = 0; row < static_cast<int32_t>(ids.size()); ++row) {
    const int64_t token = tokens_data[row];
    CHECK_GE(token, 0);
    CHECK_LE(token, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
    reset_state(ids[row],
                request_ids.empty() ? std::string() : request_ids[row],
                static_cast<int32_t>(token),
                /*position_offset=*/0);
  }
}

void TargetDecodeStateStore::write_target_bootstrap_rows(
    const std::vector<int32_t>& rows,
    const std::vector<int32_t>& embedding_ids,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& token_ids_host) {
  CHECK(token_ids_host.defined()) << "bootstrap target tokens are undefined";
  CHECK(token_ids_host.device().is_cpu())
      << "bootstrap target tokens must be on CPU";
  CHECK_EQ(token_ids_host.scalar_type(), torch::kInt)
      << "bootstrap target tokens must be int32";
  const Slice<int32_t> token_ids = tensor_slice(token_ids_host);
  for (int32_t row : rows) {
    CHECK_GE(row, 0) << "bootstrap row index should be valid";
    CHECK_LT(row, static_cast<int32_t>(embedding_ids.size()))
        << "bootstrap row index exceeds embedding ids";
    CHECK_LT(row, static_cast<int32_t>(request_ids.size()))
        << "bootstrap row index exceeds request ids";
    CHECK_LT(static_cast<int64_t>(row), token_ids_host.numel())
        << "bootstrap row index exceeds token ids";
    CHECK_GE(token_ids[static_cast<size_t>(row)], 0);
    reset_state(embedding_ids[row],
                request_ids[row],
                token_ids[static_cast<size_t>(row)],
                /*position_offset=*/0);
  }
}

void TargetDecodeStateStore::write_target_context(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& accepted_tokens) {
  CHECK(accepted_tokens.defined());
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(accepted_tokens.size(0), static_cast<int64_t>(ids.size()));
  CHECK_GT(accepted_tokens.size(1), 0)
      << "Accepted token rows must not be empty";
  CHECK(request_ids.empty() || request_ids.size() == ids.size());
  const torch::Tensor tokens = to_cpu_contiguous(accepted_tokens, torch::kLong);
  const int64_t* tokens_data = tokens.const_data_ptr<int64_t>();
  const int32_t width = static_cast<int32_t>(tokens.size(1));
  for (int32_t row = 0; row < static_cast<int32_t>(ids.size()); ++row) {
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
    reset_state(ids[row],
                request_ids.empty() ? std::string() : request_ids[row],
                last_token_id,
                last_index);
  }
}

std::vector<TargetDecodeStateStore::DecodeState>
TargetDecodeStateStore::read_decode_states(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  CHECK(!ids.empty());
  CHECK(request_ids.empty() || request_ids.size() == ids.size());
  std::vector<DecodeState> output;
  output.reserve(ids.size());
  for (int32_t row = 0; row < static_cast<int32_t>(ids.size()); ++row) {
    const DecodeState* owned = owned_state(
        ids[row], request_ids.empty() ? nullptr : &request_ids[row]);
    DecodeState state = owned != nullptr ? *owned : DecodeState();
    if (!state.valid) {
      state.token_id = 0;
      state.position_offset = 0;
    } else {
      CHECK_GE(state.token_id, 0);
    }
    output.emplace_back(std::move(state));
  }
  return output;
}

std::vector<int32_t> TargetDecodeStateStore::resolve_accepted_prefix_lengths(
    const std::vector<int32_t>& embedding_ids,
    const std::vector<std::string>& request_ids,
    int32_t num_sequences) const {
  // The explicit default vector only covers the no-ids case; owned slots
  // default to 1 below.
  if (embedding_ids.empty()) {
    return std::vector<int32_t>(num_sequences, 1);
  }
  CHECK(request_ids.empty() || request_ids.size() == embedding_ids.size());
  std::vector<int32_t> lengths;
  lengths.reserve(embedding_ids.size());
  for (int32_t row = 0; row < static_cast<int32_t>(embedding_ids.size());
       ++row) {
    const DecodeState* owned = owned_state(
        embedding_ids[row], request_ids.empty() ? nullptr : &request_ids[row]);
    lengths.emplace_back(owned == nullptr ? 1 : accepted_prefix_length(*owned));
  }
  return lengths;
}

int32_t TargetDecodeStateStore::accepted_prefix_length(
    const DecodeState& state) {
  return state.valid ? state.position_offset + 1 : 1;
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
                                         std::string request_id,
                                         int32_t token_id,
                                         int32_t position_offset) {
  DecodeState& state = detail::checked_slot(states_, embedding_id);
  // No `state = DecodeState()` first: DecodeState has exactly these four
  // fields and all four are assigned below, so the reset only served to free
  // the slot's request-id buffer that the move right after re-allocates.
  state.valid = true;
  state.request_id = std::move(request_id);
  state.token_id = token_id;
  state.position_offset = position_offset;
}

}  // namespace xllm
