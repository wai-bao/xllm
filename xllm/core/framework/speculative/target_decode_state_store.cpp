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
  CHECK_EQ(request_ids.size(), ids.size());
  const torch::Tensor tokens = to_cpu_int64_contiguous(next_tokens);
  const int64_t* tokens_data = tokens.const_data_ptr<int64_t>();
  for (int32_t row = 0; row < static_cast<int32_t>(ids.size()); ++row) {
    const int64_t token = tokens_data[row];
    CHECK_GE(token, 0);
    CHECK_LE(token, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
    reset_state(ids[row],
                request_ids[row],
                static_cast<int32_t>(token),
                /*position_offset=*/0);
  }
}

void TargetDecodeStateStore::write_target_bootstrap_context(
    int32_t embedding_id,
    const std::string& request_id,
    int32_t token_id) {
  CHECK_GE(token_id, 0);
  reset_state(embedding_id, request_id, token_id, /*position_offset=*/0);
}

void TargetDecodeStateStore::write_target_context(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& accepted_tokens,
    std::vector<int32_t>* accepted_prefix_lengths) {
  CHECK(accepted_tokens.defined());
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(accepted_tokens.size(0), static_cast<int64_t>(ids.size()));
  CHECK_GT(accepted_tokens.size(1), 0)
      << "Accepted token rows must not be empty";
  CHECK(accepted_prefix_lengths == nullptr || accepted_prefix_lengths->empty());
  CHECK_EQ(request_ids.size(), ids.size());
  const torch::Tensor tokens = to_cpu_int64_contiguous(accepted_tokens);
  const int64_t* tokens_data = tokens.const_data_ptr<int64_t>();
  const int32_t width = static_cast<int32_t>(tokens.size(1));
  if (accepted_prefix_lengths != nullptr) {
    accepted_prefix_lengths->reserve(ids.size());
  }
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
    if (accepted_prefix_lengths != nullptr) {
      accepted_prefix_lengths->emplace_back(accepted_length);
    }
    reset_state(ids[row], request_ids[row], last_token_id, last_index);
  }
}

std::vector<TargetDecodeStateStore::DecodeState>
TargetDecodeStateStore::read_decode_states(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  CHECK(!ids.empty());
  CHECK_EQ(request_ids.size(), ids.size());
  std::vector<DecodeState> output;
  output.reserve(ids.size());
  for (int32_t row = 0; row < static_cast<int32_t>(ids.size()); ++row) {
    const DecodeState* owned = owned_state(ids[row], request_ids[row]);
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

std::vector<int32_t> TargetDecodeStateStore::read_accepted_prefix_lengths(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  CHECK(!ids.empty());
  CHECK_EQ(request_ids.size(), ids.size());
  std::vector<int32_t> lengths;
  lengths.reserve(ids.size());
  for (int32_t row = 0; row < static_cast<int32_t>(ids.size()); ++row) {
    const DecodeState* owned = owned_state(ids[row], request_ids[row]);
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
    mutable_state(id) = DecodeState();
  }
}

TargetDecodeStateStore::DecodeState& TargetDecodeStateStore::mutable_state(
    int32_t embedding_id) {
  return const_cast<DecodeState&>(get_state(embedding_id));
}

const TargetDecodeStateStore::DecodeState& TargetDecodeStateStore::get_state(
    int32_t embedding_id) const {
  CHECK_GE(embedding_id, 0);
  CHECK_LT(static_cast<size_t>(embedding_id), states_.size());
  return states_[embedding_id];
}

const TargetDecodeStateStore::DecodeState* TargetDecodeStateStore::owned_state(
    int32_t embedding_id,
    const std::string& request_id) const {
  const DecodeState& state = get_state(embedding_id);
  if (!state.valid || state.request_id != request_id) {
    return nullptr;
  }
  return &state;
}

void TargetDecodeStateStore::reset_state(int32_t embedding_id,
                                         std::string request_id,
                                         int32_t token_id,
                                         int32_t position_offset) {
  DecodeState& state = mutable_state(embedding_id);
  state = DecodeState();
  state.valid = true;
  state.request_id = std::move(request_id);
  state.token_id = token_id;
  state.position_offset = position_offset;
}

}  // namespace xllm
