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

#include "core/framework/speculative/vocab_map.h"

#include <glog/logging.h>

#include <array>
#include <limits>
#include <vector>

#include "core/util/integer_math.h"
#include "core/util/tensor_helper.h"

namespace xllm {
namespace {

constexpr std::array<int32_t, 32> kPackedBitWeights = [] {
  std::array<int32_t, 32> weights{};
  for (int32_t bit = 0; bit < 31; ++bit) {
    weights[bit] = int32_t{1} << bit;
  }
  weights[31] = std::numeric_limits<int32_t>::min();
  return weights;
}();

std::vector<uint8_t> invalid_mask_rows(const torch::Tensor& active_rows) {
  const std::vector<bool> active = tensor_to_vector<bool>(active_rows);
  std::vector<uint8_t> invalid;
  invalid.reserve(active.size());
  uint8_t any_invalid = 0;
  for (bool row_active : active) {
    const uint8_t bad = row_active ? 0 : 1;
    invalid.emplace_back(bad);
    any_invalid |= bad;
  }
  if (any_invalid == 0) {
    return {};
  }
  return invalid;
}

}  // namespace

std::vector<uint8_t> VocabMap::remap_filter_masks(
    SamplingParameters& params) const {
  const torch::Device device = draft_to_target_.device();
  const int64_t target_vocab_size = target_to_draft_.numel();
  if (!params.filter_bitmask.defined()) {
    if (!params.filter_mask.defined()) {
      return {};
    }
    CHECK_EQ(params.filter_mask.dim(), 2);
    CHECK_EQ(params.filter_mask.size(1), target_vocab_size);
    CHECK(params.filter_mask.is_floating_point());
    params.filter_mask =
        params.filter_mask.to(device).index_select(1, draft_to_target_);
    const torch::Tensor active = params.filter_mask.ge(0).any(/*dim=*/1);
    std::vector<uint8_t> invalid = invalid_mask_rows(active);
    if (!invalid.empty()) {
      // index_select owns this buffer, so valid rows and the input stay intact.
      const torch::Tensor invalid_rows = active.logical_not();
      params.filter_mask.masked_fill_(invalid_rows.unsqueeze(/*dim=*/1),
                                      -std::numeric_limits<float>::infinity());
      params.filter_mask.select(/*dim=*/1, /*index=*/0)
          .masked_fill_(invalid_rows, /*value=*/0);
    }
    return invalid;
  }

  CHECK_EQ(params.filter_bitmask.dim(), 2);
  CHECK_EQ(params.filter_bitmask.scalar_type(), torch::kInt);
  CHECK_EQ(params.filter_bitmask.size(1),
           util::ceil_div(target_vocab_size, int64_t{32}));
  const int64_t batch_size = params.filter_bitmask.size(0);
  const int64_t draft_vocab_size = draft_to_target_.numel();
  // Match Sampler's precedence without reading, copying or retaining a dense
  // mask that cannot affect this draft sample.
  params.filter_mask = torch::Tensor();
  const int64_t draft_words = util::ceil_div(draft_vocab_size, int64_t{32});
  torch::Tensor values =
      params.filter_bitmask.to(device).index_select(1, mask_word_indices_);
  // Mask extraction and weighted packing avoid tensor shifts that can fall
  // back to CPU on accelerators. The immutable map owns all indices and
  // weights. index_select owns this buffer, so these updates preserve the
  // caller's mask.
  values.bitwise_and_(mask_bit_weights_).ne_(0);
  const int64_t padding = draft_words * 32 - draft_vocab_size;
  if (padding != 0) {
    values = torch::constant_pad_nd(values, {0, padding}, /*value=*/0);
  }
  // Each bit contributes once, so even the signed high bit packs in int32.
  values = values.view({batch_size, draft_words, 32});
  values.mul_(packed_bit_weights_);
  params.filter_bitmask =
      values.sum(/*dim=*/2, /*keepdim=*/false, /*dtype=*/torch::kInt);
  // Check packed words instead of rescanning every vocabulary entry.
  const torch::Tensor active = params.filter_bitmask.any(/*dim=*/1);
  std::vector<uint8_t> invalid = invalid_mask_rows(active);
  if (!invalid.empty()) {
    // Preserve batch shape; the runtime must reject these proposals or fail.
    // Inactive packed rows are already zero; only their first bit needs
    // setting.
    params.filter_bitmask.select(/*dim=*/1, /*index=*/0)
        .masked_fill_(active.logical_not(), /*value=*/1);
  }
  return invalid;
}

namespace {

void neutralize_missing_rows(torch::Tensor& penalties,
                             const torch::Tensor& inactive_rows,
                             float neutral) {
  if (!penalties.defined()) {
    return;
  }
  CHECK_EQ(penalties.dim(), 1);
  CHECK_EQ(penalties.size(0), inactive_rows.numel());
  penalties = penalties.masked_fill(inactive_rows, neutral);
}

void remap_penalty_history(SamplingParameters& params,
                           const torch::Tensor& target_to_draft,
                           const torch::Device& device,
                           const DraftTokenHistories& input_histories,
                           DraftHistoryBatch& batch) {
  if (!has_penalty_params(params)) {
    return;
  }
  // Copy only once the penalty path is known to run: the projection below
  // replaces each row with its mapped view, so it needs owned rows.
  DraftTokenHistories histories = input_histories;
  CHECK(params.unique_token_ids.defined());
  const int64_t batch_size = params.unique_token_ids.size(0);
  import_sampling_histories(params, batch_size, histories);
  int64_t active_count = 0;
  for (auto& history : histories) {
    history = history->mapped(target_to_draft);
    active_count += static_cast<int64_t>(!history->ids().empty());
  }
  if (active_count == 0) {
    params.frequency_penalties = torch::Tensor();
    params.presence_penalties = torch::Tensor();
    params.repetition_penalties = torch::Tensor();
    params.unique_token_ids = torch::Tensor();
    params.unique_token_counts = torch::Tensor();
    params.unique_token_ids_lens = torch::Tensor();
    return;
  }
  // Empty hot-token histories need neutral coefficients: padding ID 0 must
  // not introduce a repetition or frequency penalty for these rows.
  if (active_count != batch_size) {
    std::vector<bool> inactive_rows;
    inactive_rows.reserve(histories.size());
    for (const auto& history : histories) {
      inactive_rows.push_back(history->ids().empty());
    }
    const torch::Tensor inactive_dev =
        make_cpu_tensor(inactive_rows).to(device);
    neutralize_missing_rows(params.frequency_penalties,
                            inactive_dev,
                            /*neutral=*/0.0F);
    neutralize_missing_rows(params.presence_penalties,
                            inactive_dev,
                            /*neutral=*/0.0F);
    neutralize_missing_rows(params.repetition_penalties,
                            inactive_dev,
                            /*neutral=*/1.0F);
  }
  apply_draft_history_batch(batch, params, histories, device);
}

}  // namespace

VocabMap::VocabMap(torch::Tensor draft_to_target, int64_t target_vocab_size) {
  CHECK(draft_to_target.defined());
  CHECK_EQ(draft_to_target.dim(), 1);
  CHECK(draft_to_target.scalar_type() == torch::kInt ||
        draft_to_target.scalar_type() == torch::kLong);
  CHECK_GT(draft_to_target.numel(), 0);
  CHECK_GT(target_vocab_size, 0);
  CHECK_LE(draft_to_target.numel(), target_vocab_size);
  const torch::Tensor normalized = draft_to_target.to(torch::kLong);
  draft_to_target_ = normalized.is_alias_of(draft_to_target)
                         ? clone_contiguous_detached_tensor(normalized)
                         : normalized.contiguous();
  const torch::Tensor indices =
      to_cpu_contiguous(draft_to_target_, torch::kLong);
  target_to_draft_ = torch::full({target_vocab_size}, -1, torch::kLong);
  torch::Tensor word_indices = torch::empty_like(indices);
  torch::Tensor bit_weights = torch::empty({indices.numel()}, torch::kInt);
  int64_t* inverse_data = target_to_draft_.data_ptr<int64_t>();
  int64_t* word_indices_data = word_indices.data_ptr<int64_t>();
  int32_t* bit_weights_data = bit_weights.data_ptr<int32_t>();
  const int64_t* target_ids = indices.const_data_ptr<int64_t>();
  for (int64_t draft_token = 0; draft_token < indices.numel(); ++draft_token) {
    const int64_t target_token = target_ids[draft_token];
    CHECK_GE(target_token, 0)
        << "VocabMap target id is outside the target vocabulary";
    CHECK_LT(target_token, target_vocab_size)
        << "VocabMap target id is outside the target vocabulary";
    CHECK_EQ(inverse_data[target_token], -1)
        << "VocabMap contains duplicate target token ids";
    inverse_data[target_token] = draft_token;
    word_indices_data[draft_token] = target_token / 32;
    bit_weights_data[draft_token] = kPackedBitWeights[target_token % 32];
  }
  const torch::Device device = draft_to_target_.device();
  mask_word_indices_ = word_indices.to(device);
  mask_bit_weights_ = bit_weights.to(device);
  packed_bit_weights_ =
      torch::tensor(torch::ArrayRef<int32_t>(kPackedBitWeights),
                    torch::TensorOptions().dtype(torch::kInt).device(device));
}

std::vector<uint8_t> VocabMap::remap(
    SamplingParameters& params,
    DraftHistoryBatch& batch,
    const DraftTokenHistories& histories) const {
  if (!draft_to_target_.defined()) {
    return {};
  }
  if (!params.filter_mask.defined() && !params.filter_bitmask.defined() &&
      !has_penalty_params(params)) {
    return {};
  }
  std::vector<uint8_t> invalid = remap_filter_masks(params);
  remap_penalty_history(
      params, target_to_draft_, draft_to_target_.device(), histories, batch);
  return invalid;
}

}  // namespace xllm
