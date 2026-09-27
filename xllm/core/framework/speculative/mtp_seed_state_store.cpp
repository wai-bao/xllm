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

#include "core/framework/speculative/mtp_seed_state_store.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/speculative/mtp_seed_sampling.h"

namespace xllm {

MtpSeedStateStore::MtpSeedStateStore(int32_t total_nums) {
  CHECK_GT(total_nums, 0) << "No embeddings to allocate";
  states_.resize(total_nums);
}

std::shared_ptr<MtpSeedStateStore::DraftSeedBatch>
MtpSeedStateStore::make_seed_batch(const std::vector<int32_t>& ids,
                                   MtpDraftSeed seed) {
  check_mtp_draft_seed(seed);
  CHECK_EQ(seed.token_ids.size(0), static_cast<int64_t>(ids.size()));
  if (slot_generations_.empty()) {
    slot_generations_.resize(states_.size(), 0);
  }
  ++batch_generation_;
  if (batch_generation_ == 0) {
    std::fill(slot_generations_.begin(), slot_generations_.end(), 0);
    batch_generation_ = 1;
  }
  for (int32_t id : ids) {
    (void)get_draft_state(id);
    CHECK_NE(slot_generations_[id], batch_generation_)
        << "duplicate draft seed slot";
    slot_generations_[id] = batch_generation_;
  }
  // A moved Tensor handle may still alias the caller's storage or a graph
  // output buffer. The store must own a detached snapshot across rounds.
  return std::make_shared<DraftSeedBatch>(DraftSeedBatch{
      clone_mtp_draft_seed(seed), ids, static_cast<int64_t>(ids.size())});
}

std::shared_ptr<MtpSeedStateStore::DraftSeedBatch>
MtpSeedStateStore::detach_seed_batch(DraftSeedState& state) {
  auto batch = std::move(state.batch);
  if (batch != nullptr) {
    CHECK_GT(batch->live_rows, 0);
    --batch->live_rows;
  }
  return batch;
}

void MtpSeedStateStore::compact_seed_batches(
    const std::vector<std::shared_ptr<DraftSeedBatch>>& batches) {
  for (const auto& batch : batches) {
    if (batch == nullptr || batch->live_rows == 0 ||
        batch->live_rows * 2 >=
            static_cast<int64_t>(batch->embedding_ids.size())) {
      continue;
    }
    std::vector<int32_t> survivors;
    survivors.reserve(batch->live_rows);
    for (int32_t id : batch->embedding_ids) {
      if (states_[id].batch == batch) {
        survivors.emplace_back(id);
      }
    }
    CHECK_EQ(survivors.size(), batch->live_rows);
    const std::vector<MtpDraftSeed> runs = collect_draft_seed_runs(survivors);
    // Even one contiguous run must leave the old allocation behind.
    batch->seed = runs.size() == 1 ? clone_mtp_draft_seed(runs.front())
                                   : cat_mtp_draft_seeds(runs);
    batch->embedding_ids = std::move(survivors);
    batch->live_rows = static_cast<int64_t>(batch->embedding_ids.size());
    for (size_t row = 0; row < batch->embedding_ids.size(); ++row) {
      states_[batch->embedding_ids[row]].batch_row = static_cast<int64_t>(row);
    }
    // Repeated handles now see full occupancy and skip the same batch without
    // a second membership scan or an auxiliary deduplication allocation.
  }
}

std::optional<MtpDraftSeed> MtpSeedStateStore::read_draft_seed(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  return collect_draft_seeds(ids, request_ids);
}

std::vector<MtpDraftSeed> MtpSeedStateStore::collect_draft_seed_runs(
    const std::vector<int32_t>& ids) const {
  std::vector<MtpDraftSeed> runs;
  runs.reserve(ids.size());
  size_t begin = 0;
  while (begin < ids.size()) {
    const DraftSeedState& first = get_draft_state(ids[begin]);
    CHECK(first.batch != nullptr);
    size_t end = begin + 1;
    while (end < ids.size()) {
      const DraftSeedState& next = get_draft_state(ids[end]);
      if (next.batch != first.batch ||
          next.batch_row !=
              first.batch_row + static_cast<int64_t>(end - begin)) {
        break;
      }
      ++end;
    }
    runs.emplace_back(
        slice_mtp_draft_seed(first.batch->seed, first.batch_row, end - begin));
    begin = end;
  }
  return runs;
}

std::optional<MtpDraftSeed> MtpSeedStateStore::collect_draft_seeds(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  CHECK_EQ(ids.size(), request_ids.size());
  if (ids.empty()) {
    return std::nullopt;
  }
  std::vector<const DraftSeedState*> owned;
  owned.reserve(ids.size());
  for (size_t row = 0; row < ids.size(); ++row) {
    owned.emplace_back(owned_draft_state(ids[row], request_ids[row]));
  }
  return collect_draft_seeds(ids, owned, /*include_failed=*/false);
}

std::optional<MtpDraftSeed> MtpSeedStateStore::collect_draft_seeds(
    const std::vector<int32_t>& ids,
    const std::vector<const DraftSeedState*>& owned,
    bool include_failed) const {
  CHECK_EQ(ids.size(), owned.size());
  if (ids.empty()) {
    return std::nullopt;
  }
  for (size_t row = 0; row < ids.size(); ++row) {
    const DraftSeedState* state = owned[row];
    if (state == nullptr || state->batch == nullptr ||
        (state->failed && !include_failed)) {
      return std::nullopt;
    }
  }
  const DraftSeedState& first = *owned.front();
  bool contiguous = true;
  for (size_t row = 0; row < ids.size(); ++row) {
    const DraftSeedState* state = owned[row];
    contiguous =
        contiguous && state->batch == first.batch &&
        state->batch_row == first.batch_row + static_cast<int64_t>(row);
  }
  if (contiguous) {
    return slice_mtp_draft_seed(first.batch->seed, first.batch_row, ids.size());
  }
  return cat_mtp_draft_seeds(collect_draft_seed_runs(ids));
}

void MtpSeedStateStore::publish_draft_seed(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    MtpDraftSeed seed,
    MtpTokenHistories histories,
    std::vector<JsonObjectGrammarState> json_states,
    const std::vector<size_t>& failed_rows,
    bool carry_failed) {
  check_mtp_draft_seed(seed);
  CHECK_EQ(ids.size(), request_ids.size());
  CHECK_EQ(seed.token_ids.size(0), ids.size());
  CHECK(json_states.empty() || json_states.size() == ids.size());
  CHECK(histories.empty() || histories.size() == ids.size());
  std::vector<uint8_t> failed;
  if (!failed_rows.empty()) {
    failed.resize(ids.size(), 0);
    for (size_t row : failed_rows) {
      CHECK_LT(row, ids.size());
      failed[row] = 1;
    }
  }
  auto batch = make_seed_batch(ids, std::move(seed));
  std::vector<std::shared_ptr<DraftSeedBatch>> replaced;
  replaced.reserve(ids.size());
  for (size_t row = 0; row < ids.size(); ++row) {
    const DraftSeedState& previous = get_draft_state(ids[row]);
    replaced.emplace_back(detach_seed_batch(states_[ids[row]]));
    DraftSeedState state;
    state.request_id = request_ids[row];
    state.failed = (!failed.empty() && failed[row] != 0) ||
                   (carry_failed && previous.request_id == state.request_id &&
                    previous.failed);
    state.batch = batch;
    state.batch_row = static_cast<int64_t>(row);
    if (!state.failed) {
      if (!histories.empty()) {
        state.history = std::move(histories[row]);
      }
      if (!json_states.empty()) {
        state.json_state = std::move(json_states[row]);
      }
    }
    states_[ids[row]] = std::move(state);
  }
  compact_seed_batches(replaced);
}

std::optional<MtpDraftSeed> MtpSeedStateStore::take_decode_seed(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    std::vector<uint8_t>* failed_mask) {
  CHECK_EQ(ids.size(), request_ids.size());
  if (failed_mask != nullptr) {
    CHECK(failed_mask->empty());
  }
  std::vector<const DraftSeedState*> owned;
  owned.reserve(ids.size());
  // Resolve failures in the same ownership pass. Keep batches attached until
  // after the seed has been materialized; collection reads their row slices.
  for (size_t row = 0; row < ids.size(); ++row) {
    const DraftSeedState* state = owned_draft_state(ids[row], request_ids[row]);
    owned.emplace_back(state);
    if (state != nullptr && state->failed && failed_mask != nullptr) {
      if (failed_mask->empty()) {
        failed_mask->resize(ids.size(), 0);
      }
      (*failed_mask)[row] = 1;
    }
  }
  std::optional<MtpDraftSeed> seed =
      collect_draft_seeds(ids, owned, /*include_failed=*/true);
  if (!seed.has_value()) {
    return std::nullopt;
  }
  std::vector<std::shared_ptr<DraftSeedBatch>> consumed;
  consumed.reserve(ids.size());
  for (size_t row = 0; row < ids.size(); ++row) {
    if (!owned[row]->failed) {
      consumed.emplace_back(detach_seed_batch(states_[ids[row]]));
    }
  }
  compact_seed_batches(consumed);
  return seed;
}

std::vector<uint8_t> MtpSeedStateStore::failed_seed_mask(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  CHECK_EQ(ids.size(), request_ids.size());
  std::vector<uint8_t> mask;
  for (size_t row = 0; row < ids.size(); ++row) {
    const DraftSeedState* state = owned_draft_state(ids[row], request_ids[row]);
    if (state != nullptr && state->failed) {
      if (mask.empty()) {
        mask.resize(ids.size(), 0);
      }
      mask[row] = 1;
    }
  }
  return normalize_row_failure_mask(std::move(mask));
}

void MtpSeedStateStore::publish_topk_state(std::vector<int32_t> embedding_ids,
                                           std::vector<std::string> request_ids,
                                           MtpTopkStatePtr state) {
  CHECK_EQ(embedding_ids.size(), request_ids.size());
  topk_cache_.publish(
      std::move(embedding_ids), std::move(request_ids), std::move(state));
}

MtpTopkStatePtr MtpSeedStateStore::read_topk_state(
    const std::vector<int32_t>& embedding_ids,
    const std::vector<std::string>& request_ids) const {
  CHECK_EQ(embedding_ids.size(), request_ids.size());
  return topk_cache_.lookup(embedding_ids, request_ids);
}

MtpTokenHistories MtpSeedStateStore::read_histories(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  CHECK_EQ(ids.size(), request_ids.size());
  MtpTokenHistories histories;
  histories.reserve(ids.size());
  for (size_t row = 0; row < ids.size(); ++row) {
    const DraftSeedState* state = owned_draft_state(ids[row], request_ids[row]);
    histories.emplace_back(state != nullptr ? state->history : nullptr);
  }
  return histories;
}

MtpSeedStateStore::SamplingContext MtpSeedStateStore::read_sampling_context(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    std::vector<JsonObjectGrammarState>& json_states,
    bool needs_history) const {
  CHECK_EQ(ids.size(), request_ids.size());
  CHECK(json_states.empty() || json_states.size() == ids.size());
  SamplingContext context;
  if (needs_history) {
    context.histories.resize(ids.size());
  }
  for (size_t row = 0; row < ids.size(); ++row) {
    const DraftSeedState* state = owned_draft_state(ids[row], request_ids[row]);
    if (state == nullptr) {
      continue;
    }
    if (needs_history) {
      context.histories[row] = state->history;
    }
    // A default-constructed grammar state is not initialized, so the stored
    // state itself answers whether a sampling context was published.
    if (state->json_state.initialized() && !json_states.empty()) {
      json_states[row] = state->json_state;
      context.restored_grammar = true;
    }
  }
  return context;
}

void MtpSeedStateStore::write_histories(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    const MtpTokenHistories& histories) {
  CHECK_EQ(ids.size(), request_ids.size());
  CHECK_EQ(ids.size(), histories.size());
  for (size_t row = 0; row < ids.size(); ++row) {
    (void)get_draft_state(ids[row]);
    DraftSeedState& state = states_[ids[row]];
    if (state.request_id == request_ids[row]) {
      state.history = histories[row];
    }
  }
}

const MtpSeedStateStore::DraftSeedState& MtpSeedStateStore::get_draft_state(
    int32_t id) const {
  CHECK_GE(id, 0);
  CHECK_LT(static_cast<size_t>(id), states_.size());
  return states_[id];
}

const MtpSeedStateStore::DraftSeedState* MtpSeedStateStore::owned_draft_state(
    int32_t id,
    const std::string& request_id) const {
  const DraftSeedState& state = get_draft_state(id);
  return state.request_id == request_id ? &state : nullptr;
}

void MtpSeedStateStore::clear(const std::vector<int32_t>& ids) {
  std::vector<std::shared_ptr<DraftSeedBatch>> released;
  released.reserve(ids.size());
  for (int32_t id : ids) {
    (void)get_draft_state(id);
    released.emplace_back(detach_seed_batch(states_[id]));
    states_[id] = DraftSeedState();
  }
  compact_seed_batches(released);
  topk_cache_.clear(ids);
  // Released rows never come back; drop any device projection they kept alive.
  history_batch_.reclaim_unused_projection();
}

MtpSeedSamplingState MtpSeedStateStore::prepare(
    const std::vector<int32_t>& embedding_ids,
    const std::vector<std::string>& request_ids,
    SamplingParameters sampling_params,
    std::vector<JsonObjectGrammarState> json_states,
    const torch::Tensor& committed_tokens,
    const torch::Device& device) {
  MtpTokenHistories histories;
  if (has_penalty_params(sampling_params)) {
    histories = read_histories(embedding_ids, request_ids);
  }
  return prepare_mtp_seed_sampling(std::move(sampling_params),
                                   std::move(json_states),
                                   committed_tokens,
                                   device,
                                   std::move(histories),
                                   &history_batch_);
}

MtpTokenHistories MtpSeedStateStore::restore(
    const std::vector<int32_t>& embedding_ids,
    const std::vector<std::string>& request_ids,
    SamplingParameters& sampling_params,
    std::vector<JsonObjectGrammarState>& json_states,
    const torch::Device& device) {
  const bool needs_history = has_penalty_params(sampling_params);
  SamplingContext context = read_sampling_context(
      embedding_ids, request_ids, json_states, needs_history);
  if (needs_history) {
    import_mtp_sampling_histories(
        sampling_params, embedding_ids.size(), context.histories);
    write_histories(embedding_ids, request_ids, context.histories);
    history_batch_.apply(sampling_params, context.histories, device);
  }
  if (context.restored_grammar) {
    apply_draft_grammar_bitmask(sampling_params, json_states, device);
  }
  return context.histories;
}

}  // namespace xllm
