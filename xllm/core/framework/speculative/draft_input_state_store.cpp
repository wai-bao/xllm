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

#include "core/framework/speculative/draft_input_state_store.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/speculative/slot_table.h"
#include "core/util/tensor_helper.h"

namespace xllm {

DraftInputStateStore::DraftInputStateStore(int32_t total_nums) {
  CHECK_GT(total_nums, 0) << "No embeddings to allocate";
  states_.resize(total_nums);
}

std::shared_ptr<DraftInputStateStore::DraftInputBatch>
DraftInputStateStore::make_owned_draft_input_batch(
    const std::vector<int32_t>& ids,
    DraftInput draft_input) {
  // Draft-input validity is not re-checked here: write_draft_input validates
  // before calling, and clone_draft_input below re-validates before
  // snapshotting anyway.
  CHECK_EQ(draft_input.token_ids.size(0), static_cast<int64_t>(ids.size()));
  std::unordered_set<int32_t> seen;
  seen.reserve(ids.size());
  for (int32_t id : ids) {
    // The checked access doubles as the duplicate-slot precondition probe.
    (void)detail::checked_slot(states_, id);
    CHECK(seen.insert(id).second) << "duplicate draft input slot";
  }
  return std::make_shared<DraftInputBatch>(DraftInputBatch{
      clone_draft_input(draft_input), ids, static_cast<int64_t>(ids.size())});
}

std::shared_ptr<DraftInputStateStore::DraftInputBatch>
DraftInputStateStore::make_imported_draft_input_batch(
    const std::vector<int32_t>& ids,
    DraftInput draft_input) {
  // The transported draft_input owns its device storage (draft_input_to copy);
  // the snapshot moves the payload in without re-cloning tensors.
  return std::make_shared<DraftInputBatch>(DraftInputBatch{
      std::move(draft_input), ids, static_cast<int64_t>(ids.size())});
}

std::shared_ptr<DraftInputStateStore::DraftInputBatch>
DraftInputStateStore::detach_draft_input_batch(DraftInputState& state) {
  auto batch = std::move(state.batch);
  if (batch != nullptr) {
    CHECK_GT(batch->live_rows, 0);
    --batch->live_rows;
  }
  return batch;
}

void DraftInputStateStore::compact_draft_input_batches(
    const std::vector<std::shared_ptr<DraftInputBatch>>& batches) {
  for (const auto& batch : batches) {
    if (batch == nullptr || batch->live_rows == 0 ||
        batch->live_rows * 2 >=
            static_cast<int64_t>(batch->embedding_ids.size())) {
      continue;
    }
    std::vector<int32_t> survivors;
    survivors.reserve(batch->live_rows);
    for (int32_t id : batch->embedding_ids) {
      if (detail::checked_slot(states_, id).batch == batch) {
        survivors.emplace_back(id);
      }
    }
    CHECK_EQ(survivors.size(), batch->live_rows);
    const std::vector<DraftInput> runs = collect_draft_input_runs(survivors);
    // Even one contiguous run must leave the old allocation behind.
    batch->draft_input = runs.size() == 1 ? clone_draft_input(runs.front())
                                          : cat_draft_inputs(runs);
    batch->embedding_ids = std::move(survivors);
    // Compaction rebuilds full occupancy: live_rows matches the id list again.
    batch->live_rows = static_cast<int64_t>(batch->embedding_ids.size());
    for (size_t row = 0; row < batch->embedding_ids.size(); ++row) {
      detail::checked_slot(states_, batch->embedding_ids[row]).batch_row =
          static_cast<int64_t>(row);
    }
    // Repeated handles now see full occupancy and skip the same batch without
    // a second membership scan or an auxiliary deduplication allocation.
  }
}

void DraftInputStateStore::attach_draft_input_batch(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    const std::shared_ptr<DraftInputBatch>& batch) {
  std::vector<std::shared_ptr<DraftInputBatch>> replaced;
  replaced.reserve(ids.size());
  for (size_t row = 0; row < ids.size(); ++row) {
    // Update the slot identity in place: copy-assigning a same-length request
    // id reuses its buffer, while assigning a fresh DraftInputState would free
    // and re-allocate it per row per decode step. Payload fields (failed,
    // history, grammar state) stay with the callers, whose reset rules differ
    // between write and publish.
    DraftInputState& state = detail::checked_slot(states_, ids[row]);
    replaced.emplace_back(detach_draft_input_batch(state));
    state.request_id = request_ids[row];
    state.batch = batch;
    state.batch_row = static_cast<int64_t>(row);
  }
  compact_draft_input_batches(replaced);
}

void DraftInputStateStore::reset_slot_payloads(
    const std::vector<int32_t>& embedding_ids) {
  for (int32_t id : embedding_ids) {
    DraftInputState& state = detail::checked_slot(states_, id);
    state.failed = false;
    state.history.reset();
    state.json_state = JsonObjectGrammarState();
  }
}

void DraftInputStateStore::import_carried_draft_input(
    const std::vector<int32_t>& row_idxes,
    const std::vector<int32_t>& embedding_ids,
    const std::vector<std::string>& request_ids,
    DraftInput draft_input) {
  std::vector<int32_t> input_ids;
  std::vector<std::string> input_request_ids;
  input_ids.reserve(row_idxes.size());
  input_request_ids.reserve(row_idxes.size());
  for (int32_t row : row_idxes) {
    input_ids.emplace_back(embedding_ids[row]);
    input_request_ids.emplace_back(request_ids[row]);
  }
  // The transported draft_input owns fresh device storage; skip the clone the
  // plain write path applies for graph-aliasing producers.
  const std::shared_ptr<DraftInputBatch> batch =
      make_imported_draft_input_batch(input_ids, std::move(draft_input));
  reset_slot_payloads(input_ids);
  attach_draft_input_batch(input_ids, input_request_ids, batch);
}

void DraftInputStateStore::write_draft_input(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    DraftInput draft_input,
    DraftTokenHistories histories,
    std::vector<JsonObjectGrammarState> json_states,
    const std::vector<size_t>& failed_rows) {
  check_draft_input(draft_input);
  CHECK_EQ(ids.size(), request_ids.size());
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
  const std::shared_ptr<DraftInputBatch> batch =
      make_owned_draft_input_batch(ids, std::move(draft_input));
  // Payloads first, while the slots still carry the previous round's
  // identity: a row stays failed when it already failed for the same request.
  // The failed path clears the optional payloads rather than leaving the
  // previous round's behind; a non-failed row without a payload keeps what it
  // had. attach_draft_input_batch then rewrites every identity field in place.
  for (size_t row = 0; row < ids.size(); ++row) {
    DraftInputState& state = detail::checked_slot(states_, ids[row]);
    const bool row_failed =
        (!failed.empty() && failed[row] != 0) ||
        (state.request_id == request_ids[row] && state.failed);
    state.failed = row_failed;
    if (row_failed) {
      state.history.reset();
      state.json_state = JsonObjectGrammarState();
    } else {
      if (!histories.empty()) {
        state.history = std::move(histories[row]);
      }
      if (!json_states.empty()) {
        state.json_state = std::move(json_states[row]);
      }
    }
  }
  attach_draft_input_batch(ids, request_ids, batch);
}

std::optional<DraftInput> DraftInputStateStore::read_draft_input(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  return collect_draft_inputs(ids, request_ids);
}

std::vector<DraftInput> DraftInputStateStore::collect_draft_input_runs(
    const std::vector<int32_t>& ids) const {
  std::vector<DraftInput> runs;
  runs.reserve(ids.size());
  size_t begin = 0;
  while (begin < ids.size()) {
    const DraftInputState& first = detail::checked_slot(states_, ids[begin]);
    CHECK(first.batch != nullptr);
    size_t end = begin + 1;
    while (end < ids.size()) {
      const DraftInputState& next = detail::checked_slot(states_, ids[end]);
      if (next.batch != first.batch ||
          next.batch_row !=
              first.batch_row + static_cast<int64_t>(end - begin)) {
        break;
      }
      ++end;
    }
    runs.emplace_back(slice_draft_input(
        first.batch->draft_input, first.batch_row, end - begin));
    begin = end;
  }
  return runs;
}

std::optional<DraftInput> DraftInputStateStore::collect_draft_inputs(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  CHECK_EQ(ids.size(), request_ids.size());
  if (ids.empty()) {
    return std::nullopt;
  }
  return collect_draft_inputs(
      ids, resolve_owned_states(ids, request_ids), /*include_failed=*/false);
}

std::optional<DraftInput> DraftInputStateStore::collect_draft_inputs(
    const std::vector<int32_t>& ids,
    const std::vector<const DraftInputState*>& owned,
    bool include_failed) const {
  CHECK_EQ(ids.size(), owned.size());
  if (ids.empty()) {
    return std::nullopt;
  }
  for (size_t row = 0; row < ids.size(); ++row) {
    const DraftInputState* state = owned[row];
    if (state == nullptr || state->batch == nullptr ||
        (state->failed && !include_failed)) {
      return std::nullopt;
    }
  }
  // Ownership was validated above, so every run is a slice of a live batch.
  // One run means the whole request was contiguous in its snapshot batch.
  std::vector<DraftInput> runs = collect_draft_input_runs(ids);
  if (runs.size() == 1) {
    return std::move(runs.front());
  }
  return cat_draft_inputs(std::move(runs));
}

std::vector<const DraftInputStateStore::DraftInputState*>
DraftInputStateStore::resolve_owned_states(
    const std::vector<int32_t>& embedding_ids,
    const std::vector<std::string>& request_ids) const {
  CHECK_EQ(embedding_ids.size(), request_ids.size());
  std::vector<const DraftInputState*> owned;
  owned.reserve(embedding_ids.size());
  for (size_t row = 0; row < embedding_ids.size(); ++row) {
    owned.emplace_back(owned_draft_state(embedding_ids[row], request_ids[row]));
  }
  return owned;
}

void DraftInputStateStore::fill_failed_mask(
    const std::vector<const DraftInputState*>& owned,
    std::vector<uint8_t>& mask) {
  for (size_t row = 0; row < owned.size(); ++row) {
    if (owned[row] != nullptr && owned[row]->failed) {
      if (mask.empty()) {
        mask.resize(owned.size(), 0);
      }
      mask[row] = 1;
    }
  }
}

std::optional<DraftInput> DraftInputStateStore::take_draft_input(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    std::vector<uint8_t>* failed_mask) {
  CHECK_EQ(ids.size(), request_ids.size());
  if (failed_mask != nullptr) {
    CHECK(failed_mask->empty());
  }
  // Resolve failures in the same ownership pass. Keep batches attached until
  // after the draft_input has been materialized; collection reads their row
  // slices.
  const std::vector<const DraftInputState*> owned =
      resolve_owned_states(ids, request_ids);
  if (failed_mask != nullptr) {
    fill_failed_mask(owned, *failed_mask);
  }
  std::optional<DraftInput> draft_input =
      collect_draft_inputs(ids, owned, /*include_failed=*/true);
  if (!draft_input.has_value()) {
    return std::nullopt;
  }
  std::vector<std::shared_ptr<DraftInputBatch>> consumed;
  consumed.reserve(ids.size());
  for (size_t row = 0; row < ids.size(); ++row) {
    if (!owned[row]->failed) {
      consumed.emplace_back(
          detach_draft_input_batch(detail::checked_slot(states_, ids[row])));
    }
  }
  compact_draft_input_batches(consumed);
  return draft_input;
}

std::vector<uint8_t> DraftInputStateStore::failed_draft_input_mask(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  std::vector<uint8_t> mask;
  fill_failed_mask(resolve_owned_states(ids, request_ids), mask);
  return mask;
}

void DraftInputStateStore::publish_topk_state(
    const std::vector<int32_t>& embedding_ids,
    const std::vector<std::string>& request_ids,
    const MtpTopkStatePtr& state) {
  CHECK_EQ(embedding_ids.size(), request_ids.size());
  if (embedding_ids.empty()) {
    return;
  }
  // Anchor the batch-shared snapshot on the batch every row points at; rows
  // of one publication always share a batch (write/import attach whole).
  const std::shared_ptr<DraftInputBatch> batch =
      detail::checked_slot(states_, embedding_ids.front()).batch;
  CHECK(batch != nullptr) << "top-k publication requires a stored seed batch";
  for (size_t row = 0; row < embedding_ids.size(); ++row) {
    const DraftInputState& slot_state =
        detail::checked_slot(states_, embedding_ids[row]);
    CHECK(slot_state.request_id == request_ids[row])
        << "top-k publication requires the owning request";
    CHECK_EQ(slot_state.batch.get(), batch.get())
        << "top-k publication must cover one snapshot batch";
  }
  batch->topk_state = state;
}

MtpTopkStatePtr DraftInputStateStore::read_topk_state(
    const std::vector<int32_t>& embedding_ids,
    const std::vector<std::string>& request_ids) const {
  CHECK_EQ(embedding_ids.size(), request_ids.size());
  // Reuse requires the whole producing batch with unchanged ownership: every
  // row must still be owned by its publishing request and point at the same
  // batch, whose snapshot is then a single read.
  MtpTopkStatePtr snapshot;
  for (size_t row = 0; row < embedding_ids.size(); ++row) {
    const DraftInputState* state =
        owned_draft_state(embedding_ids[row], request_ids[row]);
    if (state == nullptr) {
      return nullptr;
    }
    if (row == 0) {
      snapshot = state->batch->topk_state;
    } else if (state->batch->topk_state != snapshot) {
      return nullptr;
    }
  }
  return snapshot;
}

DraftTokenHistories DraftInputStateStore::read_histories(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  CHECK_EQ(ids.size(), request_ids.size());
  DraftTokenHistories histories;
  histories.reserve(ids.size());
  for (size_t row = 0; row < ids.size(); ++row) {
    const DraftInputState* state =
        owned_draft_state(ids[row], request_ids[row]);
    histories.emplace_back(state != nullptr ? state->history : nullptr);
  }
  return histories;
}

DraftInputStateStore::SamplingContext
DraftInputStateStore::read_sampling_context(
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
    const DraftInputState* state =
        owned_draft_state(ids[row], request_ids[row]);
    if (state == nullptr) {
      continue;
    }
    context.owned_rows.push_back(row);
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

void DraftInputStateStore::write_histories(
    const std::vector<int32_t>& ids,
    const std::vector<size_t>& owned_rows,
    const DraftTokenHistories& histories) {
  CHECK_EQ(ids.size(), histories.size());
  for (const size_t row : owned_rows) {
    detail::checked_slot(states_, ids[row]).history = histories[row];
  }
}

// Ownership here is the request id alone: unlike TargetDecodeStateStore there
// is no separate validity flag, so a slot that no longer carries this request's
// id is treated as recycled. The same id is re-published every round by design.
const DraftInputStateStore::DraftInputState*
DraftInputStateStore::owned_draft_state(int32_t id,
                                        const std::string& request_id) const {
  const DraftInputState& state = detail::checked_slot(states_, id);
  return state.request_id == request_id ? &state : nullptr;
}

void DraftInputStateStore::clear(const std::vector<int32_t>& ids) {
  if (ids.empty()) {
    // launch_speculative calls clear every round even without releases; the
    // projection reclaim below is O(batch), so skip it for the empty case.
    return;
  }
  std::vector<std::shared_ptr<DraftInputBatch>> released;
  released.reserve(ids.size());
  for (int32_t id : ids) {
    DraftInputState& state = detail::checked_slot(states_, id);
    released.emplace_back(detach_draft_input_batch(state));
    state = DraftInputState();
  }
  compact_draft_input_batches(released);
  // Released rows never come back; drop any device projection they kept alive.
  history_batch_.reclaim_unused_projection();
}

DraftInputSamplingState DraftInputStateStore::prepare(
    const std::vector<int32_t>& embedding_ids,
    const std::vector<std::string>& request_ids,
    SamplingParameters sampling_params,
    std::vector<JsonObjectGrammarState> json_states,
    const torch::Tensor& committed_tokens,
    const torch::Device& device) {
  DraftTokenHistories histories;
  if (has_penalty_params(sampling_params)) {
    histories = read_histories(embedding_ids, request_ids);
  }
  return prepare_draft_input_sampling(std::move(sampling_params),
                                      std::move(json_states),
                                      committed_tokens,
                                      device,
                                      history_batch_,
                                      std::move(histories));
}

DraftTokenHistories DraftInputStateStore::restore(
    const std::vector<int32_t>& embedding_ids,
    const std::vector<std::string>& request_ids,
    SamplingParameters& sampling_params,
    std::vector<JsonObjectGrammarState>& json_states,
    const torch::Device& device) {
  const bool needs_history = has_penalty_params(sampling_params);
  SamplingContext context = read_sampling_context(
      embedding_ids, request_ids, json_states, needs_history);
  if (needs_history) {
    import_sampling_histories(
        sampling_params, embedding_ids.size(), context.histories);
    write_histories(embedding_ids, context.owned_rows, context.histories);
    history_batch_.apply(sampling_params, context.histories, device);
  }
  if (context.restored_grammar) {
    apply_draft_grammar_bitmask(sampling_params, json_states, device);
  }
  return context.histories;
}

}  // namespace xllm
