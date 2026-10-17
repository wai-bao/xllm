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
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/macros.h"
#include "core/framework/model/mtp_topk_state.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/speculative/draft_input.h"
#include "core/framework/speculative/draft_input_sampling.h"
#include "core/framework/speculative/draft_sampling_history.h"

namespace xllm {

// Autoregressive drafter proposal state (MTP/EAGLE-3). Target continuation
// state lives in TargetDecodeStateStore so other speculative algorithms do not
// inherit draft-input, grammar, or sampling-history dependencies. The store is
// the single access point for one drafter worker: draft-input lifecycle
// (write/publish/take/clear), failure masks, sampling-state preparation, the
// device history projection, and the cross-round top-k snapshot all route
// through it. Releasing a slot clears every proposal state owned by this store.
class DraftInputStateStore final {
 public:
  explicit DraftInputStateStore(int32_t total_nums);
  ~DraftInputStateStore() = default;

  DISALLOW_COPY_AND_ASSIGN(DraftInputStateStore);

  // Publish the proposal together with the committed sampling state that
  // produced it. A failed request keeps a private seed only to drain queued
  // collective phases; read_draft_input never exposes that payload. The
  // optional payload arguments default to empty for producers that only
  // carry the seed.
  void write_draft_input(const std::vector<int32_t>& embedding_ids,
                         const std::vector<std::string>& request_ids,
                         DraftInput draft_input,
                         DraftTokenHistories histories = {},
                         std::vector<JsonObjectGrammarState> json_states = {},
                         const std::vector<size_t>& failed_rows = {});
  // Imports a transported PD draft input into the store: collects the per-row
  // embedding/request ids addressed by row_idxes and writes the seed as one
  // batch snapshot. Shared by the eager worker and the task pipeline.
  void import_carried_draft_input(const std::vector<int32_t>& row_idxes,
                                  const std::vector<int32_t>& embedding_ids,
                                  const std::vector<std::string>& request_ids,
                                  DraftInput draft_input);
  // Non-consuming read: PD prefill export and tests use this; the decode
  // hot path consumes via take_draft_input.
  std::optional<DraftInput> read_draft_input(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  // Consumes the decode seed for the given rows in a single ownership pass.
  // When failed_mask is non-null it receives the per-row failure mask that
  // failed_draft_input_mask() would have produced, avoiding a second traversal.
  std::optional<DraftInput> take_draft_input(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids,
      std::vector<uint8_t>* failed_mask = nullptr);
  std::vector<uint8_t> failed_draft_input_mask(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  // The top-k snapshot is reusable only for the exact producing batch: every
  // row must still be owned by its publishing request.
  void publish_topk_state(const std::vector<int32_t>& embedding_ids,
                          const std::vector<std::string>& request_ids,
                          const MtpTopkStatePtr& state);
  MtpTopkStatePtr read_topk_state(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  void clear(const std::vector<int32_t>& embedding_ids);

  // Prepares the draft sampling state for the committed tokens: imports
  // stored CPU histories for penalized requests, replays JSON grammar rows,
  // and projects the histories through the owned device batch.
  DraftInputSamplingState prepare(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids,
      SamplingParameters sampling_params,
      std::vector<JsonObjectGrammarState> json_states,
      const torch::Tensor& committed_tokens,
      const torch::Device& device);
  // Restores the decode-phase sampling state: reads the stored sampling
  // context back into params/json_states and refreshes the device projection.
  DraftTokenHistories restore(const std::vector<int32_t>& embedding_ids,
                              const std::vector<std::string>& request_ids,
                              SamplingParameters& sampling_params,
                              std::vector<JsonObjectGrammarState>& json_states,
                              const torch::Device& device);

 private:
  struct SamplingContext {
    DraftTokenHistories histories;
    // Row indices resolved as owned during the context read.
    std::vector<size_t> owned_rows;
    bool restored_grammar = false;
  };

  DraftTokenHistories read_histories(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  SamplingContext read_sampling_context(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids,
      std::vector<JsonObjectGrammarState>& json_states,
      bool needs_history) const;
  // Writes back only the rows read_sampling_context resolved as owned; ids
  // and histories share the row order of that earlier pass.
  void write_histories(const std::vector<int32_t>& embedding_ids,
                       const std::vector<size_t>& owned_rows,
                       const DraftTokenHistories& histories);

  struct DraftInputBatch {
    DraftInput draft_input;
    std::vector<int32_t> embedding_ids;
    int64_t live_rows = 0;
    // Top-k snapshot published with this batch. Anchored here (not per slot)
    // so the batch's refcount lifecycle carries it: compaction rebuilds it
    // with the batch, releasing every slot drops it for free.
    MtpTopkStatePtr topk_state;
  };

  struct DraftInputState {
    std::string request_id;
    std::shared_ptr<DraftInputBatch> batch;
    int64_t batch_row = 0;
    std::shared_ptr<DraftTokenHistory> history;
    JsonObjectGrammarState json_state;
    bool failed = false;
  };

  std::vector<DraftInputState> states_;
  // Returns the draft state only while the slot still belongs to request_id; a
  // recycled slot (embedding_id reused by a later request) returns nullptr. All
  // draft readers route the ownership check through this single seam.
  const DraftInputState* owned_draft_state(int32_t embedding_id,
                                           const std::string& request_id) const;
  // Resolves the owned state of every row in one pass. Shared by the
  // failed-mask readers so the ownership check and the batch attachment cannot
  // drift.
  std::vector<const DraftInputState*> resolve_owned_states(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  // Fills a per-row mask (1 = failed) for the already-resolved rows. Leaves
  // the mask empty when no row failed, which is the "no failures" convention
  // the callers rely on.
  static void fill_failed_mask(const std::vector<const DraftInputState*>& owned,
                               std::vector<uint8_t>& mask);
  // Clone-and-wrap snapshot: a moved Tensor handle may still alias the
  // caller's storage or a graph output buffer, so the store owns a detached
  // copy across rounds. Probes duplicate slots before snapshotting.
  std::shared_ptr<DraftInputBatch> make_owned_draft_input_batch(
      const std::vector<int32_t>& embedding_ids,
      DraftInput draft_input);
  // Drops the per-slot sampling payloads (failure flag, history, grammar
  // state) before the import path re-attaches a batch.
  void reset_slot_payloads(const std::vector<int32_t>& embedding_ids);
  // Import-path variant: the whole seed was just transported to this device
  // (draft_input_to produces fresh storage), so the snapshot moves the payload
  // without re-cloning any tensor.
  std::shared_ptr<DraftInputBatch> make_imported_draft_input_batch(
      const std::vector<int32_t>& embedding_ids,
      DraftInput draft_input);
  static std::shared_ptr<DraftInputBatch> detach_draft_input_batch(
      DraftInputState& state);
  // Rewires every slot to a freshly built snapshot batch: detaches each
  // row's previous batch, rewrites the shared slot identity (request id,
  // batch, row) in place and compacts the detached snapshots. Per-row
  // payload (failure flag, history, grammar state) stays with the callers,
  // whose reset rules differ between write and import.
  void attach_draft_input_batch(const std::vector<int32_t>& ids,
                                const std::vector<std::string>& request_ids,
                                const std::shared_ptr<DraftInputBatch>& batch);
  // Compact below half occupancy. Retained storage is at most twice the live
  // rows; successive compactions of one snapshot copy fewer than B rows total.
  void compact_draft_input_batches(
      const std::vector<std::shared_ptr<DraftInputBatch>>& batches);
  // Group adjacent rows from the same snapshot without changing their order.
  std::vector<DraftInput> collect_draft_input_runs(
      const std::vector<int32_t>& embedding_ids) const;
  // Non-consuming read; failed requests keep a private drain payload that this
  // never exposes.
  std::optional<DraftInput> collect_draft_inputs(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  // Same as above, reusing the caller's already-resolved per-row states to
  // avoid a second ownership pass on the decode hot path.
  std::optional<DraftInput> collect_draft_inputs(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<const DraftInputState*>& owned,
      bool include_failed) const;

  // Device projection of the stored CPU histories for one drafter worker.
  DraftHistoryBatch history_batch_;
};

}  // namespace xllm
