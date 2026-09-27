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
#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/speculative/mtp_draft_seed.h"
#include "core/framework/speculative/mtp_sampling_history.h"
#include "core/framework/speculative/mtp_seed_sampling.h"
#include "core/framework/speculative/mtp_seed_topk_cache.h"

namespace xllm {

// MTP/EAGLE proposal state. Target continuation state lives in
// TargetDecodeStateStore so other speculative algorithms do not inherit seed,
// grammar, or sampling-history dependencies. The store is the single access
// point for one MTP worker: seed lifecycle (write/publish/take/clear),
// failure masks, sampling-state preparation, the device history projection,
// and the cross-round top-k snapshot all route through it. Releasing a slot
// clears every proposal state owned by this store.
class MtpSeedStateStore final {
 public:
  explicit MtpSeedStateStore(int32_t total_nums);
  ~MtpSeedStateStore() = default;

  // disable copy, move and assign
  DISALLOW_COPY_AND_ASSIGN(MtpSeedStateStore);

  // Writes a bare seed (no sampling context), e.g. a PD-imported batch.
  // Unlike publish_draft_seed, an imported seed always starts unfailed.
  void write_draft_seed(const std::vector<int32_t>& embedding_ids,
                        const std::vector<std::string>& request_ids,
                        MtpDraftSeed seed) {
    publish_draft_seed(embedding_ids,
                       request_ids,
                       std::move(seed),
                       /*histories=*/{},
                       /*json_states=*/{},
                       /*failed_rows=*/{},
                       /*carry_failed=*/false);
  }
  std::optional<MtpDraftSeed> read_draft_seed(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  // Publish the proposal together with the committed sampling state that
  // produced it. A failed request keeps a private seed only to drain queued
  // collective phases; read_draft_seed never exposes that payload. A slot
  // reused by the same still-failed request stays failed across publishes.
  void publish_draft_seed(const std::vector<int32_t>& embedding_ids,
                          const std::vector<std::string>& request_ids,
                          MtpDraftSeed seed,
                          MtpTokenHistories histories,
                          std::vector<JsonObjectGrammarState> json_states,
                          const std::vector<size_t>& failed_rows,
                          bool carry_failed = true);
  // Consumes the decode seed for the given rows in a single ownership pass.
  // When failed_mask is non-null it receives the per-row failure mask that
  // failed_seed_mask() would have produced, avoiding a second traversal.
  std::optional<MtpDraftSeed> take_decode_seed(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids,
      std::vector<uint8_t>* failed_mask = nullptr);
  std::vector<uint8_t> failed_seed_mask(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  // The top-k snapshot is reusable only for the exact producing batch.
  void publish_topk_state(std::vector<int32_t> embedding_ids,
                          std::vector<std::string> request_ids,
                          MtpTopkStatePtr state);
  MtpTopkStatePtr read_topk_state(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  void clear(const std::vector<int32_t>& embedding_ids);

  // Prepares the draft sampling state for the committed tokens: imports
  // stored CPU histories for penalized requests, replays JSON grammar rows,
  // and projects the histories through the owned device batch.
  MtpSeedSamplingState prepare(const std::vector<int32_t>& embedding_ids,
                               const std::vector<std::string>& request_ids,
                               SamplingParameters sampling_params,
                               std::vector<JsonObjectGrammarState> json_states,
                               const torch::Tensor& committed_tokens,
                               const torch::Device& device);
  // Restores the decode-phase sampling state: reads the stored sampling
  // context back into params/json_states and refreshes the device projection.
  MtpTokenHistories restore(const std::vector<int32_t>& embedding_ids,
                            const std::vector<std::string>& request_ids,
                            SamplingParameters& sampling_params,
                            std::vector<JsonObjectGrammarState>& json_states,
                            const torch::Device& device);

 private:
  struct SamplingContext {
    MtpTokenHistories histories;
    bool restored_grammar = false;
  };

  MtpTokenHistories read_histories(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  SamplingContext read_sampling_context(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids,
      std::vector<JsonObjectGrammarState>& json_states,
      bool needs_history) const;
  void write_histories(const std::vector<int32_t>& embedding_ids,
                       const std::vector<std::string>& request_ids,
                       const MtpTokenHistories& histories);

  struct DraftSeedBatch {
    MtpDraftSeed seed;
    std::vector<int32_t> embedding_ids;
    int64_t live_rows = 0;
  };

  struct DraftSeedState {
    std::string request_id;
    std::shared_ptr<DraftSeedBatch> batch;
    int64_t batch_row = 0;
    std::shared_ptr<MtpTokenHistory> history;
    JsonObjectGrammarState json_state;
    bool failed = false;
  };

  std::vector<DraftSeedState> states_;
  // Generation stamps keep duplicate checks linear without per-batch hashing.
  std::vector<uint32_t> slot_generations_;
  uint32_t batch_generation_ = 0;
  const DraftSeedState& get_draft_state(int32_t embedding_id) const;
  // Returns the draft state only while the slot still belongs to request_id; a
  // recycled slot (embedding_id reused by a later request) returns nullptr. All
  // draft readers route the ownership check through this single seam.
  const DraftSeedState* owned_draft_state(int32_t embedding_id,
                                          const std::string& request_id) const;
  std::shared_ptr<DraftSeedBatch> make_seed_batch(
      const std::vector<int32_t>& embedding_ids,
      MtpDraftSeed seed);
  static std::shared_ptr<DraftSeedBatch> detach_seed_batch(
      DraftSeedState& state);
  // Compact below half occupancy. Retained storage is at most twice the live
  // rows; successive compactions of one snapshot copy fewer than B rows total.
  void compact_seed_batches(
      const std::vector<std::shared_ptr<DraftSeedBatch>>& batches);
  // Group adjacent rows from the same snapshot without changing their order.
  std::vector<MtpDraftSeed> collect_draft_seed_runs(
      const std::vector<int32_t>& embedding_ids) const;
  // Non-consuming read; failed requests keep a private drain payload that this
  // never exposes.
  std::optional<MtpDraftSeed> collect_draft_seeds(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  // Same as above, reusing the caller's already-resolved per-row states to
  // avoid a second ownership pass on the decode hot path.
  std::optional<MtpDraftSeed> collect_draft_seeds(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<const DraftSeedState*>& owned,
      bool include_failed) const;

  // Device projection of the stored CPU histories for one MTP worker.
  MtpHistoryBatch history_batch_;
  MtpSeedTopkCache topk_cache_;
};

}  // namespace xllm
