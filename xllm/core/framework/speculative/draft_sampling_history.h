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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "core/framework/sampling/sampling_params.h"

namespace xllm {

// CPU authority for one request. Only committed tokens advance this history;
// device tensors are disposable sampler projections, never read back to update
// it.
class DraftTokenHistory final {
 public:
  DraftTokenHistory() = default;
  DraftTokenHistory(const DraftTokenHistory&) = delete;
  DraftTokenHistory& operator=(const DraftTokenHistory&) = delete;

  // Token storage is borrowed only for this call.
  void append(std::span<const int64_t> tokens);
  void append(std::initializer_list<int64_t> tokens) {
    append(std::span<const int64_t>(tokens.begin(), tokens.size()));
  }

  const std::vector<int64_t>& ids() const { return ids_; }
  const std::vector<int32_t>& counts() const { return counts_; }
  const std::vector<int64_t>& changed_columns() const {
    return changed_columns_;
  }
  uint64_t revision() const { return revision_; }

  // The loaded vocabulary map is immutable. Its per-request projection shares
  // this history's lifetime and updates only columns changed since the last
  // call.
  std::shared_ptr<DraftTokenHistory> mapped(
      const torch::Tensor& target_to_draft);

  // Copies CPU values into owned history. Source spans need only remain valid
  // for this call.
  static std::shared_ptr<DraftTokenHistory> from_host_values(
      std::span<const int64_t> ids,
      std::span<const int64_t> counts);
  static std::shared_ptr<DraftTokenHistory> from_host_values(
      std::span<const int64_t> ids,
      std::span<const int32_t> counts);

 private:
  template <typename Count>
  static std::shared_ptr<DraftTokenHistory> make_host_history(
      std::span<const int64_t> ids,
      std::span<const Count> counts);

  std::vector<int64_t> ids_;
  std::vector<int32_t> counts_;
  std::unordered_map<int64_t, int64_t> indices_;
  std::vector<int64_t> changed_columns_;
  uint64_t revision_ = 0;
  torch::Tensor mapped_vocabulary_;
  std::shared_ptr<DraftTokenHistory> mapped_history_;
  uint64_t mapped_revision_ = 0;
};

using DraftTokenHistories = std::vector<std::shared_ptr<DraftTokenHistory>>;

// Fill only missing rows. With an established CPU history this function does
// not inspect scheduler/device statistics, which may lag under overlap.
void import_sampling_histories(const SamplingParameters& params,
                               int64_t batch_size,
                               DraftTokenHistories& histories);

// A single stream owns mutations of this projection. Consumers must finish
// using its previous revision on that stream before apply() advances it.
class DraftHistoryBatch final {
 public:
  void apply(SamplingParameters& params,
             const DraftTokenHistories& histories,
             const torch::Device& device);
  void clear();
  // Release the cached tensors once none of the projected histories remains
  // owned by a request. Live rows keep their incremental-update baseline.
  void reclaim_unused_projection();

 private:
  struct ProjectedRow {
    std::weak_ptr<DraftTokenHistory> history;
    uint64_t revision = 0;
    int64_t length = 0;
    int32_t last_count = 0;
  };

  std::vector<ProjectedRow> rows_;
  torch::Tensor ids_;
  torch::Tensor counts_;
  torch::Tensor lengths_;
  // Per-round host staging scratch for the CPU and device projection paths.
  // Reuse is safe because a single stream owns projection mutations and each
  // update's non-blocking upload is enqueued before the next host write.
  torch::Tensor host_payload_scratch_;
  // Device-projection peer: the async D2H upload source must stay pinned, and
  // each incremental update reuses this allocation like the CPU scratch.
  torch::Tensor pinned_payload_scratch_;

  // Discards the current projection and re-uploads every row. Used when the
  // batch membership changed or the widest history outgrew the capacity.
  void rebuild_projection(const DraftTokenHistories& histories,
                          const torch::Device& device,
                          int64_t batch_size,
                          int64_t width,
                          bool same_batch);
  // Patches only the rows whose revision advanced, reusing the existing
  // device storage.
  void apply_incremental_update(const DraftTokenHistories& histories,
                                const torch::Device& device,
                                int64_t batch_size);
};

// Projects the histories through the caller-owned batch; owning one keeps the
// device projection alive across rounds instead of re-uploading every call.
void apply_draft_history_batch(DraftHistoryBatch& batch,
                               SamplingParameters& params,
                               const DraftTokenHistories& histories,
                               const torch::Device& device);

}  // namespace xllm
