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

#include "core/framework/speculative/mtp_sampling_history.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "core/util/tensor_helper.h"

namespace xllm {

namespace {

struct HistoryImportRow {
  int64_t index = 0;
  int64_t length = 0;
  int64_t offset = 0;
};

// Coalesces padding fills for one target matrix: consecutive rows that share a
// start column and fill value collapse into a single strided rectangle fill,
// while runs shorter than the crossover keep per-row fills. Bind the target
// once, add() each padded row in order, then flush() the trailing run.
class PaddingRunWriter final {
 public:
  explicit PaddingRunWriter(const torch::Tensor& target) : target_(target) {}

  void add(int64_t row, int64_t start, int64_t value) {
    if (rows_ > 0 && row == row_ + rows_ && start == start_ &&
        value == value_) {
      ++rows_;
      return;
    }
    flush();
    row_ = row;
    rows_ = 1;
    start_ = start;
    value_ = value;
  }

  void flush() {
    if (rows_ == 0) {
      return;
    }
    const int64_t width = target_.size(1) - start_;
    constexpr int64_t kMinCoalescedPaddingRows = 3;
    if (rows_ < kMinCoalescedPaddingRows) {
      for (int64_t row = row_; row < row_ + rows_; ++row) {
        target_.select(/*dim=*/0, row)
            .narrow(/*dim=*/0, start_, width)
            .fill_(value_);
      }
    } else {
      target_.narrow(/*dim=*/0, row_, rows_)
          .narrow(/*dim=*/1, start_, width)
          .fill_(value_);
    }
    rows_ = 0;
  }

 private:
  const torch::Tensor& target_;
  int64_t row_ = 0;
  int64_t rows_ = 0;
  int64_t start_ = 0;
  int64_t value_ = 0;
};

torch::Tensor pack_history_prefixes(const torch::Tensor& values,
                                    const std::vector<HistoryImportRow>& rows,
                                    int64_t total_tokens) {
  // Planned prefixes cannot exceed their rows, so full coverage has neither
  // padding nor established histories to exclude. Keep the original layout
  // until the host copy, where strided matrices are normalized as well.
  if (total_tokens == values.numel()) {
    return values;
  }
  std::vector<torch::Tensor> prefixes;
  prefixes.reserve(rows.size());
  for (const HistoryImportRow& row : rows) {
    if (row.length == 0) {
      continue;
    }
    prefixes.emplace_back(values.select(/*dim=*/0, row.index)
                              .narrow(/*dim=*/0, /*start=*/0, row.length));
  }
  return prefixes.size() == 1 ? prefixes.front()
                              : torch::cat(prefixes, /*dim=*/0);
}

template <typename Count>
void import_cpu_history_rows(const SamplingParameters& params,
                             const std::vector<HistoryImportRow>& rows,
                             MtpTokenHistories& histories) {
  const int64_t* ids = params.unique_token_ids.const_data_ptr<int64_t>();
  const Count* counts = params.unique_token_counts.const_data_ptr<Count>();
  const int64_t ids_stride = params.unique_token_ids.stride(/*dim=*/0);
  const int64_t counts_stride = params.unique_token_counts.stride(/*dim=*/0);
  for (const HistoryImportRow& row : rows) {
    // A zero-length row yields an empty span and thus an empty history, so no
    // separate empty-row case is needed (matching the device path).
    const size_t length = static_cast<size_t>(row.length);
    histories[row.index] = MtpTokenHistory::from_host_values(
        std::span<const int64_t>(ids + row.index * ids_stride, length),
        std::span<const Count>(counts + row.index * counts_stride, length));
  }
}

void import_history_batch(const SamplingParameters& params,
                          int64_t batch_size,
                          size_t missing_count,
                          MtpTokenHistories& histories) {
  const torch::Tensor lengths_cpu =
      to_cpu_int64_contiguous(params.unique_token_ids_lens).flatten();
  const int64_t* lengths = lengths_cpu.const_data_ptr<int64_t>();
  std::vector<HistoryImportRow> rows;
  rows.reserve(missing_count);
  int64_t total_tokens = 0;
  for (int64_t row = 0; row < batch_size; ++row) {
    if (histories[row] != nullptr) {
      continue;
    }
    const int64_t length = lengths[row];
    CHECK_GE(length, 0);
    CHECK_LE(length, params.unique_token_ids.size(1));
    rows.emplace_back(HistoryImportRow{row, length, total_tokens});
    total_tokens += length;
  }
  if (total_tokens == 0) {
    for (const HistoryImportRow& row : rows) {
      histories[row.index] = std::make_shared<MtpTokenHistory>();
    }
    return;
  }

  const torch::ScalarType count_type = params.unique_token_counts.scalar_type();
  const bool native_cpu_rows =
      params.unique_token_ids.device().is_cpu() &&
      params.unique_token_counts.device().is_cpu() &&
      params.unique_token_ids.scalar_type() == torch::kLong &&
      params.unique_token_ids.stride(/*dim=*/1) == 1 &&
      params.unique_token_counts.stride(/*dim=*/1) == 1 &&
      (count_type == torch::kInt || count_type == torch::kLong);
  if (native_cpu_rows) {
    if (count_type == torch::kInt) {
      import_cpu_history_rows<int32_t>(params, rows, histories);
    } else {
      import_cpu_history_rows<int64_t>(params, rows, histories);
    }
    return;
  }

  // Device tensors and any non-native CPU layout share one normalization:
  // select both fields, then a single host copy (to_cpu_int64_contiguous is a
  // no-op-to-copy on CPU and contiguous-normalizes strided rows). Complete
  // matrices need no device pack; other layouts copy only missing rows' valid
  // prefixes.
  const torch::Tensor packed_ids =
      pack_history_prefixes(params.unique_token_ids, rows, total_tokens);
  const torch::Tensor packed_counts =
      pack_history_prefixes(params.unique_token_counts, rows, total_tokens);
  const torch::Tensor ids_cpu = to_cpu_int64_contiguous(packed_ids);
  const torch::Tensor counts_cpu = to_cpu_int64_contiguous(packed_counts);
  CHECK_EQ(ids_cpu.numel(), total_tokens);
  CHECK_EQ(counts_cpu.numel(), total_tokens);
  const std::span<const int64_t> ids_values(ids_cpu.const_data_ptr<int64_t>(),
                                            static_cast<size_t>(total_tokens));
  const std::span<const int64_t> counts_values(
      counts_cpu.const_data_ptr<int64_t>(), static_cast<size_t>(total_tokens));
  for (const HistoryImportRow& row : rows) {
    const size_t offset = static_cast<size_t>(row.offset);
    const size_t length = static_cast<size_t>(row.length);
    histories[row.index] = MtpTokenHistory::from_host_values(
        ids_values.subspan(offset, length),
        counts_values.subspan(offset, length));
  }
}

}  // namespace

template <typename Count>
std::shared_ptr<MtpTokenHistory> MtpTokenHistory::make_host_history(
    std::span<const int64_t> ids,
    std::span<const Count> counts) {
  CHECK_EQ(ids.size(), counts.size());
  const int64_t length = static_cast<int64_t>(ids.size());
  auto history = std::make_shared<MtpTokenHistory>();
  history->ids_.reserve(length);
  history->counts_.reserve(length);
  history->indices_.reserve(length);
  for (int64_t column = 0; column < length; ++column) {
    const int64_t token = ids[column];
    const int64_t count = static_cast<int64_t>(counts[column]);
    CHECK_GE(token, 0);
    CHECK_GE(count, 0);
    CHECK_LE(count, std::numeric_limits<int32_t>::max());
    CHECK(history->indices_.emplace(token, column).second)
        << "MTP sampling history contains duplicate token ids";
    history->ids_.emplace_back(token);
    history->counts_.emplace_back(static_cast<int32_t>(count));
  }
  return history;
}

std::shared_ptr<MtpTokenHistory> MtpTokenHistory::from_host_values(
    std::span<const int64_t> ids,
    std::span<const int64_t> counts) {
  return make_host_history(ids, counts);
}

std::shared_ptr<MtpTokenHistory> MtpTokenHistory::from_host_values(
    std::span<const int64_t> ids,
    std::span<const int32_t> counts) {
  return make_host_history(ids, counts);
}

void MtpTokenHistory::append(std::span<const int64_t> tokens) {
  if (tokens.empty()) {
    return;
  }
  changed_columns_.clear();
  changed_columns_.reserve(tokens.size());
  for (int64_t token : tokens) {
    CHECK_GE(token, 0);
    auto [it, inserted] =
        indices_.try_emplace(token, static_cast<int64_t>(ids_.size()));
    if (inserted) {
      CHECK_LT(ids_.size(),
               static_cast<size_t>(std::numeric_limits<int32_t>::max()));
      ids_.emplace_back(token);
      counts_.emplace_back(0);
    }
    int32_t& count = counts_[it->second];
    CHECK_LT(count, std::numeric_limits<int32_t>::max());
    ++count;
    changed_columns_.emplace_back(it->second);
  }
  // A repeated committed token must produce one device write of its final
  // count.
  std::sort(changed_columns_.begin(), changed_columns_.end());
  changed_columns_.erase(
      std::unique(changed_columns_.begin(), changed_columns_.end()),
      changed_columns_.end());
  ++revision_;
}

std::shared_ptr<MtpTokenHistory> MtpTokenHistory::mapped(
    const torch::Tensor& target_to_draft) {
  CHECK(target_to_draft.device().is_cpu());
  CHECK_EQ(target_to_draft.scalar_type(), torch::kLong);
  CHECK_EQ(target_to_draft.dim(), 1);
  CHECK(target_to_draft.is_contiguous());
  const bool same_map = mapped_vocabulary_.defined() &&
                        mapped_vocabulary_.const_data_ptr<int64_t>() ==
                            target_to_draft.const_data_ptr<int64_t>() &&
                        mapped_vocabulary_.numel() == target_to_draft.numel();
  if (same_map && mapped_revision_ == revision_) {
    return mapped_history_;
  }
  if (!same_map) {
    mapped_history_ = std::make_shared<MtpTokenHistory>();
    mapped_vocabulary_ = target_to_draft;
  }
  auto& projected = *mapped_history_;
  const bool incremental = same_map && mapped_revision_ + 1 == revision_;
  const size_t max_changes =
      incremental ? changed_columns_.size() : ids_.size();
  projected.changed_columns_.reserve(max_changes);
  bool changed = false;
  const int64_t* inverse = target_to_draft.const_data_ptr<int64_t>();
  const auto update_column = [&](int64_t column) {
    const int64_t token = ids_[column];
    CHECK_LT(token, target_to_draft.numel());
    const int64_t draft_token = inverse[token];
    if (draft_token < 0) {
      return;
    }
    auto [it, inserted] = projected.indices_.try_emplace(
        draft_token, static_cast<int64_t>(projected.ids_.size()));
    if (inserted) {
      projected.ids_.emplace_back(draft_token);
      projected.counts_.emplace_back(counts_[column]);
    } else if (projected.counts_[it->second] == counts_[column]) {
      return;
    } else {
      projected.counts_[it->second] = counts_[column];
    }
    if (!changed) {
      // The previous delta was consumed; start this pass's delta from empty.
      projected.changed_columns_.clear();
      changed = true;
    }
    projected.changed_columns_.emplace_back(it->second);
  };
  if (incremental) {
    for (int64_t column : changed_columns_) {
      update_column(column);
    }
  } else {
    projected.ids_.reserve(ids_.size());
    projected.counts_.reserve(ids_.size());
    projected.indices_.reserve(ids_.size());
    for (int64_t column = 0; column < static_cast<int64_t>(ids_.size());
         ++column) {
      update_column(column);
    }
  }
  if (changed) {
    ++projected.revision_;
  } else if (!incremental) {
    // A cold pass that changes nothing leaves no delta to consume; drop any
    // leftover from a previous round so a later incremental pass cannot
    // replay stale columns as fresh changes.
    projected.changed_columns_.clear();
  }
  mapped_revision_ = revision_;
  return mapped_history_;
}

void import_mtp_sampling_histories(const SamplingParameters& params,
                                   int64_t batch_size,
                                   MtpTokenHistories& histories) {
  if (histories.empty()) {
    histories.resize(batch_size);
  }
  CHECK_EQ(histories.size(), batch_size);
  if (std::all_of(histories.begin(), histories.end(), [](const auto& history) {
        return history != nullptr;
      })) {
    return;
  }
  if (!params.unique_token_ids.defined()) {
    CHECK(!params.unique_token_counts.defined());
    CHECK(!params.unique_token_ids_lens.defined());
    for (auto& history : histories) {
      if (history == nullptr) {
        history = std::make_shared<MtpTokenHistory>();
      }
    }
    return;
  }
  CHECK(params.unique_token_counts.defined());
  CHECK(params.unique_token_ids_lens.defined());
  CHECK_EQ(params.unique_token_ids.dim(), 2);
  CHECK(params.unique_token_ids.sizes() == params.unique_token_counts.sizes());
  CHECK_EQ(params.unique_token_ids.size(0), batch_size);
  CHECK_EQ(params.unique_token_ids_lens.numel(), batch_size);
  const size_t missing_count = static_cast<size_t>(
      std::count(histories.begin(), histories.end(), nullptr));
  if (missing_count > 0) {
    // The batch path packs all missing rows into a single host copy; even a
    // single row saves the separate blocking lens read + ids/counts copies.
    import_history_batch(params, batch_size, missing_count, histories);
  }
}

void MtpHistoryBatch::clear() {
  rows_.clear();
  ids_ = torch::Tensor();
  counts_ = torch::Tensor();
  lengths_ = torch::Tensor();
}

void MtpHistoryBatch::reclaim_unused_projection() {
  if (std::none_of(rows_.begin(), rows_.end(), [](const ProjectedRow& row) {
        return !row.history.expired();
      })) {
    clear();
  }
}

void MtpHistoryBatch::apply(SamplingParameters& params,
                            const MtpTokenHistories& histories,
                            const torch::Device& device) {
  const int64_t batch_size = static_cast<int64_t>(histories.size());
  if (batch_size == 0) {
    clear();
    params.unique_token_ids = torch::Tensor();
    params.unique_token_counts = torch::Tensor();
    params.unique_token_ids_lens = torch::Tensor();
    return;
  }
  bool same_batch = ids_.defined() && ids_.device() == device &&
                    rows_.size() == histories.size();
  int64_t width = 0;
  for (int64_t row = 0; row < batch_size; ++row) {
    CHECK(histories[row] != nullptr);
    width = std::max(width, static_cast<int64_t>(histories[row]->ids().size()));
    same_batch = same_batch && rows_[row].history.lock() == histories[row];
  }
  const bool rebuild = !same_batch || width > ids_.size(1);
  if (rebuild) {
    rebuild_projection(histories, device, batch_size, width, same_batch);
  } else {
    apply_incremental_update(histories, device, batch_size);
  }
  params.unique_token_ids = ids_;
  params.unique_token_counts = counts_;
  params.unique_token_ids_lens = lengths_;
}

void MtpHistoryBatch::rebuild_projection(const MtpTokenHistories& histories,
                                         const torch::Device& device,
                                         int64_t batch_size,
                                         int64_t width,
                                         bool same_batch) {
  // Geometric growth only for a stable batch; rebatching sheds old capacity.
  const int64_t capacity =
      same_batch ? std::max(width, ids_.size(1) * 2) : width;
  const int64_t value_count = batch_size * capacity;
  const int64_t scalar_count = value_count + batch_size;
  // CPU projections need no upload; direct tensors avoid extra view setup.
  // Device projections pack int64 IDs followed by int32 counts and lengths.
  const bool host_projection = device.is_cpu();
  torch::Tensor host_values =
      host_projection ? torch::empty({batch_size, capacity}, torch::kLong)
                      : torch::empty({value_count + (scalar_count + 1) / 2},
                                     torch::TensorOptions()
                                         .dtype(torch::kLong)
                                         .device(torch::kCPU)
                                         .pinned_memory(true));
  torch::Tensor host_scalars =
      host_projection ? torch::empty({batch_size, capacity}, torch::kInt)
                      : host_values.view(torch::kInt);
  torch::Tensor host_lengths = host_projection
                                   ? torch::empty({batch_size}, torch::kInt)
                                   : torch::Tensor();
  int64_t* ids_data = host_values.data_ptr<int64_t>();
  int32_t* counts_data = host_scalars.data_ptr<int32_t>();
  if (!host_projection) {
    counts_data += 2 * value_count;
  }
  int32_t* lengths_data = host_projection ? host_lengths.data_ptr<int32_t>()
                                          : counts_data + value_count;
  if (!host_projection && scalar_count % 2 != 0) {
    lengths_data[batch_size] = 0;
  }
  rows_.clear();
  rows_.reserve(batch_size);
  for (int64_t row = 0; row < batch_size; ++row) {
    const auto& history = *histories[row];
    const int64_t length = static_cast<int64_t>(history.ids().size());
    const int32_t last_count = length == 0 ? 0 : history.counts().back();
    lengths_data[row] = static_cast<int32_t>(length);
    rows_.emplace_back(
        ProjectedRow{histories[row], history.revision(), length, last_count});
    if (capacity == 0) {
      continue;
    }
    int64_t* row_ids = ids_data + row * capacity;
    int32_t* row_counts = counts_data + row * capacity;
    std::copy(history.ids().begin(), history.ids().end(), row_ids);
    std::copy(history.counts().begin(), history.counts().end(), row_counts);
    // Padding repeats the last pair so penalties remain unchanged. Empty
    // histories use zeroes; a zero-capacity batch never touches data
    // pointers.
    std::fill(row_ids + length,
              row_ids + capacity,
              length == 0 ? 0 : history.ids().back());
    std::fill(row_counts + length, row_counts + capacity, last_count);
  }
  if (host_projection) {
    ids_ = std::move(host_values);
    counts_ = std::move(host_scalars);
    lengths_ = std::move(host_lengths);
  } else {
    // One pinned async upload owns all three non-overlapping contiguous
    // views; the queue is ordered, so the payload is consumed only after
    // the copy enqueued on the same stream.
    const torch::Tensor payload = host_values.to(device, /*non_blocking=*/true);
    const torch::Tensor scalars = payload.view(torch::kInt);
    ids_ = payload.narrow(/*dim=*/0, /*start=*/0, value_count)
               .view({batch_size, capacity});
    counts_ = scalars.narrow(/*dim=*/0, 2 * value_count, value_count)
                  .view({batch_size, capacity});
    lengths_ = scalars.narrow(/*dim=*/0, 3 * value_count, batch_size);
  }
}

void MtpHistoryBatch::apply_incremental_update(
    const MtpTokenHistories& histories,
    const torch::Device& device,
    int64_t batch_size) {
  const int64_t capacity = ids_.size(1);
  int64_t changed_count = 0;
  int64_t new_count = 0;
  int64_t growing_count = 0;
  for (int64_t row = 0; row < batch_size; ++row) {
    const auto& history = *histories[row];
    if (rows_[row].revision != history.revision()) {
      changed_count +=
          rows_[row].revision + 1 == history.revision()
              ? static_cast<int64_t>(history.changed_columns().size())
              : static_cast<int64_t>(history.ids().size());
      const int64_t added =
          static_cast<int64_t>(history.ids().size()) - rows_[row].length;
      new_count += added;
      growing_count += static_cast<int64_t>(added > 0);
    }
  }
  if (changed_count > 0) {
    // Counts need every changed index; immutable IDs need only new columns.
    // Put new columns first so both updates share one index array.
    // Amortize indexed update setup; small groups retain scalar writes.
    constexpr int64_t kMinBatchedLengthUpdates = 4;
    const int64_t length_update_count =
        growing_count >= kMinBatchedLengthUpdates ? growing_count : 0;
    const int64_t value_count = changed_count + new_count + length_update_count;
    const int64_t scalar_count = changed_count + length_update_count;
    const int64_t scalar_words = (scalar_count + 1) / 2;
    const int64_t payload_words = value_count + scalar_words;
    torch::Tensor host_payload;
    if (device.is_cpu()) {
      if (!host_payload_scratch_.defined() ||
          host_payload_scratch_.numel() < payload_words) {
        host_payload_scratch_ = torch::empty(
            {payload_words},
            torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
      }
      host_payload = host_payload_scratch_.narrow(/*dim=*/0, 0, payload_words);
    } else {
      host_payload = torch::empty({payload_words},
                                  torch::TensorOptions()
                                      .dtype(torch::kLong)
                                      .device(torch::kCPU)
                                      .pinned_memory(true));
    }
    // Native int32 counts and lengths share the aligned payload. Initialize
    // the unused half-word when the number of scalar values is odd.
    const torch::Tensor host_scalars = host_payload.view(torch::kInt);
    int64_t* values_data = host_payload.data_ptr<int64_t>();
    int32_t* scalar_data = host_scalars.data_ptr<int32_t>() + 2 * value_count;
    if (scalar_count % 2 != 0) {
      scalar_data[scalar_count] = 0;
    }
    int64_t next_new = 0;
    int64_t next_existing = new_count;
    int64_t next_length = 0;
    PaddingRunWriter id_padding(ids_);
    PaddingRunWriter count_padding(counts_);
    for (int64_t row = 0; row < batch_size; ++row) {
      const auto& history = *histories[row];
      auto& projected = rows_[row];
      if (projected.revision == history.revision()) {
        continue;
      }
      const auto add_column = [&](int64_t column) {
        const bool is_new = column >= projected.length;
        const int64_t offset = is_new ? next_new++ : next_existing++;
        values_data[offset] = row * capacity + column;
        scalar_data[offset] = history.counts()[column];
        if (is_new) {
          values_data[changed_count + offset] = history.ids()[column];
        }
      };
      if (projected.revision + 1 == history.revision()) {
        for (int64_t column : history.changed_columns()) {
          add_column(column);
        }
      } else {
        for (int64_t column = 0;
             column < static_cast<int64_t>(history.ids().size());
             ++column) {
          add_column(column);
        }
      }
      const int64_t length = static_cast<int64_t>(history.ids().size());
      const int32_t last_count = length == 0 ? 0 : history.counts().back();
      const bool length_changed = projected.length != length;
      if (length_changed) {
        if (length_update_count > 0) {
          values_data[changed_count + new_count + next_length] = row;
          scalar_data[changed_count + next_length] =
              static_cast<int32_t>(length);
          ++next_length;
        } else {
          lengths_.select(0, row).fill_(length);
        }
      }
      // Histories only append, so an unchanged length also preserves the last
      // token ID. Compare counts with the applied row even across revisions.
      if (length < capacity) {
        if (length_changed) {
          id_padding.add(row, length, history.ids().back());
        }
        if (length_changed || projected.last_count != last_count) {
          count_padding.add(row, length, static_cast<int64_t>(last_count));
        }
      }
      projected.revision = history.revision();
      projected.length = length;
      projected.last_count = last_count;
    }
    id_padding.flush();
    count_padding.flush();
    CHECK_EQ(next_new, new_count);
    CHECK_EQ(next_existing, changed_count);
    CHECK_EQ(next_length, length_update_count);
    // Device updates use one owned pinned async upload; CPU updates keep
    // ordinary host storage. Queued index_copy operations retain the payload.
    const torch::Tensor payload =
        host_payload.to(device, /*non_blocking=*/true);
    const torch::Tensor scalars =
        device.is_cpu() ? host_scalars : payload.view(torch::kInt);
    if (length_update_count > 0) {
      lengths_.index_copy_(
          /*dim=*/0,
          payload.narrow(
              /*dim=*/0, changed_count + new_count, length_update_count),
          scalars.narrow(
              /*dim=*/0, 2 * value_count + changed_count, length_update_count));
    }
    const torch::Tensor indices =
        payload.narrow(/*dim=*/0, /*start=*/0, changed_count);
    if (new_count > 0) {
      ids_.view({-1}).index_copy_(
          /*dim=*/0,
          indices.narrow(/*dim=*/0, /*start=*/0, new_count),
          payload.narrow(/*dim=*/0, changed_count, new_count));
    }
    counts_.view({-1}).index_copy_(
        /*dim=*/0,
        indices,
        scalars.narrow(/*dim=*/0, 2 * value_count, changed_count));
  }
}

void apply_mtp_history_batch(MtpHistoryBatch* batch,
                             SamplingParameters& params,
                             const MtpTokenHistories& histories,
                             const torch::Device& device) {
  if (batch != nullptr) {
    batch->apply(params, histories, device);
    return;
  }
  MtpHistoryBatch local_batch;
  local_batch.apply(params, histories, device);
}

}  // namespace xllm
