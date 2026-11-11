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

#include "core/framework/speculative/draft_sampling_history.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "core/framework/speculative/padded_token_row_scan.h"
#include "core/util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {

namespace {

struct HistoryImportRow {
  int64_t index = 0;
  int64_t length = 0;
  int64_t offset = 0;
};

// Coalesces padding fills: consecutive rows sharing a start column and fill
// value collapse into one strided rectangle fill. Bind the target once,
// add() each padded row in order, then flush() the trailing run.
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
  // Full coverage has neither padding nor established histories to exclude;
  // keep the original layout until the host copy normalizes strided rows.
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
                             DraftTokenHistories& histories) {
  const int64_t* ids = params.unique_token_ids().const_data_ptr<int64_t>();
  const Count* counts = params.unique_token_counts().const_data_ptr<Count>();
  const int64_t ids_stride = params.unique_token_ids().stride(/*dim=*/0);
  const int64_t counts_stride = params.unique_token_counts().stride(/*dim=*/0);
  for (const HistoryImportRow& row : rows) {
    // A zero-length row yields an empty span, so no separate empty-row case
    // is needed (matching the device path).
    const size_t length = static_cast<size_t>(row.length);
    histories[row.index] = DraftTokenHistory::from_host_values(
        std::span<const int64_t>(ids + row.index * ids_stride, length),
        std::span<const Count>(counts + row.index * counts_stride, length));
  }
}

void import_history_batch(const SamplingParameters& params,
                          int64_t batch_size,
                          DraftTokenHistories& histories) {
  const torch::Tensor lengths_cpu =
      to_cpu_contiguous(params.unique_token_ids_lens(), torch::kLong).flatten();
  const int64_t* lengths = lengths_cpu.const_data_ptr<int64_t>();
  std::vector<HistoryImportRow> rows;
  rows.reserve(batch_size);
  int64_t total_tokens = 0;
  for (int64_t row = 0; row < batch_size; ++row) {
    if (histories[row] != nullptr) {
      continue;
    }
    const int64_t length = lengths[row];
    CHECK_GE(length, 0);
    CHECK_LE(length, params.unique_token_ids().size(1));
    rows.emplace_back(HistoryImportRow{row, length, total_tokens});
    total_tokens += length;
  }
  if (total_tokens == 0) {
    for (const HistoryImportRow& row : rows) {
      histories[row.index] = std::make_shared<DraftTokenHistory>();
    }
    return;
  }

  const torch::ScalarType count_type =
      params.unique_token_counts().scalar_type();
  const bool native_cpu_rows =
      params.unique_token_ids().device().is_cpu() &&
      params.unique_token_counts().device().is_cpu() &&
      params.unique_token_ids().scalar_type() == torch::kLong &&
      params.unique_token_ids().stride(/*dim=*/1) == 1 &&
      params.unique_token_counts().stride(/*dim=*/1) == 1 &&
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
  // select both fields, then a single host copy. Complete matrices need no
  // device pack; other layouts copy only missing rows' valid prefixes.
  const torch::Tensor packed_ids =
      pack_history_prefixes(params.unique_token_ids(), rows, total_tokens);
  const torch::Tensor packed_counts =
      pack_history_prefixes(params.unique_token_counts(), rows, total_tokens);
  const torch::Tensor ids_cpu = to_cpu_contiguous(packed_ids, torch::kLong);
  const torch::Tensor counts_cpu =
      to_cpu_contiguous(packed_counts, torch::kLong);
  CHECK_EQ(ids_cpu.numel(), total_tokens);
  CHECK_EQ(counts_cpu.numel(), total_tokens);
  const std::span<const int64_t> ids_values(ids_cpu.const_data_ptr<int64_t>(),
                                            static_cast<size_t>(total_tokens));
  const std::span<const int64_t> counts_values(
      counts_cpu.const_data_ptr<int64_t>(), static_cast<size_t>(total_tokens));
  for (const HistoryImportRow& row : rows) {
    const size_t offset = static_cast<size_t>(row.offset);
    const size_t length = static_cast<size_t>(row.length);
    histories[row.index] = DraftTokenHistory::from_host_values(
        ids_values.subspan(offset, length),
        counts_values.subspan(offset, length));
  }
}

}  // namespace

namespace {

// Wire layout shared by rebuild_projection and apply_incremental_update:
// int64 values followed by int32 scalars packed two-per-word, with the
// trailing half-word zeroed when the scalar count is odd so the device-side
// gather reads a defined value.
struct PackedHistoryPayload {
  PackedHistoryPayload(int64_t value_count,
                       int64_t scalar_count,
                       const torch::TensorOptions& options)
      : payload(torch::empty(
            {value_count + xllm::util::ceil_div(scalar_count, int64_t{2})},
            options)),
        values(payload.data_ptr<int64_t>()),
        scalars(payload.view(torch::kInt).data_ptr<int32_t>() +
                2 * value_count) {
    if (scalar_count % 2 != 0) {
      scalars[scalar_count] = 0;
    }
  }

  torch::Tensor payload;
  int64_t* values;
  int32_t* scalars;
};

}  // namespace

template <typename Count>
std::shared_ptr<DraftTokenHistory> DraftTokenHistory::from_host_values_impl(
    std::span<const int64_t> ids,
    std::span<const Count> counts) {
  CHECK_EQ(ids.size(), counts.size());
  const int64_t length = static_cast<int64_t>(ids.size());
  auto history = std::make_shared<DraftTokenHistory>();
  history->ids_.reserve(length);
  history->counts_.reserve(length);
  for (int64_t column = 0; column < length; ++column) {
    const int64_t token = ids[column];
    const int64_t count = static_cast<int64_t>(counts[column]);
    CHECK(detail::is_storable_token_id(token))
        << "sampling history token id is not storable as int32";
    CHECK_GE(count, 0);
    CHECK_LE(count, std::numeric_limits<int32_t>::max());
    CHECK(!history->indices_.contains(static_cast<int32_t>(token)))
        << "sampling history contains duplicate token ids";
    history->indices_.emplace(static_cast<int32_t>(token), column);
    history->ids_.emplace_back(token);
    history->counts_.emplace_back(static_cast<int32_t>(count));
  }
  return history;
}

std::pair<int64_t, bool> DraftTokenHistory::upsert_column(int32_t key,
                                                          int64_t token) {
  const int64_t inserted_sentinel = static_cast<int64_t>(ids_.size());
  int64_t& column_ref = indices_.emplace(key, inserted_sentinel);
  if (column_ref == inserted_sentinel) {
    CHECK_LT(ids_.size(),
             static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    ids_.emplace_back(token);
    counts_.emplace_back(0);
    return {inserted_sentinel, true};
  }
  return {column_ref, false};
}

void DraftTokenHistory::append(std::span<const int64_t> tokens) {
  if (tokens.empty()) {
    return;
  }
  changed_columns_.clear();
  changed_columns_.reserve(tokens.size());
  for (int64_t token : tokens) {
    CHECK(detail::is_storable_token_id(token));
    const auto [column, inserted] =
        upsert_column(static_cast<int32_t>(token), token);
    int32_t& count = counts_[column];
    CHECK_LT(count, std::numeric_limits<int32_t>::max());
    ++count;
    changed_columns_.emplace_back(column);
  }
  // A repeated committed token must produce one device write of its final
  // count.
  std::sort(changed_columns_.begin(), changed_columns_.end());
  changed_columns_.erase(
      std::unique(changed_columns_.begin(), changed_columns_.end()),
      changed_columns_.end());
  ++revision_;
}

void import_sampling_histories(const SamplingParameters& params,
                               int64_t batch_size,
                               DraftTokenHistories& histories) {
  if (histories.empty()) {
    histories.resize(batch_size);
  }
  CHECK_EQ(histories.size(), batch_size);
  if (std::all_of(histories.begin(), histories.end(), [](const auto& history) {
        return history != nullptr;
      })) {
    return;
  }
  if (!params.unique_token_ids().defined()) {
    CHECK(!params.unique_token_counts().defined());
    CHECK(!params.unique_token_ids_lens().defined());
    for (auto& history : histories) {
      if (history == nullptr) {
        history = std::make_shared<DraftTokenHistory>();
      }
    }
    return;
  }
  CHECK(params.unique_token_counts().defined());
  CHECK(params.unique_token_ids_lens().defined());
  CHECK_EQ(params.unique_token_ids().dim(), 2);
  CHECK(params.unique_token_ids().sizes() ==
        params.unique_token_counts().sizes());
  CHECK_EQ(params.unique_token_ids().size(0), batch_size);
  CHECK_EQ(params.unique_token_ids_lens().numel(), batch_size);
  // The coalesced import short-circuits a fully-covered matrix, so it never
  // costs more than the per-row path; at least one row is missing here.
  import_history_batch(params, batch_size, histories);
}

void DraftHistoryBatch::clear() {
  rows_.clear();
  ids_ = torch::Tensor();
  counts_ = torch::Tensor();
  lengths_ = torch::Tensor();
}

void DraftHistoryBatch::reclaim_unused_projection() {
  if (std::none_of(rows_.begin(), rows_.end(), [](const ProjectedRow& row) {
        return !row.history.expired();
      })) {
    clear();
  }
}

void DraftHistoryBatch::apply(SamplingParameters& params,
                              const DraftTokenHistories& histories,
                              const torch::Device& device) {
  const int64_t batch_size = static_cast<int64_t>(histories.size());
  if (batch_size == 0) {
    clear();
    clear_penalty_history_fields(params);
    return;
  }
  bool same_batch = ids_.defined() && ids_.device() == device &&
                    rows_.size() == histories.size();
  int64_t width = 0;
  for (int64_t row = 0; row < batch_size; ++row) {
    CHECK(histories[row] != nullptr);
    width = std::max(width, static_cast<int64_t>(histories[row]->ids().size()));
    same_batch = same_batch &&
                 !rows_[row].history.owner_before(histories[row]) &&
                 !histories[row].owner_before(rows_[row].history);
  }
  const bool rebuild = !same_batch || width > ids_.size(1);
  if (rebuild) {
    rebuild_projection(histories, device, batch_size, width, same_batch);
  } else {
    apply_incremental_update(histories, device, batch_size);
  }
  params.unique_token_ids() = ids_;
  params.unique_token_counts() = counts_;
  params.unique_token_ids_lens() = lengths_;
}

void DraftHistoryBatch::rebuild_projection(const DraftTokenHistories& histories,
                                           const torch::Device& device,
                                           int64_t batch_size,
                                           int64_t width,
                                           bool same_batch) {
  // Geometric growth only for a stable batch; rebatching sheds old capacity.
  const int64_t capacity =
      same_batch ? std::max(width, ids_.size(1) * 2) : width;
  const int64_t value_count = batch_size * capacity;
  const int64_t scalar_count = value_count + batch_size;
  // CPU projections need no upload; device projections pack int64 IDs
  // followed by int32 counts and lengths (PackedHistoryPayload owns the
  // shared wire layout).
  const bool host_projection = device.is_cpu();
  torch::Tensor host_values =
      host_projection ? torch::empty({batch_size, capacity}, torch::kLong)
                      : torch::Tensor();
  torch::Tensor host_scalars =
      host_projection ? torch::empty({batch_size, capacity}, torch::kInt)
                      : torch::Tensor();
  torch::Tensor host_lengths = host_projection
                                   ? torch::empty({batch_size}, torch::kInt)
                                   : torch::Tensor();
  int64_t* ids_data = nullptr;
  int32_t* counts_data = nullptr;
  int32_t* lengths_data = nullptr;
  if (host_projection) {
    ids_data = host_values.data_ptr<int64_t>();
    counts_data = host_scalars.data_ptr<int32_t>();
    lengths_data = host_lengths.data_ptr<int32_t>();
  } else {
    PackedHistoryPayload packed(
        value_count, scalar_count, pinned_cpu_options(torch::kLong));
    host_values = std::move(packed.payload);
    ids_data = packed.values;
    counts_data = packed.scalars;
    lengths_data = counts_data + value_count;
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
    // Padding repeats the last id with a zero count: penalty kernels
    // scatter-add over the full capacity, so a zero count is an additive
    // no-op.
    std::fill(row_ids + length,
              row_ids + capacity,
              length == 0 ? 0 : history.ids().back());
    std::fill(row_counts + length, row_counts + capacity, 0);
  }
  if (host_projection) {
    ids_ = std::move(host_values);
    counts_ = std::move(host_scalars);
    lengths_ = std::move(host_lengths);
  } else {
    // One pinned async upload owns all three non-overlapping contiguous
    // views; the ordered queue consumes the payload only after the copy
    // enqueued on the same stream.
    const torch::Tensor payload = host_values.to(device, /*non_blocking=*/true);
    const torch::Tensor scalars = payload.view(torch::kInt);
    ids_ = payload.narrow(/*dim=*/0, /*start=*/0, value_count)
               .view({batch_size, capacity});
    counts_ = scalars.narrow(/*dim=*/0, 2 * value_count, value_count)
                  .view({batch_size, capacity});
    lengths_ = scalars.narrow(/*dim=*/0, 3 * value_count, batch_size);
  }
}

void DraftHistoryBatch::apply_incremental_update(
    const DraftTokenHistories& histories,
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
    // Counts need every changed index; IDs need only new columns, so new
    // columns come first and both updates share one index array.
    constexpr int64_t kMinBatchedLengthUpdates = 4;
    const int64_t length_update_count =
        growing_count >= kMinBatchedLengthUpdates ? growing_count : 0;
    const int64_t value_count = changed_count + new_count + length_update_count;
    const int64_t scalar_count = changed_count + length_update_count;
    // A fresh pinned tensor per update: a reused member scratch's next host
    // write is invisible to the caching allocator's stream tracking and can
    // race the previous round's in-flight non_blocking H2D.
    PackedHistoryPayload packed(value_count,
                                scalar_count,
                                device.is_cpu()
                                    ? pageable_cpu_options(torch::kLong)
                                    : pinned_cpu_options(torch::kLong));
    torch::Tensor host_payload = std::move(packed.payload);
    int64_t* values_data = packed.values;
    int32_t* scalar_data = packed.scalars;
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
          count_padding.add(row, length, 0);
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
    // ordinary host storage. Queued index_copy operations retain the payload,
    // and the scalar columns are read through the int32 view of the packed
    // wire layout.
    const torch::Tensor payload =
        host_payload.to(device, /*non_blocking=*/true);
    const torch::Tensor& source = device.is_cpu() ? host_payload : payload;
    const torch::Tensor scalars = source.view(torch::kInt);
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

std::shared_ptr<DraftTokenHistory> DraftTokenHistory::from_host_values(
    std::span<const int64_t> ids,
    std::span<const int64_t> counts) {
  return from_host_values_impl(ids, counts);
}

std::shared_ptr<DraftTokenHistory> DraftTokenHistory::from_host_values(
    std::span<const int64_t> ids,
    std::span<const int32_t> counts) {
  return from_host_values_impl(ids, counts);
}

template std::shared_ptr<DraftTokenHistory>
DraftTokenHistory::from_host_values_impl(std::span<const int64_t> ids,
                                         std::span<const int64_t> counts);
template std::shared_ptr<DraftTokenHistory>
DraftTokenHistory::from_host_values_impl(std::span<const int64_t> ids,
                                         std::span<const int32_t> counts);

}  // namespace xllm
