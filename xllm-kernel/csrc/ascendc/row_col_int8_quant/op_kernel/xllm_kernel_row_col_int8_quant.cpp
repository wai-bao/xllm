// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include "adv_api/reduce/reduce.h"
#include "kernel_operator.h"
#include "row_col_int8_quant_tiling.h"

namespace {
using namespace AscendC;
constexpr uint32_t kColumns = 64;
constexpr uint32_t kRows = 64;
constexpr uint32_t kMaxVector = 4096;
constexpr uint32_t kFloatsPerBlock = 8;
constexpr uint32_t kFloatsPerRepeat = 64;

__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

class RowColQuant {
 public:
  __aicore__ inline void Init(GM_ADDR x,
                              GM_ADDR q_row,
                              GM_ADDR q_col,
                              GM_ADDR row_max,
                              GM_ADDR col_max,
                              const RowColInt8QuantTilingData& tiling) {
    tiling_ = tiling;
    const uint32_t elements = tiling_.rows * tiling_.cols;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), elements);
    q_row_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(q_row), elements);
    q_col_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(q_col), elements);
    row_max_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(row_max),
                             tiling_.rows);
    col_max_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(col_max),
                             tiling_.cols);
    pipe_.InitBuffer(input_queue_, 1, kMaxVector * sizeof(float));
    pipe_.InitBuffer(output_queue_, 1, kMaxVector);
    pipe_.InitBuffer(value_, kMaxVector * sizeof(float));
    pipe_.InitBuffer(abs_, kMaxVector * sizeof(float));
    pipe_.InitBuffer(reduce_, kMaxVector * sizeof(float));
    pipe_.InitBuffer(rounded_, kMaxVector * sizeof(int16_t));
    pipe_.InitBuffer(half_, kMaxVector * sizeof(half));
    pipe_.InitBuffer(max_, kColumns * sizeof(float));
    pipe_.InitBuffer(partial_, kColumns * sizeof(float));
    pipe_.InitBuffer(scale_, kColumns * sizeof(float));
    pipe_.InitBuffer(numerator_, kColumns * sizeof(float));
    pipe_.InitBuffer(column_scale_, kMaxVector * sizeof(float));
  }

  __aicore__ inline void Run() {
    const uint32_t columns = (tiling_.cols + kColumns - 1) / kColumns;
    const uint32_t rows = tiling_.rows;
    ReduceAcrossCores(columns, rows);

    const uint32_t group = tiling_.rows_per_group;
    const uint32_t groups = (rows + group - 1) / group;
    const uint32_t start = Boundary(groups, GetBlockIdx()) * group;
    const uint32_t end = Min(rows, Boundary(groups, GetBlockIdx() + 1) * group);
    if (start < end) {
      LoadColumnScales();
    }
    const uint32_t cols = tiling_.cols;
    if (cols <= 512 && cols % kFloatsPerRepeat == 0 && rows >= kRows) {
      const uint32_t tile_rows = kMaxVector / cols;
      for (uint32_t row = start; row < end; row += tile_rows) {
        QuantizeAlignedRows(row, Min(tile_rows, end - row));
      }
    } else {
      for (uint32_t row = start; row < end; ++row) {
        QuantizeRow(row);
      }
    }
  }

 private:
  __aicore__ inline void ReduceAcrossCores(uint32_t columns, uint32_t rows) {
    const uint32_t first_tile = Boundary(columns, GetBlockIdx());
    const uint32_t end_tile = Boundary(columns, GetBlockIdx() + 1);
    // The unwritten row maxima hold the partials until the cores finish
    // merging.
    const uint32_t shards =
        Min(GetBlockNum() / columns, rows / (columns * kColumns));
    if (shards > 1) {
      const uint32_t worker = GetBlockIdx();
      if (worker < columns * shards) {
        const uint32_t tile = worker % columns;
        const uint32_t shard = worker / columns;
        const uint32_t first_row = rows * shard / shards;
        const uint32_t end_row = rows * (shard + 1) / shards;
        ReduceColumns(
            tile * kColumns, first_row, end_row, row_max_[worker * kColumns]);
      }
      SyncAll<true>();
      for (uint32_t tile = first_tile; tile < end_tile; ++tile) {
        MergeColumns(tile, columns, shards);
      }
    } else {
      for (uint32_t tile = first_tile; tile < end_tile; ++tile) {
        ReduceColumns(tile * kColumns, 0, rows, col_max_[tile * kColumns]);
      }
    }
    if (GetBlockNum() > 1) {
      SyncAll<true>();
    } else {
      PipeBarrier<PIPE_ALL>();
    }
  }

  __aicore__ inline uint32_t Boundary(uint32_t groups, uint32_t worker) {
    return groups * worker / GetBlockNum();
  }

  __aicore__ inline void Barrier() { PipeBarrier<PIPE_V>(); }

  __aicore__ inline void StoreMax(const GlobalTensor<float>& dst,
                                  const LocalTensor<float>& src,
                                  uint32_t count) {
    auto local = output_queue_.AllocTensor<float>();
    Adds(local, src, 0.0f, count);
    output_queue_.EnQue(local);
    local = output_queue_.DeQue<float>();
    DataCopyPad(dst, local, DataCopyExtParams{1, count * 4, 0, 0, 0});
    output_queue_.FreeTensor(local);
  }

  __aicore__ inline LocalTensor<int8_t> Round(const LocalTensor<float>& value,
                                              uint32_t count) {
    auto rounded = rounded_.Get<int16_t>();
    auto narrow = half_.Get<half>();
    auto result = output_queue_.AllocTensor<int8_t>();
    // INT32-to-FP16 would depend on the core's dequantization scale.
    Cast(rounded, value, RoundMode::CAST_RINT, count);
    Barrier();
    Cast(narrow, rounded, RoundMode::CAST_RINT, count);
    Barrier();
    Cast(result, narrow, RoundMode::CAST_RINT, count);
    output_queue_.EnQue(result);
    return output_queue_.DeQue<int8_t>();
  }

  __aicore__ inline void ReadRows(uint32_t row, uint32_t count) {
    auto local = input_queue_.AllocTensor<half>();
    DataCopyPad(local,
                x_[row * tiling_.cols],
                DataCopyExtParams{1, count * 2, 0, 0, 0},
                DataCopyPadExtParams<half>{false, 0, 0, 0});
    input_queue_.EnQue(local);
    local = input_queue_.DeQue<half>();
    Cast(value_.Get<float>(), local, RoundMode::CAST_NONE, count);
    input_queue_.FreeTensor(local);
    Barrier();
  }

  __aicore__ inline void ComputeScale(uint32_t count) {
    auto scale = scale_.Get<float>();
    auto numerator = numerator_.Get<float>();
    Maxs(scale, max_.Get<float>(), 1e-30f, count);
    Duplicate(numerator, 127.0f, count);
    Barrier();
    Div(scale, numerator, scale, count);
    Barrier();
  }

  __aicore__ inline void LoadColumnScales() {
    const uint32_t cols = tiling_.cols;
    auto loaded = input_queue_.AllocTensor<float>();
    DataCopyPad(loaded,
                col_max_,
                DataCopyExtParams{1, cols * 4, 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    input_queue_.EnQue(loaded);
    loaded = input_queue_.DeQue<float>();
    auto scale = column_scale_.Get<float>();
    auto numerator = abs_.Get<float>();
    Maxs(scale, loaded, 1e-30f, cols);
    input_queue_.FreeTensor(loaded);
    Duplicate(numerator, 127.0f, cols);
    Barrier();
    Div(scale, numerator, scale, cols);
    Barrier();
  }

  __aicore__ inline void QuantizeRow(uint32_t row) {
    const uint32_t cols = tiling_.cols;
    auto values = value_.Get<float>();
    auto magnitudes = abs_.Get<float>();
    auto max = max_.Get<float>();
    ReadRows(row, cols);
    Mul(magnitudes, values, column_scale_.Get<float>(), cols);
    Barrier();
    StoreQuantized(q_col_[row * cols], magnitudes, cols);
    Abs(magnitudes, values, cols);
    Barrier();
    ReduceMax(max, magnitudes, reduce_.Get<float>(), cols, false);
    Barrier();
    StoreMax(row_max_[row], max, 1);
    ComputeScale(1);
    const int32_t event = pipe_.FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
    const float scale = scale_.Get<float>().GetValue(0);
    Muls(values, values, scale, cols);
    Barrier();
    StoreQuantized(q_row_[row * cols], values, cols);
  }

  __aicore__ inline void StoreQuantized(const GlobalTensor<int8_t>& dst,
                                        const LocalTensor<float>& src,
                                        uint32_t count) {
    auto local = Round(src, count);
    DataCopyPad(dst, local, DataCopyExtParams{1, count, 0, 0, 0});
    output_queue_.FreeTensor(local);
  }

  __aicore__ inline void QuantizeAlignedRows(uint32_t row, uint32_t rows) {
    const uint32_t cols = tiling_.cols;
    const uint32_t count = rows * cols;
    const auto row_stride = static_cast<uint8_t>(cols / kFloatsPerBlock);
    auto values = value_.Get<float>();
    auto scratch = abs_.Get<float>();
    auto column_scale = column_scale_.Get<float>();
    ReadRows(row, count);

    // Each repeat handles one row and reads the same 64 column scales.
    const BinaryRepeatParams columns{1, 1, 1, row_stride, row_stride, 0};
    for (uint32_t col = 0; col < cols; col += kFloatsPerRepeat) {
      Mul(scratch[col],
          values[col],
          column_scale[col],
          kFloatsPerRepeat,
          rows,
          columns);
    }
    Barrier();
    StoreQuantized(q_col_[row * cols], scratch, count);

    Abs(scratch, values, count);
    Barrier();
    const uint32_t shape[2] = {rows, cols};
    auto max = max_.Get<float>();
    ReduceMax<float, Pattern::Reduce::AR>(
        max, scratch, reduce_.Get<uint8_t>(), shape, false);
    Barrier();
    StoreMax(row_max_[row], max, rows);
    ComputeScale(rows);

    // One 32-byte block per row supplies its scale to the whole repeat.
    Brcb(scratch,
         scale_.Get<float>(),
         (rows + kFloatsPerBlock - 1) / kFloatsPerBlock,
         BrcbRepeatParams{1, 8});
    Barrier();
    const BinaryRepeatParams row_scales{1, 1, 0, row_stride, row_stride, 1};
    for (uint32_t col = 0; col < cols; col += kFloatsPerRepeat) {
      Mul(values[col],
          values[col],
          scratch,
          kFloatsPerRepeat,
          rows,
          row_scales);
    }
    Barrier();
    StoreQuantized(q_row_[row * cols], values, count);
  }

  __aicore__ inline void ReadColumns(uint32_t row,
                                     uint32_t col,
                                     uint32_t rows,
                                     uint32_t width) {
    const uint32_t cols = tiling_.cols;
    auto local = input_queue_.AllocTensor<half>();
    const uint32_t local_stride =
        (kColumns * 2 - (width * 2 + 31) / 32 * 32) / 32;
    const DataCopyExtParams copy{static_cast<uint16_t>(rows),
                                 width * 2,
                                 (cols - width) * 2,
                                 local_stride,
                                 0};
    DataCopyPad(local,
                x_[row * cols + col],
                copy,
                DataCopyPadExtParams<half>{false, 0, 0, 0});
    input_queue_.EnQue(local);
    local = input_queue_.DeQue<half>();
    Cast(value_.Get<float>(), local, RoundMode::CAST_NONE, rows * kColumns);
    input_queue_.FreeTensor(local);
    Barrier();
  }

  __aicore__ inline void ReduceColumns(uint32_t col,
                                       uint32_t first_row,
                                       uint32_t end_row,
                                       const GlobalTensor<float>& result) {
    const uint32_t width = Min(kColumns, tiling_.cols - col);
    auto values = value_.Get<float>();
    auto max = max_.Get<float>();
    auto partial = partial_.Get<float>();
    auto magnitudes = abs_.Get<float>();
    Duplicate(max, 0.0f, kColumns);
    Barrier();
    for (uint32_t row = first_row; row < end_row; row += kRows) {
      const uint32_t rows = Min(kRows, end_row - row);
      ReadColumns(row, col, rows, width);
      Abs(magnitudes, values, rows * kColumns);
      Barrier();
      const uint32_t shape[2] = {rows, kColumns};
      ReduceMax<float, Pattern::Reduce::RA>(
          partial, magnitudes, reduce_.Get<uint8_t>(), shape, false);
      Barrier();
      Max(max, max, partial, width);
      Barrier();
    }
    StoreMax(result, max, width);
  }

  __aicore__ inline void MergeColumns(uint32_t tile,
                                      uint32_t columns,
                                      uint32_t shards) {
    const uint32_t col = tile * kColumns;
    const uint32_t width = Min(kColumns, tiling_.cols - col);
    const uint32_t transfer_bytes = (width * 4 + 31) / 32 * 32;
    const DataCopyExtParams copy{static_cast<uint16_t>(shards),
                                 transfer_bytes,
                                 columns * kColumns * 4 - transfer_bytes,
                                 (kColumns * 4 - transfer_bytes) / 32,
                                 0};
    auto values = input_queue_.AllocTensor<float>();
    DataCopyPad(values,
                row_max_[col],
                copy,
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    input_queue_.EnQue(values);
    values = input_queue_.DeQue<float>();
    const uint32_t shape[2] = {shards, kColumns};
    auto result = partial_.Get<float>();
    ReduceMax<float, Pattern::Reduce::RA>(
        result, values, reduce_.Get<uint8_t>(), shape, false);
    input_queue_.FreeTensor(values);
    Barrier();
    StoreMax(col_max_[col], result, width);
  }

  RowColInt8QuantTilingData tiling_;
  TPipe pipe_;
  TQue<TPosition::VECIN, 1> input_queue_;
  TQue<TPosition::VECOUT, 1> output_queue_;
  TBuf<TPosition::VECCALC> value_, abs_, reduce_, rounded_, half_;
  TBuf<TPosition::VECCALC> max_, partial_, scale_, numerator_, column_scale_;
  GlobalTensor<half> x_;
  GlobalTensor<int8_t> q_row_, q_col_;
  GlobalTensor<float> row_max_, col_max_;
};
}  // namespace

extern "C" __global__ __aicore__ void xllm_kernel_row_col_int8_quant(
    GM_ADDR x,
    GM_ADDR q_row,
    GM_ADDR q_col,
    GM_ADDR row_max,
    GM_ADDR col_max,
    GM_ADDR,
    GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(RowColInt8QuantTilingData);
  GET_TILING_DATA(tiling_data, tiling);
  RowColQuant op;
  op.Init(x, q_row, q_col, row_max, col_max, tiling_data);
  op.Run();
}
