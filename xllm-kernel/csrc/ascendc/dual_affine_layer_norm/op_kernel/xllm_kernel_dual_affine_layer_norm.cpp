// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include "dual_affine_layer_norm_tiling.h"
#include "kernel_operator.h"

namespace {
using namespace AscendC;

template <typename T>
class DualAffine {
 public:
  __aicore__ inline void Init(GM_ADDR x,
                              GM_ADDR g1,
                              GM_ADDR b1,
                              GM_ADDR g2,
                              GM_ADDR b2,
                              GM_ADDR y1,
                              GM_ADDR y2,
                              const DualAffineLayerNormTilingData& tiling) {
    tiling_ = tiling;
    const uint32_t width = tiling_.width;
    const uint32_t aligned = (width + 15) / 16 * 16;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), tiling_.rows * width);
    gamma_[0].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(g1), width);
    beta_[0].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b1), width);
    gamma_[1].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(g2), width);
    beta_[1].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b2), width);
    out_[0].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y1),
                            tiling_.rows * width);
    out_[1].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y2),
                            tiling_.rows * width);
    pipe_.InitBuffer(in_queue_, 1, aligned * sizeof(T));
    pipe_.InitBuffer(out_queue_, 1, aligned * sizeof(T));
    pipe_.InitBuffer(normalized_, aligned * sizeof(float));
    pipe_.InitBuffer(affine_, aligned * sizeof(float));
    pipe_.InitBuffer(parameter_, aligned * sizeof(float));
    pipe_.InitBuffer(reduce_workspace_, aligned * sizeof(float));
    pipe_.InitBuffer(scalar_, 32);
  }

  __aicore__ inline void Run() {
    const uint32_t group_rows = tiling_.rows_per_group;
    const uint32_t groups = (tiling_.rows + group_rows - 1) / group_rows;
    const uint32_t first = groups * GetBlockIdx() / GetBlockNum() * group_rows;
    const uint32_t last_group = groups * (GetBlockIdx() + 1) / GetBlockNum();
    const uint32_t limit = last_group * group_rows;
    const uint32_t last = limit < tiling_.rows ? limit : tiling_.rows;
    for (uint32_t row = first; row < last; ++row) {
      Normalize(row);
      Affine(row, 0);
      Affine(row, 1);
    }
  }

 private:
  __aicore__ inline void Read(const LocalTensor<float>& dest,
                              const GlobalTensor<T>& src) {
    const DataCopyExtParams copy{
        1, tiling_.width * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    auto typed = in_queue_.AllocTensor<T>();
    DataCopyPad(typed, src, copy, DataCopyPadExtParams<T>{false, 0, 0, 0});
    in_queue_.EnQue(typed);
    typed = in_queue_.DeQue<T>();
    Cast(dest, typed, RoundMode::CAST_NONE, tiling_.width);
    in_queue_.FreeTensor(typed);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline float ReadScalar(const LocalTensor<float>& src) {
    const int32_t event = pipe_.FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
    return src.GetValue(0);
  }

  __aicore__ inline float Sum(const LocalTensor<float>& src) {
    auto scalar = scalar_.Get<float>();
    ReduceSum(scalar, src, reduce_workspace_.Get<float>(), tiling_.width);
    return ReadScalar(scalar);
  }

  __aicore__ inline void Normalize(uint32_t row) {
    auto z = normalized_.Get<float>();
    auto square = affine_.Get<float>();
    auto scalar = scalar_.Get<float>();
    const uint32_t width = tiling_.width;
    const float reciprocal_width = tiling_.reciprocal_width;
    Read(z, x_[row * width]);
    // A power-of-two reduction preserves a repeated FP16 or BF16 value
    // exactly. Shift other widths so constant rows also center exactly.
    if ((width & (width - 1)) != 0) {
      Adds(z, z, -ReadScalar(z), width);
      PipeBarrier<PIPE_V>();
    }
    Adds(z, z, -Sum(z) * reciprocal_width, width);
    PipeBarrier<PIPE_V>();
    Mul(square, z, z, width);
    PipeBarrier<PIPE_V>();
    const float variance = Sum(square) * reciprocal_width;
    const float shifted_variance = variance + tiling_.epsilon;
    Duplicate(scalar, shifted_variance, 8);
    PipeBarrier<PIPE_V>();
    Rsqrt(scalar, scalar, 8);
    const float estimate = ReadScalar(scalar);
    const float refinement =
        estimate * (1.5f - 0.5f * shifted_variance * estimate * estimate);
    Muls(z, z, refinement, width);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Affine(uint32_t row, uint32_t branch) {
    auto parameter = parameter_.Get<float>();
    auto result = affine_.Get<float>();
    const uint32_t width = tiling_.width;
    Read(parameter, gamma_[branch]);
    Mul(result, normalized_.Get<float>(), parameter, width);
    PipeBarrier<PIPE_V>();
    Read(parameter, beta_[branch]);
    Add(result, result, parameter, width);
    PipeBarrier<PIPE_V>();
    auto typed = out_queue_.AllocTensor<T>();
    Cast(typed, result, RoundMode::CAST_RINT, width);
    out_queue_.EnQue(typed);
    typed = out_queue_.DeQue<T>();
    const DataCopyExtParams copy{
        1, width * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(out_[branch][row * width], typed, copy);
    out_queue_.FreeTensor(typed);
  }

  DualAffineLayerNormTilingData tiling_;
  TPipe pipe_;
  TQue<TPosition::VECIN, 1> in_queue_;
  TQue<TPosition::VECOUT, 1> out_queue_;
  TBuf<TPosition::VECCALC> normalized_, affine_, parameter_, reduce_workspace_,
      scalar_;
  GlobalTensor<T> x_, gamma_[2], beta_[2], out_[2];
};
}  // namespace

extern "C" __global__ __aicore__ void xllm_kernel_dual_affine_layer_norm(
    GM_ADDR x,
    GM_ADDR gamma1,
    GM_ADDR beta1,
    GM_ADDR gamma2,
    GM_ADDR beta2,
    GM_ADDR y1,
    GM_ADDR y2,
    GM_ADDR,
    GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(DualAffineLayerNormTilingData);
  GET_TILING_DATA(tiling_data, tiling);
  DualAffine<DTYPE_X> op;
  op.Init(x, gamma1, beta1, gamma2, beta2, y1, y2, tiling_data);
  op.Run();
}
