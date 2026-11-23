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

#include "qwen2_vision_layer.h"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace xllm {
namespace layer {
namespace {

constexpr double kVisionRopeTheta = 10000.0;

}  // namespace

Qwen2_VisionPatchEmbedImpl::Qwen2_VisionPatchEmbedImpl(
    const ModelContext& context) {
  const auto& args = context.get_model_args();
  const int64_t patch_size = args.mm_patch_size();
  const int64_t in_features = args.mm_num_channels() *
                              args.mm_temporal_patch_size() * patch_size *
                              patch_size;
  proj_ =
      register_module(/*name=*/"proj",
                      torch::nn::Linear(torch::nn::LinearOptions(
                                            in_features, args.mm_hidden_size())
                                            .bias(/*new_bias=*/false)));
  proj_->weight.set_data(proj_->weight.to(context.get_tensor_options()));
}

void Qwen2_VisionPatchEmbedImpl::load_state_dict(const StateDict& state_dict) {
  auto weight = state_dict.get_tensor(/*tensor_name=*/"proj.weight");
  if (weight.defined()) {
    weight = weight.reshape({weight.size(/*dim=*/0), -1});
    DCHECK_EQ(proj_->weight.sizes(), weight.sizes())
        << "proj weight size mismatch for " << name();
    proj_->weight.data().copy_(weight);
  }
}

Qwen2_VisionRotaryEmbeddingImpl::Qwen2_VisionRotaryEmbeddingImpl(
    const ModelContext& context)
    : dim_(context.get_model_args().mm_head_dim() / 2),
      spatial_merge_size_(context.get_model_args().mm_spatial_merge_size()) {
  const auto options = context.get_tensor_options().dtype(torch::kFloat32);
  inv_freq_ = register_buffer(
      /*name=*/"inv_freq",
      1.0 / torch::pow(
                kVisionRopeTheta,
                torch::arange(/*start=*/0, dim_, /*step=*/2, options) / dim_));
}

void Qwen2_VisionRotaryEmbeddingImpl::update_freqs_cache(int64_t seqlen) {
  if (seqlen <= seq_len_cached_) {
    return;
  }

  seqlen *= 2;
  seq_len_cached_ = seqlen;
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(inv_freq_.device());
  inv_freq_ =
      1.0 /
      torch::pow(kVisionRopeTheta,
                 torch::arange(/*start=*/0, dim_, /*step=*/2, options) / dim_);
  freqs_cached_ = torch::outer(torch::arange(seqlen, options), inv_freq_);
}

std::tuple<torch::Tensor, torch::Tensor>
Qwen2_VisionRotaryEmbeddingImpl::forward(const torch::Tensor& grid_thw) {
  const int64_t count = grid_thw.size(/*dim=*/0);
  const auto grid_thw_cpu = grid_thw.cpu();
  const auto options =
      torch::TensorOptions().dtype(torch::kLong).device(grid_thw.device());

  std::vector<torch::Tensor> position_ids;
  position_ids.reserve(count);
  int64_t max_grid_size = 0;
  for (int64_t idx = 0; idx < count; ++idx) {
    const auto t = grid_thw_cpu[idx][0].item<int64_t>();
    const auto h = grid_thw_cpu[idx][1].item<int64_t>();
    const auto w = grid_thw_cpu[idx][2].item<int64_t>();
    max_grid_size = std::max({max_grid_size, h, w});

    auto hpos_ids =
        torch::arange(h, options).unsqueeze(/*dim=*/1).expand({-1, w});
    auto wpos_ids =
        torch::arange(w, options).unsqueeze(/*dim=*/0).expand({h, -1});
    const std::array<int64_t, 4> shape{h / spatial_merge_size_,
                                       spatial_merge_size_,
                                       w / spatial_merge_size_,
                                       spatial_merge_size_};
    hpos_ids = hpos_ids.reshape(shape).permute({0, 2, 1, 3}).flatten();
    wpos_ids = wpos_ids.reshape(shape).permute({0, 2, 1, 3}).flatten();
    position_ids.push_back(
        torch::stack({hpos_ids, wpos_ids}, /*dim=*/-1).repeat({t, 1}));
  }
  auto pos_ids = torch::cat(position_ids, /*dim=*/0);
  update_freqs_cache(max_grid_size);
  auto freqs = freqs_cached_.slice(/*dim=*/0, /*start=*/0, max_grid_size);
  return {freqs.index({pos_ids}).flatten(/*start_dim=*/1), std::move(pos_ids)};
}

torch::Tensor build_vision_cu_seqlens(const torch::Tensor& grid_thw) {
  auto frame_lengths =
      torch::repeat_interleave(grid_thw.select(/*dim=*/1, /*index=*/1) *
                                   grid_thw.select(/*dim=*/1, /*index=*/2),
                               grid_thw.select(/*dim=*/1, /*index=*/0));
  namespace F = torch::nn::functional;
  return F::pad(
      frame_lengths.cumsum(/*dim=*/0, torch::kInt32),
      F::PadFuncOptions({1, 0}).mode(torch::kConstant).value(/*new_value=*/0));
}

torch::nn::Sequential build_vision_merger_mlp(
    int64_t hidden_size,
    int64_t output_size,
    const torch::TensorOptions& options) {
  auto fc1 =
      torch::nn::Linear(torch::nn::LinearOptions(hidden_size, hidden_size)
                            .bias(/*new_bias=*/true));
  fc1->weight.set_data(fc1->weight.to(options));
  fc1->bias.set_data(fc1->bias.to(options));
  auto fc2 =
      torch::nn::Linear(torch::nn::LinearOptions(hidden_size, output_size)
                            .bias(/*new_bias=*/true));
  fc2->weight.set_data(fc2->weight.to(options));
  fc2->bias.set_data(fc2->bias.to(options));
  return torch::nn::Sequential(fc1, torch::nn::GELU(), fc2);
}

void load_vision_merger_mlp(const StateDict& state_dict,
                            const torch::nn::Sequential& mlp,
                            const std::string& module_name) {
  const auto load_linear = [&](const char* prefix, size_t index) {
    const auto dict = state_dict.get_dict_with_prefix(prefix);
    const auto& linear = mlp->at<torch::nn::LinearImpl>(index);
    if (auto weight = dict.get_tensor(/*tensor_name=*/"weight");
        weight.defined()) {
      CHECK_EQ(linear.weight.sizes(), weight.sizes())
          << "weight size mismatch for " << module_name;
      linear.weight.data().copy_(weight);
    }
    if (auto bias = dict.get_tensor(/*tensor_name=*/"bias"); bias.defined()) {
      CHECK_EQ(linear.bias.sizes(), bias.sizes())
          << "bias size mismatch for " << module_name;
      linear.bias.data().copy_(bias);
    }
  };
  load_linear(/*prefix=*/"mlp.0.", /*index=*/0);
  load_linear(/*prefix=*/"mlp.2.", /*index=*/2);
}

Qwen2_VisionLayerImpl::Qwen2_VisionLayerImpl(const ModelContext& context)
    : Qwen2_5_VisionLayerImpl(context, true) {}

void Qwen2_VisionLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
  mlp_->load_state_dict(
      state_dict.get_dict_with_prefix("mlp."), {"fc1."}, "fc2.");
  norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
  norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
}

}  // namespace layer
}  // namespace xllm
