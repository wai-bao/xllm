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

#include <tuple>

#include "qwen2_5_vision_layer.h"

namespace xllm {
namespace layer {

class Qwen2_VisionPatchEmbedImpl final : public torch::nn::Module {
 public:
  explicit Qwen2_VisionPatchEmbedImpl(const ModelContext& context);

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    return proj_->forward(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict);

 private:
  torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(Qwen2_VisionPatchEmbed);

class Qwen2_VisionRotaryEmbeddingImpl final : public torch::nn::Module {
 public:
  explicit Qwen2_VisionRotaryEmbeddingImpl(const ModelContext& context);
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& grid_thw);

 private:
  void update_freqs_cache(int64_t seqlen);

  const int64_t dim_;
  const int64_t spatial_merge_size_;
  int64_t seq_len_cached_ = 0;
  torch::Tensor inv_freq_;
  torch::Tensor freqs_cached_;
};
TORCH_MODULE(Qwen2_VisionRotaryEmbedding);

torch::Tensor build_vision_cu_seqlens(const torch::Tensor& grid_thw);

torch::nn::Sequential build_vision_merger_mlp(
    int64_t hidden_size,
    int64_t output_size,
    const torch::TensorOptions& options);
void load_vision_merger_mlp(const StateDict& state_dict,
                            const torch::nn::Sequential& mlp,
                            const std::string& module_name);

class Qwen2_VisionLayerImpl : public Qwen2_5_VisionLayerImpl {
 public:
  Qwen2_VisionLayerImpl(const ModelContext& context);
  void load_state_dict(const StateDict& state_dict);
};
TORCH_MODULE(Qwen2_VisionLayer);

}  // namespace layer
}  // namespace xllm
