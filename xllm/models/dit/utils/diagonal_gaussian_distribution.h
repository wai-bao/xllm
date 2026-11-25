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

#include <torch/types.h>

#include <optional>
#include <vector>

#include "models/dit/utils/util.h"

namespace xllm::dit {

class DiagonalGaussianDistribution final {
 public:
  explicit DiagonalGaussianDistribution(torch::Tensor parameters,
                                        bool deterministic = false)
      : parameters_(std::move(parameters)), deterministic_(deterministic) {
    auto chunks = parameters_.chunk(2, 1);
    mean_ = chunks[0];
    logvar_ = chunks[1];

    logvar_ = torch::clamp(logvar_, -30.0f, 20.0f);

    std_ = torch::exp(0.5f * logvar_);

    if (deterministic_) {
      std_.fill_(0.0f);
    }
  }

  torch::Tensor sample(int64_t seed) const {
    torch::Generator generator =
        torch::make_generator<torch::CPUGeneratorImpl>().clone();
    generator.set_current_seed(seed);
    return sample(generator, torch::kBFloat16);
  }

  torch::Tensor sample(torch::Generator& generator,
                       torch::ScalarType noise_type = torch::kFloat32) const {
    std::vector<int64_t> shape(mean_.sizes().begin(), mean_.sizes().end());
    return mean_ +
           std_ * randn_tensor(shape, generator, mean_.options(), noise_type);
  }

  torch::Tensor mode() const { return mean_; }

  const torch::Tensor& mean() const { return mean_; }

 private:
  torch::Tensor parameters_;
  torch::Tensor mean_;
  torch::Tensor logvar_;
  torch::Tensor std_;
  bool deterministic_;
};

}  // namespace xllm::dit
