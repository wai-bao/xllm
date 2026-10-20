/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/sampling/gumbel_sampling.h"

#include <glog/logging.h>

#include <limits>

namespace xllm {

namespace {

// Gumbel(0, 1) = -log(-log(U)), U ~ Uniform(0, 1); the clamp guards -log(0).
// Entirely in place: zero extra allocations.
torch::Tensor gumbel_from_uniform(torch::Tensor uniform) {
  uniform.clamp_min_(std::numeric_limits<float>::min());
  return uniform.log_().neg_().log_().neg_();
}

}  // namespace

torch::Tensor gumbel_noise_like(const torch::Tensor& like,
                                std::optional<torch::Generator> generator) {
  torch::TensorOptions options = like.options().dtype(torch::kFloat32);
  return gumbel_from_uniform(torch::rand(like.sizes(), generator, options));
}

torch::Tensor greedy_draft_rows(const SamplingParameters& sampling_params) {
  CHECK(sampling_params.do_sample.defined());
  // In GREEDY mode do_sample may carry mixed values from the request (gumbel
  // noise and the sampling-group synchronize still branch on it), so the
  // all-greedy shortcut treats every row as greedy.
  if (sampling_params.all_greedy_sample) {
    return torch::ones({sampling_params.do_sample.size(0)}, torch::kBool);
  }
  return sampling_params.do_sample.logical_not();
}

torch::Tensor sample_gumbel_noise(int64_t num_steps,
                                  int64_t num_classes,
                                  const torch::Tensor& greedy_rows,
                                  bool all_greedy,
                                  const torch::Device& device) {
  CHECK(greedy_rows.defined());
  CHECK_EQ(greedy_rows.dim(), 1);
  const int64_t batch_size = greedy_rows.size(0);
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_steps, 0);
  CHECK_GT(num_classes, 0);
  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  if (all_greedy) {
    return torch::zeros({batch_size, num_steps, num_classes}, float_options);
  }
  torch::Tensor gumbel = gumbel_from_uniform(
      torch::rand({batch_size, num_steps, num_classes}, float_options));
  gumbel.masked_fill_(greedy_rows.to(device).view({batch_size, 1, 1}), 0.0);
  return gumbel;
}

void apply_selector_temperatures(torch::Tensor& logits,
                                 const torch::Tensor& temperatures,
                                 int64_t batch_size) {
  CHECK(temperatures.defined());
  CHECK_EQ(temperatures.dim(), 1);
  CHECK_EQ(temperatures.size(0), batch_size);
  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(logits.device());
  torch::Tensor scaled =
      temperatures.to(float_options).view({batch_size, 1, 1, 1});
  scaled = scaled.masked_fill(scaled.eq(0), 1.0);
  logits.div_(scaled);
}

torch::Tensor gumbel_argmax(const torch::Tensor& logits,
                            const torch::Tensor& gumbel_noise) {
  return (logits + gumbel_noise).argmax(/*dim=*/-1);
}

}  // namespace xllm
