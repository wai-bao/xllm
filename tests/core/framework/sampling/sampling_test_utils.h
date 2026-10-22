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
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/framework/sampling/rejection_sampler.h"
#include "core/framework/sampling/sampling_params.h"

namespace xllm {
namespace sampling_test_utils {

// All-greedy sampler; SamplingParameters defaults to all-greedy.
inline RejectionSampler make_greedy_rejection_sampler() {
  const SamplingParameters params;
  return RejectionSampler(params, /*logprobs=*/false, /*max_top_logprobs=*/0);
}

// Derive SamplingMode from a do_sample tensor with a single host transfer.
inline SamplingMode sampling_mode_from_do_sample(
    const torch::Tensor& do_sample) {
  const torch::Tensor mask = do_sample.to(torch::kCPU);
  const bool any = mask.any().item<bool>();
  const bool all = mask.all().item<bool>();
  if (all) {
    return SamplingMode::RANDOM;
  }
  return any ? SamplingMode::MIXED : SamplingMode::GREEDY;
}

// SamplingParameters carrying the given do_sample and the matching mode.
inline SamplingParameters make_do_sample_params(
    const torch::Tensor& do_sample) {
  SamplingParameters params;
  params.do_sample = do_sample;
  params.sampling_mode = sampling_mode_from_do_sample(do_sample);
  return params;
}

}  // namespace sampling_test_utils
}  // namespace xllm
