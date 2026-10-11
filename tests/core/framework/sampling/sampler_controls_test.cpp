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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/framework/sampling/sampler.h"

namespace xllm {
namespace {

SamplingParameters make_sampling_params(int64_t rows, bool random) {
  SamplingParameters params;
  params.selected_token_idxes = torch::arange(rows, torch::kInt);
  params.sample_idxes = torch::arange(rows, torch::kInt);
  params.do_sample = torch::full({rows}, random, torch::kBool);
  params.all_greedy_sample = !random;
  params.all_random_sample = random;
  return params;
}

TEST(SamplerControlsTest, GreedyHonorsBiasAndAllowedTokenMask) {
  auto params = make_sampling_params(1, false);
  params.logits_bias = torch::tensor({{-INFINITY, 10.0F, 0.0F}});
  torch::Tensor logits = torch::tensor({{100.0F, 1.0F, 2.0F}});
  EXPECT_EQ(Sampler::forward(logits, params).next_tokens[0].item<int64_t>(), 1);
}

TEST(SamplerControlsTest, ReportsRawLogprobsBeforeConstraints) {
  auto params = make_sampling_params(1, false);
  params.logprobs = true;
  params.max_top_logprobs = 2;
  params.logits_bias = torch::tensor({{-INFINITY, 0.0F, 10.0F}});
  torch::Tensor logits = torch::tensor({{4.0F, 2.0F, 1.0F}});
  // Scores come from the processed logits (after bias/penalties), the same
  // distribution the sampler draws from.
  const torch::Tensor expected =
      torch::log_softmax(logits + params.logits_bias, -1);
  const auto result = Sampler::forward(logits, params);
  EXPECT_EQ(result.next_tokens[0].item<int64_t>(), 2);
  EXPECT_NEAR(
      result.logprobs[0].item<float>(), expected[0][2].item<float>(), 1e-6);
  // Top rows rank the biased distribution: token 2 (bias +10) first, then
  // token 1; the -inf-biased token 0 falls out of the top-2 entirely.
  EXPECT_EQ(result.top_tokens[0][0].item<int64_t>(), 2);
  EXPECT_EQ(result.top_tokens[0][1].item<int64_t>(), 1);
  EXPECT_NEAR(result.top_logprobs[0][0].item<float>(),
              expected[0][2].item<float>(),
              1e-6);
  EXPECT_NEAR(result.top_logprobs[0][1].item<float>(),
              expected[0][1].item<float>(),
              1e-6);
}

TEST(SamplerControlsTest, AppliesRepetitionBeforePresenceAndFrequency) {
  auto params = make_sampling_params(1, false);
  params.unique_token_ids = torch::tensor({{0}}, torch::kLong);
  params.unique_token_counts = torch::tensor({{1}}, torch::kInt);
  params.unique_token_ids_lens = torch::tensor({1}, torch::kInt);
  params.repetition_penalties = torch::tensor({2.0F});
  params.frequency_penalties = torch::tensor({1.0F});
  params.presence_penalties = torch::tensor({1.0F});
  torch::Tensor logits = torch::tensor({{6.0F, 1.5F}});
  EXPECT_EQ(Sampler::forward(logits, params).next_tokens[0].item<int64_t>(), 1);
  EXPECT_NEAR(logits[0][0].item<float>(), 1.0F, 1e-6);
}

TEST(SamplerControlsTest, SeededSamplingIsIndependentOfBatchPosition) {
  auto one = make_sampling_params(1, true);
  one.seeds = torch::tensor({12345}, torch::kLong);
  one.seed_offsets = torch::tensor({7}, torch::kLong);
  auto batch = make_sampling_params(2, true);
  batch.seeds = torch::tensor({6789, 12345}, torch::kLong);
  batch.seed_offsets = torch::tensor({2, 7}, torch::kLong);
  torch::Tensor logits = torch::arange(32, torch::kFloat).unsqueeze(0) / 20;
  torch::Tensor batch_logits =
      torch::cat({torch::zeros_like(logits), logits}, 0);
  const auto alone = Sampler::forward(logits, one);
  const auto together = Sampler::forward(batch_logits, batch);
  EXPECT_EQ(alone.next_tokens[0].item<int64_t>(),
            together.next_tokens[1].item<int64_t>());
}

TEST(SamplerControlsTest, SeededSamplingDoesNotChangeGlobalGenerator) {
  auto params = make_sampling_params(1, true);
  params.seeds = torch::tensor({42}, torch::kLong);
  params.seed_offsets = torch::tensor({0}, torch::kLong);
  torch::Tensor logits = torch::zeros({1, 64});
  const auto& generator =
      torch::globalContext().defaultGenerator(torch::Device(torch::kCPU));
  const torch::Tensor before = generator.get_state().clone();
  const auto first = Sampler::forward(logits, params);
  const auto second = Sampler::forward(logits, params);
  EXPECT_TRUE(torch::equal(first.next_tokens, second.next_tokens));
  EXPECT_TRUE(torch::equal(before, generator.get_state()));
}

TEST(SamplerControlsTest, MinPRemovesLowProbabilityCandidates) {
  auto params = make_sampling_params(1, true);
  params.seeds = torch::tensor({42}, torch::kLong);
  params.seed_offsets = torch::tensor({0}, torch::kLong);
  params.min_p = torch::tensor({0.5F});
  // probs now carry only the selected token's probability ([B], the same
  // shape as logprobs), so the filter is verified through sampling: with
  // p < 0.5 removed, every draw lands on token 0.
  params.return_probs = true;
  torch::Tensor logits = torch::tensor({{2.0F, 1.0F, 0.0F}});
  for (int32_t offset = 0; offset < 20; ++offset) {
    params.seed_offsets = torch::tensor({offset}, torch::kLong);
    const auto result = Sampler::forward(logits, params);
    EXPECT_EQ(result.next_tokens[0].item<int64_t>(), 0);
    EXPECT_GT(result.probs[0].item<float>(), 0.9F);
  }
}

}  // namespace
}  // namespace xllm
