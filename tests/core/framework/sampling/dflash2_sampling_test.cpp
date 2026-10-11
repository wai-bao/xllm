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

#include "core/framework/sampling/dflash2_sampling.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/framework/model/causal_lm.h"
#include "core/framework/sampling/draft_sampling_mode.h"
#include "core/framework/sampling/sampling_params.h"

namespace xllm {
namespace {

DFlash2CandidateOutput make_candidates() {
  DFlash2CandidateOutput candidates;
  candidates.candidate_ids =
      torch::tensor({2, 5, 3, 7}, torch::kInt64).view({1, 2, 2});
  // Step zero chooses candidate 1. Step one's row 1 chooses candidate 0,
  // while row 0 would choose candidate 1, exposing a broken path recurrence.
  candidates.edge_logits =
      torch::tensor({0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 4.0f, 0.0f})
          .view({1, 2, 2, 2});
  return candidates;
}

TEST(DFlash2SamplingTest, GreedyFollowsPreviouslySelectedCandidate) {
  const auto candidates = make_candidates();
  SamplingParameters params;
  params.all_greedy_sample = true;
  params.all_random_sample = false;
  params.do_sample = torch::zeros({1}, torch::kBool);
  params.temperatures = torch::zeros({1});
  const auto result = sample_dflash2_path(candidates,
                                          params,
                                          DraftSamplingMode::PROBABILISTIC,
                                          torch::zeros({1, 2, 2}));
  EXPECT_TRUE(torch::equal(result.token_ids(),
                           torch::tensor({5, 3}, torch::kInt64).view({1, 2})));
  EXPECT_FALSE(result.logits().has_value());
}

TEST(DFlash2SamplingTest, MixedRowsPreserveExactProposalDistribution) {
  auto candidates = make_candidates();
  candidates.candidate_ids = candidates.candidate_ids.repeat({2, 1, 1});
  candidates.edge_logits = candidates.edge_logits.repeat({2, 1, 1, 1});
  SamplingParameters params;
  params.all_greedy_sample = false;
  params.all_random_sample = false;
  params.do_sample = torch::tensor({false, true}, torch::kBool);
  params.temperatures = torch::tensor({0.0f, 2.0f});
  // Fixed noise forces the random row down candidate 0 then candidate 1.
  const auto noise =
      torch::tensor({0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f})
          .view({2, 2, 2});
  const auto result = sample_dflash2_path(
      candidates, params, DraftSamplingMode::PROBABILISTIC, noise);
  EXPECT_TRUE(
      torch::equal(result.token_ids(),
                   torch::tensor({5, 3, 2, 7}, torch::kInt64).view({2, 2})));
  ASSERT_EQ(result.logits()->sizes(), torch::IntArrayRef({2, 2, 2}));
  EXPECT_TRUE(torch::equal(result.vocab_ids(), candidates.candidate_ids));
  EXPECT_TRUE(torch::allclose(result.logits()->index({1}),
                              torch::tensor({{0.0f, 1.0f}, {0.0f, 1.5f}})));
  EXPECT_TRUE(
      torch::equal(result.sampled_indices(),
                   torch::tensor({1, 0, 0, 1}, torch::kLong).view({2, 2})));
}

TEST(DFlash2SamplingTest, RandomRowsReturnConditionalSoftmaxForRejection) {
  const auto candidates = make_candidates();
  SamplingParameters params;
  params.all_greedy_sample = false;
  params.all_random_sample = true;
  params.do_sample = torch::ones({1}, torch::kBool);
  const auto result = sample_dflash2_path(candidates,
                                          params,
                                          DraftSamplingMode::PROBABILISTIC,
                                          torch::zeros({1, 2, 2}));
  const auto probabilities = torch::softmax(torch::tensor({4.0f, 0.0f}), 0);
  const auto actual = torch::softmax(result.logits().value(), -1);
  EXPECT_FLOAT_EQ(actual.index({0, 1, 0}).item<float>(),
                  probabilities[0].item<float>());
  EXPECT_FLOAT_EQ(actual.index({0, 1, 1}).item<float>(),
                  probabilities[1].item<float>());
  EXPECT_TRUE(torch::allclose(actual.sum(-1), torch::ones({1, 2})));
}
}  // namespace
}  // namespace xllm
