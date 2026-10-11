/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "rejection_sampler.h"

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>

#include "platform/device.h"
#include "platform/platform.h"
#include "sampler.h"

namespace xllm {

namespace {
// Exercise the active accelerator backend when available.
torch::Device get_test_device() {
  if ((Platform::is_npu() || Platform::is_mlu() || Platform::is_cuda()) &&
      Platform::device_count() > 0) {
    return torch::Device(Platform::type_torch(), 0);
  }
  return torch::Device(torch::kCPU);
}

// Helper function to get test tensor options with automatic device selection
torch::TensorOptions get_test_options(
    torch::ScalarType dtype = torch::kFloat32) {
  return torch::dtype(dtype).device(get_test_device());
}
}  // namespace

TEST(RejectionSamplerTest, Basic) {
  // test with hand-crafted example
  const auto options = get_test_options(torch::kFloat32);

  // set random seed
  torch::manual_seed(100);

  const auto draft_token_ids =
      torch::tensor({{1, 2, 3}}, options.dtype(torch::kInt64));

  // shape: [1, 3, 5]
  auto draft_probs = torch::tensor({{0.2104, 0.2163, 0.1912, 0.1937, 0.1884},
                                    {0.2100, 0.1803, 0.2398, 0.2088, 0.1610},
                                    {0.1838, 0.2079, 0.2270, 0.2451, 0.1362}},
                                   options)
                         .reshape({1, 3, 5});
  // shape: [1, 3, 5]
  auto target_probs = torch::tensor({{0.1299, 0.2462, 0.1821, 0.1354, 0.3064},
                                     {0.1159, 0.2839, 0.1603, 0.2451, 0.1949},
                                     {0.0002, 0.0433, 0.6629, 0.1469, 0.1467}},
                                    options)
                          .reshape({1, 3, 5});

  // selected_target_probs:  [0.2462  0.1603  0.1469]
  // selected_draft_probs:   [0.2163  0.2398  0.2451]
  // acceptance_probs:       [1.1382  0.6685  0.5993]
  // log(uniform_rand):      [-0.7371 -0.4172 -0.0620]
  // accepted:               [  1        1       0  ]
  auto uniform_rand = torch::tensor({{0.4785, 0.6589, 0.9399}}, options).log();
  auto bonus_token_ids = torch::tensor({{5}}, options.dtype(torch::kInt64));
  DraftProposal draft_proposal =
      DraftProposal(draft_token_ids, draft_probs.log());

  auto output = RejectionSampler::random_sample(
      draft_proposal, target_probs.log(), uniform_rand, bonus_token_ids);
  auto desired_masked_output =
      torch::tensor({{1, 2, 2, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::allclose(output, desired_masked_output));
}

TEST(RejectionSamplerTest, GreedyDraftUsesDeltaProposalDistribution) {
  const auto options = get_test_options(torch::kFloat32);

  const auto draft_token_ids =
      torch::tensor({{0, 1, 2, 1}}, options.dtype(torch::kInt64));

  const auto target_probs = torch::tensor({{{0.8f, 0.2f, 0.0f},
                                            {0.0f, 0.2f, 0.8f},
                                            {1.0f, 0.0f, 0.0f},
                                            {0.0f, 1.0f, 0.0f}}},
                                          options);
  const auto uniform_rand =
      torch::tensor({{0.5f, 0.5f, 0.5f, 0.5f}}, options).log();
  const auto bonus_token_ids =
      torch::tensor({{2}}, options.dtype(torch::kInt64));
  DraftProposal draft_proposal = DraftProposal(draft_token_ids);

  auto output = RejectionSampler::random_sample(
      draft_proposal, target_probs.log(), uniform_rand, bonus_token_ids);
  auto desired_masked_output =
      torch::tensor({{0, 2, -1, -1, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::allclose(output, desired_masked_output));
}

TEST(RejectionSamplerTest, Greedy) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();

  int64_t batch_size = 2;
  int64_t n_speculative_tokens = 3;
  int64_t vocab_size = 4;
  int64_t n_bonus_tokens = 1;

  const auto draft_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_speculative_tokens},
                     torch::dtype(torch::kInt64).device(device));
  auto target_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options)
          .softmax(/*dim=*/-1, /*dtype=*/torch::kFloat32);
  const auto bonus_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_bonus_tokens},
                     torch::dtype(torch::kInt64).device(device));

  // The production path consumes pre-sampled target ids; produce them with
  // an independent argmax here, then feed the token-match kernel directly.
  const auto desired_output = target_probs.argmax(/*dim=*/-1);
  auto output = RejectionSampler::greedy_sample_from_token_ids(
      draft_token_ids, desired_output, bonus_token_ids);

  // Independent reference: a prefix survives iff every earlier draft matched,
  // which is the cumulative product of the accepted indicators.
  const auto committed =
      torch::cat({torch::ones({batch_size, 1},
                              torch::dtype(torch::kInt64).device(device)),
                  (desired_output == draft_token_ids).to(torch::kInt64)},
                 /*dim=*/-1)
          .cumprod(/*dim=*/1);
  const auto expected =
      torch::where(committed > 0,
                   torch::cat({desired_output, bonus_token_ids}, /*dim=*/-1),
                   -1);
  EXPECT_TRUE(torch::equal(output, expected));
}

TEST(RejectionSamplerTest, GreedyFromTokenIds) {
  const auto device = get_test_device();
  const auto draft_token_ids = torch::tensor({{1, 2, 3}, {3, 1, 0}}, device);
  const auto target_token_ids = torch::tensor({{1, 4, 3}, {3, 2, 0}}, device);
  const auto bonus_token_ids = torch::tensor({{5}, {6}}, device);

  auto masked_output = RejectionSampler::greedy_sample_from_token_ids(
      draft_token_ids, target_token_ids, bonus_token_ids);

  EXPECT_TRUE(torch::equal(
      masked_output, torch::tensor({{1, 4, -1, -1}, {3, 2, -1, -1}}, device)));
}

TEST(RejectionSamplerTest, GreedyForwardAllowsUndefinedDraftProbs) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();
  const auto do_sample = torch::tensor({false, false}, device);
  SamplingParameters params;
  params.do_sample = do_sample;
  const auto draft_token_ids = torch::tensor({{1, 2}, {3, 4}}, device);
  const auto target_logits = torch::tensor({{{0.0f, 4.0f, 1.0f, 2.0f, 3.0f},
                                             {0.0f, 1.0f, 2.0f, 4.0f, 3.0f},
                                             {4.0f, 0.0f, 1.0f, 2.0f, 3.0f}},
                                            {{0.0f, 1.0f, 2.0f, 4.0f, 3.0f},
                                             {0.0f, 1.0f, 2.0f, 3.0f, 4.0f},
                                             {0.0f, 4.0f, 1.0f, 2.0f, 3.0f}}},
                                           options);
  const auto bonus_token_ids = torch::tensor({{0}, {1}}, device);

  DraftProposal draft_proposal = DraftProposal(draft_token_ids);
  // Verify matches target token ids [B, n + 1]: greedy argmax of the last
  // column supplies the bonus, the rest come from target_logits argmax.
  const auto target_tokens = torch::tensor({{1, 3, 0}, {3, 4, 1}}, device);
  SampleOutput output = RejectionSampler::verify(
      params, draft_proposal, target_logits, target_tokens);

  EXPECT_TRUE(torch::equal(output.next_tokens,
                           torch::tensor({{1, 3, -1}, {3, 4, 1}}, device)));
}

TEST(RejectionSamplerTest, GreedyRejectsGrammarMaskedDraftToken) {
  const auto options = get_test_options(torch::kFloat32);

  // The JSON grammar mask removes draft token 0 from the target distribution.
  const auto draft_token_ids =
      torch::tensor({{0}}, options.dtype(torch::kInt64));
  const auto target_probs = torch::tensor({{{0.0F, 1.0F, 0.0F}}}, options);
  const auto bonus_token_ids =
      torch::tensor({{2}}, options.dtype(torch::kInt64));

  const auto target_token_ids = target_probs.argmax(/*dim=*/-1);
  auto masked_output = RejectionSampler::greedy_sample_from_token_ids(
      draft_token_ids, target_token_ids, bonus_token_ids);

  const auto expected_masked_output =
      torch::tensor({{1, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::equal(masked_output, expected_masked_output));
}

TEST(RejectionSamplerTest, RandomRejectsGrammarMaskedDraftToken) {
  const auto options = get_test_options(torch::kFloat32);

  // The JSON grammar mask removes draft token 0 from the target distribution.
  const auto draft_token_ids =
      torch::tensor({{0}}, options.dtype(torch::kInt64));
  const auto draft_probs = torch::tensor({{{1.0F, 0.0F, 0.0F}}}, options);
  const auto target_probs = torch::tensor({{{0.0F, 1.0F, 0.0F}}}, options);
  const auto uniform_rand = torch::tensor({{0.5F}}, options).log();
  const auto bonus_token_ids =
      torch::tensor({{2}}, options.dtype(torch::kInt64));

  DraftProposal draft_proposal =
      DraftProposal(draft_token_ids, draft_probs.log());
  auto masked_output = RejectionSampler::random_sample(
      draft_proposal, target_probs.log(), uniform_rand, bonus_token_ids);

  const auto expected_masked_output =
      torch::tensor({{1, -1}}, options.dtype(torch::kInt64));
  EXPECT_TRUE(torch::equal(masked_output, expected_masked_output));
}

TEST(RejectionSamplerTest, RandomForwardUsesGreedyDraftWithoutDraftProbs) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();
  const auto do_sample = torch::tensor({true}, device);
  SamplingParameters params;
  params.do_sample = do_sample;
  params.all_random_sample = true;
  params.all_greedy_sample = false;

  DraftProposal draft_proposal = DraftProposal(torch::tensor({{1, 2}}, device));
  const auto target_logits = torch::tensor({{{-1000.0f, 1000.0f, -1000.0f},
                                             {-1000.0f, -1000.0f, 1000.0f},
                                             {1000.0f, -1000.0f, -1000.0f}}},
                                           options);
  const auto bonus_token_ids = torch::tensor({{0}}, device);

  auto output = RejectionSampler::verify(
      params,
      draft_proposal,
      target_logits,
      torch::cat({torch::tensor({{1, 2}}, device), bonus_token_ids}, -1));

  EXPECT_TRUE(
      torch::equal(output.next_tokens, torch::tensor({{1, 2, 0}}, device)));
}

TEST(RejectionSamplerTest, RandomFallsBackToUniformOnDegenerateResidual) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();

  const int64_t vocab_size = 3;
  const auto draft_token_ids = torch::tensor({{0}}, device);
  const auto draft_probs = torch::tensor({{{1.0f, 0.0f, 0.0f}}}, options);
  const auto target_probs = torch::tensor({{{1.0f, 0.0f, 0.0f}}}, options);
  const auto uniform_rand = torch::tensor({{1.0f}}, options).log();
  const auto bonus_token_ids = torch::tensor({{0}}, device);

  DraftProposal draft_proposal =
      DraftProposal(draft_token_ids, draft_probs.log());
  auto masked_output = RejectionSampler::random_sample(
      draft_proposal, target_probs.log(), uniform_rand, bonus_token_ids);

  const auto resampled = masked_output.index({0, 0}).item<int64_t>();
  EXPECT_GE(resampled, 0);
  EXPECT_LT(resampled, vocab_size);
  EXPECT_EQ(masked_output.index({0, 1}).item<int64_t>(), -1);
}

TEST(RejectionSamplerTest, LogProbs) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();
  const auto do_sample = torch::tensor({false, true, false, true}, device);
  const int64_t max_top_logprobs = 2;
  SamplingParameters params;
  params.do_sample = do_sample;
  params.logprobs = true;
  params.max_top_logprobs = max_top_logprobs;

  int64_t batch_size = 4;
  int64_t n_speculative_tokens = 4;
  int64_t vocab_size = 8;

  const auto draft_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_speculative_tokens},
                     torch::dtype(torch::kInt64).device(device));
  auto draft_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options)
          .softmax(/*dim=*/-1);

  auto target_logits =
      torch::randn({batch_size, n_speculative_tokens + 1, vocab_size}, options);
  const auto bonus_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, 1},
                     torch::dtype(torch::kInt64).device(device));

  DraftProposal draft_proposal =
      DraftProposal(draft_token_ids, draft_probs.log());
  const auto target_tokens =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_speculative_tokens + 1},
                     options.dtype(torch::kInt64));
  auto output = RejectionSampler::verify(
      params, draft_proposal, target_logits, target_tokens);

  const auto logprobs =
      torch::log_softmax(target_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  // Independent reference: gather the committed token's logprob at surviving
  // positions only (indices kept valid via a where on the token ids), and
  // check masked positions are exactly zero as a separate expectation.
  const auto committed = output.next_tokens.ge(0);
  const auto reference_ids = torch::where(
      committed, output.next_tokens, torch::zeros_like(output.next_tokens));
  const auto selected_logprobs =
      logprobs.gather(/*dim=*/-1, reference_ids.unsqueeze(/*dim=*/-1))
          .squeeze(/*dim=*/-1);
  // gather - logsumexp differs from log_softmax by fp32 rounding order;
  // compare with tolerance instead of bitwise equality.
  EXPECT_TRUE(torch::allclose(output.logprobs.masked_select(committed),
                              selected_logprobs.masked_select(committed),
                              /*rtol=*/1e-5,
                              /*atol=*/1e-6));
  EXPECT_TRUE(torch::equal(output.logprobs.masked_fill(committed, 0.0),
                           torch::zeros_like(output.logprobs)));

  auto [top_k_values, top_k_indices] = logprobs.topk(
      max_top_logprobs, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
  EXPECT_TRUE(torch::allclose(
      output.top_logprobs, top_k_values, /*rtol=*/1e-5, /*atol=*/1e-6));
  EXPECT_TRUE(torch::equal(output.top_tokens, top_k_indices));
}

TEST(RejectionSamplerTest,
     ReusingDoSampleAfterRejectionSamplerKeepsSamplerOutput1D) {
  const auto options = get_test_options(torch::kFloat32);
  const auto device = get_test_device();
  auto do_sample = torch::tensor({false, true}, torch::device(device));

  SamplingParameters params;
  params.selected_token_idxes =
      torch::tensor({0, 1}, torch::dtype(torch::kInt64).device(device));
  params.sample_idxes =
      torch::tensor({0, 1}, torch::dtype(torch::kInt64).device(device));
  params.do_sample = do_sample;
  params.all_random_sample = false;
  params.all_greedy_sample = false;
  params.return_probs = true;

  auto logits =
      torch::tensor({{3.0f, 1.0f, 0.5f}, {0.1f, 0.2f, 4.0f}}, options);
  auto output = Sampler::forward(logits, params);

  // probs is the selected token's probability per sequence, not the full
  // vocab distribution.
  EXPECT_EQ(output.probs.dim(), 1);
  EXPECT_EQ(output.probs.size(0), 2);
  EXPECT_EQ(output.next_tokens.dim(), 1);
  EXPECT_EQ(output.next_tokens.size(0), 2);
}

TEST(RejectionSamplerTest, Random) {
  const auto options = get_test_options(torch::kFloat32);

  // set random seed
  torch::manual_seed(100);

  int64_t vocab_size = 50;
  int64_t num_samples = 500000;

  auto target_prob = torch::randn({vocab_size}, options).softmax(/*dim=*/-1);
  auto target_probs =
      target_prob.reshape({1, 1, -1}).repeat({num_samples, 1, 1});

  auto draft_probs =
      torch::randn({num_samples, 1, vocab_size}, options).softmax(/*dim=*/-1);
  auto draft_token_ids = Sampler::random_sample(draft_probs);

  // not used
  auto bonus_token_ids =
      torch::ones({num_samples, 1}, options.dtype(torch::kInt64));

  auto uniform_rand = torch::rand(draft_token_ids.sizes(), options).log();
  DraftProposal draft_proposal =
      DraftProposal(draft_token_ids, draft_probs.log());
  auto output = RejectionSampler::random_sample(
      draft_proposal, target_probs.log(), uniform_rand, bonus_token_ids);

  // remove bonus token
  auto token_ids = output
                       .slice(/*dim=*/-1,
                              /*start=*/0,
                              /*end=*/-1)
                       .flatten();

  // calculate the probability of each sampled token
  auto bincount = token_ids.bincount(/*weights=*/torch::nullopt,
                                     /*minlength=*/vocab_size);
  auto sample_prob = bincount.to(torch::kFloat) / num_samples;

  EXPECT_TRUE(torch::allclose(target_prob,
                              sample_prob,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2));
}

}  // namespace xllm
