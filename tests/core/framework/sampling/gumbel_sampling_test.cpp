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

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm {

TEST(GumbelSamplingTest, ReturnsZeroNoiseForGreedyBatch) {
  torch::Tensor noise = sample_gumbel_noise(/*num_steps=*/3,
                                            /*num_classes=*/4,
                                            /*greedy_rows=*/
                                            torch::ones({2}, torch::kBool),
                                            /*all_greedy=*/false,
                                            torch::Device(torch::kCPU));

  EXPECT_EQ(noise.sizes(), torch::IntArrayRef({2, 3, 4}));
  EXPECT_TRUE(noise.eq(0).all().item<bool>());
}

TEST(GumbelSamplingTest, SkipsDrawForAllGreedyBatch) {
  torch::Tensor noise = sample_gumbel_noise(/*num_steps=*/3,
                                            /*num_classes=*/4,
                                            /*greedy_rows=*/
                                            torch::ones({2}, torch::kBool),
                                            /*all_greedy=*/true,
                                            torch::Device(torch::kCPU));

  EXPECT_EQ(noise.sizes(), torch::IntArrayRef({2, 3, 4}));
  EXPECT_TRUE(noise.eq(0).all().item<bool>());
}

TEST(GumbelSamplingTest, MasksGreedyRowsInMixedBatch) {
  torch::manual_seed(2026);

  torch::Tensor noise =
      sample_gumbel_noise(/*num_steps=*/3,
                          /*num_classes=*/4,
                          /*greedy_rows=*/
                          torch::tensor({true, false}, torch::kBool),
                          /*all_greedy=*/false,
                          torch::Device(torch::kCPU));

  EXPECT_TRUE(noise.select(/*dim=*/0, /*index=*/0).eq(0).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(noise).all().item<bool>());
  EXPECT_TRUE(noise.select(/*dim=*/0, /*index=*/1).ne(0).any().item<bool>());
}

}  // namespace xllm
