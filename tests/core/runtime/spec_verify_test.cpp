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

#include "core/framework/speculative/spec_verify.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm::spec_verify {
namespace {

TEST(SpecVerifyTest, MaterializesDraftColumnsForEagerFallback) {
  torch::Tensor verify_tokens =
      torch::tensor({10, -1, -1, 20, -1, -1}, torch::kInt);
  const std::vector<torch::Tensor> draft_sources = {
      torch::tensor({11, 21}, torch::kLong),
      torch::tensor({12, 22}, torch::kLong)};

  materialize_speculative_verify_tokens(verify_tokens, draft_sources);

  EXPECT_TRUE(torch::equal(
      verify_tokens, torch::tensor({10, 11, 12, 20, 21, 22}, torch::kInt)));

  const torch::Tensor ordinary_tokens = torch::tensor({10, 11}, torch::kInt);
  materialize_speculative_verify_tokens(ordinary_tokens, {});
  EXPECT_TRUE(
      torch::equal(ordinary_tokens, torch::tensor({10, 11}, torch::kInt)));
}

}  // namespace
}  // namespace xllm::spec_verify
