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

#include "core/framework/speculative/draft_extend_input.h"

#include <gtest/gtest.h>

#include "core/framework/model/model_output.h"
#include "core/framework/sampling/sampling_params.h"

namespace xllm {

namespace {

bool tensor_equal(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  return lhs.defined() && rhs.defined() && torch::equal(lhs, rhs);
}

TEST(DraftExtendInputTest, SpeculativeHiddenSelectionPreservesDrafterSource) {
  const torch::Tensor primary = torch::tensor({{10.0f, 11.0f}, {20.0f, 21.0f}});
  const torch::Tensor aux = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
  const torch::Tensor indices = torch::tensor({1}, torch::kInt32);
  const torch::Tensor gathered_primary = torch::tensor({{20.0f, 21.0f}});
  const torch::Tensor selected_aux = torch::tensor({{3.0f, 4.0f}});
  const ModelOutput model_output(primary, torch::Tensor(), aux);

  for (const bool is_target_prefill : {false, true}) {
    for (const bool cp_enabled : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "prefill=" << is_target_prefill << ", cp=" << cp_enabled);
      SampleOutput output;
      output_spec_hidden_states(output,
                                model_output,
                                indices,
                                gathered_primary,
                                is_target_prefill,
                                cp_enabled);
      EXPECT_TRUE(tensor_equal(
          is_target_prefill ? output.selected_embeddings : output.embeddings,
          selected_aux));
      if (is_target_prefill) {
        EXPECT_TRUE(tensor_equal(output.embeddings, aux));
      }
    }
  }

  SampleOutput output;
  output_spec_hidden_states(output,
                            ModelOutput(primary.narrow(0, 0, 1)),
                            indices,
                            gathered_primary,
                            /*is_target_prefill=*/false,
                            /*cp_enabled=*/true);
  EXPECT_TRUE(tensor_equal(output.embeddings, gathered_primary));
}

}  // namespace
}  // namespace xllm
