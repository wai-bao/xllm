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

#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/speculative/draft_input.h"

namespace xllm::test {

// Small shared grammar for MTP tests: a single-object JSON schema with a
// dedicated stop token, so state transitions and invalid-suffix rejection can
// be exercised with token ids 0-5.
inline JsonObjectGrammar make_mtp_draft_input_test_grammar() {
  return JsonObjectGrammar({"{", "\"a\"", ":", "1", "}", "stop"},
                           /*stop_token_ids=*/{5});
}

// Moves both draft_inputs to CPU and compares every tensor member plus the
// point_mass_q flag. Undefined probs/selected_probs are compared by their
// defined() state only.
inline void expect_draft_input_equal(const DraftInput& actual,
                                     const DraftInput& expected) {
  const DraftInput actual_cpu =
      draft_input_to(actual, torch::Device(torch::kCPU));
  const DraftInput expected_cpu =
      draft_input_to(expected, torch::Device(torch::kCPU));
  EXPECT_TRUE(torch::equal(actual_cpu.token_ids, expected_cpu.token_ids));
  EXPECT_EQ(actual_cpu.hidden_states.scalar_type(),
            expected_cpu.hidden_states.scalar_type());
  EXPECT_TRUE(
      torch::equal(actual_cpu.hidden_states, expected_cpu.hidden_states));
  EXPECT_EQ(actual_cpu.probs.defined(), expected_cpu.probs.defined());
  if (actual_cpu.probs.defined() && expected_cpu.probs.defined()) {
    EXPECT_TRUE(torch::equal(actual_cpu.probs, expected_cpu.probs));
  }
  EXPECT_EQ(actual_cpu.selected_probs.defined(),
            expected_cpu.selected_probs.defined());
  if (actual_cpu.selected_probs.defined() &&
      expected_cpu.selected_probs.defined()) {
    EXPECT_TRUE(
        torch::equal(actual_cpu.selected_probs, expected_cpu.selected_probs));
  }
  EXPECT_EQ(actual_cpu.point_mass_q, expected_cpu.point_mass_q);
}

}  // namespace xllm::test
