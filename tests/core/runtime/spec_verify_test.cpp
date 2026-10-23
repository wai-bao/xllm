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

#include "core/runtime/spec_verify.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <string_view>

#include "core/framework/config/speculative_config.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/capture_hidden_mode.h"
#include "core/framework/speculative/target_decode_state_store.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/speculative_output.h"

namespace xllm::spec_verify {
namespace {

TEST(SpecVerifyTest, ResolvesCaptureHiddenModeSeparatelyForTargetAndDraft) {
  const CaptureHiddenRole target = CaptureHiddenRole::TARGET;
  const CaptureHiddenRole draft = CaptureHiddenRole::DRAFT;
  EXPECT_EQ(resolve_capture_hidden_mode("MTP", target),
            CaptureHiddenMode::FINAL_OUTPUT);
  EXPECT_EQ(resolve_capture_hidden_mode("MTP", draft),
            CaptureHiddenMode::FINAL_OUTPUT);
  EXPECT_EQ(resolve_capture_hidden_mode("MTP", target, /*model_hook=*/true),
            CaptureHiddenMode::MODEL_HOOK);
  EXPECT_EQ(resolve_capture_hidden_mode("MTP", draft, /*model_hook=*/true),
            CaptureHiddenMode::MODEL_HOOK);
  EXPECT_EQ(resolve_capture_hidden_mode("Eagle3", target),
            CaptureHiddenMode::CAPTURED_LAYERS);
  EXPECT_EQ(resolve_capture_hidden_mode("Eagle3", draft),
            CaptureHiddenMode::MODEL_PROVIDED);
  for (std::string_view algorithm : {"DFlash", "DFlash2", "DSpark"}) {
    EXPECT_EQ(resolve_capture_hidden_mode(algorithm, target),
              CaptureHiddenMode::CAPTURED_LAYERS);
    EXPECT_EQ(resolve_capture_hidden_mode(algorithm, draft),
              CaptureHiddenMode::FINAL_OUTPUT);
  }
  EXPECT_EQ(resolve_capture_hidden_mode("Suffix", target),
            CaptureHiddenMode::FINAL_OUTPUT);
}

TEST(SpecVerifyTest, ClassifiesSpeculativeAlgorithmsCaseInsensitively) {
  EXPECT_TRUE(SpeculativeConfig::is_eagle3_algorithm("eAgLe3"));
  EXPECT_TRUE(SpeculativeConfig::is_dflash_algorithm("dflash"));
  EXPECT_TRUE(SpeculativeConfig::is_dflash2_algorithm("DFLASH2"));
  EXPECT_TRUE(SpeculativeConfig::is_dspark_algorithm("dspark"));
  EXPECT_TRUE(SpeculativeConfig::is_suffix_algorithm("suffix"));
  EXPECT_TRUE(SpeculativeConfig::is_mtp_algorithm("mtp"));
  EXPECT_TRUE(SpeculativeConfig::requires_aux_hidden_capture("eagle3"));
  EXPECT_TRUE(SpeculativeConfig::supports_host_kv_cache("mtp"));
  EXPECT_FALSE(SpeculativeConfig::is_eagle3_algorithm("mtp"));
}

TEST(SpecVerifyTest, FinalOutputCaptureBorrowsPrimaryOutput) {
  const torch::Tensor hidden =
      torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}});
  const torch::Tensor aux = torch::full({3, 6}, 99.0F);
  const torch::Tensor rows = torch::tensor({2, 0}, torch::kLong);
  const torch::Tensor gathered = torch::tensor({{5.0F, 6.0F}, {1.0F, 2.0F}});
  SampleOutput output;
  output_captured_hidden_states(output,
                                hidden,
                                aux,
                                rows,
                                gathered,
                                CaptureHiddenMode::FINAL_OUTPUT,
                                /*is_target_prefill=*/true,
                                /*cp_enabled=*/false);
  EXPECT_EQ(output.embeddings.data_ptr(), hidden.data_ptr());
  EXPECT_EQ(output.selected_embeddings.data_ptr(), gathered.data_ptr());
}

TEST(SpecVerifyTest, PrefillTargetStateKeepsHiddenOnlyForEmbeddingRequests) {
  TargetDecodeStateStore target_state_store(/*total_nums=*/2);
  LlmForwardInput input;
  input.input_params.embedding.embedding_ids = {1};
  input.input_params.embedding.request_ids = {"prefill-request"};

  const torch::Tensor selected_hidden = torch::tensor({{3.0F, 4.0F}});
  for (bool is_embeddings : {false, true}) {
    input.sampling_params.is_embeddings = is_embeddings;
    ForwardOutput output;
    output.sample_output.next_tokens = torch::tensor({17}, torch::kLong);
    output.sample_output.embeddings = torch::tensor({{1.0F, 2.0F}});
    output.sample_output.selected_embeddings = selected_hidden;

    prepare_target_prefill_state(target_state_store, input, output);

    const auto states =
        target_state_store.read_decode_states({1}, {"prefill-request"});
    ASSERT_EQ(states.size(), 1);
    EXPECT_TRUE(states[0].valid);
    EXPECT_EQ(states[0].token_id, 17);
    EXPECT_EQ(states[0].position_offset, 0);
    EXPECT_FALSE(output.sample_output.selected_embeddings.defined());
    if (is_embeddings) {
      ASSERT_TRUE(output.sample_output.embeddings.defined());
      EXPECT_TRUE(
          torch::equal(output.sample_output.embeddings, selected_hidden));
    } else {
      EXPECT_FALSE(output.sample_output.embeddings.defined());
    }
  }
}

TEST(SpecVerifyTest, CapturedAndModelProvidedSourcesSelectExplicitly) {
  const torch::Tensor hidden = torch::arange(6, torch::kFloat).view({3, 2});
  const torch::Tensor aux = torch::arange(18, torch::kFloat).view({3, 6});
  const torch::Tensor rows = torch::tensor({2, 0}, torch::kLong);
  SampleOutput output;
  output_captured_hidden_states(output,
                                hidden,
                                aux,
                                rows,
                                torch::Tensor(),
                                CaptureHiddenMode::CAPTURED_LAYERS,
                                /*is_target_prefill=*/true,
                                /*cp_enabled=*/false);
  EXPECT_EQ(output.embeddings.data_ptr(), aux.data_ptr());
  EXPECT_TRUE(torch::equal(output.selected_embeddings,
                           aux.index_select(/*dim=*/0, rows)));

  output_captured_hidden_states(output,
                                hidden,
                                aux,
                                rows,
                                torch::Tensor(),
                                CaptureHiddenMode::MODEL_HOOK,
                                /*is_target_prefill=*/false,
                                /*cp_enabled=*/false);
  EXPECT_TRUE(
      torch::equal(output.embeddings, aux.index_select(/*dim=*/0, rows)));

  output_captured_hidden_states(output,
                                hidden,
                                aux,
                                rows,
                                torch::Tensor(),
                                CaptureHiddenMode::MODEL_PROVIDED,
                                /*is_target_prefill=*/false,
                                /*cp_enabled=*/false);
  EXPECT_TRUE(
      torch::equal(output.embeddings, aux.index_select(/*dim=*/0, rows)));
  output_captured_hidden_states(output,
                                hidden,
                                torch::Tensor(),
                                rows,
                                torch::Tensor(),
                                CaptureHiddenMode::MODEL_PROVIDED,
                                /*is_target_prefill=*/false,
                                /*cp_enabled=*/false);
  EXPECT_TRUE(
      torch::equal(output.embeddings, hidden.index_select(/*dim=*/0, rows)));
  EXPECT_DEATH(output_captured_hidden_states(output,
                                             hidden,
                                             torch::Tensor(),
                                             rows,
                                             torch::Tensor(),
                                             CaptureHiddenMode::CAPTURED_LAYERS,
                                             /*is_target_prefill=*/true,
                                             /*cp_enabled=*/false),
               "requires captured auxiliary hidden states");
  EXPECT_DEATH(output_captured_hidden_states(output,
                                             hidden,
                                             torch::Tensor(),
                                             rows,
                                             torch::Tensor(),
                                             CaptureHiddenMode::MODEL_HOOK,
                                             /*is_target_prefill=*/false,
                                             /*cp_enabled=*/false),
               "requires model-provided hidden states");
}

TEST(SpecVerifyTest, AuxSourcesKeepGlobalRowsUnderContextParallelDecode) {
  const torch::Tensor hidden = torch::arange(6, torch::kFloat).view({3, 2});
  const torch::Tensor model_hook =
      torch::arange(18, torch::kFloat).view({3, 6});
  const torch::Tensor rows = torch::tensor({2, 0}, torch::kLong);
  const torch::Tensor gathered_primary = hidden.index_select(/*dim=*/0, rows);
  SampleOutput output;
  output_captured_hidden_states(output,
                                hidden,
                                model_hook,
                                rows,
                                gathered_primary,
                                CaptureHiddenMode::MODEL_HOOK,
                                /*is_target_prefill=*/false,
                                /*cp_enabled=*/true);
  EXPECT_TRUE(torch::equal(output.embeddings,
                           model_hook.index_select(/*dim=*/0, rows)));
  EXPECT_NE(output.embeddings.data_ptr(), gathered_primary.data_ptr());

  output_captured_hidden_states(output,
                                hidden,
                                model_hook,
                                rows,
                                gathered_primary,
                                CaptureHiddenMode::CAPTURED_LAYERS,
                                /*is_target_prefill=*/false,
                                /*cp_enabled=*/true);
  EXPECT_TRUE(torch::equal(output.embeddings,
                           model_hook.index_select(/*dim=*/0, rows)));
  EXPECT_NE(output.embeddings.data_ptr(), gathered_primary.data_ptr());

  output_captured_hidden_states(output,
                                hidden,
                                model_hook,
                                rows,
                                gathered_primary,
                                CaptureHiddenMode::FINAL_OUTPUT,
                                /*is_target_prefill=*/false,
                                /*cp_enabled=*/true);
  EXPECT_EQ(output.embeddings.data_ptr(), gathered_primary.data_ptr());
}

TEST(SpecVerifyTest, ResolvesClosedTargetExecutionCapabilities) {
  const TargetSpecVerifyCapabilities qwen =
      resolve_target_spec_verify_capabilities(
          TargetSpecVerifyMode::UNIFORM_EXPANDED_VERIFY, false);
  EXPECT_TRUE(qwen.supports_explicit_replay_update);
  EXPECT_TRUE(qwen.requires_uniform_validate_width);
  EXPECT_TRUE(qwen.uses_chunked_prefill);

  const TargetSpecVerifyCapabilities native_deepseek =
      resolve_target_spec_verify_capabilities(
          TargetSpecVerifyMode::PYTHON_EXPANDED_VERIFY, false);
  EXPECT_FALSE(native_deepseek.supports_explicit_replay_update);
  EXPECT_FALSE(native_deepseek.requires_uniform_validate_width);
  EXPECT_FALSE(native_deepseek.uses_chunked_prefill);

  const TargetSpecVerifyCapabilities python_deepseek =
      resolve_target_spec_verify_capabilities(
          TargetSpecVerifyMode::PYTHON_EXPANDED_VERIFY, true);
  EXPECT_TRUE(python_deepseek.supports_explicit_replay_update);
  EXPECT_TRUE(python_deepseek.uses_chunked_prefill);
}

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
