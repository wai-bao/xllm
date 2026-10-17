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

#include "models/model_registry.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace xllm {
namespace {

TEST(ModelRegistryTest, DeepseekV4DSparkUsesTorchBackend) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("deepseek_v4_dspark",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v4_dspark");
  EXPECT_TRUE(error_message.empty());

  EXPECT_TRUE(resolve_model_registration("deepseek_v4_dspark",
                                         "TORCH",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v4_dspark");

  EXPECT_FALSE(resolve_model_registration("deepseek_v4_dspark",
                                          "ATB",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message,
            "Model type deepseek_v4_dspark only supports "
            "--npu_kernel_backend=TORCH.");
}

TEST(ModelRegistryTest, GlmMoeDsaMtpSupportsAtbAndTorchBackends) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("glm_moe_dsa_mtp",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "ATB");
  EXPECT_EQ(resolved_name, "glm_moe_dsa_mtp");
  EXPECT_TRUE(error_message.empty());

  EXPECT_TRUE(resolve_model_registration("glm_moe_dsa_mtp",
                                         "TORCH",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "glm_moe_dsa_mtp");

  EXPECT_TRUE(resolve_model_registration("glm_moe_dsa_mtp",
                                         "ATB",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "ATB");
  EXPECT_EQ(resolved_name, "glm_moe_dsa_mtp");
}

TEST(ModelRegistryTest, TargetVerificationModesAreDeclaredByModels) {
  EXPECT_EQ(ModelRegistry::get_target_spec_verify_mode("qwen3_5_text"),
            TargetSpecVerifyMode::UNIFORM_EXPANDED_VERIFY);
  EXPECT_EQ(ModelRegistry::get_target_spec_verify_mode("deepseek_v32"),
            TargetSpecVerifyMode::PYTHON_EXPANDED_VERIFY);
  EXPECT_EQ(ModelRegistry::get_target_spec_verify_mode("glm_moe_dsa"),
            TargetSpecVerifyMode::PYTHON_EXPANDED_VERIFY);
  EXPECT_EQ(ModelRegistry::get_target_spec_verify_mode("qwen3_5_mtp"),
            TargetSpecVerifyMode::GENERIC);
}

TEST(ModelRegistryTest, MtpPairsOwnArgumentConversion) {
  for (const std::string target : {"qwen3_5_text", "qwen3_5_moe_text"}) {
    ModelArgs args;
    args.model_type(target);
    args.n_layers(32);
    args.num_nextn_predict_layers(2);
    args.layer_types({"linear_attention", "full_attention"});
    args.full_attention_interval(4);
    ASSERT_TRUE(ModelRegistry::convert_to_mtp_draft(args));
    EXPECT_FALSE(args.model_type().empty());
    EXPECT_EQ(ModelRegistry::get_mtp_target_model_type(args.model_type()),
              target);
    EXPECT_EQ(args.n_layers(), 2);
    EXPECT_EQ(args.num_nextn_predict_layers(), 2);
    EXPECT_EQ(args.full_attention_interval(), 1);
    EXPECT_EQ(args.layer_types(),
              (std::vector<std::string>{"full_attention", "full_attention"}));
    // A draft is not a target, so conversion cannot run twice.
    EXPECT_FALSE(ModelRegistry::convert_to_mtp_draft(args));
  }
}

TEST(ModelRegistryTest, UnpairedModelIsNotConvertedByItsPrefix) {
  ModelArgs args;
  args.model_type("qwen3_5_future_text");
  args.n_layers(32);
  EXPECT_FALSE(ModelRegistry::convert_to_mtp_draft(args));
  EXPECT_EQ(args.model_type(), "qwen3_5_future_text");
  EXPECT_EQ(args.n_layers(), 32);
  EXPECT_TRUE(
      ModelRegistry::get_mtp_target_model_type(args.model_type()).empty());
}

}  // namespace
}  // namespace xllm
