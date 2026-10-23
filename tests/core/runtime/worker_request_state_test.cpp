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
#include <unistd.h>

#include <string>

#include "core/framework/config/execution_config.h"
#include "core/runtime/forward_shared_memory_manager.h"

namespace xllm {
namespace {

class WorkerRequestStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& config = ExecutionConfig::get_instance();
    contiguous_ = config.use_contiguous_input_buffer();
    config.use_contiguous_input_buffer(false);
  }

  void TearDown() override {
    ExecutionConfig::get_instance().use_contiguous_input_buffer(contiguous_);
  }

  static LlmForwardInput input() {
    LlmForwardInput result;
    result.token_ids = torch::tensor({11, 12}, torch::kInt);
    result.positions = torch::tensor({0, 0}, torch::kInt);
    result.input_params.meta = {BatchForwardType::DECODE, 2, 2, 1, 1, 9, false};
    result.input_params.embedding.embedding_ids = {7, 4};
    result.input_params.embedding.request_ids = {"request-b", "request-a"};
    result.input_params.embedding.extra_token_ids = {13, -1};
    result.request_state.released_embedding_ids = {2};
    result.request_state.target_decode_bootstrap_row_idxes = {1};
    result.request_state.draft_input_row_idxes = {0};
    return result;
  }

 private:
  bool contiguous_ = false;
};

TEST_F(WorkerRequestStateTest, DeviceConversionPreservesIndependentIdentities) {
  const LlmForwardInput source = input();
  LlmForwardInput converted =
      source.to(torch::Device(torch::kCPU), torch::kFloat);
  EXPECT_EQ(converted.input_params.embedding.embedding_ids,
            source.input_params.embedding.embedding_ids);
  EXPECT_EQ(converted.input_params.embedding.request_ids,
            source.input_params.embedding.request_ids);
  EXPECT_EQ(converted.input_params.embedding.extra_token_ids,
            source.input_params.embedding.extra_token_ids);
  EXPECT_EQ(converted.request_state.target_decode_bootstrap_row_idxes,
            source.request_state.target_decode_bootstrap_row_idxes);
  EXPECT_EQ(converted.request_state.draft_input_row_idxes,
            source.request_state.draft_input_row_idxes);
  converted.input_params.embedding.request_ids[0] = "replacement";
  EXPECT_EQ(source.input_params.embedding.request_ids[0], "request-b");
}

TEST_F(WorkerRequestStateTest,
       SharedMemoryPreservesRowOrderAndClearsOldIdentity) {
  LlmForwardInput source = input();
  const std::string name = "worker_request_state_" + std::to_string(getpid());
  bool writer_created = false;
  bool reader_created = false;
  ForwardSharedMemoryManager writer(
      name, /*size=*/1 << 20, writer_created, ForwardType::RAW_INPUT);
  ForwardSharedMemoryManager reader(
      name, /*size=*/1 << 20, reader_created, ForwardType::RAW_INPUT);
  ASSERT_TRUE(writer.input_write(source));
  LlmForwardInput received;
  reader.input_read(received, torch::Device(torch::kCPU));
  EXPECT_TRUE(torch::equal(received.token_ids, source.token_ids));
  EXPECT_EQ(received.input_params.embedding.embedding_ids,
            source.input_params.embedding.embedding_ids);
  EXPECT_EQ(received.input_params.embedding.request_ids,
            source.input_params.embedding.request_ids);
  EXPECT_EQ(received.input_params.embedding.extra_token_ids,
            source.input_params.embedding.extra_token_ids);
  EXPECT_EQ(received.request_state.target_decode_bootstrap_row_idxes,
            source.request_state.target_decode_bootstrap_row_idxes);
  // released_embedding_ids crosses the transport so the worker's per-step
  // speculative cleanup sees it; draft_input_row_idxes is filled by the
  // builder after the SHM read and stays cleared.
  EXPECT_EQ(received.request_state.released_embedding_ids,
            source.request_state.released_embedding_ids);
  EXPECT_TRUE(received.request_state.draft_input_row_idxes.empty());

  source.request_state = {};
  source.input_params.embedding.embedding_ids.clear();
  source.input_params.embedding.request_ids.clear();
  source.input_params.embedding.extra_token_ids.clear();
  ASSERT_TRUE(writer.input_write(source));
  reader.input_read(received, torch::Device(torch::kCPU));
  EXPECT_TRUE(received.input_params.embedding.embedding_ids.empty());
  EXPECT_TRUE(received.input_params.embedding.request_ids.empty());
  EXPECT_TRUE(received.input_params.embedding.extra_token_ids.empty());
  EXPECT_TRUE(received.request_state.released_embedding_ids.empty());
  EXPECT_TRUE(received.request_state.target_decode_bootstrap_row_idxes.empty());
  EXPECT_TRUE(received.request_state.draft_input_row_idxes.empty());
}

}  // namespace
}  // namespace xllm
