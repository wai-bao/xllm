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

#include <torch/types.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace xllm {

struct SamplingParameters;

namespace mtp_verify {
struct ValidateInputLayout;
struct ValidateInputPlan;
}  // namespace mtp_verify

// Owns the stable NPU sources used by MTP target verification. The worker
// supplies the current layout and model-dependent limits; this object keeps
// graph input addresses and greedy sampling controls alive across rounds.
class MtpNpuValidateInputState final {
 public:
  struct Config {
    torch::Device device;
    int64_t verify_block_table_width = 0;
    int32_t max_seqs_per_batch = 0;
    bool supports_explicit_replay_update = false;
  };

  bool try_prepare_stable_greedy_sampling(SamplingParameters& sampling_params,
                                          int32_t total_num_val_tokens,
                                          bool use_explicit_replay_update,
                                          const torch::Device& device);

  template <typename TargetInput>
  void prepare_attention(TargetInput& validate_input,
                         mtp_verify::ValidateInputLayout& layout,
                         const mtp_verify::ValidateInputPlan& plan,
                         std::vector<int32_t>& accepted_prefix_lengths,
                         std::vector<int32_t>& expanded_linear_state_ids,
                         bool static_graph_tasks_prepared,
                         const Config& config);

 private:
  template <typename TargetInput>
  void prepare_explicit_replay_attention(
      TargetInput& validate_input,
      mtp_verify::ValidateInputLayout& layout,
      const mtp_verify::ValidateInputPlan& plan,
      std::vector<int32_t>& accepted_prefix_lengths,
      std::vector<int32_t>& expanded_linear_state_ids,
      bool static_graph_tasks_prepared,
      const Config& config);

  torch::Tensor attention_host_buffer_;
  torch::Tensor attention_device_buffer_;
  uint64_t attention_buffer_capacity_ = 0;
  std::shared_ptr<int> attention_buffer_owner_ = std::make_shared<int>(0);
  torch::Tensor greedy_indices_;
  torch::Tensor greedy_do_sample_;
  // Retained across rounds: rebuilt in full every decode step, so reusing the
  // storage avoids one CPU allocation and one zero-fill pass each. Safe to
  // retain because it is only a pack source, never handed to the caller.
  std::vector<int32_t> expanded_block_tables_dense_;
};

}  // namespace xllm
