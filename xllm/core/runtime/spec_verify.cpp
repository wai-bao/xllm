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

#include <glog/logging.h>

#include <algorithm>

#include "core/framework/sampling/rejection_sampler.h"
#include "core/runtime/forward_params.h"
#include "core/util/tensor_helper.h"
#include "core/util/utils.h"

namespace xllm::spec_verify {

TargetSpecVerifyCapabilities resolve_target_spec_verify_capabilities(
    TargetSpecVerifyMode mode,
    bool is_python_model_impl) {
  TargetSpecVerifyCapabilities capabilities;
  capabilities.requires_uniform_validate_width =
      mode == TargetSpecVerifyMode::UNIFORM_EXPANDED_VERIFY;
  capabilities.supports_explicit_replay_update =
      mode == TargetSpecVerifyMode::UNIFORM_EXPANDED_VERIFY ||
      (mode == TargetSpecVerifyMode::PYTHON_EXPANDED_VERIFY &&
       is_python_model_impl);
  capabilities.uses_chunked_prefill =
      mode == TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL ||
      capabilities.supports_explicit_replay_update;
  return capabilities;
}

int64_t speculative_verify_block_table_capacity(int64_t max_position_embeddings,
                                                int64_t block_size) {
  CHECK_GT(max_position_embeddings, 0);
  CHECK_GT(block_size, 0);
  return util::ceil_div(max_position_embeddings, block_size) + 1;
}

void materialize_speculative_verify_tokens(
    const torch::Tensor& verify_tokens,
    const std::vector<torch::Tensor>& draft_token_sources) {
  if (draft_token_sources.empty()) {
    return;
  }
  CHECK(verify_tokens.defined());
  CHECK_EQ(verify_tokens.dim(), 1);
  const int64_t verify_width =
      static_cast<int64_t>(draft_token_sources.size()) + 1;
  CHECK_EQ(verify_tokens.numel() % verify_width, 0);
  const int64_t batch_size = verify_tokens.numel() / verify_width;
  torch::Tensor verify_rows = verify_tokens.view({batch_size, verify_width});
  for (size_t step = 0; step < draft_token_sources.size(); ++step) {
    const torch::Tensor& source = draft_token_sources[step];
    CHECK(source.defined());
    CHECK_EQ(source.numel(), batch_size);
    verify_rows.select(/*dim=*/1, static_cast<int64_t>(step) + 1)
        .copy_(source.flatten(), /*non_blocking=*/true);
  }
}

bool has_speculative_verify_block_table_layout(
    const torch::Tensor& block_tables,
    const std::vector<torch::Tensor>& multi_block_tables,
    int64_t num_sequences) {
  if (num_sequences <= 0) {
    return false;
  }
  const auto is_compatible = [num_sequences](const torch::Tensor& table) {
    return table.defined() && table.dim() == 2 &&
           table.size(0) == num_sequences && table.size(1) > 0 &&
           table.scalar_type() == torch::kInt32;
  };
  if (block_tables.defined()) {
    return is_compatible(block_tables);
  }
  return !multi_block_tables.empty() && std::all_of(multi_block_tables.begin(),
                                                    multi_block_tables.end(),
                                                    is_compatible);
}

// In-place variant of materialize_speculative_verify_tokens for graph replay:
// writes into whichever token tensor the graph owns and returns it.
torch::Tensor materialize_graph_speculative_verify_tokens(
    const torch::Tensor& tokens,
    const GraphInputView& graph_input) {
  torch::Tensor verify_tokens = graph_input.input_tokens_override.defined()
                                    ? graph_input.input_tokens_override
                                    : tokens;
  materialize_speculative_verify_tokens(
      verify_tokens, graph_input.spec_verify_draft_token_sources);
  return verify_tokens;
}

SampleOutput run_rejection_sampling(const SamplingParameters& sampling_params,
                                    const DraftProposal& draft_proposal,
                                    const torch::Tensor& target_logits,
                                    const ForwardOutput& target_output) {
  const RejectionSampler sampler(
      sampling_params, target_output.logprobs, target_output.max_top_logprobs);

  SampleOutput sample_output = sampler.forward(draft_proposal, target_logits);

  const torch::Tensor& embeddings = target_output.sample_output.embeddings;
  sample_output.embeddings = embeddings.view(
      {target_logits.size(0), target_logits.size(1), embeddings.size(-1)});
  return sample_output;
}

}  // namespace xllm::spec_verify
