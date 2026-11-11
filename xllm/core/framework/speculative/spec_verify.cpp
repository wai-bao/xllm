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

#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "core/framework/model/model_params_view.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/speculative/rejection_sampler.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/vlm_forward_params.h"
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
  capabilities.rewrites_validate_linear_state_ids = is_python_model_impl;
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

// `target_hidden_states`, when defined, is attached to the output as the
// [batch, val_tokens, hidden] view.
SampleOutput run_rejection_sampling(SamplingParameters sampling_params,
                                    const DraftProposal& draft_proposal,
                                    torch::Tensor target_logits,
                                    const torch::Tensor& target_hidden_states) {
  if (target_logits.dim() == 3) {
    // [batch, val_tokens, vocab] callers (the eager MTP validate path) flatten
    // into the [batch * val_tokens, vocab] rows rejection_sample expects.
    target_logits = target_logits.flatten(0, 1).contiguous();
  }
  SampleOutput sample_output = rejection_sample(
      std::move(sampling_params), draft_proposal, std::move(target_logits));

  if (target_hidden_states.defined()) {
    // Reuse next_tokens' [batch, num_val_tokens] shape instead of
    // re-deriving the width.
    const torch::Tensor& next_tokens = sample_output.next_tokens;
    sample_output.embeddings =
        target_hidden_states.view({next_tokens.size(0),
                                   next_tokens.size(1),
                                   target_hidden_states.size(-1)});
  }
  return sample_output;
}

}  // namespace xllm::spec_verify

namespace xllm {

template <typename Params>
void publish_accepted_prefix_lengths(
    Params& input_params,
    const std::vector<int32_t>& accepted_prefix_lengths,
    const torch::TensorOptions& token_options,
    bool publish_device_tensor) {
  // num_accepted_tokens is a logical checkpoint index, not the physical
  // conv-state capacity. Clamping it commits a stale checkpoint (issue #2247).
  input_params.num_accepted_tokens_host.assign(accepted_prefix_lengths.begin(),
                                               accepted_prefix_lengths.end());
  if (publish_device_tensor) {
    input_params.num_accepted_tokens =
        async_h2d_tensor(accepted_prefix_lengths, token_options);
  }
}

#if defined(USE_NPU)
template <typename Params>
bool prepare_expanded_spec_verify_graph_host_input(Params& input_params) {
  input_params.graph.use_expanded_decode_for_spec_verify_attention = false;
  input_params.graph.expanded_kv_seq_lens = torch::Tensor();
  input_params.graph.expanded_block_tables = torch::Tensor();
  input_params.graph.expanded_paged_kv_indptr = torch::Tensor();
  input_params.graph.expanded_paged_kv_indices = torch::Tensor();
  input_params.graph.expanded_paged_kv_last_page_len = torch::Tensor();
  input_params.graph.expanded_tiling_data = torch::Tensor();
  input_params.graph.expanded_kv_seq_lens_vec.clear();
  if (!input_params.is_spec_verify ||
      !input_params.meta.batch_forward_type.is_chunked_prefill()) {
    return false;
  }

  const std::vector<int32_t>& q_seq_lens =
      input_params.attention.host.q_seq_lens;
  const std::vector<int32_t>& kv_seq_lens =
      input_params.attention.host.kv_seq_lens;
  if (q_seq_lens.empty() || kv_seq_lens.empty()) {
    return false;
  }
  input_params.graph.expanded_kv_seq_lens_vec =
      layer::ExpandedDecodeMetadataBuilder::build_tokenwise_kv_seq_lens(
          q_seq_lens, kv_seq_lens);
  if (input_params.graph.expanded_kv_seq_lens_vec.empty()) {
    return false;
  }
  input_params.graph.use_expanded_decode_for_spec_verify_attention = true;
  return true;
}

namespace {
template <typename Params>
void bind_expanded_spec_verify_graph_input(Params& input_params,
                                           const torch::Device& device,
                                           int32_t block_size) {
  if (!input_params.graph.use_expanded_decode_for_spec_verify_attention) {
    return;
  }
  CHECK(input_params.attention.device.block_tables.defined())
      << "spec verify block tables must be rebuilt before graph input";
  const std::vector<int32_t>& q_seq_lens =
      input_params.attention.host.q_seq_lens;
  CHECK_GE(input_params.attention.device.block_tables.size(0),
           static_cast<int64_t>(q_seq_lens.size()))
      << "spec verify block table rows are fewer than sequences";

  if (!input_params.graph.expanded_kv_seq_lens.defined()) {
    input_params.graph.expanded_kv_seq_lens =
        async_h2d_tensor(input_params.graph.expanded_kv_seq_lens_vec, device);
  }
  // One repeat_interleave replaces per-token select + stack: a single kernel
  // with no intermediate row-view TensorImpls.
  torch::Tensor expanded_block_tables =
      input_params.attention.device.block_tables.repeat_interleave(
          async_h2d_tensor(
              q_seq_lens,
              torch::TensorOptions()
                  .dtype(torch::kInt64)
                  .device(input_params.attention.device.block_tables.device())),
          /*dim=*/0);
  layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      ModelParamsView(input_params),
      input_params.graph.expanded_kv_seq_lens,
      expanded_block_tables,
      input_params.graph.expanded_kv_seq_lens_vec,
      block_size);
}
}  // namespace

template <typename Params>
void build_expanded_spec_verify_graph_input(Params& input_params,
                                            const torch::Device& device,
                                            int32_t block_size) {
  CHECK(prepare_expanded_spec_verify_graph_host_input(input_params))
      << "spec-verify chunked prefill requires q/kv seq lens";
  bind_expanded_spec_verify_graph_input(input_params, device, block_size);
}

// Explicit instantiations for the parameter-view templates shared by the
// LLM/VLM MTP workers and DFlash.
template void publish_accepted_prefix_lengths(LlmModelParams&,
                                              const std::vector<int32_t>&,
                                              const torch::TensorOptions&,
                                              bool);
template void publish_accepted_prefix_lengths(VlmModelParams&,
                                              const std::vector<int32_t>&,
                                              const torch::TensorOptions&,
                                              bool);
template void build_expanded_spec_verify_graph_input(LlmModelParams&,
                                                     const torch::Device&,
                                                     int32_t);
template void build_expanded_spec_verify_graph_input(VlmModelParams&,
                                                     const torch::Device&,
                                                     int32_t);
template bool prepare_expanded_spec_verify_graph_host_input(LlmModelParams&);
template bool prepare_expanded_spec_verify_graph_host_input(VlmModelParams&);
#endif  // USE_NPU

}  // namespace xllm
