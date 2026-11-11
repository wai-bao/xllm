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
#include <vector>

// Runtime target-verification policy and output transformation contract.

namespace xllm {

// Defined in models/model_registry.h; forward-declared to keep this
// lightweight header out of the model-registry dependency cone.
enum class TargetSpecVerifyMode : int8_t;

class DraftProposal;
class GraphInputView;
class SampleOutput;
class ForwardOutput;
class SamplingParameters;

namespace spec_verify {

// Immutable target-side execution contract resolved once after model loading.
// The worker consumes these capabilities without branching on model type.
struct TargetSpecVerifyCapabilities {
  bool supports_explicit_replay_update = false;
  bool requires_uniform_validate_width = false;
  bool uses_chunked_prefill = false;
  // Python-impl targets rewrite the validate input's linear-state ids: the
  // uniform fused verifier keeps them sequence-scoped, every other mode
  // replicates them per verify token.
  bool rewrites_validate_linear_state_ids = false;
};

TargetSpecVerifyCapabilities resolve_target_spec_verify_capabilities(
    TargetSpecVerifyMode mode,
    bool is_python_model_impl);

// Expanded verify needs a primary block-table tensor for its generic graph
// metadata. Model-managed cache layouts instead carry every real cache table
// in multi_block_tables, so the primary table may be intentionally absent.
bool has_speculative_verify_block_table_layout(
    const torch::Tensor& block_tables,
    const std::vector<torch::Tensor>& multi_block_tables,
    int64_t num_sequences);

// Shared allocation/launch width for target verification block tables. The
// extra entry covers the speculative token that can cross a block boundary.
int64_t speculative_verify_block_table_capacity(int64_t max_position_embeddings,
                                                int64_t block_size);

// Eager execution materializes the same proposer columns used by graph replay.
// Writes the draft columns into verify_tokens in place; allocates nothing.
void materialize_speculative_verify_tokens(
    const torch::Tensor& verify_tokens,
    const std::vector<torch::Tensor>& draft_token_sources);

// In-place graph replay variant: returns whichever token tensor the graph
// owns (the override when present) after filling its deferred draft columns.
torch::Tensor materialize_graph_speculative_verify_tokens(
    const torch::Tensor& tokens,
    const GraphInputView& graph_input);

// `target_hidden_states`, when defined, is attached to the output as the
// [batch, val_tokens, hidden] view.
SampleOutput run_rejection_sampling(SamplingParameters sampling_params,
                                    const DraftProposal& draft_proposal,
                                    torch::Tensor target_logits,
                                    const torch::Tensor& target_hidden_states);

}  // namespace spec_verify

// Keep the logical accepted count for recurrent checkpoint selection. The
// device tensor is materialized only when replay does not supply it directly.
template <typename Params>
void publish_accepted_prefix_lengths(
    Params& input_params,
    const std::vector<int32_t>& accepted_prefix_lengths,
    const torch::TensorOptions& token_options,
    bool publish_device_tensor);
#if defined(USE_NPU)
// Build the host and device parts of expanded target-verification metadata.
// The host layout may be prepared before the packed replay inputs are bound.
template <typename Params>
bool prepare_expanded_spec_verify_graph_host_input(Params& input_params);
// Prepares the host layout (CHECKing that the required seq lens exist) and
// binds the device graph inputs. Callers decide when expanded metadata is
// needed: MTP also uses it in eager mode, while DFlash uses it in graph mode.
template <typename Params>
void build_expanded_spec_verify_graph_input(Params& input_params,
                                            const torch::Device& device,
                                            int32_t block_size);
#endif

}  // namespace xllm
