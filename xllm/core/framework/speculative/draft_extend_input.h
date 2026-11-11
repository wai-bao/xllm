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

#include <vector>

#include "core/framework/model/mtp_topk_state.h"
#include "core/framework/speculative/draft_input.h"

namespace xllm {

struct SamplingParameters;

class SampleOutput;
struct ModelOutput;
class LlmForwardInput;
class VlmForwardInput;
class ForwardOutput;

// Selects the drafter's feed from `get_hidden_states_for_drafter(model_output)`
// and exposes it through `embeddings` (plus `selected_embeddings` on target
// prefill).
void output_spec_hidden_states(
    SampleOutput& sample_output,
    const ModelOutput& model_output,
    const torch::Tensor& logits_indices,
    const torch::Tensor& gathered_sample_hidden_states,
    bool is_target_prefill,
    bool cp_enabled);

void clear_all_output_embeddings(ForwardOutput& output);

// Exposes the target prefill's selected hidden as `embeddings` for the PD
// handoff to the first draft-extend step.
void prepare_first_draft_inputs(ForwardOutput& output);

// Reuses the NPU MTP draft-step row-selection flow for every backend. Backend
// state implementations only interpret index_select_rows() for their payload.
MtpTopkStatePtr select_mtp_topk_state_for_next_step(
    const MtpTopkStatePtr& state,
    const SamplingParameters& sampling_params);

// Shift each prefill query left and append its target-sampled root token into
// caller-owned CPU storage (pinned for H2D).
template <typename Params>
void fill_mtp_prefill_token_ids(const torch::Tensor& token_ids_host,
                                const Params& params,
                                const std::vector<int32_t>& extra_token_ids,
                                torch::Tensor& destination);

struct MtpDraftExtendRows {
  torch::Tensor token_ids;
  torch::Tensor hidden_states;
  torch::Tensor logits_indices;
};

// Pair committed next tokens with their producing target rows; only the last
// accepted row is sampled. Every sequence contributes the same fixed query
// length.
MtpDraftExtendRows build_mtp_draft_extend_rows(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& target_hidden_states);

// Construct a draft input from the verifier's row layout. Target execution,
// graph, transfer and recurrent-state controls never cross this boundary; the
// caller orders this snapshot after verification on the metadata stream.
template <typename TargetInput>
TargetInput build_mtp_draft_extend_input(const TargetInput& validate_input,
                                         const MtpDraftExtendRows& rows,
                                         SamplingParameters sampling_params);

// Copies the refresh metadata independently on the current stream; the caller
// is responsible for synchronizing with updates to the source buffers.
template <typename TargetInput>
void clone_mtp_draft_metadata(TargetInput& input);

}  // namespace xllm
