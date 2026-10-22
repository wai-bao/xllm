/* Copyright 2025-2026 The xLLM Authors.

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
#include <optional>
#include <vector>

#include "core/framework/batch/forward_input_builder.h"
#include "core/framework/model/mtp_topk_state.h"
#include "core/framework/sampling/draft_proposal.h"
#include "util/slice.h"

namespace xllm {

class ModelParamsView;
class LlmForwardInput;
struct SamplingParameters;

namespace specBuilder {

// Empty DP ranks participate in block-draft execution with a complete query
// block. Fill their execution counts on every rank and materialize an idle
// rank's host input without writing any real cache slots.
void materialize_empty_block_draft_rows(LlmForwardInput& input,
                                        int32_t query_width,
                                        int32_t mask_token_id);

// Reuses the NPU MTP draft-step row-selection flow for every backend. Backend
// state implementations only interpret index_select_rows() for their payload.
MtpTopkStatePtr select_mtp_topk_state_for_next_step(
    const MtpTopkStatePtr& state,
    const SamplingParameters& sampling_params);

}  // namespace specBuilder

}  // namespace xllm
