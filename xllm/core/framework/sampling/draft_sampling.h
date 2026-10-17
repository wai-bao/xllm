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

#include "core/framework/sampling/draft_sampling_mode.h"

namespace xllm {

struct SamplingParameters;

// Configures proposal sampling without mutating tensors shared with the target.
// A supplied greedy_do_sample is a caller-owned, all-false mask with the sample
// row layout; pipelines can reuse fixed storage instead of allocating a mask.
// require_probs_for_greedy preserves adaptive confidence for probabilistic
// drafts whose target batch is greedy. Forced-greedy drafts never return q.
void configure_draft_sampling(SamplingParameters& params,
                              DraftSamplingMode mode,
                              bool require_probs_for_greedy,
                              torch::Tensor greedy_do_sample = torch::Tensor());

}  // namespace xllm
