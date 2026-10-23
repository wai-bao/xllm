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

#include "core/framework/sampling/draft_sampling.h"

#include <utility>

#include "core/framework/sampling/sampling_params.h"

namespace xllm {

void configure_draft_sampling(SamplingParameters& params,
                              DraftSamplingMode mode,
                              torch::Tensor greedy_do_sample) {
  // Target-only constraints never apply to draft proposals.
  params.use_beam_search = false;
  // vLLM applies minimum length to target sampling, not draft proposals.
  params.min_token_positions = torch::Tensor();
  params.min_tokens_remaining = torch::Tensor();
  params.min_token_ids = torch::Tensor();
  if (mode == DraftSamplingMode::GREEDY) {
    if (greedy_do_sample.defined()) {
      params.do_sample = std::move(greedy_do_sample);
    } else if (params.do_sample.defined()) {
      params.do_sample = torch::zeros_like(params.do_sample);
    }
    params.sampling_mode = SamplingMode::GREEDY;
    clear_sampling_transforms(params);
  }
  params.logprobs = false;
  params.max_top_logprobs = 0;
  params.return_logits = mode == DraftSamplingMode::PROBABILISTIC;
}

}  // namespace xllm
