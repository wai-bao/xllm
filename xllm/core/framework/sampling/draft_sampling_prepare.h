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

#include <utility>

#include "framework/sampling/draft_sampling_mode.h"
#include "framework/sampling/sampling_params.h"

namespace xllm {

// Verification gathers scores after final token selection: suppress the
// validate-side sampler's logprob and prob outputs. Dispatch flags stay
// untouched (they are re-derived from do_sample downstream).
inline void suppress_score_outputs(SamplingParameters& params) {
  params.logprobs = false;
  params.max_top_logprobs = 0;
  params.return_probs = false;
}

// Force the greedy dispatch policy on already-populated params: the flag
// pair plus the zeroed do_sample mask (greedy_do_sample preserves device and
// dtype when supplied; undefined falls back to zeros_like). Score flags pass
// through untouched — verify-side forcing keeps them, draft-side
// prepare_draft_sampling suppresses them.
inline void force_greedy_dispatch(SamplingParameters& params,
                                  torch::Tensor greedy_do_sample = {}) {
  params.all_random_sample = false;
  params.all_greedy_sample = true;
  if (params.do_sample.defined()) {
    params.do_sample = greedy_do_sample.defined()
                           ? std::move(greedy_do_sample)
                           : torch::zeros_like(params.do_sample);
  }
}

// Keep proposal distributions separate from adaptive selected-token scores:
// draft-side sampling suppresses the score outputs, then GREEDY dispatches
// through force_greedy_dispatch (which owns the greedy_do_sample contract).
inline void prepare_draft_sampling(SamplingParameters& params,
                                   DraftSamplingMode mode,
                                   bool need_selected_probs = false,
                                   torch::Tensor greedy_do_sample = {}) {
  suppress_score_outputs(params);
  if (mode == DraftSamplingMode::GREEDY) {
    force_greedy_dispatch(params, std::move(greedy_do_sample));
  } else {
    params.return_probs = need_selected_probs;
  }
}

}  // namespace xllm
