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

#include <vector>

#include "framework/sampling/sampling_params.h"
#if defined(USE_NPU)
#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#endif
#include "kernels/ops_api.h"

namespace xllm {

// One FP32 materialization of the logits plus its shared logsumexp
// normalizer: the selected gather and the top-k rows both subtract it, so
// the full-vocab reduction runs once per call. Convert before reduction:
// casting a BF16 normalizer afterward cannot recover lost precision.
struct NormalizedLogits {
  torch::Tensor values;
  torch::Tensor normalizer;
};
NormalizedLogits normalized_logits(const torch::Tensor& logits);

// The gather-selected-then-subtract kernel: the single home for "selected
// token logprob = value at the token id minus its row normalizer". Negative
// ids clamp to row 0 (callers mask the gathered slots afterwards).
torch::Tensor selected_logprobs(const NormalizedLogits& normalized,
                                const torch::Tensor& token_ids);

// Normalize only selected tokens, reducing in FP32 without a dense logprob
// output. token_ids has logits' leading dimensions and excludes the vocab dim.
torch::Tensor compute_selected_logprobs(const torch::Tensor& logits,
                                        const torch::Tensor& token_ids);

// Selected-token scores [B] in the log domain, preferring an already
// normalized tensor (logprobs, then probs) over re-normalizing the logits.
// Callers derive probs as scores.exp().
torch::Tensor compute_selected_scores(const torch::Tensor& logprobs,
                                      const torch::Tensor& logits,
                                      const torch::Tensor& sample_indices,
                                      const torch::Tensor& probs = {});

// The selected-score emission contract: probs ≡ scores.exp() cast to the
// logits dtype, logprobs the raw view. One owner so every sampler and the
// rejection verify path cannot drift.
void emit_selected_scores(SampleOutput& output,
                          const SamplingParameters& params,
                          const torch::Tensor& scores,
                          torch::Dtype logits_dtype);

void apply_frequency_presence_penalties(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& unique_token_counts,
    const torch::Tensor& frequency_penalties,
    const torch::Tensor& presence_penalties);

void apply_repetition_penalties(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& penalties,
    const torch::Tensor& unique_token_ids_lens = {});

void apply_temperatures(torch::Tensor& logits,
                        const torch::Tensor& temperatures);

void apply_min_p(torch::Tensor& logits, const torch::Tensor& min_p);

void apply_top_k_top_p(torch::Tensor& logits,
                       const torch::Tensor& temperatures,
                       const torch::Tensor& top_k,
                       const torch::Tensor& top_p);

}  // namespace xllm
