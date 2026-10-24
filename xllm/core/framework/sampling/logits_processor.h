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

#include "core/framework/sampling/sampling_params.h"
#include "core/util/tensor_helper.h"

namespace xllm {

// Generic transform stages shared by LogitsProcessor::process_logits and the
// rejection sampler's verify path. Two contracts:
// - Penalty stages (apply_*_penalties, apply_min_token_block) are
//   rank-agnostic: they operate along the last dim and take per-sequence
//   tables, broadcast through stride-0 views on the grouped domain.
// - Argmax-preserving filters (apply_temperatures, apply_min_p,
//   apply_top_k_top_p) take flat 2-D [rows, vocab] logits with per-row
//   parameter tensors (the verify caller expands the per-sequence scalars
//   once); the flat contract is what the fused device kernels accept.

// Shape of a [batch] per-sequence tensor padded so it broadcasts against
// [batch, ..., vocab] logits: right-aligned broadcasting would align the
// batch dim with the inner dims, so the leading dims are padded to the
// logits' rank instead. Shared by the temperature/min-p transforms and the
// greedy-row noise masking.
inline std::vector<int64_t> pad_batch_shape(int64_t batch, int64_t rank) {
  std::vector<int64_t> shape{/*batch=*/batch};
  shape.insert(shape.end(), rank - 1, 1);
  return shape;
}

void apply_frequency_presence_penalties(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& unique_token_counts,
    const torch::Tensor& frequency_penalties,
    const torch::Tensor& presence_penalties);
void apply_min_token_block(torch::Tensor& logits,
                           const torch::Tensor& min_token_ids,
                           const torch::Tensor& min_tokens_remaining);
void apply_repetition_penalties(torch::Tensor& logits,
                                const torch::Tensor& unique_token_ids,
                                const torch::Tensor& penalties,
                                const torch::Tensor& unique_token_ids_lens);
void apply_min_p(torch::Tensor& logits, const torch::Tensor& min_p);
void apply_top_k_top_p(torch::Tensor& logits,
                       const torch::Tensor& top_k,
                       const torch::Tensor& top_p);
void apply_token_bitmask_inplace(torch::Tensor& logits,
                                 const torch::Tensor& bitmask);

// Stateless owner of the in-place sampling transforms applied to logits
// before sampling (bias, penalties, temperature, min-p, top-k/top-p, token
// filters): the transform inputs arrive per call in SamplingParameters, so
// every consumer shares one fixed transform order. The Sampler stays a pure
// consumer of the already-processed logits.
class LogitsProcessor final {
 public:
  // In-place sampling transforms on [batch_size, vocab_size] logits:
  // penalties first, then the argmax-preserving transforms (temperature,
  // min-p, top-k/top-p) and the token filter mask/bitmask. A greedy batch
  // that reports no distribution values skips the argmax-preserving stage.
  static void process_logits(torch::Tensor& logits,
                             const SamplingParameters& params,
                             const torch::Tensor& filter_mask = {});

  // In-place logits bias and repetition/frequency/presence penalties. The
  // first stage of process_logits, shared by paths that skip the later
  // transforms (e.g. the RecSampler fast path).
  static void apply_sampling_penalties(torch::Tensor& logits,
                                       const SamplingParameters& params);

  // In-place temperature only: for callers outside the full transform
  // pipeline (e.g. the DFlash2 selector's top-k distribution, where the
  // target-side penalties and grammar constraints must not apply). Broadcasts
  // the [batch] temperatures against [batch, ..., vocab] logits of any rank.
  static void apply_temperatures(torch::Tensor& logits,
                                 const torch::Tensor& temperatures);
};

}  // namespace xllm
