/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/sampling/sampling_params.h"

namespace xllm {

// Gumbel(0, 1) noise with the shape and device of `like` (fp32). Uses the
// global RNG unless a generator is given (the seeded sampler path passes its
// private one).
torch::Tensor gumbel_noise_like(
    const torch::Tensor& like,
    std::optional<torch::Generator> generator = std::nullopt);

// Greedy row mask [batch] for draft sampling: all-greedy batches treat every
// row as greedy (do_sample may carry request-side values in GREEDY mode),
// mixed batches invert do_sample.
torch::Tensor greedy_draft_rows(const SamplingParameters& sampling_params);

// Draws Gumbel(0, 1) noise shaped [batch_size, num_steps, num_classes] on
// `device`. `greedy_rows` [batch_size] bool zeroes those rows' noise so
// gumbel_argmax collapses onto the plain argmax for them; `all_greedy` skips
// the draw entirely (all-greedy rows only ever argmax the zero-noise logits:
// deterministic zeros on every rank, so the draw and the cross-rank consensus
// on it are both skippable). The caller owns cross-rank consensus: per-rank
// RNG divergence would fork the sampled path, so under TP > 1 the caller must
// broadcast the result from a single root before feeding it to gumbel_argmax.
torch::Tensor sample_gumbel_noise(int64_t num_steps,
                                  int64_t num_classes,
                                  const torch::Tensor& greedy_rows,
                                  bool all_greedy,
                                  const torch::Device& device);

// Applies temperatures in-place to `logits` [batch, ..., num_classes] with the
// generic sampler's semantics: temperature 0 means greedy and must not divide,
// so it is substituted with 1 before scaling.
void apply_selector_temperatures(torch::Tensor& logits,
                                 const torch::Tensor& temperatures,
                                 int64_t batch_size);

// Single-row Gumbel-max sampling: argmax(logits + g) draws from exactly
// categorical(softmax(logits)). Unnormalized logits are fine — softmax is
// monotone, so any constant per-row offset cancels in the argmax; -inf
// entries stay -inf and never win. Returns sampled class indices
// [batch_size] on the logits device. Stays on the current stream and never
// blocks the host, which is what makes it ACL-graph safe.
torch::Tensor gumbel_argmax(const torch::Tensor& logits,
                            const torch::Tensor& gumbel_noise);

}  // namespace xllm
