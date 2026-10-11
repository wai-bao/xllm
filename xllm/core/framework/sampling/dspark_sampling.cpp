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

#include "core/framework/sampling/dspark_sampling.h"

#include "core/framework/sampling/draft_sampling_prepare.h"
#include "core/framework/sampling/sampler.h"

namespace xllm {

DraftProposal sample_dspark_block(
    torch::Tensor base_logits,
    const torch::Tensor& anchor_token_ids,
    SamplingParameters sampling,
    bool need_draft_logits,
    const std::function<torch::Tensor(const torch::Tensor&)>& markov_bias,
    const std::function<void(torch::Tensor&)>& synchronize_tokens,
    torch::Tensor tokens) {
  CHECK_EQ(base_logits.dim(), 3);
  CHECK_EQ(anchor_token_ids.dim(), 1);
  CHECK_EQ(anchor_token_ids.size(0), base_logits.size(0));
  CHECK_EQ(anchor_token_ids.scalar_type(), torch::kInt64);
  CHECK_EQ(anchor_token_ids.device(), base_logits.device());
  const int64_t rows = base_logits.size(0);
  const int64_t steps = base_logits.size(1);
  const int64_t vocab = base_logits.size(2);
  CHECK_GT(steps, 0);
  sampling.selected_token_idxes =
      torch::empty({0}, anchor_token_ids.options().dtype(torch::kInt32));
  sampling.sample_idxes = sampling.selected_token_idxes;
  suppress_score_outputs(sampling);
  sampling.use_beam_search = false;
  if (!tokens.defined()) {
    tokens = torch::empty({rows, steps}, anchor_token_ids.options());
  } else {
    CHECK_EQ(tokens.sizes(), torch::IntArrayRef({rows, steps}));
    CHECK_EQ(tokens.scalar_type(), torch::kInt64);
    CHECK_EQ(tokens.device(), base_logits.device());
  }
  // Write each step's sampled logits straight into the [B, steps, V] result
  // slice instead of stacking per-step tensors afterwards: the stack would
  // allocate a second [B, steps, V] buffer while every step tensor is still
  // alive.
  torch::Tensor stacked_logits =
      need_draft_logits
          ? torch::empty({rows, steps, vocab}, base_logits.options())
          : torch::Tensor();
  auto previous = anchor_token_ids;
  for (int64_t step = 0; step < steps; ++step) {
    auto bias = markov_bias(previous);
    CHECK_EQ(bias.sizes(), torch::IntArrayRef({rows, vocab}))
        << "DSpark reduced vocabulary requires draft-to-target remapping.";
    // Sample in place inside the stacked slice when logits are needed: the
    // sampler mutates its input through the reference, so the per-step [B, V]
    // temporary (select + bias) and the post-sample copy both disappear.
    // The greedy path mutates base_logits directly (out-of-place before).
    torch::Tensor step_input = need_draft_logits
                                   ? stacked_logits.select(1, step)
                                   : base_logits.select(1, step);
    if (need_draft_logits) {
      // One fused kernel writes bias + base into the private stacked slice.
      torch::add_out(step_input, base_logits.select(1, step), bias);
    } else {
      step_input.add_(bias);
    }
    auto sampled = Sampler::forward(step_input, sampling);
    synchronize_tokens(sampled.next_tokens);
    tokens.select(1, step).copy_(sampled.next_tokens);
    previous = sampled.next_tokens;
  }
  return need_draft_logits
             ? DraftProposal(std::move(tokens), std::move(stacked_logits))
             : DraftProposal(std::move(tokens));
}

}  // namespace xllm
