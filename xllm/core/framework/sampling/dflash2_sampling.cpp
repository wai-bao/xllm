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

#include "core/framework/sampling/dflash2_sampling.h"

#include <glog/logging.h>

#include "core/framework/model/causal_lm.h"
#include "core/framework/sampling/draft_sampling_mode.h"
#include "core/framework/sampling/gumbel_sampling.h"
#include "core/framework/sampling/sampling_params.h"

namespace xllm {

DraftProposal sample_dflash2_path(const DFlash2CandidateOutput& candidates,
                                  const SamplingParameters& sampling_params,
                                  DraftSamplingMode mode,
                                  const torch::Tensor& gumbel_noise) {
  const bool need_logits =
      draft_logits_required(mode, sampling_params.verify_mode());
  CHECK_EQ(candidates.candidate_ids.dim(), 3);
  CHECK_EQ(candidates.edge_logits.dim(), 4);
  const int64_t batch_size = candidates.candidate_ids.size(0);
  const int64_t num_steps = candidates.candidate_ids.size(1);
  const int64_t top_k = candidates.candidate_ids.size(2);
  CHECK_EQ(candidates.edge_logits.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k, top_k}));
  CHECK_EQ(gumbel_noise.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k}));
  const torch::Device device = candidates.edge_logits.device();
  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  // A row-wise normalizer does not change Gumbel-max. Retain the selected
  // conditional logits and compact candidate support for lazy verification.
  torch::Tensor edge_logits = candidates.edge_logits.to(torch::kFloat32);
  if (sampling_params.temperatures.defined()) {
    apply_selector_temperatures(
        edge_logits, sampling_params.temperatures, batch_size);
  }

  torch::Tensor token_ids =
      torch::empty({batch_size, num_steps}, candidates.candidate_ids.options());
  torch::Tensor path_logits;
  if (need_logits) {
    path_logits = torch::empty({batch_size, num_steps, top_k}, float_options);
  }
  torch::Tensor path_indices =
      need_logits ? torch::empty_like(token_ids) : torch::Tensor();
  torch::Tensor previous_indices =
      torch::zeros({batch_size}, candidates.candidate_ids.options());

  for (int64_t step = 0; step < num_steps; ++step) {
    torch::Tensor edge = edge_logits.select(/*dim=*/1, /*index=*/step);
    torch::Tensor gather_indices = previous_indices.view({batch_size, 1, 1})
                                       .expand({batch_size, 1, top_k});
    // Gather straight into the stored path slice (the same
    // slice-destination pattern the dspark chain uses): no per-step
    // [B, top_k] temporary, no copy back.
    torch::Tensor row_logits;
    if (need_logits) {
      row_logits = path_logits.select(/*dim=*/1, /*index=*/step).unsqueeze(1);
      torch::gather_out(row_logits, edge, /*dim=*/1, gather_indices);
      row_logits = row_logits.squeeze(/*dim=*/1);
    } else {
      row_logits = edge.gather(/*dim=*/1, gather_indices).squeeze(/*dim=*/1);
    }
    torch::Tensor sampled_indices = gumbel_argmax(
        row_logits, gumbel_noise.select(/*dim=*/1, /*index=*/step));

    torch::Tensor step_candidates =
        candidates.candidate_ids.select(/*dim=*/1, /*index=*/step);
    torch::Tensor sampled_tokens =
        step_candidates.gather(/*dim=*/1, sampled_indices.view({-1, 1}))
            .view({-1});
    if (need_logits) {
      path_indices.select(/*dim=*/1, /*index=*/step).copy_(sampled_indices);
    }
    token_ids.select(/*dim=*/1, /*index=*/step).copy_(sampled_tokens);
    previous_indices = sampled_indices;
  }

  if (!need_logits) {
    return DraftProposal(std::move(token_ids));
  }
  return DraftProposal(std::move(token_ids),
                       std::move(path_logits),
                       candidates.candidate_ids,
                       std::move(path_indices));
}

}  // namespace xllm
