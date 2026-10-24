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

#include "core/framework/speculative/draft_proposal.h"
#include "runtime/dflash_worker_impl.h"

namespace xllm {

class ProcessGroup;

// Samples a path through the DFlash2 candidate tree with Gumbel-noise
// argmax; undefined gumbel noise (the greedy-draft shortcut) degrades every
// step to the raw argmax. Shared by the DFlash2 worker and the task
// pipeline; TP noise consensus is the caller's.
DraftProposal sample_candidate_path(
    const DFlash2CandidateOutput& candidates,
    const SamplingParameters& sampling_params,
    const torch::Tensor& gumbel_noise,
    int64_t vocab_size,
    torch::Tensor token_ids_out = torch::Tensor());
// Draws the [num_sample_seqs, num_steps, num_classes] selector noise for
// probabilistic block drafts (the DFlash2 candidate walk). An all-greedy
// batch under probabilistic mode keeps its greedy semantics — no noise,
// mirroring Sampler::forward's GREEDY leg — and returns an undefined tensor.
// When `sampling_group` is non-null the noise is broadcast from rank 0: the
// noise drives path selection, so every rank must walk the same candidates.
// Lives here (not in framework/sampling) because the TP consensus policy is
// DFlash2-walk knowledge; gumbel_noise itself stays communication-free.
torch::Tensor draw_draft_gumbel_noise(const SamplingParameters& sampling,
                                      int64_t num_steps,
                                      int64_t num_classes,
                                      const torch::Device& device,
                                      ProcessGroup* sampling_group);

class DFlash2WorkerImpl final : public DFlashWorkerImpl {
 public:
  DFlash2WorkerImpl(const ParallelArgs& parallel_args,
                    const torch::Device& device,
                    const runtime::Options& options);

  ~DFlash2WorkerImpl() override = default;

 protected:
  DraftProposal run_decode_draft(const LlmForwardInput& input,
                                 LlmForwardInput& validate_input) override;
};

}  // namespace xllm
