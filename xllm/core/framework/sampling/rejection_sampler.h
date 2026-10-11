/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/sampling/draft_proposal.h"
#include "logits_utils.h"
#include "sampling_params.h"

namespace xllm {

// Stateless verification primitives for speculative decoding. The single
// production entry is verify(); the sample kernels it dispatches to are
// exposed for reuse and testing.
class RejectionSampler final {
 public:
  // Common exact-chain verification. Inputs use batch-major layout and
  // already-processed target logits. Greedy no-logprobs batches match target
  // token ids [batch, n_spec + 1] directly (the NPU kernel converts ids in
  // UB); other modes fall back to the logits rejection path. Optional
  // lengths and invalid positions terminate the draft prefix before output
  // logprobs are gathered. When target_embeddings is given, the output
  // carries it as a batch-major [B, n_spec + 1, H] view — the single home
  // for the verify-output embedding contract.
  static SampleOutput verify(const SamplingParameters& params,
                             const DraftProposal& proposal,
                             const torch::Tensor& target_logits,
                             const torch::Tensor& target_tokens,
                             const torch::Tensor& draft_lengths = {},
                             const torch::Tensor& invalid_draft = {},
                             const torch::Tensor& target_embeddings = {});

  // Returns the prefix-masked sampled tokens [B, n_spec + 1]: each row keeps
  // its accepted drafts, the first-reject resample, and (when fully accepted)
  // the bonus; every later position is -1. uniform_rand arrives already in
  // the log domain: the production dispatch takes the log in place on the
  // fresh random tensor. When normalized is provided (verify's full-width
  // FP32 pass over the same logits), the acceptance gather consumes it
  // instead of re-normalizing; the standalone path normalizes alone.
  static torch::Tensor random_sample(
      const DraftProposal& draft_proposal,
      const torch::Tensor& target_logits,
      const torch::Tensor& uniform_rand,
      const torch::Tensor& bonus_token_ids,
      const NormalizedLogits* normalized = nullptr);

  // NPU requires int32-representable int32/int64 IDs and returns int32;
  // this value-range precondition is not validated per element.
  // Input views retain their dtype and strides; conversion happens in
  // kernel UB. Returns the prefix-masked output.
  static torch::Tensor greedy_sample_from_token_ids(
      const torch::Tensor& draft_token_ids,
      const torch::Tensor& target_token_ids,
      const torch::Tensor& bonus_token_ids);
};

}  // namespace xllm
