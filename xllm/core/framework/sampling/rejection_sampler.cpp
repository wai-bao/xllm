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

#include "core/framework/sampling/rejection_sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <limits>
#include <optional>
#include <utility>

#include "core/framework/sampling/gumbel_sampling.h"
#include "core/framework/sampling/logits_utils.h"
#include "core/framework/sampling/sampler.h"
#include "core/util/tensor_helper.h"

#if defined(USE_NPU)
#include <c10/core/DeviceType.h>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#endif

namespace xllm {

namespace {

torch::Tensor mask_out_after_first_reject(
    const torch::Tensor& first_reject,
    const torch::Tensor& accepted_token_ids) {
  const auto indices =
      arange_indices(accepted_token_ids.size(1), accepted_token_ids.device())
          .unsqueeze(/*dim=*/0);
  return torch::where(indices <= first_reject, accepted_token_ids, -1);
}

// First rejected draft position per sequence; all-accepted sequences point
// past the last draft row (the bonus position).
torch::Tensor first_rejected_indices(const torch::Tensor& accepted) {
  auto accepted_int64 = accepted.to(torch::kInt64);
  auto bonus_mask =
      torch::zeros({accepted.size(0), 1}, accepted_int64.options());
  auto combined_mask = torch::cat({accepted_int64, bonus_mask}, /*dim=*/-1);
  return combined_mask.argmin(/*dim=*/1, /*keepdim=*/true);
}

// Samples the recovered token from the residual (p - q)+ at each sequence's
// first rejected step, entirely in log space. `target_log_probs` and
// `draft_log_probs` are the [B, V] rows at that step with the row logsumexp
// already subtracted, consumed in place; sentinel (-1) drafts carry
// log q = -inf. The residual log-weight is
// log w = a + log1p(-exp(b - a)) with d = b - a clamped at 0 (p < q rows
// have zero residual, and the clamp also absorbs fp32 rounding that pushes a
// zero residual slightly negative), so a Gumbel argmax over log w draws from
// exactly Categorical(normalize(w)) without materializing probs. All -inf
// rows (p == q) argmax to a deterministic index; the caller guarantees those
// rows are masked out downstream.
torch::Tensor sample_recovered_tokens(torch::Tensor target_log_probs,
                                      torch::Tensor draft_log_probs,
                                      const torch::Tensor& rejected_tokens) {
  if (draft_log_probs.defined()) {
    // Sentinel (-1) drafts carry q = 0, i.e. log q = -inf: the residual is
    // the full target row there.
    draft_log_probs.masked_fill_(rejected_tokens.lt(0),
                                 -std::numeric_limits<float>::infinity());
    // log w = a + log1p(-exp(b - a)): work in the moved-in draft buffer,
    // in place. d = b - a clamped at 0 (p < q rows have zero residual, and
    // the clamp also absorbs fp32 rounding that pushes a zero residual
    // slightly negative); NaN from -inf - (-inf) (p = q = 0) is 0 under the
    // same clamp.
    draft_log_probs.sub_(target_log_probs)
        .nan_to_num_(/*nan=*/0.0)
        .clamp_max_(0)
        .exp_()
        .neg_()
        .log1p_();
    target_log_probs.add_(draft_log_probs);
  } else {
    // Greedy q is a delta: log w = a except -inf at the drafted token.
    target_log_probs.scatter_(
        /*dim=*/-1,
        rejected_tokens.clamp_min(0),
        -std::numeric_limits<float>::infinity());
  }
  torch::Tensor gumbel_noise = gumbel_noise_like(target_log_probs);
  return gumbel_argmax(target_log_probs, gumbel_noise);
}

}  // namespace

RejectionSampler::RejectionSampler(SamplingParameters sampling_params)
    : sampling_params_(std::move(sampling_params)) {
  CHECK(sampling_params_.do_sample.defined());
}

SampleOutput RejectionSampler::forward(const DraftProposal& draft_proposal,
                                       torch::Tensor target_logits) const {
  const torch::Tensor& draft_token_ids = draft_proposal.token_ids();
  const int64_t num_draft_tokens = draft_token_ids.size(1);
  CHECK_EQ(target_logits.size(1), num_draft_tokens + 1)
      << "target logits must carry one verification row per draft token "
         "plus "
         "a bonus row";

  // The single application of the sampling transforms: penalties,
  // temperature, top-k/top-p, filter mask, in place on the raw logits.
  Sampler::process_logits(target_logits, sampling_params_);

  torch::Tensor bonus_token_ids;
  torch::Tensor full_greedy_token_ids;
  if (sampling_params_.all_greedy_sample) {
    full_greedy_token_ids = Sampler::greedy_sample(target_logits);
    bonus_token_ids =
        full_greedy_token_ids.select(/*dim=*/1, -1).unsqueeze(/*dim=*/-1);
  } else {
    bonus_token_ids =
        Sampler::random_sample(target_logits.select(/*dim=*/1, -1))
            .to(torch::kLong)
            .view({target_logits.size(0), 1});
    if (!sampling_params_.all_random_sample) {
      full_greedy_token_ids = Sampler::greedy_sample(target_logits);
      const torch::Tensor greedy_bonus =
          full_greedy_token_ids.select(/*dim=*/1, -1).unsqueeze(/*dim=*/-1);
      bonus_token_ids =
          torch::where(row_sample_mask(), bonus_token_ids, greedy_bonus);
    }
  }

  SampleOutput output;
  if (num_draft_tokens == 0) {
    output.next_tokens = bonus_token_ids;
    return fill_logprob_outputs(output, target_logits);
  }
  // Only the random/mixed verification legs read the lse; greedy batches
  // (the default draft mode) skip the [B, K, V] fp32 reduction entirely.
  const torch::Tensor target_lse =
      sampling_params_.all_greedy_sample
          ? torch::Tensor()
          : torch::logsumexp(
                target_logits
                    .narrow(
                        /*dim=*/1, /*start=*/0, /*length=*/num_draft_tokens)
                    .to(torch::kFloat32),
                /*dim=*/-1,
                /*keepdim=*/false);
  output.next_tokens = verify(draft_proposal,
                              target_logits,
                              target_lse,
                              bonus_token_ids,
                              full_greedy_token_ids);
  return fill_logprob_outputs(output, target_logits);
}

SampleOutput RejectionSampler::fill_logprob_outputs(
    SampleOutput output,
    const torch::Tensor& target_logits) const {
  if (sampling_params_.logprobs) {
    // log_softmax is equivalent to log(softmax) but more numerically stable
    // [batch_size, n_speculative_tokens + 1, vocab_size]
    auto target_logprobs = torch::log_softmax(
        target_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);

    // -1 sentinels cannot index the gather; their logprobs are zeroed.
    auto selected_logprobs =
        gather_from_logits(target_logprobs, output.next_tokens);
    selected_logprobs.masked_fill_(output.next_tokens.lt(0), 0);
    output.logprobs = std::move(selected_logprobs);

    if (sampling_params_.max_top_logprobs > 0) {
      auto [values, indices] =
          target_logprobs.topk(sampling_params_.max_top_logprobs, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }
  return output;
}

torch::Tensor RejectionSampler::verify(
    const DraftProposal& draft_proposal,
    const torch::Tensor& target_logits,
    const torch::Tensor& target_lse,
    const torch::Tensor& bonus_token_ids,
    const torch::Tensor& target_greedy_token_ids) const {
  const torch::Tensor& draft_token_ids = draft_proposal.token_ids();
  const int64_t num_draft_tokens = draft_token_ids.size(1);
  draft_proposal.validate(
      /*expected_batch_size=*/sampling_params_.do_sample.size(0),
      /*expected_vocab_size=*/target_logits.size(-1),
      /*expected_num_speculative_tokens=*/target_logits.size(1) - 1);
  const torch::Tensor greedy_target_token_ids =
      target_greedy_token_ids.defined()
          ? target_greedy_token_ids.narrow(
                /*dim=*/1, /*start=*/0, /*length=*/num_draft_tokens)
          : torch::Tensor();

  auto random_sample_dispatch = [&]() -> torch::Tensor {
    torch::Tensor uniform_rand =
        torch::rand(draft_token_ids.sizes(),
                    target_logits.options().dtype(torch::kFloat32));
    return random_sample(
        draft_proposal,
        target_logits.narrow(
            /*dim=*/1, /*start=*/0, /*length=*/num_draft_tokens),
        target_lse,
        uniform_rand,
        bonus_token_ids);
  };

  auto greedy_sample_dispatch = [&]() -> torch::Tensor {
    CHECK_EQ(greedy_target_token_ids.sizes(), draft_token_ids.sizes())
        << "target and draft token shapes must match";
#if defined(USE_NPU)
    if (greedy_target_token_ids.device().type() ==
        c10::DeviceType::PrivateUse1) {
      return kernel::npu::tilelang::greedy_prefix_verify(
          draft_token_ids, greedy_target_token_ids, bonus_token_ids);
    }
#endif
    torch::Tensor greedy_accepted =
        torch::cat({greedy_target_token_ids, bonus_token_ids}, /*dim=*/-1);
    auto accepted = (greedy_target_token_ids == draft_token_ids);
    return mask_out_after_first_reject(first_rejected_indices(accepted),
                                       greedy_accepted);
  };

  if (sampling_params_.all_greedy_sample) {
    return greedy_sample_dispatch();
  }
  if (sampling_params_.all_random_sample) {
    return random_sample_dispatch();
  }
  // mixed sample: dispatch both legs and select per row.
  return torch::where(
      row_sample_mask(), random_sample_dispatch(), greedy_sample_dispatch());
}

torch::Tensor RejectionSampler::row_sample_mask() const {
  return sampling_params_.do_sample.unsqueeze(/*dim=*/-1);
}

torch::Tensor RejectionSampler::random_sample(
    const DraftProposal& draft_proposal,
    const torch::Tensor& target_logits,
    const torch::Tensor& target_lse,
    torch::Tensor uniform_rand,
    const torch::Tensor& bonus_token_ids) {
  const torch::Tensor& draft_token_ids = draft_proposal.token_ids();
  const std::optional<torch::Tensor>& draft_logits = draft_proposal.logits();

  // Acceptance in log space: p/q > u becomes target_log_prob >
  // draft_log_prob + log u (greedy q = 1).
  torch::Tensor target_log_prob =
      gather_log_probs_from_logits(target_logits, target_lse, draft_token_ids);
  torch::Tensor draft_lse;
  if (draft_logits.has_value()) {
    draft_lse = torch::logsumexp(
        draft_logits->to(torch::kFloat32), /*dim=*/-1, /*keepdim=*/false);
    target_log_prob.sub_(gather_log_probs_from_logits(
        *draft_logits, draft_lse, draft_token_ids));
  }
  auto accepted = (uniform_rand.log_().lt(target_log_prob))
                      .logical_and(draft_token_ids.ge(0));

  // All-accepted sequences clamp back into range; their recovered token
  // is masked out below.
  const torch::Tensor first_reject = first_rejected_indices(accepted);
  const torch::Tensor draft_first_reject =
      first_reject.clamp_max(draft_token_ids.size(1) - 1);
  const torch::Tensor rejected_tokens =
      draft_token_ids.gather(/*dim=*/1, draft_first_reject);
  // Row draft_first_reject[i] of each [B, K, V] tensor as [B, V], kept as
  // log-probs (row logsumexp subtracted) — the residual recovery below works
  // in log space, so no probs are materialized.
  const auto row_log_probs = [&](const torch::Tensor& logits,
                                 const torch::Tensor& lse) {
    torch::Tensor gathered_logits =
        index_select_2d(logits,
                        /*dim=*/1,
                        draft_first_reject.expand(
                            {target_logits.size(0), target_logits.size(-1)}))
            .to(torch::kFloat32);
    return gathered_logits.sub_(lse.gather(/*dim=*/1, draft_first_reject));
  };
  torch::Tensor target_log_probs_at_reject =
      row_log_probs(target_logits, target_lse);
  torch::Tensor draft_log_probs_at_reject;
  if (draft_logits.has_value()) {
    draft_log_probs_at_reject = row_log_probs(*draft_logits, draft_lse);
  }
  auto recovered_token_ids =
      sample_recovered_tokens(std::move(target_log_probs_at_reject),
                              std::move(draft_log_probs_at_reject),
                              rejected_tokens)
          .unsqueeze(/*dim=*/-1)
          .to(torch::kLong);

  // Accepted rows carry non-sentinel ids, so the clamp in the
  // rejected-token gather is the identity there.
  auto combined = torch::where(accepted, draft_token_ids, recovered_token_ids);
  auto accepted_token_ids = torch::cat({combined, bonus_token_ids}, /*dim=*/-1);
  return mask_out_after_first_reject(first_reject, accepted_token_ids);
}

}  // namespace xllm
