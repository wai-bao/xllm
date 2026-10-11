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

#include "rejection_sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <limits>

#include "sampler.h"

#if defined(USE_NPU)
#include <c10/core/DeviceType.h>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#endif

namespace xllm {

namespace {

torch::Tensor normalize_recovered_probs(torch::Tensor recovered_probs) {
  const auto sums = recovered_probs.sum(/*dim=*/-1, /*keepdim=*/true);
  // A zero residual has no valid normalization; fall back to uniform. The
  // clamp keeps ratios untouched for positive sums; renormalize below fixes
  // the clamped rows.
  const double uniform_prob =
      1.0 / static_cast<double>(recovered_probs.size(-1));
  recovered_probs.div_(sums.clamp_min(std::numeric_limits<float>::min()));
  recovered_probs.masked_fill_(sums <= 0, uniform_prob);
  return recovered_probs;
}

// [B, 1] per-request first-reject index; the appended zero column sends
// fully-accepted rows past the last draft slot so the bonus column survives
// the mask.
torch::Tensor first_reject_index(const torch::Tensor& accepted) {
  const auto combined =
      torch::constant_pad_nd(accepted.to(torch::kInt64), {0, 1}, 0);
  return combined.argmin(/*dim=*/1, /*keepdim=*/true);
}

// Only the first rejection needs a full target distribution. For compact
// proposals, subtract q on its support without expanding it to [B, N, V].
// The caller's shared normalization, when available, replaces the softmax:
// recovered = exp(values - normalizer) at the reject slot, one gather on an
// already-normalized pair instead of a fresh full-vocab softmax.
torch::Tensor resample_first_reject(const DraftProposal& proposal,
                                    const torch::Tensor& target_logits,
                                    const torch::Tensor& reject_index,
                                    const NormalizedLogits* normalized) {
  const auto& tokens = proposal.token_ids();
  const auto row_slot = reject_index.clamp_max(tokens.size(1) - 1);
  const auto slot = row_slot.unsqueeze(-1);
  torch::Tensor recovered_probs;
  if (normalized != nullptr) {
    // In-place on the fresh gather: no chained [B, 1, V] temporaries.
    auto values = normalized->values.take_along_dim(slot, 1);
    recovered_probs =
        values.sub_(normalized->normalizer.take_along_dim(slot.squeeze(-1), 1))
            .exp_();
  } else {
    recovered_probs = torch::softmax(
        target_logits.take_along_dim(slot, 1), -1, torch::kFloat32);
  }
  if (proposal.is_greedy()) {
    recovered_probs.scatter_(-1, tokens.gather(1, row_slot).unsqueeze(-1), 0);
  } else {
    const auto q = torch::softmax(
        proposal.logits()->take_along_dim(slot, 1), -1, torch::kFloat32);
    if (proposal.vocab_ids().defined()) {
      recovered_probs.scatter_add_(
          -1, proposal.support_at(slot, q).to(torch::kLong), -q);
      recovered_probs.clamp_min_(0);
    } else {
      recovered_probs.sub_(q).clamp_min_(0);
    }
  }
  return Sampler::random_sample(normalize_recovered_probs(recovered_probs))
      .to(tokens.scalar_type());
}

// Selected-token logprobs for the verify output. The negative-token
// clamp/mask convention lives here alone: rejected positions gather from
// row 0 and are zeroed afterwards, so every caller of this helper (logprobs
// output, probs output) shares one definition. With the caller's shared
// normalization the top-k tail reads the dense pair; a null pointer (greedy
// scores-only batches, no top-k) takes the gather-first path and never
// materializes the dense FP32 copy.
void populate_selected_scores(SampleOutput& output,
                              const NormalizedLogits* normalized,
                              const torch::Tensor& target_logits,
                              int64_t max_top_logprobs) {
  if (normalized != nullptr) {
    output.logprobs = selected_logprobs(*normalized, output.next_tokens);
  } else {
    output.logprobs =
        compute_selected_logprobs(target_logits, output.next_tokens);
  }
  output.logprobs.masked_fill_(output.next_tokens.lt(0), 0);
  if (normalized != nullptr && max_top_logprobs > 0) {
    const int64_t k = std::min(max_top_logprobs, normalized->values.size(-1));
    auto [top_values, top_indices] = normalized->values.topk(k, -1);
    output.top_logprobs = top_values - normalized->normalizer.unsqueeze(-1);
    output.top_tokens = top_indices;
  }
}

// Prefix-mask [B, n_tokens + 1] token ids: each row keeps its positions
// 0..reject_index and becomes -1 afterwards. The sentinel index n_tokens from
// first_reject_index lets fully-accepted rows keep the appended bonus column.
// In-place on the caller's freshly built tensor (both call sites pass a local
// torch::cat result): no [B, N+1] temporary.
torch::Tensor mask_out_after_first_reject(torch::Tensor accepted_token_ids,
                                          const torch::Tensor& reject_index) {
  const int64_t n_tokens = accepted_token_ids.size(1) - 1;
  const auto indices =
      torch::arange(n_tokens + 1, reject_index.device()).unsqueeze(/*dim=*/0);
  accepted_token_ids.masked_fill_(indices.gt(reject_index), -1);
  return accepted_token_ids;
}

}  // namespace

torch::Tensor RejectionSampler::random_sample(
    const DraftProposal& proposal,
    const torch::Tensor& target_logits,
    const torch::Tensor& uniform_rand,
    const torch::Tensor& bonus_token_ids,
    const NormalizedLogits* normalized) {
  const auto& tokens = proposal.token_ids();
  if (tokens.size(1) == 0) {
    return bonus_token_ids;
  }
  // One FP32 materialization of the target logits serves both the acceptance
  // gather here and the score outputs in verify: normalized (when provided)
  // is the caller's full-width [B, N+1, V] pass over the same logits, so the
  // draft-position gather slices the N draft columns out of it; the
  // standalone path normalizes alone.
  torch::Tensor log_acceptance;
  if (normalized != nullptr) {
    // The caller's pass covers [B, N+1, V]; the draft-position gather slices
    // the N draft columns via zero-copy narrow views of the shared pair.
    log_acceptance =
        selected_logprobs({normalized->values.narrow(1, 0, tokens.size(1)),
                           normalized->normalizer.narrow(1, 0, tokens.size(1))},
                          tokens);
  } else {
    log_acceptance = compute_selected_logprobs(target_logits, tokens);
  }
  if (!proposal.is_greedy()) {
    const auto& indices =
        proposal.vocab_ids().defined() ? proposal.sampled_indices() : tokens;
    log_acceptance -=
        compute_selected_logprobs(proposal.logits().value(), indices);
  }
  // uniform_rand arrives already in the log domain from the dispatch caller.
  const auto accepted = uniform_rand < log_acceptance;
  const auto reject_index = first_reject_index(accepted);
  const auto recovered =
      resample_first_reject(proposal, target_logits, reject_index, normalized);
  return mask_out_after_first_reject(
      torch::cat({torch::where(accepted, tokens, recovered), bonus_token_ids},
                 -1),
      reject_index);
}

SampleOutput RejectionSampler::verify(const SamplingParameters& params,
                                      const DraftProposal& proposal,
                                      const torch::Tensor& target_logits,
                                      const torch::Tensor& target_tokens,
                                      const torch::Tensor& draft_lengths,
                                      const torch::Tensor& invalid_draft,
                                      const torch::Tensor& target_embeddings) {
  CHECK_EQ(target_tokens.dim(), 2);
  CHECK_EQ(target_logits.dim(), 3);
  CHECK_EQ(target_logits.size(0), target_tokens.size(0));
  CHECK_EQ(target_logits.size(1), target_tokens.size(1));
  CHECK_EQ(params.do_sample.numel(), target_tokens.size(0));
  const int64_t batch_size = target_tokens.size(0);
  const int64_t num_steps = target_tokens.size(1) - 1;
  proposal.validate(batch_size, target_logits.size(-1), num_steps);
  torch::Tensor cut;
  const bool has_cut = draft_lengths.defined() || invalid_draft.defined();
  auto bonus = target_tokens.narrow(1, num_steps, 1);
  if (has_cut) {
    const auto cut_options = target_tokens.options().dtype(torch::kLong);
    if (draft_lengths.defined()) {
      CHECK_EQ(draft_lengths.numel(), batch_size);
      cut = draft_lengths.to(cut_options)
                .view({batch_size, 1})
                .clamp(0, num_steps);
    }
    if (invalid_draft.defined()) {
      CHECK_EQ(invalid_draft.sizes(), proposal.token_ids().sizes());
      const auto invalid_reject = first_reject_index(
          ~invalid_draft.to(target_tokens.device(), torch::kBool));
      cut =
          cut.defined() ? torch::minimum(cut, invalid_reject) : invalid_reject;
    }
    bonus = target_tokens.gather(1, cut);
  }
  SampleOutput output;
  // Coerce the draft ids to the target device once at the shared entry: both
  // the token-match fast path and the logits rejection path need it. Every
  // member .to() short-circuits to shared storage on a matching device, so
  // the copy is only paid when actually needed.
  const DraftProposal coerced = proposal.coerce_to(target_tokens);
  const torch::Tensor& draft_token_ids = coerced.token_ids();
  // The bonus row is sampled by the target worker and does not participate in
  // rejection sampling, so drop it before scoring draft positions.
  const torch::Tensor target_draft_logits =
      target_logits.narrow(1, 0, num_steps);
  // Fresh random tensor, log taken in place to avoid a second [B, N]
  // allocation; -log(U) ~ Exp(1) is the acceptance threshold's other side.
  const auto draw_acceptance_noise = [&]() -> torch::Tensor {
    return torch::rand(draft_token_ids.sizes(),
                       target_logits.options().dtype(torch::kFloat32))
        .log_();
  };
  // Dense [B, N+1, V] FP32 normalization ownership: built only when some
  // consumer needs the full width — top-k scores, or the RANDOM/MIXED
  // acceptance gather and first-reject recovery that share it (one
  // materialization per step, never normalized twice). A greedy
  // scores-only batch consumes just the selected gather and takes the
  // gather-first path with no dense copy; without scores nothing
  // materializes at all.
  const bool needs_scores = params.needs_scores();
  const bool needs_dense_normalization =
      needs_scores && (params.max_top_logprobs > 0 ||
                       params.verify_mode() != VerifyMode::GREEDY);
  const std::optional<NormalizedLogits> normalized =
      needs_dense_normalization
          ? std::make_optional(normalized_logits(target_logits))
          : std::nullopt;
  // The greedy leg's target rows and the score-bearing normalization are
  // shared across verify-mode arms; hoist them so the arms stay in sync.
  const NormalizedLogits* normalized_scores =
      normalized ? &*normalized : nullptr;
  const torch::Tensor target_draft_tokens =
      target_tokens.narrow(1, 0, num_steps);
  switch (params.verify_mode()) {
    case VerifyMode::GREEDY:
      // Greedy batches match target token ids directly instead of
      // re-deriving them from the logits.
      output.next_tokens = greedy_sample_from_token_ids(
          draft_token_ids, target_draft_tokens, bonus);
      break;
    case VerifyMode::RANDOM:
      output.next_tokens = random_sample(coerced,
                                         target_draft_logits,
                                         draw_acceptance_noise(),
                                         bonus,
                                         normalized_scores);
      break;
    case VerifyMode::MIXED: {
      // Mixed batch: sample both paths, then choose per row from do_sample.
      // The greedy leg reads the target sampler's own greedy rows out of
      // target_tokens (they are argmax of the same logits) instead of paying
      // a full-vocab argmax; the do_sample rows overwrite them below.
      const auto random = random_sample(coerced,
                                        target_draft_logits,
                                        draw_acceptance_noise(),
                                        bonus,
                                        normalized_scores);
      const auto greedy = greedy_sample_from_token_ids(
          draft_token_ids, target_draft_tokens, bonus);
      output.next_tokens =
          torch::where(params.do_sample.unsqueeze(-1), random, greedy);
      break;
    }
  }
  if (has_cut) {
    // An earlier rejection wins. Only a still-reachable boundary is replaced
    // by its independently sampled target token; positions past the cut take
    // the shared -1 sentinel, applied in place (one masked_fill launch).
    const auto columns =
        torch::arange(num_steps + 1, cut.options()).view({1, -1});
    output.next_tokens =
        torch::where((columns == cut) & output.next_tokens.ge(0),
                     target_tokens,
                     output.next_tokens);
    output.next_tokens.masked_fill_(columns > cut, -1);
  }
  // Score outputs share one definition of the rejected-position convention:
  // populate_selected_scores clamps the gather to row 0 and zeroes the mask,
  // and probs ≡ logprobs.exp() wherever both exist. The logprobs field stays
  // undefined unless the request asked for it (downstream padding and
  // serialization branch on its definedness).
  if (needs_scores) {
    // The null normalization is the greedy scores-only case (no top-k):
    // populate_selected_scores takes the gather-first path for it.
    populate_selected_scores(
        output, normalized_scores, target_logits, params.max_top_logprobs);
    // probs ≡ logprobs.exp() under the shared emission contract; exp(0) at
    // a rejected slot would read as certainty 1.0, so re-mask it.
    emit_selected_scores(
        output, params, output.logprobs, target_logits.scalar_type());
    if (output.probs.defined()) {
      output.probs.masked_fill_(output.next_tokens.lt(0), 0);
    }
    if (!params.logprobs) {
      output.logprobs = torch::Tensor();
    }
  }
  // The verify output carries the target's embeddings as a batch-major
  // [B, n_spec + 1, H] view; callers never re-attach them.
  if (target_embeddings.defined()) {
    output.embeddings = target_embeddings.view({target_tokens.size(0),
                                                target_tokens.size(1),
                                                target_embeddings.size(-1)});
  }
  return output;
}

torch::Tensor RejectionSampler::greedy_sample_from_token_ids(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& target_token_ids,
    const torch::Tensor& bonus_token_ids) {
  CHECK_EQ(target_token_ids.sizes(), draft_token_ids.sizes())
      << "target and draft token shapes must match";
#if defined(USE_NPU)
  if (target_token_ids.device().type() == c10::DeviceType::PrivateUse1) {
    return kernel::npu::tilelang::greedy_prefix_verify(
        draft_token_ids, target_token_ids, bonus_token_ids);
  }
#endif
  // [batch_size, n_speculative_tokens + 1]
  torch::Tensor accepted_token_ids =
      torch::cat({target_token_ids, bonus_token_ids}, /*dim=*/-1);
  const auto accepted = (target_token_ids == draft_token_ids);
  return mask_out_after_first_reject(accepted_token_ids,
                                     first_reject_index(accepted));
}

}  // namespace xllm
