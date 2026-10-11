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

#include "core/framework/sampling/sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <mutex>

#if defined(USE_NPU)
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#endif

#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/sampling/logits_utils.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/kernels/ops_api.h"

namespace xllm {
namespace {

uint64_t sampling_seed(int64_t seed, int64_t offset) {
  uint64_t value = static_cast<uint64_t>(seed) +
                   static_cast<uint64_t>(offset) * 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

torch::Generator make_sampling_generator(const torch::Device& device) {
#if defined(USE_NPU)
  if (device.type() == torch::kPrivateUse1) {
    // Native execution does not register the generic default-generator hook.
    return at_npu::detail::createNPUGenerator(device.index());
  }
#endif
  auto generator = torch::globalContext().defaultGenerator(device);
  std::lock_guard<std::mutex> lock(generator.mutex());
  return generator.clone();
}

// Per-request seeded sampling: rows with a seed >= 0 draw from a private
// generator (independent of the global RNG); seed == -1 falls back to the
// global stream, and greedy rows take the argmax. Row-by-row because the
// per-row generator API leaves no batched form. One D2H transfer per tensor
// up front, then host-local reads: per-row .item() would issue a device
// sync on every access.
torch::Tensor seeded_random_sample(const torch::Tensor& probs,
                                   const SamplingParameters& params) {
  const auto seeds = to_cpu_contiguous(params.seeds);
  const auto offsets = to_cpu_contiguous(params.seed_offsets);
  const auto sample_rows = to_cpu_contiguous(params.do_sample);
  const auto seeds_a = seeds.accessor<int64_t, 1>();
  const auto offsets_a = offsets.accessor<int64_t, 1>();
  const auto sample_a = sample_rows.accessor<bool, 1>();
  std::vector<torch::Tensor> rows;
  rows.reserve(probs.size(0));
  for (int64_t row = 0; row < probs.size(0); ++row) {
    const torch::Tensor probabilities = probs.select(0, row);
    if (!sample_a[row]) {
      rows.emplace_back(probabilities.argmax(-1).view({1}));
      continue;
    }
    const int64_t seed = seeds_a[row];
    if (seed == -1) {
      rows.emplace_back(
          Sampler::random_sample(probabilities.unsqueeze(0)).view({1}));
      continue;
    }
    // A private generator keeps this request independent from the global RNG.
    auto generator = make_sampling_generator(probs.device());
    generator.set_current_seed(sampling_seed(seed, offsets_a[row]));
    rows.emplace_back(probabilities.multinomial(/*num_samples=*/1,
                                                /*replacement=*/false,
                                                generator));
  }
  return torch::cat(rows, /*dim=*/0);
}

// Qwen3-reranker verdict token ids ("no", "yes") in the served tokenizer
// vocabulary; the sampler reads the P(yes | {yes, no}) verdict score from
// these columns instead of logprobs.
constexpr int64_t kRerankNoTokenId = 2152;
constexpr int64_t kRerankYesTokenId = 9693;

// Reranker requests sample greedily and carry the verdict score in place of
// logprobs. The fast path and general path both reach here — rerank requests
// set no top_k/top_p or filter masks, so both stages see the same logits.
SampleOutput reranker_output(const torch::Tensor& sample_logits) {
  SampleOutput output;
  output.next_tokens = Sampler::greedy_sample(sample_logits).to(torch::kLong);
  output.logits = sample_logits;
  // P(yes | {yes, no}) = sigmoid(yes - no): algebraically equal to the
  // two-column softmax·exp, one [B] allocation instead of stack +
  // log_softmax + select + exp. Verdict columns read as strided scalars: no
  // device index tensor, no H2D copy.
  output.logprobs =
      (sample_logits.select(/*dim=*/1, /*index=*/kRerankYesTokenId) -
       sample_logits.select(/*dim=*/1, /*index=*/kRerankNoTokenId))
          .to(torch::kFloat32)
          .sigmoid_();
  return output;
}

}  // namespace

SampleOutput Sampler::forward(torch::Tensor& logits,
                              const SamplingParameters& params,
                              const torch::Tensor& filter_mask) {
  const torch::Tensor& effective_filter_mask =
      filter_mask.defined() ? filter_mask : params.filter_mask;
  SampleOutput output;
  if (params.logits_bias.defined()) {
    logits.add_(params.logits_bias);
  }
  // Apply repetition before frequency and presence penalties.
  if (params.repetition_penalties.defined()) {
    apply_repetition_penalties(logits,
                               params.unique_token_ids,
                               params.repetition_penalties,
                               params.unique_token_ids_lens);
  }

  if (params.frequency_penalties.defined()) {
    apply_frequency_presence_penalties(logits,
                                       params.unique_token_ids,
                                       params.unique_token_counts,
                                       params.frequency_penalties,
                                       params.presence_penalties);
  }

  torch::Tensor sample_logits = logits;
  torch::Tensor sample_temperatures = params.temperatures;
  torch::Tensor sample_top_k = params.top_k;
  torch::Tensor sample_top_p = params.top_p;
  torch::Tensor sample_min_p = params.min_p;
  torch::Tensor sample_filter_bitmask = params.filter_bitmask;
  const bool use_sample_indices =
      params.selected_token_idxes.numel() != params.sample_idxes.numel();
  if (use_sample_indices) {
    sample_logits = logits.index_select(/*dim=*/0, params.sample_idxes);
    if (params.temperatures.defined()) {
      sample_temperatures = params.temperatures.index_select(
          /*dim=*/0, params.sample_idxes);
    }
    if (params.top_k.defined()) {
      sample_top_k = params.top_k.index_select(
          /*dim=*/0, params.sample_idxes);
    }
    if (params.top_p.defined()) {
      sample_top_p = params.top_p.index_select(
          /*dim=*/0, params.sample_idxes);
    }
    if (params.min_p.defined()) {
      sample_min_p = params.min_p.index_select(/*dim=*/0, params.sample_idxes);
    }
    if (sample_filter_bitmask.defined()) {
      sample_filter_bitmask = sample_filter_bitmask.index_select(
          /*dim=*/0, params.sample_idxes);
    }
  }

  if (sample_top_k.defined() && sample_top_p.defined()) {
    // Native top-k/top-p consumes top-p in the logits dtype.
    sample_top_p = sample_top_p.to(sample_logits.scalar_type());
  }

  if (sample_filter_bitmask.defined()) {
    apply_token_bitmask_inplace(sample_logits, sample_filter_bitmask);
  } else if (effective_filter_mask.defined()) {
    CHECK_EQ(effective_filter_mask.dim(), 2)
        << "filter_mask must be 2-D, dim=" << effective_filter_mask.dim();
    CHECK_EQ(effective_filter_mask.size(0), sample_logits.size(0))
        << "filter_mask batch mismatch, filter_mask.size(0)="
        << effective_filter_mask.size(0)
        << ", sample_logits.size(0)=" << sample_logits.size(0);
    CHECK_EQ(effective_filter_mask.size(1), sample_logits.size(1))
        << "filter_mask vocab mismatch, filter_mask.size(1)="
        << effective_filter_mask.size(1)
        << ", sample_logits.size(1)=" << sample_logits.size(1);
    sample_logits.add_(effective_filter_mask);
  }

  const VerifyMode mode = params.verify_mode();
  // Rerank requests: greedy verdict score replaces the logprobs output. One
  // dispatch before any processing covers both entry shapes: rerank batches
  // carry no top_k/top_p or filter masks, so the processing stages below are
  // no-ops for them either way.
  if (params.is_rerank && mode == VerifyMode::GREEDY) {
    return reranker_output(sample_logits);
  }
  const bool needs_scores = params.needs_scores();
  // The fast path samples raw logits; any modifier that would rewrite them
  // before sampling (min_p included) must fall through to the general path so
  // the sampled token and its scores stay consistent.
  const bool needs_score_processing =
      needs_scores &&
      (sample_temperatures.defined() || sample_top_k.defined() ||
       sample_top_p.defined() || sample_min_p.defined());
  if (mode == VerifyMode::GREEDY && params.max_top_logprobs == 0 &&
      !use_sample_indices && !filter_mask.defined() &&
      !needs_score_processing) {
    auto sample_indices = greedy_sample(sample_logits).to(torch::kLong);
    output.next_tokens = sample_indices;
    output.logits = sample_logits;
    if (needs_scores) {
      // Same score-gather contract as the general path: one owner in
      // logits_utils, one emission owner in emit_selected_scores.
      const auto log_probs =
          compute_selected_logprobs(sample_logits, sample_indices);
      emit_selected_scores(output, params, log_probs, logits.scalar_type());
    }
    return output;
  }

  // apply temperatures, top-k and top-p
  if (sample_temperatures.defined()) {
    apply_temperatures(sample_logits, sample_temperatures);
  }
  apply_min_p(sample_logits, sample_min_p);
  apply_top_k_top_p(sample_logits, torch::Tensor(), sample_top_k, sample_top_p);
  // apply_top_k_top_p may rebind sample_logits to a fresh tensor (MLU);
  // always write the result back so the caller sees the processed logits.
  if (use_sample_indices) {
    logits.index_copy_(/*dim=*/0, params.sample_idxes, sample_logits);
  } else {
    logits = sample_logits;
  }

  CHECK(params.do_sample.defined()) << "params.do_sample must be defined";
  CHECK_EQ(params.do_sample.dim(), 1) << "params.do_sample must be 1D "
                                         "[num_seqs], got "
                                      << params.do_sample.sizes();
  // same batch size
  CHECK_EQ(sample_logits.size(0), params.do_sample.size(0));

  // Sampling needs probabilities (multinomial/vendor kernels reject the log
  // domain); argmax is domain-invariant, so a fully greedy batch samples
  // straight from the logits.
  // Materialize only the distributions requested by sampling or consumers.
  const bool need_probs = mode != VerifyMode::GREEDY;
  torch::Tensor logprobs;
  torch::Tensor probs;

  if (params.logprobs) {
    logprobs = torch::log_softmax(sample_logits,
                                  /*dim=*/-1,
                                  /*dtype=*/torch::kFloat32);
    if (need_probs) {
      probs = logprobs.exp();
    }
  } else if (need_probs) {
    probs = torch::softmax(sample_logits,
                           /*dim=*/-1,
                           /*dtype=*/torch::kFloat32);
  }
  torch::Tensor samples;
  if (mode == VerifyMode::GREEDY) {
    samples = greedy_sample(sample_logits);
  } else if (params.seeds.defined()) {
    // seeded_random_sample dispatches per row: greedy rows argmax inside, so
    // the result is complete for MIXED batches — no full-vocab argmax + where.
    samples = seeded_random_sample(probs, params);
  } else {
    const torch::Tensor random = random_sample(probs);
    samples = mode == VerifyMode::MIXED
                  ? torch::where(
                        params.do_sample, random, greedy_sample(sample_logits))
                  : random;
  }
  const auto sample_indices = samples.to(torch::kLong);
  output.next_tokens = sample_indices;
  output.logits = sample_logits;

  // One selected-score gather serves both outputs (probs ≡ logprobs.exp()).
  const auto selected_scores =
      needs_scores ? compute_selected_scores(
                         logprobs, sample_logits, sample_indices, probs)
                   : torch::Tensor();
  emit_selected_scores(output, params, selected_scores, logits.scalar_type());
  if (params.logprobs) {
    if (params.max_top_logprobs > 0) {
      // max_top_logprobs is the batch-wide max, so one request asking for more
      // than the vocabulary would make topk throw and fail every request in the
      // batch. The factories reject such requests up front
      // (RequestSamplingParam::top_logprobs_vocab_error); clamp here as the
      // last line of defense so the sampler can never be the failure point.
      // Every consumer sizes by the returned row width, so a shorter row is
      // safe.
      const int64_t k =
          std::min<int64_t>(params.max_top_logprobs, logprobs.size(-1));
      auto [values, indices] = logprobs.topk(k, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }

  return output;
}

torch::Tensor Sampler::greedy_sample(const torch::Tensor& probs) {
  return probs.argmax(/*dim=*/-1);
}

torch::Tensor Sampler::random_sample(const torch::Tensor& probs) {
#if defined(USE_MLU) || defined(USE_CUDA) || defined(USE_DCU)
  xllm::kernel::RandomSampleParams params;
  params.logits = probs;
  return xllm::kernel::random_sample(params);
#endif
  if (probs.dim() == 3) {
    auto batch_size = probs.size(0);
    auto seq_len = probs.size(1);
    auto vocab_size = probs.size(2);
    auto flat_probs = probs.reshape({-1, vocab_size});
    auto sampled =
        flat_probs.multinomial(/*num_samples=*/1, /*replacement=*/false);
    return sampled.reshape({batch_size, seq_len});
  } else {
    return probs.multinomial(/*num_samples=*/1, /*replacement=*/false)
        .flatten();
  }
}

}  // namespace xllm
