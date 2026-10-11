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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "framework/sampling/verify_mode.h"
#include "util/tensor_helper.h"  // repeat_interleave_if_defined

namespace xllm {

// Requests asking for top logprobs must also enable logprobs. Free function
// so scalar-only callers (result binding) share the member predicate's
// semantics.
inline bool inconsistent_top_logprobs(bool logprobs, int64_t top_logprobs) {
  return !logprobs && top_logprobs != 0;
}

struct Token;

// A class (not struct) per the code style: it carries behavior
// (enable_beam_search / top_logprobs_vocab_error) beyond plain aggregation.
class RequestSamplingParam final {
 public:
  float frequency_penalty = 0.0;
  float presence_penalty = 0.0;
  float repetition_penalty = 1.0;
  float temperature = 1.0;
  float top_p = 1.0;
  int64_t top_k = 0;
  float min_p = 0.0;
  std::optional<int64_t> seed;
  uint32_t min_tokens = 0;
  int64_t vocab_size = 0;
  std::unordered_map<int32_t, float> logit_bias;
  std::optional<std::vector<int32_t>> allowed_token_ids;
  std::vector<std::vector<int32_t>> bad_words_token_ids;
  std::vector<int32_t> all_stop_token_ids;
  bool logprobs = false;
  int64_t top_logprobs = 0;
  bool do_sample = false;
  bool is_embeddings = false;
  bool is_rerank = false;
  bool json_object = false;
  int32_t beam_width = 0;
  int32_t num_return_sequences = 0;

  // Turns on beam search with the given width and enforces what the beam
  // machinery needs. Both the on-device BeamSearcher kernel and the host-side
  // SequencesGroup::process_beam_search() pick the next beams from the
  // sampler's top-k candidates, and the sampler produces exactly top_logprobs
  // of them per sequence (the batch takes max(top_logprobs) as its top-k). So
  // logprobs must be on, and top_logprobs must be at least beam_width: the
  // search starts from a single sequence, so with fewer candidates the first
  // step can only fan out to top_logprobs beams and the lower-ranked initial
  // candidates are discarded for good, yielding a narrower search than asked
  // for. A smaller explicit value is therefore raised to the width (the client
  // still receives at least the logprob count it requested); an unset one is
  // derived from it, matching the RequestParams-level default. Every factory
  // that enables beam search goes through here so the invariant lives in one
  // place. Out-of-range counts are rejected up front by
  // RequestParams::verify_params. A width <= 1 just records the value (no beam
  // search).
  void enable_beam_search(int32_t width) {
    beam_width = width;
    if (beam_width > 1) {
      logprobs = true;
      top_logprobs =
          std::max<int64_t>(top_logprobs, static_cast<int64_t>(beam_width));
    }
  }

  // Whenever logprobs are on, the sampler runs topk(top_logprobs) over the
  // vocabulary, and torch::topk throws for k > vocab. The batch uses
  // max(top_logprobs) across its requests, so a single oversized request would
  // take down every request in the batch. RequestParams::verify_params only
  // knows the model-agnostic 2000 cap; this is the model-aware check for the
  // factories, which know the vocabulary. Returns an error message when the
  // effective top-k exceeds it, std::nullopt when it fits or the vocabulary is
  // unknown (vocab_size <= 0). Call after enable_beam_search so the beam-raised
  // value is what gets checked.
  std::optional<std::string> top_logprobs_vocab_error(
      int64_t vocab_size) const {
    if (!logprobs || vocab_size <= 0 || top_logprobs <= vocab_size) {
      return std::nullopt;
    }
    std::string error = "top_logprobs (" + std::to_string(top_logprobs) + ")";
    if (beam_width > 1 && top_logprobs == static_cast<int64_t>(beam_width)) {
      // The value came from beam_width (unset or smaller top_logprobs was
      // raised to the width), so point the client at the field it actually set.
      error = "beam_width (" + std::to_string(beam_width) + ")";
    }
    return error + " must not exceed the model vocabulary size (" +
           std::to_string(vocab_size) + ")";
  }
};

// A class (not struct) per the code style: it carries behavior
// (init/to/concat/verify_mode) beyond plain
// aggregation; no aggregate initialization of it exists, so the public
// data members stay directly initializable.
class SamplingParameters final {
 public:
  void init(const std::vector<const RequestSamplingParam*>& req_sampling_params,
            const std::vector<int32_t>& selected_token_idxes,
            const std::vector<int32_t>& sample_idxes,
            const std::vector<std::vector<int64_t>>& unique_token_ids_vec,
            const std::vector<std::vector<int32_t>>& unique_token_counts_vec,
            const std::vector<int32_t>& unique_token_lens_vec,
            const std::vector<torch::Tensor>& filter_mask_rows = {});

  SamplingParameters to(const torch::Device& device,
                        torch::ScalarType dtype) const;

  // concat two SamplingParameters into one
  void concat(const SamplingParameters& param);

  // The batch-level greedy/random/mixed dispatch mode. Greedy takes priority
  // when an empty do_sample leaves both flags true.
  VerifyMode verify_mode() const {
    if (all_greedy_sample) {
      return VerifyMode::GREEDY;
    }
    return all_random_sample ? VerifyMode::RANDOM : VerifyMode::MIXED;
  }

  // Whether any optional sampling-modifier tensor is defined for this batch:
  // the single enumeration for callers that gate on "plain greedy, nothing
  // else requested" (graph capture, replay reuse). A new modifier field must
  // join this list, not per-caller copies. Seeds are excluded deliberately:
  // they are consumed only by RANDOM/MIXED sampling, and every caller of
  // this predicate already gates on GREEDY — including them would silently
  // drop seeded-but-greedy batches (e.g. benchmark defaults) from graph
  // capture.
  bool has_sampling_modifiers() const {
    return temperatures.defined() || top_k.defined() || top_p.defined() ||
           min_p.defined() || logits_bias.defined() ||
           frequency_penalties.defined() || presence_penalties.defined() ||
           repetition_penalties.defined() || unique_token_ids.defined() ||
           unique_token_counts.defined() || unique_token_ids_lens.defined() ||
           filter_mask.defined() || filter_bitmask.defined();
  }

  // Whether any consumer needs selected-token scores (logprobs or probs).
  // The single owner of the score-request predicate: gating sites (sampler,
  // rejection sampler, consensus sync) must read this, not per-caller copies.
  bool needs_scores() const { return logprobs || return_probs; }

  // Requests asking for top logprobs must also enable logprobs: the single
  // owner of the request-consistency predicate shared by admission checks
  // and the result binding. The free-function form also serves callers
  // (slot_buffer) that hold plain scalars instead of a SamplingParameters.
  bool inconsistent_top_logprobs() const {
    return xllm::inconsistent_top_logprobs(logprobs, max_top_logprobs);
  }

  // Repeat every per-request tensor field along dim 0 for speculative
  // validation: the one enumeration of the repeatable field list. Repeats is
  // either a scalar count (uniform width) or a per-sequence tensor. A zero
  // scalar count leaves the fields unrepeated (skip the empty dispatch).
  template <typename Repeats>
  void repeat_tensors(const Repeats& repeats) {
    if constexpr (std::is_integral_v<std::decay_t<Repeats>>) {
      if (repeats == 0) {
        return;
      }
    }
    repeat_interleave_if_defined(frequency_penalties, repeats);
    repeat_interleave_if_defined(presence_penalties, repeats);
    repeat_interleave_if_defined(repetition_penalties, repeats);
    repeat_interleave_if_defined(temperatures, repeats);
    repeat_interleave_if_defined(top_p, repeats);
    repeat_interleave_if_defined(min_p, repeats);
    repeat_interleave_if_defined(logits_bias, repeats);
    repeat_interleave_if_defined(seeds, repeats);
    repeat_interleave_if_defined(seed_offsets, repeats);
    repeat_interleave_if_defined(top_k, repeats);
    repeat_interleave_if_defined(unique_token_ids, repeats);
    repeat_interleave_if_defined(unique_token_counts, repeats);
    repeat_interleave_if_defined(unique_token_ids_lens, repeats);
    repeat_interleave_if_defined(do_sample, repeats);
    // Today's JSON path installs per-request masks after the repeat, so this
    // is a no-op; any future per-request filter source must join this list.
    repeat_interleave_if_defined(filter_mask, repeats);
    repeat_interleave_if_defined(filter_bitmask, repeats);
  }

  // selected tokens are tokens for sampling the next token,
  // including the generated tokens and the last prompt token
  // IntTensor
  torch::Tensor selected_token_idxes;

  // Dense additive mask for token-level structured output constraints. Zero
  // entries are allowed and negative entries are forbidden.
  torch::Tensor filter_mask;

  // Compact allowed-token bitmask [num_tokens, ceil(vocab/32)] int32. When
  // defined, Sampler prefers this over filter_mask (smaller H2D).
  torch::Tensor filter_bitmask;

  // [num_tokens] FloatTensor
  torch::Tensor frequency_penalties;

  // [num_tokens] FloatTensor
  torch::Tensor presence_penalties;

  // [num_tokens] FloatTensor
  torch::Tensor repetition_penalties;

  // [num_tokens] FloatTensor
  torch::Tensor temperatures;

  // [num_tokens] FloatTensor
  torch::Tensor top_p;

  // Minimum probability relative to the largest probability in each row.
  torch::Tensor min_p;

  // Per-selected-row additive constraints and independent random streams.
  torch::Tensor logits_bias;
  torch::Tensor seeds;
  torch::Tensor seed_offsets;

  // [num_tokens] LongTensor
  torch::Tensor top_k;

  // the unique token id and count of each sequence in the batch.
  // [num_tokens, max_unique_tokens] LongTensor
  torch::Tensor unique_token_ids;

  // [num_tokens, max_unique_tokens] IntTensor
  torch::Tensor unique_token_counts;

  // the number of unique tokens in each sequence.
  // [num_tokens] IntTensor
  torch::Tensor unique_token_ids_lens;

  // the last index of the selected tokens for sampling.
  // [num_seqs] IntTensor
  torch::Tensor sample_idxes;

  // whether to sample for each sequence.
  // [num_seqs] BoolTensor
  torch::Tensor do_sample;

  // Beam search accumulated log probability.
  // [num_seq, 1] FloatTensor
  torch::Tensor acc_logprob;

  bool all_random_sample = false;
  bool all_greedy_sample = true;

  // whether to output logprobs for each generated token.
  bool logprobs = false;

  // Whether a consumer needs selected-token probabilities.
  bool return_probs = false;

  // whether to get the embeddings of the tokens. used by embeddings model.
  bool is_embeddings = false;

  // whether any request in the batch is a Qwen3 rerank request.
  bool is_rerank = false;

  // max number of top logprobs in the batch.
  // only used when logprobs is true.
  int64_t max_top_logprobs = 0;

  // requested final beam result width for request-level beam search output.
  int32_t num_return_sequences = 0;

  // for beam search
  bool use_beam_search = false;
};

// Draft proposals and acceptances executed by the verifier, before output
// truncation. Accepted drafts exclude the target replacement/bonus token.
struct SpeculativeTokenStats {
  int64_t accepted_tokens = 0;
  int64_t proposed_tokens = 0;
};

class SampleOutput final {
 public:
  // Host access only: tensors must be CPU-contiguous. next_tokens is int32 or
  // int64 with shape [num_seq] for ordinary sampling, or [num_seq, width] for
  // speculative sampling. Optional float32 logprobs match that shape; paired
  // int64 top_tokens/float32 top_logprobs append a top-k dimension.
  // Returned top-k slices borrow this output's tensor storage; keep it alive
  // and unchanged while using the Token. The overload must match token rank.
  Token token_at(int64_t output_idx) const;
  Token token_at(int64_t output_idx, int64_t token_idx) const;

  // [num_seq, ...] LongTensor
  torch::Tensor next_tokens;

  // Actual post-processor sampling logits. Tensor handles retain the sampling
  // storage without caching a normalized distribution.
  torch::Tensor logits;
  // Compact proposal support and its sampled indices, when tokens are remapped.
  torch::Tensor vocab_ids;
  torch::Tensor sampled_indices;

  // Selected-token probabilities, computed only for explicit score consumers.
  torch::Tensor probs;

  // [num_seq, ...] FloatTensor
  torch::Tensor logprobs;

  // [num_seq, ..., top_k] FloatTensor
  torch::Tensor top_logprobs;
  // [num_seq, ..., top_k] LongTensor
  torch::Tensor top_tokens;

  // [num_seq, ..., embed_dim] FloatTensor (full forward hidden on the
  // speculative target-prefill path; see `selected_embeddings`).
  torch::Tensor embeddings;

  // Per-sequence target-prefill hidden: one row per sequence, in
  // `logits_indices` order. Separate from `embeddings` because under CP
  // `embeddings` holds only the local shard, whose rows cannot be re-indexed
  // per-sequence by the all-gather-space `logits_indices`.
  torch::Tensor selected_embeddings;

  std::vector<std::vector<torch::Tensor>> mm_embeddings;
};

}  // namespace xllm
