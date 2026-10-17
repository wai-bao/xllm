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

#include "core/framework/speculative/draft_input_sampling.h"

#include <glog/logging.h>

#include <algorithm>
#include <optional>
#include <utility>

#include "core/framework/sampling/logits_utils.h"
#include "core/framework/speculative/mtp_json_object_state.h"
#include "core/framework/speculative/padded_token_row_scan.h"
#include "core/util/tensor_helper.h"

namespace xllm {

namespace {

// Which rung of the confidence ladder serves a sample output: the carried
// draft_input's selected probs, then the sampler's probs (1-D/[batch,1]
// passthrough or 2-D gather), then a per-row softmax over logits+next_tokens.
// Resolving the source separately from computing it keeps the availability
// probe tensor-free while the extractor pays the single real compute; the two
// can never disagree because both switch on this enum.
enum class DraftInputConfidenceSource : uint8_t {
  CARRIED_SELECTED_PROBS,
  FLAT_PROBS,
  GATHERED_PROBS,
  LOGITS_SOFTMAX,
  NONE,
};

DraftInputConfidenceSource resolve_draft_input_confidence_source(
    const SampleOutput& sample_output,
    const torch::Tensor& logits) {
  if (sample_output.draft_input.has_value() &&
      sample_output.draft_input->selected_probs.defined()) {
    return DraftInputConfidenceSource::CARRIED_SELECTED_PROBS;
  }
  const bool has_probs = sample_output.probs.defined();
  // 1-D probs and [batch,1] probs are already per-row selected values.
  if (has_probs &&
      (sample_output.probs.dim() == 1 ||
       (sample_output.probs.dim() == 2 && sample_output.probs.size(1) == 1))) {
    return DraftInputConfidenceSource::FLAT_PROBS;
  }
  if (!sample_output.next_tokens.defined()) {
    return DraftInputConfidenceSource::NONE;
  }
  if (has_probs) {
    return DraftInputConfidenceSource::GATHERED_PROBS;
  }
  if (!logits.defined()) {
    return DraftInputConfidenceSource::NONE;
  }
  return DraftInputConfidenceSource::LOGITS_SOFTMAX;
}

torch::Tensor select_from_source(DraftInputConfidenceSource source,
                                 const SampleOutput& sample_output,
                                 const torch::Tensor& logits,
                                 const torch::Tensor& sample_idxes) {
  switch (source) {
    case DraftInputConfidenceSource::CARRIED_SELECTED_PROBS:
      return sample_output.draft_input->selected_probs.to(torch::kFloat);
    case DraftInputConfidenceSource::FLAT_PROBS:
      return sample_output.probs.flatten().to(torch::kFloat);
    case DraftInputConfidenceSource::GATHERED_PROBS: {
      const torch::Tensor selected =
          sample_output.next_tokens.to(torch::kLong).view({-1, 1});
      return sample_output.probs.gather(/*dim=*/1, selected)
          .flatten()
          .to(torch::kFloat);
    }
    case DraftInputConfidenceSource::LOGITS_SOFTMAX: {
      const torch::Tensor selected =
          sample_output.next_tokens.to(torch::kLong).view({-1, 1});
      torch::Tensor sample_logits = logits;
      if (sample_logits.size(0) != selected.size(0)) {
        sample_logits =
            sample_logits.index_select(0, sample_idxes.to(torch::kLong));
      }
      // Keep a single probability per row to avoid allocating a full
      // [batch, vocab] softmax.
      return selected_softmax_prob(sample_logits, selected).flatten();
    }
    case DraftInputConfidenceSource::NONE:
      break;
  }
  LOG(FATAL) << "no confidence source available for sample output";
}

}  // namespace

bool has_input_confidence(const SampleOutput& sample_output,
                          const torch::Tensor& logits) {
  return resolve_draft_input_confidence_source(sample_output, logits) !=
         DraftInputConfidenceSource::NONE;
}

torch::Tensor select_input_confidence(const SampleOutput& sample_output,
                                      const torch::Tensor& logits,
                                      const torch::Tensor& sample_idxes) {
  return select_from_source(
      resolve_draft_input_confidence_source(sample_output, logits),
      sample_output,
      logits,
      sample_idxes);
}

DraftInputSamplingState prepare_draft_input_sampling(
    SamplingParameters params,
    std::vector<JsonObjectGrammarState> states,
    const torch::Tensor& committed_tokens,
    const torch::Device& device,
    DraftHistoryBatch& batch,
    DraftTokenHistories histories) {
  const bool update_history = has_penalty_params(params);
  const bool update_grammar =
      !update_history && needs_committed_tokens(params, states);
  DraftInputSamplingState result{
      std::move(params), std::move(states), {}, std::move(histories)};
  if (!update_history && !update_grammar) {
    return result;
  }

  CHECK(committed_tokens.defined());
  CHECK_EQ(committed_tokens.dim(), 2);
  CHECK(committed_tokens.scalar_type() == torch::kInt ||
        committed_tokens.scalar_type() == torch::kLong);
  const int64_t batch_size = committed_tokens.size(0);
  const int64_t width = committed_tokens.size(1);
  CHECK(result.json_object_states.empty() ||
        result.json_object_states.size() == static_cast<size_t>(batch_size));
  result.invalid_rows.assign(batch_size, 0);
  const torch::Tensor tokens_cpu =
      to_cpu_contiguous(committed_tokens, torch::kLong);
  const int64_t* data = tokens_cpu.const_data_ptr<int64_t>();
  if (update_history) {
    import_sampling_histories(
        result.sampling_params, batch_size, result.histories);
  }
  for (int64_t row = 0; row < batch_size; ++row) {
    const detail::PaddedTokenRowScan scan =
        detail::scan_padded_token_row(data + row * width, width);
    const int64_t length = scan.length;
    if (scan.over_int32_range || !scan.tail_all_padding || length == 0) {
      result.invalid_rows[row] = 1;
      continue;
    }
    bool row_failed = false;
    if (update_grammar && result.json_object_states[row].initialized()) {
      // Move the row out before replaying its tokens: the committed history
      // is O(seq_len) and would otherwise be deep-copied per row per step.
      // On failure the partially-mutated state is discarded with the row
      // marked invalid, so nothing moves the moved-from slot back.
      JsonObjectGrammarState state = std::move(result.json_object_states[row]);
      const std::span<const int64_t> row_tokens(data + row * width,
                                                static_cast<size_t>(length));
      for (int64_t token : row_tokens) {
        const int32_t token_id = static_cast<int32_t>(token);
        // accept_token commits only when the token is accepted; a false
        // return marks the row failed here (the caller discards the partially
        // mutated state), avoiding can_accept_token's extra matcher fork +
        // FSM pass per token.
        if (!state.accept_token(token_id)) {
          row_failed = true;
          break;
        }
      }
      if (!row_failed) {
        result.json_object_states[row] = std::move(state);
      }
    }
    if (row_failed) {
      // A grammar-failed row must not commit its partial history either:
      // the draft_input store drops failed rows' state on publish, and
      // appending here would leave the returned projection (and any reused
      // batch) claiming tokens the row never validly accepted.
      result.invalid_rows[row] = 1;
    } else if (update_history) {
      result.histories[row]->append(std::span<const int64_t>(
          data + row * width, static_cast<size_t>(length)));
    }
  }

  if (update_history) {
    apply_draft_history_batch(
        batch, result.sampling_params, result.histories, device);
  }
  if (update_grammar) {
    apply_draft_grammar_bitmask(
        result.sampling_params, result.json_object_states, device);
  }
  return result;
}

void apply_draft_grammar_bitmask(
    SamplingParameters& params,
    const std::vector<JsonObjectGrammarState>& json_states,
    const torch::Device& device) {
  params.filter_bitmask = build_json_object_filter_bitmask(
      json_states, device, JsonObjectMaskBuildPhase::DRAFT);
  params.filter_mask = torch::Tensor();
}

void normalize_mixed_proposal_q(const torch::Tensor& token_ids,
                                torch::Tensor& probs,
                                const torch::Tensor& do_sample) {
  CHECK(token_ids.defined() && probs.defined() && do_sample.defined());
  CHECK_EQ(token_ids.dim(), 1);
  CHECK_EQ(probs.dim(), 2);
  CHECK_EQ(do_sample.dim(), 1);
  CHECK_EQ(probs.size(0), token_ids.size(0));
  CHECK_EQ(do_sample.size(0), token_ids.size(0));
  CHECK_EQ(do_sample.scalar_type(), torch::kBool);
  probs.mul_(do_sample.to(probs.device(), probs.scalar_type()).view({-1, 1}));
  probs.scatter_add_(
      /*dim=*/1,
      token_ids.view({-1, 1}),
      do_sample.logical_not()
          .to(probs.device(), probs.scalar_type())
          .view({-1, 1}));
}

bool finalize_mixed_proposal_q(SampleOutput& sample,
                               const SamplingParameters& params,
                               bool probabilistic) {
  // An all-greedy probabilistic batch carries the point-mass marker instead
  // of a dense q; a non-probabilistic batch carries neither. This is the
  // single authoritative drop point; eager EAGLE3 applies only the
  // non-probabilistic half inline (see remap_draft_sample_output) because the
  // all-greedy drop needs the sampling params.
  const bool point_mass_q = probabilistic && params.all_greedy_sample;
  if (point_mass_q || !probabilistic) {
    sample.probs = torch::Tensor();
    return point_mass_q;
  }
  if (params.all_random_sample) {
    return false;
  }
  normalize_mixed_proposal_q(
      sample.next_tokens, sample.probs, params.do_sample);
  return false;
}

DraftInput finish_draft_input(SampleOutput& sample,
                              torch::Tensor hidden_states,
                              const SamplingParameters& params,
                              bool probabilistic,
                              torch::Tensor selected_probs,
                              int64_t vocab_size,
                              const DraftInputModelSpec& spec) {
  const bool point_mass_q =
      finalize_mixed_proposal_q(sample, params, probabilistic);
  DraftInput draft_input{sample.next_tokens,
                         std::move(hidden_states),
                         sample.probs,
                         std::move(selected_probs),
                         point_mass_q};
  check_draft_input_model(draft_input, spec);
  check_draft_input_mode(draft_input, probabilistic, vocab_size);
  return draft_input;
}

}  // namespace xllm
