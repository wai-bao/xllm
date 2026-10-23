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
#include "core/framework/speculative/padded_token_row_scan.h"
#include "core/util/tensor_helper.h"

namespace xllm {

namespace {

// Confidence ladder: the carried draft_input's selected probs, then the
// sampler's probs, then a per-row softmax over logits+next_tokens. The enum
// exists so the availability probe stays tensor-free while the extractor
// pays the single real compute.
enum class DraftInputConfidenceSource : uint8_t {
  CARRIED_SELECTED_PROBS,
  FLAT_DRAFT_LOGITS,
  LOGITS_SOFTMAX,
  NONE,
};

DraftInputConfidenceSource resolve_draft_input_confidence_source(
    const SampleOutput& sample_output,
    const std::optional<DraftInput>& draft_input,
    const torch::Tensor& logits) {
  if (draft_input.has_value() && draft_input->selected_probs.defined()) {
    return DraftInputConfidenceSource::CARRIED_SELECTED_PROBS;
  }
  // 1-D draft logits are already per-row selected confidences (the sampler's
  // compact export); pass them through without a softmax.
  if (sample_output.draft_logits.defined() &&
      sample_output.draft_logits.dim() == 1) {
    return DraftInputConfidenceSource::FLAT_DRAFT_LOGITS;
  }
  if (!sample_output.next_tokens.defined()) {
    return DraftInputConfidenceSource::NONE;
  }
  if (sample_output.draft_logits.defined() || logits.defined()) {
    return DraftInputConfidenceSource::LOGITS_SOFTMAX;
  }
  return DraftInputConfidenceSource::NONE;
}

torch::Tensor select_from_source(DraftInputConfidenceSource source,
                                 const SampleOutput& sample_output,
                                 const std::optional<DraftInput>& draft_input,
                                 const torch::Tensor& logits,
                                 const torch::Tensor& sample_idxes) {
  switch (source) {
    case DraftInputConfidenceSource::CARRIED_SELECTED_PROBS:
      return draft_input->selected_probs.to(torch::kFloat);
    case DraftInputConfidenceSource::FLAT_DRAFT_LOGITS:
      return sample_output.draft_logits.flatten().to(torch::kFloat);
    case DraftInputConfidenceSource::LOGITS_SOFTMAX: {
      // The sampler's own export reflects any draft-to-target remap, so it
      // wins over the caller-provided forward logits. Confidence is the
      // selected token's raw logit (confidence-head semantics); no softmax
      // materialization.
      torch::Tensor sample_logits = sample_output.draft_logits.defined()
                                        ? sample_output.draft_logits
                                        : logits;
      const torch::Tensor selected =
          sample_output.next_tokens.to(torch::kLong).view({-1, 1});
      if (sample_logits.size(0) != selected.size(0)) {
        sample_logits =
            sample_logits.index_select(0, sample_idxes.to(torch::kLong));
      }
      if (sample_output.draft_logits.defined()) {
        return sample_logits.gather(/*dim=*/-1, selected)
            .flatten()
            .to(torch::kFloat);
      }
      return gather_probs_from_logits(sample_logits,
                                      selected.squeeze(/*dim=*/-1));
    }
    case DraftInputConfidenceSource::NONE:
      break;
  }
  LOG(FATAL) << "no confidence source available for sample output";
}

}  // namespace

bool has_input_confidence(const SampleOutput& sample_output,
                          const std::optional<DraftInput>& draft_input,
                          const torch::Tensor& logits) {
  return resolve_draft_input_confidence_source(
             sample_output, draft_input, logits) !=
         DraftInputConfidenceSource::NONE;
}

torch::Tensor select_input_confidence(
    const SampleOutput& sample_output,
    const std::optional<DraftInput>& draft_input,
    const torch::Tensor& logits,
    const torch::Tensor& sample_idxes) {
  return select_from_source(
      resolve_draft_input_confidence_source(sample_output, draft_input, logits),
      sample_output,
      draft_input,
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
  // The two consumers are independent: the history projection appends the
  // committed tokens for penalties while the grammar loop replays the same
  // tokens for FSM transitions. A grammar-failed row skips its history
  // append below, so the two can never disagree on what was committed.
  DraftInputSamplingState result{
      std::move(params), std::move(states), {}, std::move(histories)};
  if (!needs_committed_tokens(result.sampling_params,
                              result.json_object_states)) {
    return result;
  }
  const bool update_history = has_penalty_params(result.sampling_params);
  const bool update_grammar = std::any_of(
      result.json_object_states.begin(),
      result.json_object_states.end(),
      [](const JsonObjectGrammarState& state) { return state.initialized(); });
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
    batch.apply(result.sampling_params, result.histories, device);
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

std::vector<uint8_t> filter_vocabulary_failures(
    std::vector<uint8_t> invalid_rows,
    const std::vector<JsonObjectGrammarState>& json_states) {
  if (invalid_rows.empty() || json_states.empty()) {
    return invalid_rows;
  }
  CHECK_EQ(invalid_rows.size(), json_states.size());
  bool any_failed = false;
  for (size_t row = 0; row < invalid_rows.size(); ++row) {
    const auto& state = json_states[row];
    invalid_rows[row] =
        invalid_rows[row] && (!state.initialized() || !state.is_valid());
    any_failed = any_failed || invalid_rows[row] != 0;
  }
  if (!any_failed) {
    invalid_rows.clear();
  }
  return invalid_rows;
}

DraftInput finish_draft_input(SampleOutput& sample,
                              torch::Tensor hidden_states,
                              const SamplingParameters& params,
                              bool probabilistic,
                              torch::Tensor selected_probs,
                              const DraftInputModelSpec& spec) {
  // An all-greedy probabilistic batch carries the point-mass marker instead
  // of a dense q; a non-probabilistic batch carries neither. The greedy drop
  // must happen here rather than in remap_draft_sample_output because only
  // this tail sees the sampling params.
  const bool point_mass_q =
      probabilistic && params.sampling_mode == SamplingMode::GREEDY;
  if (point_mass_q) {
    sample.draft_logits = torch::Tensor();
  }
  DraftInput draft_input{sample.next_tokens,
                         std::move(hidden_states),
                         std::move(selected_probs),
                         sample.draft_logits,
                         point_mass_q};
  check_draft_input_model(draft_input, spec);
  check_draft_input_mode(draft_input, probabilistic);
  return draft_input;
}

}  // namespace xllm
