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

#include "core/framework/speculative/draft_input.h"
#include "core/framework/speculative/padded_token_row_scan.h"
#include "core/util/tensor_helper.h"

namespace xllm {

torch::Tensor select_input_confidence(
    const SampleOutput& sample_output,
    const std::optional<DraftInput>& draft_input,
    const torch::Tensor& logits,
    const torch::Tensor& sample_idxes) {
  // Confidence ladder: the carried seed's selected probs, then the sampler's
  // flat per-row confidences, then a gather (the sampler's own export wins
  // because it reflects any remap).
  if (draft_input.has_value() && draft_input->selected_probs.defined()) {
    return draft_input->selected_probs.to(torch::kFloat);
  }
  if (sample_output.draft_logits.defined() &&
      sample_output.draft_logits.dim() == 1) {
    return sample_output.draft_logits.flatten().to(torch::kFloat);
  }
  const bool can_gather =
      sample_output.next_tokens.defined() &&
      (sample_output.draft_logits.defined() || logits.defined());
  CHECK(can_gather) << "no confidence source available for sample output";
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
    // Confidence-head semantics: the selected token's raw logit.
    return gather_from_logits(sample_logits, selected.squeeze(-1))
        .to(torch::kFloat);
  }
  return gather_probs_from_logits(sample_logits, selected.squeeze(/*dim=*/-1));
}

DraftInputSamplingState prepare_draft_input_sampling(
    SamplingParameters params,
    std::vector<JsonObjectGrammarState> states,
    const torch::Tensor& committed_tokens,
    const torch::Device& device,
    DraftHistoryBatch& batch,
    DraftTokenHistories histories) {
  // The history projection and the grammar replay are independent; a
  // grammar-failed row skips its history append.
  DraftInputSamplingState result{
      std::move(params), std::move(states), {}, std::move(histories)};
  const bool update_history = has_penalty_params(result.sampling_params);
  const bool update_grammar = needs_committed_tokens_for_grammar(
      std::span<const JsonObjectGrammarState>(result.json_object_states));
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
      // Move the row out: the committed history is O(seq_len) and would
      // otherwise be deep-copied per row per step.
      JsonObjectGrammarState state = std::move(result.json_object_states[row]);
      const std::span<const int64_t> row_tokens(data + row * width,
                                                static_cast<size_t>(length));
      for (int64_t token : row_tokens) {
        const int32_t token_id = static_cast<int32_t>(token);
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
      // A failed row must not commit its partial history either.
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
  params.filter_bitmask() = build_json_object_filter_bitmask(
      json_states, device, JsonObjectMaskBuildPhase::DRAFT);
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

std::vector<uint8_t> merge_invalid_rows(const std::vector<uint8_t>& lhs,
                                        const std::vector<uint8_t>& rhs) {
  if (lhs.empty()) {
    return rhs;
  }
  if (rhs.empty()) {
    return lhs;
  }
  CHECK_EQ(lhs.size(), rhs.size());
  std::vector<uint8_t> merged = lhs;
  for (size_t row = 0; row < merged.size(); ++row) {
    merged[row] = merged[row] != 0 || rhs[row] != 0 ? 1 : 0;
  }
  return merged;
}

DraftInput finish_draft_input(SampleOutput& sample,
                              torch::Tensor hidden_states,
                              const SamplingParameters& params,
                              bool probabilistic,
                              torch::Tensor selected_probs,
                              const DraftInputModelSpec& spec) {
  // The greedy drop must happen here rather than in remap_draft_sample_output:
  // only this tail sees the sampling params.
  const bool point_mass_q =
      probabilistic && params.sampling_mode() == SamplingMode::GREEDY;
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
