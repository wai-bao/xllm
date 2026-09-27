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

#include "core/framework/speculative/mtp_seed_sampling.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "core/framework/speculative/mtp_json_object_state.h"
#include "core/framework/speculative/padded_token_row_scan.h"
#include "core/util/tensor_helper.h"

namespace xllm {

MtpConfidenceSource mtp_seed_confidence_source(
    const SampleOutput& sample_output,
    const torch::Tensor& logits) {
  if (sample_output.mtp_draft_seed.has_value() &&
      sample_output.mtp_draft_seed->selected_probs.defined()) {
    return MtpConfidenceSource::SELECTED_PROBS;
  }
  if (sample_output.probs.defined()) {
    return MtpConfidenceSource::PROBS;
  }
  // The logits rung needs both the logits and the selected tokens, mirroring
  // the CHECKs select_mtp_seed_confidence would raise on this input.
  if (logits.defined() && sample_output.next_tokens.defined()) {
    return MtpConfidenceSource::LOGITS;
  }
  return MtpConfidenceSource::NONE;
}

torch::Tensor select_mtp_seed_confidence(const SampleOutput& sample_output,
                                         const torch::Tensor& logits,
                                         const torch::Tensor& sample_idxes) {
  switch (mtp_seed_confidence_source(sample_output, logits)) {
    case MtpConfidenceSource::SELECTED_PROBS:
      return sample_output.mtp_draft_seed->selected_probs.to(torch::kFloat);
    case MtpConfidenceSource::PROBS: {
      const torch::Tensor& probs = sample_output.probs;
      // 1-D probs and [batch,1] probs are already per-row selected values.
      if (probs.dim() == 1 || (probs.dim() == 2 && probs.size(1) == 1)) {
        return probs.flatten().to(torch::kFloat);
      }
      const torch::Tensor selected =
          sample_output.next_tokens.to(torch::kLong).view({-1, 1});
      const torch::Tensor confidence =
          probs.gather(/*dim=*/1, selected).flatten();
      return confidence.to(torch::kFloat);
    }
    case MtpConfidenceSource::LOGITS: {
      const torch::Tensor selected =
          sample_output.next_tokens.to(torch::kLong).view({-1, 1});
      torch::Tensor sample_logits = logits;
      if (sample_logits.size(0) != selected.size(0)) {
        CHECK(sample_idxes.defined());
        sample_logits =
            sample_logits.index_select(0, sample_idxes.to(torch::kLong));
      }
      // Keep a single probability per row to avoid allocating a full
      // [batch, vocab] softmax.
      return selected_softmax_prob(sample_logits, selected);
    }
    case MtpConfidenceSource::NONE:
      break;
  }
  LOG(FATAL) << "No confidence source available for MTP seed sampling";
}

MtpSeedSamplingState prepare_mtp_seed_sampling(
    SamplingParameters params,
    std::vector<JsonObjectGrammarState> states,
    const torch::Tensor& committed_tokens,
    const torch::Device& device,
    MtpTokenHistories histories,
    MtpHistoryBatch* batch) {
  const bool update_history = has_penalty_params(params);
  const bool update_grammar =
      std::any_of(states.begin(), states.end(), [](const auto& state) {
        return state.initialized();
      });
  MtpSeedSamplingState result{
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
  const torch::Tensor tokens_cpu = to_cpu_int64_contiguous(committed_tokens);
  const int64_t* data = tokens_cpu.const_data_ptr<int64_t>();
  if (update_history) {
    import_mtp_sampling_histories(
        result.sampling_params, batch_size, result.histories);
  }
  for (int64_t row = 0; row < batch_size; ++row) {
    const detail::PaddedTokenRowScan scan =
        detail::scan_padded_token_row(data + row * width, width);
    const int64_t length = scan.length;
    if (scan.over_int32_range || !scan.tail_all_padding || length == 0) {
      result.invalid_rows[row] = 1;
    }
    if (update_grammar && result.invalid_rows[row] == 0 &&
        result.json_object_states[row].initialized()) {
      // Move the row out before replaying its tokens: the committed history
      // is O(seq_len) and would otherwise be deep-copied per row per step.
      // On failure the partially-mutated state is discarded with the row
      // marked invalid, so nothing moves the moved-from slot back.
      JsonObjectGrammarState state = std::move(result.json_object_states[row]);
      const std::span<const int64_t> row_tokens(data + row * width,
                                                static_cast<size_t>(length));
      bool row_invalid = false;
      for (int64_t token : row_tokens) {
        const int32_t token_id = static_cast<int32_t>(token);
        // accept_token both trials and commits the transition: a false
        // return leaves the row invalid, avoiding can_accept_token's extra
        // matcher fork + FSM pass per token.
        if (!state.accept_token(token_id)) {
          row_invalid = true;
          break;
        }
      }
      if (!row_invalid) {
        result.json_object_states[row] = std::move(state);
      } else {
        result.invalid_rows[row] = 1;
      }
    }
    if (update_history && result.invalid_rows[row] == 0) {
      result.histories[row]->append(std::span<const int64_t>(
          data + row * width, static_cast<size_t>(length)));
    }
  }

  if (update_history) {
    apply_mtp_history_batch(
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

bool finalize_mixed_mtp_proposal_q(SampleOutput& sample,
                                   const SamplingParameters& params,
                                   bool probabilistic) {
  // All-greedy batches carry the point-mass marker instead of a dense q;
  // non-probabilistic seeds never carry q at all.
  if (!probabilistic || params.all_greedy_sample) {
    sample.probs = torch::Tensor();
    return probabilistic;
  }
  if (params.all_random_sample) {
    return false;
  }
  // Mixed greedy/random batch: the sampler emitted softmax for every row.
  // Overwrite the greedy rows with point masses in place. Must run after any
  // draft-to-target vocabulary remapping, which consumes raw sampler q.
  write_point_mass_rows(sample.next_tokens, sample.probs, params.do_sample);
  return false;
}

}  // namespace xllm
