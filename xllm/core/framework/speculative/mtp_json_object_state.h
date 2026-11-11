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

#pragma once

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/constrained/json_object_grammar.h"
#include "core/framework/constrained/json_object_output_error.h"
#include "core/framework/speculative/padded_token_row_scan.h"

namespace xllm::detail {

struct JsonDraftValidationScratch final {
  // Matcher-only state after each draft step, indexed [draft_step][sequence].
  std::vector<std::vector<JsonObjectGrammarState>> states_after;
  // Invalid suffix flags in [draft_step][sequence] order.
  std::vector<uint8_t> invalid_draft_step_major;
};

inline bool append_json_draft_step(
    std::vector<JsonObjectGrammarState>& current_states,
    std::vector<uint8_t>& invalid_suffix,
    const std::vector<int32_t>& token_ids,
    JsonDraftValidationScratch& scratch) {
  CHECK_EQ(token_ids.size(), current_states.size())
      << "JSON draft token count must match grammar state count";
  CHECK_EQ(invalid_suffix.size(), current_states.size())
      << "JSON invalid suffix count must match grammar state count";

  bool halt_json_draft = false;
  for (size_t state_idx = 0; state_idx < current_states.size(); ++state_idx) {
    const JsonObjectGrammarState& state = current_states[state_idx];
    bool invalid = invalid_suffix[state_idx] != 0;
    if (!invalid && state.initialized()) {
      // try_accept_token leaves a rejected row's fingerprint untouched, so no
      // can_accept_token pre-fork is needed here.
      invalid =
          !current_states[state_idx].try_accept_token(token_ids[state_idx]);
      if (invalid) {
        halt_json_draft = true;
      }
    }
    invalid_suffix[state_idx] = static_cast<uint8_t>(invalid);
    scratch.invalid_draft_step_major.push_back(static_cast<uint8_t>(invalid));
  }
  // Snapshot only the matcher view: a full clone would copy O(seq_len) per row
  // per draft step.
  scratch.states_after.emplace_back();
  scratch.states_after.back().reserve(current_states.size());
  for (const JsonObjectGrammarState& state : current_states) {
    scratch.states_after.back().emplace_back(state.fork_matcher_only());
  }
  return halt_json_draft;
}

inline std::vector<JsonObjectGrammarState> build_json_validation_states(
    const std::vector<JsonObjectGrammarState>& initial_states,
    JsonDraftValidationScratch scratch,
    std::vector<uint8_t>& invalid_draft) {
  CHECK(!scratch.states_after.empty())
      << "JSON validation requires at least one draft state";
  const size_t num_sequences = initial_states.size();
  const size_t num_draft_steps = scratch.states_after.size();
  for (const auto& states : scratch.states_after) {
    CHECK_EQ(states.size(), num_sequences)
        << "JSON draft state rows must match validation sequences";
  }
  CHECK_EQ(scratch.invalid_draft_step_major.size(),
           num_sequences * num_draft_steps)
      << "JSON invalid draft flags must cover every draft row";

  std::vector<JsonObjectGrammarState> validation_states;
  validation_states.reserve(num_sequences * (num_draft_steps + 1));
  invalid_draft.clear();
  invalid_draft.reserve(num_sequences * num_draft_steps);
  for (size_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    for (size_t draft_idx = 0; draft_idx < num_draft_steps; ++draft_idx) {
      if (draft_idx == 0) {
        validation_states.emplace_back(initial_states[seq_idx]);
      } else {
        validation_states.emplace_back(
            std::move(scratch.states_after[draft_idx - 1][seq_idx]));
      }
      invalid_draft.push_back(
          scratch
              .invalid_draft_step_major[draft_idx * num_sequences + seq_idx]);
    }
    validation_states.emplace_back(
        std::move(scratch.states_after.back()[seq_idx]));
  }
  return validation_states;
}

inline std::vector<JsonTokenMismatch> find_json_accepted_token_mismatches(
    const std::vector<JsonObjectGrammarState>& initial_states,
    const int64_t* accepted_tokens,
    size_t num_rows,
    size_t row_width) {
  CHECK_EQ(initial_states.size(), num_rows)
      << "JSON accepted output rows must match grammar state rows";
  CHECK(accepted_tokens != nullptr || num_rows * row_width == 0)
      << "JSON accepted output token buffer must be defined";

  std::vector<JsonTokenMismatch> mismatches;
  mismatches.reserve(num_rows);
  for (size_t sequence_index = 0; sequence_index < num_rows; ++sequence_index) {
    const JsonObjectGrammarState& initial = initial_states[sequence_index];
    if (!initial.initialized()) {
      continue;
    }
    JsonObjectGrammarState state = initial.fork_matcher_only();

    const size_t row_offset = sequence_index * row_width;
    for (size_t token_offset = 0; token_offset < row_width; ++token_offset) {
      const int64_t raw_token_id = accepted_tokens[row_offset + token_offset];
      if (raw_token_id < 0) {
        break;
      }
      CHECK(detail::is_storable_token_id(raw_token_id))
          << "JSON accepted output token id exceeds int32 range";
      const int32_t token_id = static_cast<int32_t>(raw_token_id);
      if (!state.accept_token(token_id)) {
        // A failed transition may partially mutate the disposable matcher.
        // Reconstruct only this error row to preserve the pre-failure
        // diagnostic fingerprint without a second parse per valid token.
        JsonObjectGrammarState before = initial.fork_matcher_only();
        for (size_t prior = 0; prior < token_offset; ++prior) {
          before.accept_token(
              static_cast<int32_t>(accepted_tokens[row_offset + prior]));
        }
        mismatches.push_back({sequence_index,
                              token_offset,
                              token_id,
                              initial.committed_token_count() + token_offset,
                              before.fingerprint()});
        break;
      }
    }
  }
  return mismatches;
}

// Unions the sample-sequence ids flagged by input and output grammar errors;
// shared by the draft failure paths so the error shape cannot drift.
inline std::unordered_set<std::string> failed_sequence_ids_from_errors(
    const std::vector<JsonObjectOutputError>& input_errors,
    const std::vector<JsonObjectOutputError>& output_errors,
    bool require_non_empty_id) {
  std::unordered_set<std::string> failed_ids;
  failed_ids.reserve(input_errors.size() + output_errors.size());
  for (const auto* errors : {&input_errors, &output_errors}) {
    const std::string_view phase = errors == &input_errors ? "input" : "output";
    for (const JsonObjectOutputError& error : *errors) {
      if (require_non_empty_id) {
        CHECK(!error.sample_sequence_id.empty())
            << "JSON " << phase << " error requires a sampled sequence id";
      }
      failed_ids.emplace(error.sample_sequence_id);
    }
  }
  return failed_ids;
}

}  // namespace xllm::detail
