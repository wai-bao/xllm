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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/framework/sampling/json_object_grammar.h"
#include "core/util/tensor_helper.h"

namespace xllm::detail {

struct JsonDraftValidationScratch final {
  // Matcher-only state after each draft step, indexed [draft_step][sequence].
  std::vector<std::vector<JsonObjectGrammarState>> states_after;
  // Invalid suffix flags in [draft_step][sequence] order.
  std::vector<uint8_t> invalid_draft_step_major;
};

struct JsonAcceptedTokenMismatch final {
  size_t sequence_index = 0;
  size_t token_offset = 0;
  int32_t token_id = -1;
  size_t committed_tokens = 0;
  uint64_t state_fingerprint = 0;
};

inline std::vector<int32_t> copy_json_draft_token_ids(const int64_t* token_ids,
                                                      size_t token_count) {
  std::vector<int32_t> result(token_count);
  if (token_count == 0) {
    return result;
  }
  CHECK(token_ids != nullptr) << "JSON draft token buffer must be defined";
  std::transform(
      token_ids, token_ids + token_count, result.begin(), [](int64_t token_id) {
        return static_cast<int32_t>(token_id);
      });
  return result;
}

// Shared D2H read of draft tokens: fuses the dtype conversion into the
// device-to-host copy, then reads the host buffer in one data_ptr pass
// instead of per-element item() calls. Both the generic draft loop and the
// NPU handoff fallback route through this single implementation.
inline std::vector<int32_t> read_json_draft_tokens_host(
    const torch::Tensor& next_tokens) {
  const torch::Tensor host_tokens =
      to_cpu_contiguous(next_tokens.flatten(), torch::kLong);
  return copy_json_draft_token_ids(host_tokens.const_data_ptr<int64_t>(),
                                   static_cast<size_t>(host_tokens.numel()));
}

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
  // The current batch already owns mutable states. Snapshot only the matcher
  // view of each row for validation instead of cloning both the next batch and
  // its snapshot: fingerprints, the filter bitmask, and FSM replay never read
  // the committed-token history, so a full state clone would copy an O(seq_len)
  // history per row on every draft step.
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

inline std::vector<JsonAcceptedTokenMismatch>
find_json_accepted_token_mismatches(
    const std::vector<JsonObjectGrammarState>& initial_states,
    const int64_t* accepted_tokens,
    size_t num_rows,
    size_t row_width) {
  CHECK_EQ(initial_states.size(), num_rows)
      << "JSON accepted output rows must match grammar state rows";
  CHECK(accepted_tokens != nullptr || num_rows * row_width == 0)
      << "JSON accepted output token buffer must be defined";

  std::vector<JsonAcceptedTokenMismatch> mismatches;
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
      CHECK_LE(raw_token_id,
               static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
          << "JSON accepted output token id exceeds int32 range";
      const int32_t token_id = static_cast<int32_t>(raw_token_id);
      if (!state.accept_token(token_id)) {
        // A failed transition may partially mutate the disposable matcher.
        // Reconstruct only this error row to preserve the pre-failure
        // diagnostic fingerprint without a second parse on every valid token.
        JsonObjectGrammarState before = initial.fork_matcher_only();
        for (size_t prior = 0; prior < token_offset; ++prior) {
          CHECK(before.accept_token(
              static_cast<int32_t>(accepted_tokens[row_offset + prior])));
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

}  // namespace xllm::detail
