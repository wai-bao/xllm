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

#include "core/framework/speculative/mtp_json_object_state.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "core/framework/constrained/json_object_grammar.h"
#include "tests/core/framework/speculative/mtp_draft_input_test_utils.h"

namespace xllm {
namespace {

using test::make_mtp_draft_input_test_grammar;

// Matcher-only fingerprint of a fresh state with `tokens` accepted in order.
uint64_t replay_fingerprint(const JsonObjectGrammar& grammar,
                            std::initializer_list<int32_t> tokens) {
  JsonObjectGrammarState state = grammar.initial_state();
  for (const int32_t token : tokens) {
    state.accept_token(token);
  }
  return state.fingerprint();
}

TEST(MtpJsonObjectStateTest, BuildsSequenceMajorRowsForMixedAcceptedDrafts) {
  JsonObjectGrammar grammar = make_mtp_draft_input_test_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state(), JsonObjectGrammarState()};
  std::vector<JsonObjectGrammarState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(2, 0);
  detail::JsonDraftValidationScratch scratch;

  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {0, 4}, scratch));
  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {1, 4}, scratch));
  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {2, 4}, scratch));

  const auto validation_states =
      detail::build_json_validation_states(initial_states, scratch);

  ASSERT_EQ(validation_states.size(), 8u);
  EXPECT_EQ(validation_states[0].fingerprint(),
            initial_states[0].fingerprint());
  EXPECT_EQ(validation_states[1].fingerprint(),
            scratch.states_after[0][0].fingerprint());
  EXPECT_EQ(validation_states[2].fingerprint(),
            scratch.states_after[1][0].fingerprint());
  EXPECT_EQ(validation_states[3].fingerprint(),
            scratch.states_after[2][0].fingerprint());
  for (size_t row = 4; row < validation_states.size(); ++row) {
    EXPECT_FALSE(validation_states[row].initialized());
  }
  EXPECT_EQ(initial_states[0].snapshot().token_ids, std::vector<int32_t>({}));
}
TEST(MtpJsonObjectStateTest, FreezesFirstInvalidDraftAndItsSuffix) {
  JsonObjectGrammar grammar = make_mtp_draft_input_test_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state(), JsonObjectGrammarState()};
  std::vector<JsonObjectGrammarState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(2, 0);
  detail::JsonDraftValidationScratch scratch;

  EXPECT_TRUE(detail::append_json_draft_step(
      current_states, invalid_suffix, {4, 0}, scratch));
  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {-1, 0}, scratch));
  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {-1, 0}, scratch));

  const auto validation_states =
      detail::build_json_validation_states(initial_states, scratch);

  for (size_t row = 0; row < 4; ++row) {
    EXPECT_EQ(validation_states[row].fingerprint(),
              initial_states[0].fingerprint());
  }
}

TEST(MtpJsonObjectStateTest, PreservesAcceptedPrefixAfterMiddleInvalidDraft) {
  JsonObjectGrammar grammar = make_mtp_draft_input_test_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state()};
  std::vector<JsonObjectGrammarState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(1, 0);
  detail::JsonDraftValidationScratch scratch;

  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {0}, scratch));
  EXPECT_TRUE(detail::append_json_draft_step(
      current_states, invalid_suffix, {2}, scratch));
  EXPECT_FALSE(detail::append_json_draft_step(
      current_states, invalid_suffix, {-1}, scratch));

  const auto validation_states =
      detail::build_json_validation_states(initial_states, scratch);

  ASSERT_EQ(validation_states.size(), 4u);
  EXPECT_EQ(validation_states[1].fingerprint(),
            scratch.states_after[0][0].fingerprint());
  EXPECT_EQ(validation_states[2].fingerprint(),
            scratch.states_after[0][0].fingerprint());
  EXPECT_EQ(validation_states[3].fingerprint(),
            scratch.states_after[0][0].fingerprint());
}

TEST(MtpJsonObjectStateTest, FiveDraftRowsReplayFromCommittedState) {
  JsonObjectGrammar grammar = make_mtp_draft_input_test_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state()};
  std::vector<JsonObjectGrammarState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(1, 0);
  detail::JsonDraftValidationScratch scratch;
  const std::vector<int32_t> draft_tokens = {0, 1, 2, 3, 4};

  for (const int32_t token_id : draft_tokens) {
    EXPECT_FALSE(detail::append_json_draft_step(
        current_states, invalid_suffix, {token_id}, scratch));
  }

  const std::vector<JsonObjectGrammarState> validation_states =
      detail::build_json_validation_states(initial_states, scratch);

  ASSERT_EQ(validation_states.size(), 6u);
  JsonObjectGrammarState committed_state = initial_states[0];
  for (size_t token_idx = 0; token_idx < draft_tokens.size(); ++token_idx) {
    EXPECT_EQ(validation_states[token_idx].fingerprint(),
              committed_state.fingerprint());
    ASSERT_TRUE(
        validation_states[token_idx].can_accept_token(draft_tokens[token_idx]));
    ASSERT_TRUE(committed_state.accept_token(draft_tokens[token_idx]));
  }
  EXPECT_EQ(validation_states.back().fingerprint(),
            committed_state.fingerprint());
  EXPECT_TRUE(validation_states.back().can_accept_token(/*stop=*/5));
  EXPECT_TRUE(committed_state.accept_token(/*stop=*/5));
}

TEST(MtpJsonObjectStateTest,
     ReasoningDisabledValidationRowsRejectReasoningEndMarker) {
  JsonObjectGrammar grammar(
      {"{", "\"a\"", ":", "1", "}", "stop", "<think>", "</think>"},
      /*stop_token_ids=*/{5},
      /*reasoning_end_token_ids=*/{7});
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state(/*reasoning_phase=*/false)};
  std::vector<JsonObjectGrammarState> current_states = initial_states;
  std::vector<uint8_t> invalid_suffix(1, 0);
  detail::JsonDraftValidationScratch scratch;
  const std::vector<int32_t> draft_tokens = {0, 1, 2, 3, 4};

  for (const int32_t token_id : draft_tokens) {
    EXPECT_FALSE(detail::append_json_draft_step(
        current_states, invalid_suffix, {token_id}, scratch));
  }

  const std::vector<JsonObjectGrammarState> validation_states =
      detail::build_json_validation_states(initial_states, scratch);

  ASSERT_EQ(validation_states.size(), 6u);
  for (const JsonObjectGrammarState& state : validation_states) {
    EXPECT_FALSE(state.in_reasoning());
    EXPECT_FALSE(state.can_accept_token(/*reasoning_end=*/7));
  }
}

TEST(MtpJsonObjectStateTest, ReplaysFiveStepMixedAcceptedOutputRows) {
  JsonObjectGrammar grammar = make_mtp_draft_input_test_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state(), JsonObjectGrammarState()};
  const int64_t accepted_tokens[] = {
      0,
      1,
      2,
      3,
      4,
      5,
      5,
      4,
      3,
      2,
      1,
      0,
  };

  const std::vector<JsonTokenMismatch> mismatches =
      detail::find_json_accepted_token_mismatches(
          initial_states, accepted_tokens, /*num_rows=*/2, /*row_width=*/6);

  EXPECT_TRUE(mismatches.empty());
}

TEST(MtpJsonObjectStateTest, ReportsFirstAcceptedOutputMismatchPerSequence) {
  JsonObjectGrammar grammar = make_mtp_draft_input_test_grammar();
  JsonObjectGrammarState committed_state = grammar.initial_state();
  ASSERT_TRUE(committed_state.accept_token(/*open_object=*/0));
  const std::vector<JsonObjectGrammarState> initial_states = {
      committed_state, JsonObjectGrammarState()};
  const int64_t accepted_tokens[] = {
      1,
      4,
      3,
      -1,
      -1,
      -1,
      5,
      4,
      3,
      2,
      1,
      0,
  };

  const std::vector<JsonTokenMismatch> mismatches =
      detail::find_json_accepted_token_mismatches(
          initial_states, accepted_tokens, /*num_rows=*/2, /*row_width=*/6);

  ASSERT_EQ(mismatches.size(), 1u);
  EXPECT_EQ(mismatches[0].sequence_index, 0u);
  EXPECT_EQ(mismatches[0].token_offset, 1u);
  EXPECT_EQ(mismatches[0].token_id, 4);
  EXPECT_EQ(mismatches[0].committed_tokens, 2u);
  JsonObjectGrammarState before_mismatch = committed_state;
  ASSERT_TRUE(before_mismatch.accept_token(/*key=*/1));
  EXPECT_EQ(mismatches[0].state_fingerprint, before_mismatch.fingerprint());
}
TEST(MtpJsonObjectStateTest, AcceptedOutputStopsAtAnyNegativePadding) {
  JsonObjectGrammar grammar = make_mtp_draft_input_test_grammar();
  const std::vector<JsonObjectGrammarState> initial_states = {
      grammar.initial_state()};
  const int64_t accepted_tokens[] = {0, -2};

  const std::vector<JsonTokenMismatch> mismatches =
      detail::find_json_accepted_token_mismatches(
          initial_states, accepted_tokens, /*num_rows=*/1, /*row_width=*/2);
  EXPECT_TRUE(mismatches.empty());
}

}  // namespace
}  // namespace xllm
