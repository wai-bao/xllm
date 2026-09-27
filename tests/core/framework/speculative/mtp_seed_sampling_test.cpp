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

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <span>
#include <vector>

#include "core/framework/sampling/logits_utils.h"
#include "core/framework/speculative/mtp_draft_seed.h"
#include "core/runtime/adaptive_pruning_helpers.h"
#include "core/runtime/forward_params.h"

namespace xllm {
namespace {

SamplingParameters make_penalized_params(int64_t batch_size) {
  SamplingParameters params;
  params.frequency_penalties = torch::ones({batch_size});
  params.presence_penalties = torch::ones({batch_size});
  params.repetition_penalties = torch::full({batch_size}, 2.0F);
  params.selected_token_idxes = torch::arange(batch_size, torch::kInt);
  params.sample_idxes = params.selected_token_idxes;
  return params;
}

JsonObjectGrammar make_seed_grammar() {
  return JsonObjectGrammar({"{", "\"a\"", ":", "1", "}", "stop"},
                           /*stop_token_ids=*/{5});
}

TEST(MtpSeedSamplingTest, HostHistoryOwnsValuesAndIndexesCommittedTokens) {
  std::vector<int64_t> ids{4, 2, 9};
  std::vector<int64_t> counts{3, 0, 5};
  auto history = MtpTokenHistory::from_host_values(ids, counts);
  ids.assign(ids.size(), -1);
  counts.assign(counts.size(), -1);
  EXPECT_EQ(history->ids(), std::vector<int64_t>({4, 2, 9}));
  EXPECT_EQ(history->counts(), std::vector<int32_t>({3, 0, 5}));
  EXPECT_EQ(history->revision(), 0);

  history->append({2, 4, 2, 7});
  EXPECT_EQ(history->ids(), std::vector<int64_t>({4, 2, 9, 7}));
  EXPECT_EQ(history->counts(), std::vector<int32_t>({4, 2, 5, 1}));
  EXPECT_EQ(history->changed_columns(), std::vector<int64_t>({0, 1, 3}));
  EXPECT_EQ(history->revision(), 1);
}

TEST(MtpSeedSamplingTest, HostHistoryAcceptsEmptyAndSlicedValues) {
  auto empty =
      MtpTokenHistory::from_host_values({}, std::span<const int64_t>{});
  EXPECT_TRUE(empty->ids().empty());
  EXPECT_TRUE(empty->counts().empty());
  EXPECT_EQ(empty->revision(), 0);
  empty->append({5});
  EXPECT_EQ(empty->counts(), std::vector<int32_t>({1}));

  constexpr int32_t kMaxCount = std::numeric_limits<int32_t>::max();
  const std::array<int64_t, 5> ids{-1, 9, 3, 5, -1};
  const std::array<int64_t, 5> counts{-1, 0, 7, kMaxCount, -1};
  const auto history = MtpTokenHistory::from_host_values(
      std::span<const int64_t>(ids).subspan(/*offset=*/1, /*count=*/3),
      std::span<const int64_t>(counts).subspan(/*offset=*/1, /*count=*/3));
  EXPECT_EQ(history->ids(), std::vector<int64_t>({9, 3, 5}));
  EXPECT_EQ(history->counts(), std::vector<int32_t>({0, 7, kMaxCount}));
}

TEST(MtpSeedSamplingTest, HostHistoryAcceptsNativeCountsWithoutBorrowingThem) {
  constexpr int32_t kMaxCount = std::numeric_limits<int32_t>::max();
  const std::array<int64_t, 3> ids{4, 2, 9};
  std::array<int32_t, 3> counts{3, 0, kMaxCount};
  auto history = MtpTokenHistory::from_host_values(ids, counts);
  counts.fill(-1);
  EXPECT_EQ(history->ids(), std::vector<int64_t>({4, 2, 9}));
  EXPECT_EQ(history->counts(), std::vector<int32_t>({3, 0, kMaxCount}));
  history->append({2, 4, 2, 7});
  EXPECT_EQ(history->ids(), std::vector<int64_t>({4, 2, 9, 7}));
  EXPECT_EQ(history->counts(), std::vector<int32_t>({4, 2, kMaxCount, 1}));
}

TEST(MtpSeedSamplingTest, AppendsBorrowedTokenPrefixesWithoutRetainingStorage) {
  MtpTokenHistory history;
  std::array<int64_t, 6> tokens{-1, 7, 4, 7, 9, -1};
  history.append(
      std::span<const int64_t>(tokens).subspan(/*offset=*/1, /*count=*/4));
  tokens.fill(-1);
  EXPECT_EQ(history.ids(), std::vector<int64_t>({7, 4, 9}));
  EXPECT_EQ(history.counts(), std::vector<int32_t>({2, 1, 1}));
  EXPECT_EQ(history.changed_columns(), std::vector<int64_t>({0, 1, 2}));
  EXPECT_EQ(history.revision(), 1);

  history.append(std::span<const int64_t>{});
  EXPECT_EQ(history.revision(), 1);
  EXPECT_EQ(history.changed_columns(), std::vector<int64_t>({0, 1, 2}));
  const std::vector<int64_t> repeated{4, 7};
  history.append(repeated);
  EXPECT_EQ(history.counts(), std::vector<int32_t>({3, 2, 1}));
  EXPECT_EQ(history.changed_columns(), std::vector<int64_t>({0, 1}));
  EXPECT_EQ(history.revision(), 2);
}

TEST(MtpSeedSamplingTest, CpuHistoryBatchHandlesEmptyAndBroadcastRows) {
  SamplingParameters params;
  params.unique_token_ids = torch::empty({3, 0}, torch::kLong);
  params.unique_token_counts = torch::empty({3, 0}, torch::kInt);
  params.unique_token_ids_lens = torch::zeros({3}, torch::kInt);
  MtpTokenHistories histories;
  import_mtp_sampling_histories(params, /*batch_size=*/3, histories);
  ASSERT_EQ(histories.size(), 3);
  for (const auto& history : histories) {
    ASSERT_NE(history, nullptr);
    EXPECT_TRUE(history->ids().empty());
    EXPECT_TRUE(history->counts().empty());
  }

  params.unique_token_ids =
      torch::tensor({{7, 9}}, torch::kLong).expand({3, 2});
  params.unique_token_counts =
      torch::tensor({{1, 2}}, torch::kInt).expand({3, 2});
  params.unique_token_ids_lens = torch::full({3}, 2, torch::kInt);
  ASSERT_EQ(params.unique_token_ids.stride(0), 0);
  histories.clear();
  import_mtp_sampling_histories(params, /*batch_size=*/3, histories);
  EXPECT_NE(histories[0], histories[1]);
  histories[0]->append({7});
  EXPECT_EQ(histories[0]->counts(), std::vector<int32_t>({2, 2}));
  EXPECT_EQ(histories[1]->counts(), std::vector<int32_t>({1, 2}));
  EXPECT_EQ(histories[2]->counts(), std::vector<int32_t>({1, 2}));
}

TEST(MtpSeedSamplingTest, CpuHistoryBatchConvertsNonNativeRowLayouts) {
  for (torch::ScalarType id_type : {torch::kInt, torch::kLong}) {
    SamplingParameters params;
    params.unique_token_ids =
        torch::arange(12, id_type).view({4, 3}).transpose(0, 1);
    params.unique_token_counts =
        torch::arange(1, 13, torch::kInt).view({4, 3}).transpose(0, 1);
    params.unique_token_ids_lens = torch::tensor({4, 2, 1}, torch::kInt);
    ASSERT_NE(params.unique_token_ids.stride(1), 1);
    MtpTokenHistories histories;
    import_mtp_sampling_histories(params, /*batch_size=*/3, histories);
    EXPECT_EQ(histories[0]->ids(), std::vector<int64_t>({0, 3, 6, 9}));
    EXPECT_EQ(histories[0]->counts(), std::vector<int32_t>({1, 4, 7, 10}));
    EXPECT_EQ(histories[1]->ids(), std::vector<int64_t>({1, 4}));
    EXPECT_EQ(histories[1]->counts(), std::vector<int32_t>({2, 5}));
    EXPECT_EQ(histories[2]->ids(), std::vector<int64_t>({2}));
    EXPECT_EQ(histories[2]->counts(), std::vector<int32_t>({3}));
  }
}

TEST(MtpTokenHistoryDeathTest, RejectsMalformedHostValues) {
  const std::array<int64_t, 2> ids{1, 2};
  const std::array<int64_t, 2> counts{3, 4};
  const std::array<int64_t, 1> short_counts{3};
  const std::array<int64_t, 2> negative_ids{-1, 2};
  const std::array<int64_t, 2> duplicate_ids{1, 1};
  const std::array<int64_t, 2> negative_counts{-1, 4};
  const std::array<int64_t, 2> oversized_counts{
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1, 4};
  const std::array<int32_t, 1> short_native_counts{3};
  const std::array<int32_t, 2> negative_native_counts{-1, 4};
  EXPECT_DEATH(MtpTokenHistory::from_host_values(ids, short_counts),
               "ids.size");
  EXPECT_DEATH(MtpTokenHistory::from_host_values(negative_ids, counts),
               "token");
  EXPECT_DEATH(MtpTokenHistory::from_host_values(duplicate_ids, counts),
               "duplicate token ids");
  EXPECT_DEATH(MtpTokenHistory::from_host_values(ids, negative_counts),
               "count");
  EXPECT_DEATH(MtpTokenHistory::from_host_values(ids, oversized_counts),
               "count");
  EXPECT_DEATH(MtpTokenHistory::from_host_values(ids, short_native_counts),
               "ids.size");
  EXPECT_DEATH(MtpTokenHistory::from_host_values(ids, negative_native_counts),
               "count");
}

TEST(MtpSeedSamplingTest, UnconstrainedPathDoesNotReadDeviceTokens) {
  SamplingParameters params;
  params.selected_token_idxes = torch::tensor({3}, torch::kInt);
  params.sample_idxes = torch::tensor({0}, torch::kInt);
  params.temperatures = torch::tensor({0.7F});
  const torch::Tensor tokens = torch::empty(
      {1, 1}, torch::TensorOptions().dtype(torch::kLong).device(torch::kMeta));

  MtpSeedSamplingState result =
      prepare_mtp_seed_sampling(params, {}, tokens, torch::kCPU);

  EXPECT_TRUE(result.invalid_rows.empty());
  EXPECT_TRUE(result.json_object_states.empty());
  EXPECT_FALSE(result.sampling_params.unique_token_ids.defined());
  EXPECT_EQ(result.sampling_params.temperatures.data_ptr(),
            params.temperatures.data_ptr());
  EXPECT_EQ(result.sampling_params.selected_token_idxes.data_ptr(),
            params.selected_token_idxes.data_ptr());
}

TEST(MtpSeedSamplingTest, PrefillRootKeepsPromptCountsAtZero) {
  SamplingParameters params = make_penalized_params(/*batch_size=*/1);
  params.unique_token_ids = torch::tensor({{4, 9}}, torch::kLong);
  params.unique_token_counts = torch::tensor({{0, 0}}, torch::kInt);
  params.unique_token_ids_lens = torch::tensor({2}, torch::kInt);

  MtpSeedSamplingState result = prepare_mtp_seed_sampling(
      params, {}, torch::tensor({{4}}, torch::kLong), torch::kCPU);

  EXPECT_EQ(result.invalid_rows, std::vector<uint8_t>({0}));
  EXPECT_TRUE(torch::equal(result.sampling_params.unique_token_ids,
                           torch::tensor({{4, 9}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(result.sampling_params.unique_token_counts,
                           torch::tensor({{1, 0}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.unique_token_counts,
                           torch::tensor({{0, 0}}, torch::kInt)));
  EXPECT_NE(result.sampling_params.unique_token_counts.data_ptr(),
            params.unique_token_counts.data_ptr());
}

TEST(MtpSeedSamplingTest, AppendsAcceptedTokensAndCorrectionExactlyOnce) {
  SamplingParameters params = make_penalized_params(/*batch_size=*/2);
  params.unique_token_ids = torch::tensor({{4, 1, 0}, {2, 7, 0}}, torch::kLong);
  params.unique_token_counts =
      torch::tensor({{1, 0, 0}, {0, 1, 0}}, torch::kInt);
  params.unique_token_ids_lens = torch::tensor({2, 2}, torch::kInt);
  const torch::Tensor tokens =
      torch::tensor({{5, 6, -1}, {7, 7, 8}}, torch::kLong);

  MtpSeedSamplingState result =
      prepare_mtp_seed_sampling(params, {}, tokens, torch::kCPU);

  EXPECT_TRUE(
      torch::equal(result.sampling_params.unique_token_ids,
                   torch::tensor({{4, 1, 5, 6}, {2, 7, 8, 8}}, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(result.sampling_params.unique_token_counts,
                   torch::tensor({{1, 0, 1, 1}, {0, 3, 1, 1}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(result.sampling_params.unique_token_ids_lens,
                           torch::tensor({4, 3}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.unique_token_counts,
                           torch::tensor({{1, 0, 0}, {0, 1, 0}}, torch::kInt)));
}

TEST(MtpSeedSamplingTest, PaddingPreservesTokenZeroAndRepetitionPenalties) {
  SamplingParameters params = make_penalized_params(/*batch_size=*/2);
  params.unique_token_ids = torch::tensor({{0}, {1}}, torch::kLong);
  params.unique_token_counts = torch::tensor({{2}, {0}}, torch::kInt);
  params.unique_token_ids_lens = torch::tensor({1, 1}, torch::kInt);
  MtpSeedSamplingState result = prepare_mtp_seed_sampling(
      params, {}, torch::tensor({{2, -1}, {3, 4}}, torch::kInt), torch::kCPU);
  const SamplingParameters& updated = result.sampling_params;

  torch::Tensor frequency_logits = torch::zeros({2, 5});
  apply_frequency_presence_penalties(frequency_logits,
                                     updated.unique_token_ids,
                                     updated.unique_token_counts,
                                     updated.frequency_penalties,
                                     updated.presence_penalties);
  EXPECT_TRUE(torch::equal(frequency_logits,
                           torch::tensor({{-3.0F, 0.0F, -2.0F, 0.0F, 0.0F},
                                          {0.0F, 0.0F, 0.0F, -2.0F, -2.0F}})));

  torch::Tensor repetition_logits = torch::full({2, 5}, 4.0F);
  apply_repetition_penalties(repetition_logits,
                             updated.unique_token_ids,
                             updated.repetition_penalties);
  EXPECT_TRUE(torch::equal(repetition_logits,
                           torch::tensor({{2.0F, 4.0F, 2.0F, 4.0F, 4.0F},
                                          {4.0F, 2.0F, 4.0F, 2.0F, 2.0F}})));
}

TEST(MtpSeedSamplingTest, PrefillGrammarMaskFollowsNewRoot) {
  JsonObjectGrammar grammar = make_seed_grammar();
  const std::vector<JsonObjectGrammarState> states = {grammar.initial_state()};
  SamplingParameters params;
  params.filter_mask = torch::zeros({1, 6});

  MtpSeedSamplingState result = prepare_mtp_seed_sampling(
      params, states, torch::tensor({{0}}, torch::kLong), torch::kCPU);

  EXPECT_EQ(result.invalid_rows, std::vector<uint8_t>({0}));
  EXPECT_TRUE(states[0].can_accept_token(/*open_object=*/0));
  EXPECT_TRUE(result.json_object_states[0].can_accept_token(/*key=*/1));
  EXPECT_FALSE(
      result.json_object_states[0].can_accept_token(/*open_object=*/0));
  EXPECT_FALSE(result.sampling_params.filter_mask.defined());
  EXPECT_TRUE(params.filter_mask.defined());
  EXPECT_TRUE(torch::equal(
      result.sampling_params.filter_bitmask,
      build_json_object_filter_bitmask(result.json_object_states,
                                       torch::kCPU,
                                       JsonObjectMaskBuildPhase::DRAFT)));
}

TEST(MtpSeedSamplingTest, VerifyGrammarIncludesAcceptedPrefixAndCorrection) {
  JsonObjectGrammar grammar = make_seed_grammar();
  JsonObjectGrammarState state = grammar.initial_state();
  ASSERT_TRUE(state.accept_token(/*open_object=*/0));

  MtpSeedSamplingState result = prepare_mtp_seed_sampling(
      {}, {state}, torch::tensor({{1, 2, 3, -1}}, torch::kLong), torch::kCPU);

  EXPECT_EQ(result.invalid_rows, std::vector<uint8_t>({0}));
  EXPECT_EQ(result.json_object_states[0].snapshot().token_ids,
            std::vector<int32_t>({0, 1, 2, 3}));
  EXPECT_TRUE(
      result.json_object_states[0].can_accept_token(/*close_object=*/4));
  EXPECT_EQ(state.snapshot().token_ids, std::vector<int32_t>({0}));
}

TEST(MtpSeedSamplingTest, InvalidGrammarRowsDoNotCommitPartialHistory) {
  JsonObjectGrammar grammar = make_seed_grammar();
  const std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                      grammar.initial_state()};
  SamplingParameters params = make_penalized_params(/*batch_size=*/2);
  params.unique_token_ids = torch::tensor({{5}, {5}}, torch::kLong);
  params.unique_token_counts = torch::zeros({2, 1}, torch::kInt);
  params.unique_token_ids_lens = torch::ones({2}, torch::kInt);

  MtpSeedSamplingState result = prepare_mtp_seed_sampling(
      params,
      states,
      torch::tensor({{0, 2, -1}, {0, 1, -1}}, torch::kLong),
      torch::kCPU);

  EXPECT_EQ(result.invalid_rows, std::vector<uint8_t>({1, 0}));
  EXPECT_EQ(result.json_object_states[0].fingerprint(),
            states[0].fingerprint());
  EXPECT_TRUE(result.json_object_states[1].can_accept_token(/*colon=*/2));
  EXPECT_TRUE(torch::equal(result.sampling_params.unique_token_ids_lens,
                           torch::tensor({1, 3}, torch::kInt)));
  EXPECT_TRUE(torch::equal(result.sampling_params.unique_token_counts,
                           torch::tensor({{0, 0, 0}, {0, 1, 1}}, torch::kInt)));
}

TEST(MtpSeedSamplingTest, MarksMalformedOrEmptyCommittedRowsInvalid) {
  SamplingParameters params = make_penalized_params(/*batch_size=*/3);
  params.unique_token_ids = torch::tensor({{4}, {4}, {4}}, torch::kLong);
  params.unique_token_counts = torch::ones({3, 1}, torch::kInt);
  params.unique_token_ids_lens = torch::ones({3}, torch::kInt);

  MtpSeedSamplingState result = prepare_mtp_seed_sampling(
      params,
      {},
      torch::tensor({{5, -1, 6}, {-2, -1, -1}, {-1, -1, -1}}, torch::kLong),
      torch::kCPU);

  EXPECT_EQ(result.invalid_rows, std::vector<uint8_t>({1, 1, 1}));
  EXPECT_TRUE(torch::equal(result.sampling_params.unique_token_counts,
                           params.unique_token_counts));
}

TEST(MtpSeedSamplingTest, BorrowedPrefixesRespectStridesAndRowValidation) {
  for (torch::ScalarType dtype : {torch::kInt, torch::kLong}) {
    SamplingParameters params = make_penalized_params(/*batch_size=*/4);
    MtpTokenHistories histories(4);
    for (auto& history : histories) {
      history = std::make_shared<MtpTokenHistory>();
      history->append({9});
    }
    torch::Tensor storage =
        torch::tensor({{1, 2, 0, -1}, {1, -1, 0, -1}, {-1, 3, 4, -1}}, dtype);
    const torch::Tensor tokens = storage.transpose(/*dim0=*/0, /*dim1=*/1);
    ASSERT_FALSE(tokens.is_contiguous());
    auto result =
        prepare_mtp_seed_sampling(params, {}, tokens, torch::kCPU, histories);
    storage.fill_(-1);
    EXPECT_EQ(result.invalid_rows, std::vector<uint8_t>({0, 1, 0, 1}));
    EXPECT_EQ(histories[0]->ids(), std::vector<int64_t>({9, 1}));
    EXPECT_EQ(histories[0]->counts(), std::vector<int32_t>({1, 2}));
    EXPECT_EQ(histories[1]->ids(), std::vector<int64_t>({9}));
    EXPECT_EQ(histories[1]->revision(), 1);
    EXPECT_EQ(histories[2]->ids(), std::vector<int64_t>({9, 0, 4}));
    EXPECT_EQ(histories[2]->counts(), std::vector<int32_t>({1, 2, 1}));
    EXPECT_EQ(histories[3]->revision(), 1);
  }
}

TEST(MtpSeedSamplingTest, EmptyWidthAndOversizedTokensDoNotAdvanceState) {
  JsonObjectGrammar grammar = make_seed_grammar();
  const std::vector<JsonObjectGrammarState> states(2, grammar.initial_state());
  SamplingParameters params = make_penalized_params(/*batch_size=*/2);
  MtpTokenHistories histories(2);
  for (auto& history : histories) {
    history = std::make_shared<MtpTokenHistory>();
    history->append({5});
  }
  constexpr int64_t kOversizedToken =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  for (const torch::Tensor& tokens :
       {torch::empty({2, 0}, torch::kLong),
        torch::tensor(std::vector<int64_t>{0, kOversizedToken, 0, -2},
                      torch::kLong)
            .view({2, 2})}) {
    const auto result = prepare_mtp_seed_sampling(
        params, states, tokens, torch::kCPU, histories);
    EXPECT_EQ(result.invalid_rows, std::vector<uint8_t>({1, 1}));
    for (size_t row = 0; row < histories.size(); ++row) {
      EXPECT_EQ(histories[row]->ids(), std::vector<int64_t>({5}));
      EXPECT_EQ(histories[row]->counts(), std::vector<int32_t>({1}));
      EXPECT_EQ(histories[row]->revision(), 1);
      EXPECT_EQ(result.json_object_states[row].fingerprint(),
                states[row].fingerprint());
    }
  }
}

TEST(MtpSeedSamplingTest,
     CpuHistorySurvivesRebatchWithoutReadingDeviceHistory) {
  SamplingParameters original = make_penalized_params(/*batch_size=*/2);
  original.unique_token_ids =
      torch::tensor({{0, 2, 2}, {1, 3, 4}}, torch::kLong);
  original.unique_token_counts =
      torch::tensor({{2, 1, 1}, {0, 1, 1}}, torch::kInt);
  original.unique_token_ids_lens = torch::tensor({2, 3}, torch::kInt);
  MtpTokenHistories histories;
  import_mtp_sampling_histories(original, 2, histories);
  original.unique_token_counts.fill_(99);
  // Meta statistics cannot be copied to CPU. An established history must ignore
  // them, including when requests arrive in a different batch order.
  original.unique_token_ids = original.unique_token_ids.to(torch::kMeta);
  original.unique_token_counts = original.unique_token_counts.to(torch::kMeta);
  original.unique_token_ids_lens =
      original.unique_token_ids_lens.to(torch::kMeta);
  std::swap(histories[0], histories[1]);
  MtpHistoryBatch batch;
  const auto result =
      prepare_mtp_seed_sampling(original,
                                {},
                                torch::tensor({{4}, {2}}, torch::kLong),
                                torch::kCPU,
                                histories,
                                &batch);
  EXPECT_TRUE(
      torch::equal(result.sampling_params.unique_token_ids,
                   torch::tensor({{1, 3, 4}, {0, 2, 2}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(result.sampling_params.unique_token_counts,
                           torch::tensor({{0, 1, 2}, {2, 2, 2}}, torch::kInt)));
}

TEST(MtpSeedSamplingTest, ReusesProjectionAndUpdatesRepeatedPaddingCounts) {
  SamplingParameters params = make_penalized_params(/*batch_size=*/2);
  MtpHistoryBatch batch;
  auto state = prepare_mtp_seed_sampling(
      params,
      {},
      torch::tensor({{1, -1, -1}, {2, 3, 4}}, torch::kLong),
      torch::kCPU,
      {},
      &batch);
  const void* ids_storage = state.sampling_params.unique_token_ids.data_ptr();
  const void* counts_storage =
      state.sampling_params.unique_token_counts.data_ptr();
  state =
      prepare_mtp_seed_sampling(params,
                                {},
                                torch::tensor({{1, 1}, {3, 4}}, torch::kLong),
                                torch::kCPU,
                                state.histories,
                                &batch);
  EXPECT_EQ(state.sampling_params.unique_token_ids.data_ptr(), ids_storage);
  EXPECT_EQ(state.sampling_params.unique_token_counts.data_ptr(),
            counts_storage);
  EXPECT_TRUE(torch::equal(state.sampling_params.unique_token_counts,
                           torch::tensor({{3, 3, 3}, {1, 2, 2}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(state.sampling_params.unique_token_ids_lens,
                           torch::tensor({1, 3}, torch::kInt)));
  batch.apply(params, state.histories, torch::kCPU);
  EXPECT_EQ(params.unique_token_counts.data_ptr(), counts_storage);
  EXPECT_TRUE(torch::equal(params.unique_token_counts,
                           state.sampling_params.unique_token_counts));
}

TEST(MtpSeedSamplingTest, ReleasingOtherRequestKeepsLiveProjection) {
  SamplingParameters params = make_penalized_params(/*batch_size=*/1);
  MtpHistoryBatch batch;
  MtpTokenHistories histories{std::make_shared<MtpTokenHistory>()};
  histories.front()->append({7, 8});
  batch.apply(params, histories, torch::kCPU);
  const torch::Tensor original_ids = params.unique_token_ids;
  const void* ids_storage = original_ids.data_ptr();

  // Releasing an unrelated request must not discard the current batch.
  MtpTokenHistories other{std::make_shared<MtpTokenHistory>()};
  other.clear();
  batch.reclaim_unused_projection();
  histories.front()->append({7});
  batch.apply(params, histories, torch::kCPU);
  EXPECT_EQ(params.unique_token_ids.data_ptr(), ids_storage);
  EXPECT_TRUE(torch::equal(params.unique_token_counts,
                           torch::tensor({{2, 1}}, torch::kInt)));

  histories.clear();
  batch.reclaim_unused_projection();
  MtpTokenHistories replacement{std::make_shared<MtpTokenHistory>()};
  replacement.front()->append({9});
  batch.apply(params, replacement, torch::kCPU);
  EXPECT_NE(params.unique_token_ids.data_ptr(), ids_storage);
  EXPECT_TRUE(torch::equal(params.unique_token_ids,
                           torch::tensor({{9}}, torch::kLong)));
}

TEST(MtpSeedSamplingTest, ProjectionGrowthAndSkippedRevisionsKeepEveryCount) {
  SamplingParameters params = make_penalized_params(/*batch_size=*/1);
  MtpHistoryBatch batch;
  auto state = prepare_mtp_seed_sampling(params,
                                         {},
                                         torch::tensor({{1, 2}}, torch::kLong),
                                         torch::kCPU,
                                         {},
                                         &batch);
  state.histories[0]->append({3});
  batch.apply(params, state.histories, torch::kCPU);
  EXPECT_EQ(params.unique_token_ids.size(1), 4);
  EXPECT_TRUE(torch::equal(params.unique_token_ids,
                           torch::tensor({{1, 2, 3, 3}}, torch::kLong)));
  state.histories[0]->append({2, 2});
  state.histories[0]->append({4, 3});
  batch.apply(params, state.histories, torch::kCPU);
  EXPECT_TRUE(torch::equal(params.unique_token_counts,
                           torch::tensor({{1, 3, 2, 1}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.unique_token_ids_lens,
                           torch::tensor({4}, torch::kInt)));
  std::weak_ptr<MtpTokenHistory> weak = state.histories.front();
  state.histories.clear();
  EXPECT_TRUE(weak.expired());
  MtpTokenHistories replacement{std::make_shared<MtpTokenHistory>()};
  replacement[0]->append({9});
  batch.apply(params, replacement, torch::kCPU);
  EXPECT_TRUE(torch::equal(params.unique_token_ids,
                           torch::tensor({{9}}, torch::kLong)));
}

TEST(MtpSeedSamplingTest, MixedConfidenceKeepsFullProposalDistributions) {
  MtpDraftSeed first{torch::tensor({1}, torch::kLong),
                     torch::ones({1, 2}),
                     torch::tensor({{0.4F, 0.6F}}),
                     torch::tensor({0.6F})};
  MtpDraftSeed second{torch::tensor({0}, torch::kLong),
                      torch::zeros({1, 2}),
                      torch::tensor({{0.7F, 0.3F}}),
                      torch::Tensor()};
  const MtpDraftSeed batch = cat_mtp_draft_seeds({first, second});
  EXPECT_FALSE(batch.selected_probs.defined());
  EXPECT_TRUE(
      torch::equal(batch.token_ids, torch::tensor({1, 0}, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(batch.probs, torch::tensor({{0.4F, 0.6F}, {0.7F, 0.3F}})));
  EXPECT_TRUE(first.selected_probs.defined());
}

TEST(MtpSeedSamplingTest, MixedGreedyRowsUsePointMassProposalQ) {
  const torch::Tensor token_ids = torch::tensor({1, 0, 2}, torch::kLong);
  const torch::Tensor do_sample =
      torch::tensor({false, true, false}, torch::kBool);
  torch::Tensor probs = torch::tensor(
      {{0.2F, 0.7F, 0.1F}, {0.6F, 0.3F, 0.1F}, {0.2F, 0.2F, 0.6F}});
  write_point_mass_rows(token_ids, probs, do_sample);
  EXPECT_TRUE(torch::allclose(
      probs,
      torch::tensor(
          {{0.0F, 1.0F, 0.0F}, {0.6F, 0.3F, 0.1F}, {0.0F, 0.0F, 1.0F}})));
}

TEST(MtpSeedSamplingTest, FinalizedProposalQTracksSamplingMode) {
  SamplingParameters params;
  params.all_greedy_sample = true;
  SampleOutput greedy;
  greedy.next_tokens = torch::tensor({1}, torch::kLong);
  greedy.probs = torch::tensor({{0.2F, 0.8F}});
  EXPECT_TRUE(finalize_mixed_mtp_proposal_q(greedy, params, true));
  EXPECT_FALSE(greedy.probs.defined());

  SampleOutput deterministic;
  deterministic.next_tokens = greedy.next_tokens;
  deterministic.probs = torch::tensor({{0.2F, 0.8F}});
  EXPECT_FALSE(finalize_mixed_mtp_proposal_q(deterministic, params, false));
  EXPECT_FALSE(deterministic.probs.defined());

  params.all_greedy_sample = false;
  params.all_random_sample = false;
  params.do_sample = torch::tensor({false, true}, torch::kBool);
  SampleOutput mixed;
  mixed.next_tokens = torch::tensor({1, 0}, torch::kLong);
  mixed.probs = torch::tensor({{0.2F, 0.8F}, {0.7F, 0.3F}});
  EXPECT_FALSE(finalize_mixed_mtp_proposal_q(mixed, params, true));
  EXPECT_TRUE(torch::allclose(mixed.probs,
                              torch::tensor({{0.0F, 1.0F}, {0.7F, 0.3F}})));
}

TEST(MtpSeedSamplingTest, ConfidenceFromDenseProbabilitiesPreservesProposalQ) {
  SampleOutput sample;
  sample.next_tokens = torch::tensor({1, 0}, torch::kLong);
  sample.probs = torch::tensor({{0.1F, 0.6F, 0.3F}, {0.5F, 0.25F, 0.25F}});
  const torch::Tensor original_probs = sample.probs.clone();
  const void* original_storage = sample.probs.data_ptr();

  const torch::Tensor confidence = select_mtp_seed_confidence(sample, {}, {});

  EXPECT_TRUE(torch::allclose(confidence, torch::tensor({0.6F, 0.5F})));
  EXPECT_EQ(sample.probs.data_ptr(), original_storage);
  EXPECT_TRUE(torch::equal(sample.probs, original_probs));
  EXPECT_TRUE(
      torch::equal(sample.next_tokens, torch::tensor({1, 0}, torch::kLong)));
}

TEST(MtpSeedSamplingTest, SelectedConfidenceDoesNotRequireTokenIndices) {
  for (torch::ScalarType dtype : {torch::kFloat, torch::kBFloat16}) {
    SampleOutput sample;
    sample.probs = torch::tensor({0.25F, 9.0F, 0.75F, 9.0F}, dtype)
                       .slice(/*dim=*/0, /*start=*/0, /*end=*/4, /*step=*/2);
    const torch::Tensor original = sample.probs.clone();
    const torch::Tensor unused_logits = torch::empty(
        {2, 4},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kMeta));
    const torch::Tensor confidence =
        select_mtp_seed_confidence(sample, unused_logits, {});
    EXPECT_EQ(confidence.scalar_type(), torch::kFloat);
    EXPECT_TRUE(torch::equal(confidence, torch::tensor({0.25F, 0.75F})));
    EXPECT_TRUE(torch::equal(sample.probs, original));
    EXPECT_FALSE(sample.next_tokens.defined());
    if (dtype == torch::kFloat) {
      EXPECT_TRUE(confidence.is_alias_of(sample.probs));
      EXPECT_EQ(confidence.stride(0), 2);
    }
  }
  SampleOutput empty;
  empty.probs = torch::empty({0}, torch::kFloat);
  EXPECT_EQ(select_mtp_seed_confidence(empty, {}, {}).numel(), 0);
}

TEST(MtpSeedSamplingTest, ConfidenceFromLogitsMatchesDenseSoftmax) {
  SampleOutput sample;
  sample.next_tokens = torch::tensor({0, 1}, torch::kLong);
  const torch::Tensor logits =
      torch::tensor({{100.0F, -100.0F, 99.0F}, {-50.0F, -51.0F, -53.0F}});
  const torch::Tensor original_logits = logits.clone();
  const torch::Tensor expected =
      torch::softmax(logits, /*dim=*/1)
          .gather(1, sample.next_tokens.view({-1, 1}))
          .flatten();

  const torch::Tensor confidence =
      select_mtp_seed_confidence(sample, logits, {});

  EXPECT_TRUE(torch::allclose(confidence, expected));
  EXPECT_TRUE(torch::equal(logits, original_logits));
  EXPECT_FALSE(sample.probs.defined());
  EXPECT_EQ(confidence.scalar_type(), torch::kFloat);
}

TEST(MtpSeedSamplingTest, ConfidenceSelectsSampleRowsBeforeReduction) {
  SampleOutput sample;
  sample.next_tokens = torch::tensor({2, 1}, torch::kLong);
  const torch::Tensor logits = torch::tensor(
      {{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}, {7.0F, 8.0F, 9.0F}},
      torch::kBFloat16);
  const torch::Tensor sample_idxes = torch::tensor({2, 0}, torch::kInt);
  const torch::Tensor original_logits = logits.clone();
  const torch::Tensor expected =
      torch::softmax(logits.to(torch::kFloat), /*dim=*/1)
          .index_select(0, sample_idxes.to(torch::kLong))
          .gather(1, sample.next_tokens.view({-1, 1}))
          .flatten();

  const torch::Tensor confidence =
      select_mtp_seed_confidence(sample, logits, sample_idxes);

  EXPECT_TRUE(torch::allclose(confidence, expected));
  EXPECT_EQ(confidence.scalar_type(), torch::kFloat);
  EXPECT_TRUE(torch::equal(logits, original_logits));
  EXPECT_FALSE(sample.probs.defined());
}

TEST(MtpSeedSamplingTest, ConfidenceKeepsSelectedProbabilityPayloadForRebatch) {
  SampleOutput prefill;
  prefill.next_tokens = torch::tensor({1}, torch::kLong);
  prefill.probs = torch::tensor({0.75F}, torch::kBFloat16);
  const torch::Tensor original_probs = prefill.probs.clone();
  SampleOutput extend;
  extend.next_tokens = torch::tensor({0}, torch::kLong);
  const torch::Tensor prefill_confidence =
      select_mtp_seed_confidence(prefill, {}, {});
  const torch::Tensor extend_confidence =
      select_mtp_seed_confidence(extend, torch::zeros({1, 2}), {});

  const MtpDraftSeed combined = cat_mtp_draft_seeds(
      {{prefill.next_tokens, torch::zeros({1, 3}), {}, prefill_confidence},
       {extend.next_tokens, torch::zeros({1, 3}), {}, extend_confidence}});

  EXPECT_TRUE(
      torch::allclose(combined.selected_probs, torch::tensor({0.75F, 0.5F})));
  EXPECT_EQ(combined.selected_probs.scalar_type(), torch::kFloat);
  EXPECT_FALSE(combined.probs.defined());
  EXPECT_TRUE(torch::equal(prefill.probs, original_probs));
}

TEST(MtpSeedSamplingTest, AdaptiveConfidenceUsesSeedSelectedProbabilities) {
  ForwardOutput seed_output;
  seed_output.sample_output.next_tokens = torch::tensor({1, 3}, torch::kLong);
  seed_output.sample_output.probs = torch::full({2, 4}, 0.25F);
  seed_output.sample_output.mtp_draft_seed =
      MtpDraftSeed{seed_output.sample_output.next_tokens,
                   torch::zeros({2, 3}),
                   seed_output.sample_output.probs,
                   torch::tensor({0.8F, 0.6F})};
  ForwardOutput ordinary_output;
  ordinary_output.sample_output.probs = torch::tensor({0.7F, 0.5F});

  EXPECT_TRUE(adaptive_pruning::has_selected_probs_by_step(
      {seed_output, ordinary_output}));
  EXPECT_TRUE(torch::allclose(
      adaptive_pruning::selected_probs_by_step({seed_output, ordinary_output}),
      torch::tensor({{0.8F, 0.7F}, {0.6F, 0.5F}})));

  seed_output.sample_output.probs = torch::Tensor();
  EXPECT_TRUE(adaptive_pruning::has_selected_probs_by_step({seed_output}));
  EXPECT_TRUE(
      torch::allclose(adaptive_pruning::selected_probs_by_step({seed_output}),
                      torch::tensor({{0.8F}, {0.6F}})));
}

TEST(MtpSeedSamplingTest, AdaptiveConfidenceKeepsOrdinaryLogitsFallback) {
  ForwardOutput output;
  output.sample_output.next_tokens = torch::tensor({1}, torch::kLong);
  output.logits = torch::tensor({{0.0F, 0.0F}});

  EXPECT_TRUE(adaptive_pruning::has_selected_probs_by_step({output}));
  EXPECT_TRUE(
      torch::allclose(adaptive_pruning::selected_probs_by_step({output}),
                      torch::tensor({{0.5F}})));
}

TEST(MtpSeedSamplingTest, AdaptiveConfidenceAcceptsColumnProbabilities) {
  ForwardOutput output;
  output.sample_output.next_tokens = torch::tensor({3, 4}, torch::kLong);
  output.sample_output.probs = torch::tensor({{0.7F}, {0.4F}});

  EXPECT_TRUE(adaptive_pruning::has_selected_probs_by_step({output}));
  EXPECT_TRUE(
      torch::allclose(adaptive_pruning::selected_probs_by_step({output}),
                      torch::tensor({{0.7F}, {0.4F}})));
}

TEST(AdaptivePruningDecisionTest, SkipsPruningWithoutTargetWorkReduction) {
  using adaptive_pruning::reduces_validate_work;
  EXPECT_FALSE(reduces_validate_work({3}, 3, 3, true));
  EXPECT_FALSE(reduces_validate_work({1}, 3, 3, true));
  EXPECT_FALSE(reduces_validate_work({3, 3}, 3, 3, false));
  EXPECT_TRUE(reduces_validate_work({1}, 3, 1, true));
  EXPECT_TRUE(reduces_validate_work({1, 3}, 3, 3, false));
}

TEST(AdaptivePruningDecisionTest, FullReplayCheckpointSkipsSelection) {
  std::vector<TargetDecodeStateStore::DecodeState> states(1);
  states.front().valid = true;
  states.front().position_offset = 2;
  bool selected = false;
  const auto plan =
      adaptive_pruning::plan_mtp_validate(states,
                                          /*num_speculative_tokens=*/3,
                                          /*supports_replay_update=*/true,
                                          /*requires_uniform_width=*/true,
                                          [&]() {
                                            selected = true;
                                            return std::vector<int32_t>{1};
                                          });
  EXPECT_TRUE(plan.use_static_tail);
  EXPECT_FALSE(selected);
  EXPECT_EQ(plan.effective_speculative_tokens, 3);
}

TEST(AdaptivePruningDecisionTest, UniformReplayFloorPreservesCheckpoint) {
  std::vector<TargetDecodeStateStore::DecodeState> states(2);
  states.front().valid = true;
  states.front().position_offset = 1;
  states.back().valid = true;
  states.back().position_offset = 0;
  const auto plan = adaptive_pruning::plan_mtp_validate(
      states,
      /*num_speculative_tokens=*/4,
      /*supports_replay_update=*/true,
      /*requires_uniform_width=*/true,
      []() { return std::vector<int32_t>{1, 1}; });
  EXPECT_FALSE(plan.use_static_tail);
  EXPECT_EQ(plan.effective_speculative_tokens, 2);
  EXPECT_EQ(plan.per_seq_val_tokens, (std::vector<int32_t>{3, 3}));
  EXPECT_EQ(plan.prefix_lengths, (std::vector<int32_t>{1, 1}));
}

TEST(AdaptivePruningDecisionTest, UniformNoShrinkUsesStaticTail) {
  std::vector<TargetDecodeStateStore::DecodeState> states(2);
  states.front().valid = true;
  states.front().position_offset = 2;
  const auto plan = adaptive_pruning::plan_mtp_validate(
      states,
      /*num_speculative_tokens=*/4,
      /*supports_replay_update=*/true,
      /*requires_uniform_width=*/true,
      []() { return std::vector<int32_t>{1, 1}; });
  EXPECT_FALSE(plan.use_static_tail);
  EXPECT_EQ(plan.effective_speculative_tokens, 3);
  EXPECT_EQ(plan.per_seq_val_tokens, (std::vector<int32_t>{4, 4}));

  const auto full_plan = adaptive_pruning::plan_mtp_validate(
      states,
      /*num_speculative_tokens=*/4,
      /*supports_replay_update=*/true,
      /*requires_uniform_width=*/true,
      []() { return std::vector<int32_t>{1, 4}; });
  EXPECT_TRUE(full_plan.use_static_tail);
}

TEST(AdaptivePruningDecisionTest, VariableWidthKeepsPerSequenceBenefit) {
  const std::vector<TargetDecodeStateStore::DecodeState> states(2);
  const auto plan = adaptive_pruning::plan_mtp_validate(
      states,
      /*num_speculative_tokens=*/3,
      /*supports_replay_update=*/false,
      /*requires_uniform_width=*/false,
      []() { return std::vector<int32_t>{1, 3}; });
  EXPECT_FALSE(plan.use_static_tail);
  EXPECT_EQ(plan.effective_speculative_tokens, 3);
  EXPECT_EQ(plan.per_seq_val_tokens, (std::vector<int32_t>{2, 4}));
}

TEST(AdaptivePruningDecisionTest, VariableReplayWidthCoversEachCheckpoint) {
  std::vector<TargetDecodeStateStore::DecodeState> states(2);
  states.front().valid = true;
  states.front().position_offset = 2;
  states.back().valid = true;
  states.back().position_offset = 0;
  const auto plan = adaptive_pruning::plan_mtp_validate(
      states,
      /*num_speculative_tokens=*/4,
      /*supports_replay_update=*/true,
      /*requires_uniform_width=*/false,
      []() { return std::vector<int32_t>{1, 1}; });
  EXPECT_FALSE(plan.use_static_tail);
  EXPECT_EQ(plan.effective_speculative_tokens, 3);
  EXPECT_EQ(plan.per_seq_val_tokens, (std::vector<int32_t>{4, 2}));
  EXPECT_EQ(plan.prefix_lengths, (std::vector<int32_t>{1, 1}));
}

}  // namespace
}  // namespace xllm
