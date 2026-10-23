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

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <span>
#include <vector>

#include "core/framework/sampling/logits_utils.h"
#include "core/framework/speculative/draft_input.h"
#include "core/framework/speculative/draft_input_sampling.h"
#include "core/runtime/forward_params.h"
#include "tests/core/framework/speculative/mtp_draft_input_test_utils.h"

namespace xllm {
namespace {

using test::make_mtp_draft_input_test_grammar;

SamplingParameters make_penalized_params(int64_t batch_size) {
  SamplingParameters params;
  params.frequency_penalties = torch::ones({batch_size});
  params.presence_penalties = torch::ones({batch_size});
  params.repetition_penalties = torch::full({batch_size}, 2.0F);
  params.selected_token_idxes = torch::arange(batch_size, torch::kInt);
  return params;
}

TEST(MtpSeedSamplingTest, HostHistoryOwnsValuesAndIndexesCommittedTokens) {
  std::vector<int64_t> ids{4, 2, 9};
  std::vector<int64_t> counts{3, 0, 5};
  auto history = DraftTokenHistory::from_host_values(ids, counts);
  ids.assign(ids.size(), -1);
  counts.assign(counts.size(), -1);
  EXPECT_EQ(history->ids(), std::vector<int64_t>({4, 2, 9}));
  EXPECT_EQ(history->counts(), std::vector<int32_t>({3, 0, 5}));
  EXPECT_EQ(history->revision(), 0);

  history->append(
      std::span<const int64_t>(std::vector<int64_t>{2, 4, 2, 7}.data(), 4));
  EXPECT_EQ(history->ids(), std::vector<int64_t>({4, 2, 9, 7}));
  EXPECT_EQ(history->counts(), std::vector<int32_t>({4, 2, 5, 1}));
  EXPECT_EQ(history->changed_columns(), std::vector<int64_t>({0, 1, 3}));
  EXPECT_EQ(history->revision(), 1);
}

TEST(MtpSeedSamplingTest, HostHistoryAcceptsEmptyAndSlicedValues) {
  auto empty =
      DraftTokenHistory::from_host_values({}, std::span<const int64_t>{});
  EXPECT_TRUE(empty->ids().empty());
  EXPECT_TRUE(empty->counts().empty());
  EXPECT_EQ(empty->revision(), 0);
  empty->append(std::span<const int64_t>(std::vector<int64_t>{5}.data(), 1));
  EXPECT_EQ(empty->counts(), std::vector<int32_t>({1}));

  constexpr int32_t kMaxCount = std::numeric_limits<int32_t>::max();
  const std::array<int64_t, 5> ids{-1, 9, 3, 5, -1};
  const std::array<int64_t, 5> counts{-1, 0, 7, kMaxCount, -1};
  const auto history = DraftTokenHistory::from_host_values(
      std::span<const int64_t>(ids).subspan(/*offset=*/1, /*count=*/3),
      std::span<const int64_t>(counts).subspan(/*offset=*/1, /*count=*/3));
  EXPECT_EQ(history->ids(), std::vector<int64_t>({9, 3, 5}));
  EXPECT_EQ(history->counts(), std::vector<int32_t>({0, 7, kMaxCount}));
}

TEST(MtpSeedSamplingTest, HostHistoryAcceptsNativeCountsWithoutBorrowingThem) {
  constexpr int32_t kMaxCount = std::numeric_limits<int32_t>::max();
  const std::array<int64_t, 3> ids{4, 2, 9};
  std::array<int32_t, 3> counts{3, 0, kMaxCount};
  auto history = DraftTokenHistory::from_host_values(ids, counts);
  counts.fill(-1);
  EXPECT_EQ(history->ids(), std::vector<int64_t>({4, 2, 9}));
  EXPECT_EQ(history->counts(), std::vector<int32_t>({3, 0, kMaxCount}));
  history->append(
      std::span<const int64_t>(std::vector<int64_t>{2, 4, 2, 7}.data(), 4));
  EXPECT_EQ(history->ids(), std::vector<int64_t>({4, 2, 9, 7}));
  EXPECT_EQ(history->counts(), std::vector<int32_t>({4, 2, kMaxCount, 1}));
}

TEST(MtpSeedSamplingTest, AppendsBorrowedTokenPrefixesWithoutRetainingStorage) {
  DraftHistoryBatch batch;
  DraftTokenHistory history;
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
  DraftHistoryBatch batch;
  SamplingParameters params;
  params.unique_token_ids = torch::empty({3, 0}, torch::kLong);
  params.unique_token_counts = torch::empty({3, 0}, torch::kInt);
  params.unique_token_ids_lens = torch::zeros({3}, torch::kInt);
  DraftTokenHistories histories;
  import_sampling_histories(params, /*batch_size=*/3, histories);
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
  import_sampling_histories(params, /*batch_size=*/3, histories);
  EXPECT_NE(histories[0], histories[1]);
  histories[0]->append(
      std::span<const int64_t>(std::vector<int64_t>{7}.data(), 1));
  EXPECT_EQ(histories[0]->counts(), std::vector<int32_t>({2, 2}));
  EXPECT_EQ(histories[1]->counts(), std::vector<int32_t>({1, 2}));
  EXPECT_EQ(histories[2]->counts(), std::vector<int32_t>({1, 2}));
}

TEST(MtpSeedSamplingTest, CpuHistoryBatchConvertsNonNativeRowLayouts) {
  DraftHistoryBatch batch;
  for (torch::ScalarType id_type : {torch::kInt, torch::kLong}) {
    SamplingParameters params;
    params.unique_token_ids =
        torch::arange(12, id_type).view({4, 3}).transpose(0, 1);
    params.unique_token_counts =
        torch::arange(1, 13, torch::kInt).view({4, 3}).transpose(0, 1);
    params.unique_token_ids_lens = torch::tensor({4, 2, 1}, torch::kInt);
    ASSERT_NE(params.unique_token_ids.stride(1), 1);
    DraftTokenHistories histories;
    import_sampling_histories(params, /*batch_size=*/3, histories);
    EXPECT_EQ(histories[0]->ids(), std::vector<int64_t>({0, 3, 6, 9}));
    EXPECT_EQ(histories[0]->counts(), std::vector<int32_t>({1, 4, 7, 10}));
    EXPECT_EQ(histories[1]->ids(), std::vector<int64_t>({1, 4}));
    EXPECT_EQ(histories[1]->counts(), std::vector<int32_t>({2, 5}));
    EXPECT_EQ(histories[2]->ids(), std::vector<int64_t>({2}));
    EXPECT_EQ(histories[2]->counts(), std::vector<int32_t>({3}));
  }
}

TEST(DraftTokenHistoryDeathTest, RejectsMalformedHostValues) {
  DraftHistoryBatch batch;
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
  EXPECT_DEATH(DraftTokenHistory::from_host_values(ids, short_counts),
               "ids.size");
  EXPECT_DEATH(DraftTokenHistory::from_host_values(negative_ids, counts),
               "token");
  EXPECT_DEATH(DraftTokenHistory::from_host_values(duplicate_ids, counts),
               "duplicate token ids");
  EXPECT_DEATH(DraftTokenHistory::from_host_values(ids, negative_counts),
               "count");
  EXPECT_DEATH(DraftTokenHistory::from_host_values(ids, oversized_counts),
               "count");
  EXPECT_DEATH(DraftTokenHistory::from_host_values(ids, short_native_counts),
               "ids.size");
  EXPECT_DEATH(DraftTokenHistory::from_host_values(ids, negative_native_counts),
               "count");
}

TEST(MtpSeedSamplingTest, UnconstrainedPathDoesNotReadDeviceTokens) {
  DraftHistoryBatch batch;
  SamplingParameters params;
  params.selected_token_idxes = torch::tensor({3}, torch::kInt);
  params.temperatures = torch::tensor({0.7F});
  const torch::Tensor tokens = torch::empty(
      {1, 1}, torch::TensorOptions().dtype(torch::kLong).device(torch::kMeta));

  DraftInputSamplingState result =
      prepare_draft_input_sampling(params, {}, tokens, torch::kCPU, batch);

  EXPECT_TRUE(result.invalid_rows.empty());
  EXPECT_TRUE(result.json_object_states.empty());
  EXPECT_FALSE(result.sampling_params.unique_token_ids.defined());
  EXPECT_EQ(result.sampling_params.temperatures.data_ptr(),
            params.temperatures.data_ptr());
  EXPECT_EQ(result.sampling_params.selected_token_idxes.data_ptr(),
            params.selected_token_idxes.data_ptr());
}

TEST(MtpSeedSamplingTest, PrefillRootKeepsPromptCountsAtZero) {
  DraftHistoryBatch batch;
  SamplingParameters params = make_penalized_params(/*batch_size=*/1);
  params.unique_token_ids = torch::tensor({{4, 9}}, torch::kLong);
  params.unique_token_counts = torch::tensor({{0, 0}}, torch::kInt);
  params.unique_token_ids_lens = torch::tensor({2}, torch::kInt);

  DraftInputSamplingState result = prepare_draft_input_sampling(
      params, {}, torch::tensor({{4}}, torch::kLong), torch::kCPU, batch);

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
  DraftHistoryBatch batch;
  SamplingParameters params = make_penalized_params(/*batch_size=*/2);
  params.unique_token_ids = torch::tensor({{4, 1, 0}, {2, 7, 0}}, torch::kLong);
  params.unique_token_counts =
      torch::tensor({{1, 0, 0}, {0, 1, 0}}, torch::kInt);
  params.unique_token_ids_lens = torch::tensor({2, 2}, torch::kInt);
  const torch::Tensor tokens =
      torch::tensor({{5, 6, -1}, {7, 7, 8}}, torch::kLong);

  DraftInputSamplingState result =
      prepare_draft_input_sampling(params, {}, tokens, torch::kCPU, batch);

  EXPECT_TRUE(
      torch::equal(result.sampling_params.unique_token_ids,
                   torch::tensor({{4, 1, 5, 6}, {2, 7, 8, 8}}, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(result.sampling_params.unique_token_counts,
                   torch::tensor({{1, 0, 1, 1}, {0, 3, 1, 0}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(result.sampling_params.unique_token_ids_lens,
                           torch::tensor({4, 3}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.unique_token_counts,
                           torch::tensor({{1, 0, 0}, {0, 1, 0}}, torch::kInt)));
}

TEST(MtpSeedSamplingTest, PaddingPreservesTokenZeroAndRepetitionPenalties) {
  DraftHistoryBatch batch;
  SamplingParameters params = make_penalized_params(/*batch_size=*/2);
  params.unique_token_ids = torch::tensor({{0}, {1}}, torch::kLong);
  params.unique_token_counts = torch::tensor({{2}, {0}}, torch::kInt);
  params.unique_token_ids_lens = torch::tensor({1, 1}, torch::kInt);
  DraftInputSamplingState result = prepare_draft_input_sampling(
      params,
      {},
      torch::tensor({{2, -1}, {3, 4}}, torch::kInt),
      torch::kCPU,
      batch);
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
                             updated.repetition_penalties,
                             updated.unique_token_ids_lens);
  EXPECT_TRUE(torch::equal(repetition_logits,
                           torch::tensor({{2.0F, 4.0F, 2.0F, 4.0F, 4.0F},
                                          {4.0F, 2.0F, 4.0F, 2.0F, 2.0F}})));
}

TEST(MtpSeedSamplingTest, PrefillGrammarMaskFollowsNewRoot) {
  DraftHistoryBatch batch;
  JsonObjectGrammar grammar = make_mtp_draft_input_test_grammar();
  const std::vector<JsonObjectGrammarState> states = {grammar.initial_state()};
  SamplingParameters params;
  params.filter_mask = torch::zeros({1, 6});

  DraftInputSamplingState result = prepare_draft_input_sampling(
      params, states, torch::tensor({{0}}, torch::kLong), torch::kCPU, batch);

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
  DraftHistoryBatch batch;
  JsonObjectGrammar grammar = make_mtp_draft_input_test_grammar();
  JsonObjectGrammarState state = grammar.initial_state();
  ASSERT_TRUE(state.accept_token(/*open_object=*/0));

  DraftInputSamplingState result =
      prepare_draft_input_sampling({},
                                   {state},
                                   torch::tensor({{1, 2, 3, -1}}, torch::kLong),
                                   torch::kCPU,
                                   batch);

  EXPECT_EQ(result.invalid_rows, std::vector<uint8_t>({0}));
  EXPECT_EQ(result.json_object_states[0].snapshot().token_ids,
            std::vector<int32_t>({0, 1, 2, 3}));
  EXPECT_TRUE(
      result.json_object_states[0].can_accept_token(/*close_object=*/4));
  EXPECT_EQ(state.snapshot().token_ids, std::vector<int32_t>({0}));
}

TEST(MtpSeedSamplingTest, InvalidGrammarRowsDoNotCommitPartialHistory) {
  DraftHistoryBatch batch;
  JsonObjectGrammar grammar = make_mtp_draft_input_test_grammar();
  const std::vector<JsonObjectGrammarState> states = {grammar.initial_state(),
                                                      grammar.initial_state()};
  SamplingParameters params = make_penalized_params(/*batch_size=*/2);
  params.unique_token_ids = torch::tensor({{5}, {5}}, torch::kLong);
  params.unique_token_counts = torch::zeros({2, 1}, torch::kInt);
  params.unique_token_ids_lens = torch::ones({2}, torch::kInt);

  DraftInputSamplingState result = prepare_draft_input_sampling(
      params,
      states,
      torch::tensor({{0, 2, -1}, {0, 1, -1}}, torch::kLong),
      torch::kCPU,
      batch);

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
  DraftHistoryBatch batch;
  SamplingParameters params = make_penalized_params(/*batch_size=*/3);
  params.unique_token_ids = torch::tensor({{4}, {4}, {4}}, torch::kLong);
  params.unique_token_counts = torch::ones({3, 1}, torch::kInt);
  params.unique_token_ids_lens = torch::ones({3}, torch::kInt);

  DraftInputSamplingState result = prepare_draft_input_sampling(
      params,
      {},
      torch::tensor({{5, -1, 6}, {-2, -1, -1}, {-1, -1, -1}}, torch::kLong),
      torch::kCPU,
      batch);

  EXPECT_EQ(result.invalid_rows, std::vector<uint8_t>({1, 1, 1}));
  EXPECT_TRUE(torch::equal(result.sampling_params.unique_token_counts,
                           params.unique_token_counts));
}

TEST(MtpSeedSamplingTest, BorrowedPrefixesRespectStridesAndRowValidation) {
  DraftHistoryBatch batch;
  for (torch::ScalarType dtype : {torch::kInt, torch::kLong}) {
    SamplingParameters params = make_penalized_params(/*batch_size=*/4);
    DraftTokenHistories histories(4);
    for (auto& history : histories) {
      history = std::make_shared<DraftTokenHistory>();
      history->append(
          std::span<const int64_t>(std::vector<int64_t>{9}.data(), 1));
    }
    torch::Tensor storage =
        torch::tensor({{1, 2, 0, -1}, {1, -1, 0, -1}, {-1, 3, 4, -1}}, dtype);
    const torch::Tensor tokens = storage.transpose(/*dim0=*/0, /*dim1=*/1);
    ASSERT_FALSE(tokens.is_contiguous());
    auto result = prepare_draft_input_sampling(
        params, {}, tokens, torch::kCPU, batch, histories);
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
  DraftHistoryBatch batch;
  JsonObjectGrammar grammar = make_mtp_draft_input_test_grammar();
  const std::vector<JsonObjectGrammarState> states(2, grammar.initial_state());
  SamplingParameters params = make_penalized_params(/*batch_size=*/2);
  DraftTokenHistories histories(2);
  for (auto& history : histories) {
    history = std::make_shared<DraftTokenHistory>();
    history->append(
        std::span<const int64_t>(std::vector<int64_t>{5}.data(), 1));
  }
  constexpr int64_t kOversizedToken =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  for (const torch::Tensor& tokens :
       {torch::empty({2, 0}, torch::kLong),
        torch::tensor(std::vector<int64_t>{0, kOversizedToken, 0, -2},
                      torch::kLong)
            .view({2, 2})}) {
    const auto result = prepare_draft_input_sampling(
        params, states, tokens, torch::kCPU, batch, histories);
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
  DraftHistoryBatch batch;
  SamplingParameters original = make_penalized_params(/*batch_size=*/2);
  original.unique_token_ids =
      torch::tensor({{0, 2, 2}, {1, 3, 4}}, torch::kLong);
  original.unique_token_counts =
      torch::tensor({{2, 1, 1}, {0, 1, 1}}, torch::kInt);
  original.unique_token_ids_lens = torch::tensor({2, 3}, torch::kInt);
  DraftTokenHistories histories;
  import_sampling_histories(original, 2, histories);
  original.unique_token_counts.fill_(99);
  // Meta statistics cannot be copied to CPU. An established history must ignore
  // them, including when requests arrive in a different batch order.
  original.unique_token_ids = original.unique_token_ids.to(torch::kMeta);
  original.unique_token_counts = original.unique_token_counts.to(torch::kMeta);
  original.unique_token_ids_lens =
      original.unique_token_ids_lens.to(torch::kMeta);
  std::swap(histories[0], histories[1]);
  const auto result =
      prepare_draft_input_sampling(original,
                                   {},
                                   torch::tensor({{4}, {2}}, torch::kLong),
                                   torch::kCPU,
                                   batch,
                                   histories);
  EXPECT_TRUE(
      torch::equal(result.sampling_params.unique_token_ids,
                   torch::tensor({{1, 3, 4}, {0, 2, 2}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(result.sampling_params.unique_token_counts,
                           torch::tensor({{0, 1, 2}, {2, 2, 0}}, torch::kInt)));
}

TEST(MtpSeedSamplingTest, ReleasingOtherRequestKeepsLiveProjection) {
  SamplingParameters params = make_penalized_params(/*batch_size=*/1);
  DraftHistoryBatch batch;
  DraftTokenHistories histories{std::make_shared<DraftTokenHistory>()};
  histories.front()->append(
      std::span<const int64_t>(std::vector<int64_t>{7, 8}.data(), 2));
  batch.apply(params, histories, torch::kCPU);
  const torch::Tensor original_ids = params.unique_token_ids;
  const void* ids_storage = original_ids.data_ptr();

  // Releasing an unrelated request must not discard the current batch.
  DraftTokenHistories other{std::make_shared<DraftTokenHistory>()};
  other.clear();
  batch.reclaim_unused_projection();
  histories.front()->append(
      std::span<const int64_t>(std::vector<int64_t>{7}.data(), 1));
  batch.apply(params, histories, torch::kCPU);
  EXPECT_EQ(params.unique_token_ids.data_ptr(), ids_storage);
  EXPECT_TRUE(torch::equal(params.unique_token_counts,
                           torch::tensor({{2, 1}}, torch::kInt)));

  histories.clear();
  batch.reclaim_unused_projection();
  DraftTokenHistories replacement{std::make_shared<DraftTokenHistory>()};
  replacement.front()->append(
      std::span<const int64_t>(std::vector<int64_t>{9}.data(), 1));
  batch.apply(params, replacement, torch::kCPU);
  EXPECT_NE(params.unique_token_ids.data_ptr(), ids_storage);
  EXPECT_TRUE(torch::equal(params.unique_token_ids,
                           torch::tensor({{9}}, torch::kLong)));
}

TEST(MtpSeedSamplingTest, MixedDraftInputsCatLogitsAndConfidence) {
  DraftInput first{torch::tensor({1}, torch::kLong),
                   torch::ones({1, 2}),
                   torch::tensor({0.6F}),
                   torch::tensor({{0.4F, 0.6F}})};
  DraftInput second{torch::tensor({0}, torch::kLong),
                    torch::zeros({1, 2}),
                    torch::Tensor(),
                    torch::tensor({{0.7F, 0.3F}})};
  const DraftInput batch = cat_draft_inputs({first, second});
  EXPECT_FALSE(batch.selected_probs.defined());
  EXPECT_TRUE(
      torch::equal(batch.token_ids, torch::tensor({1, 0}, torch::kLong)));
  EXPECT_TRUE(torch::equal(batch.draft_logits,
                           torch::tensor({{0.4F, 0.6F}, {0.7F, 0.3F}})));
  EXPECT_FALSE(batch.point_mass_q);
}

TEST(MtpSeedSamplingTest, SelectedConfidenceDoesNotRequireTokenIndices) {
  for (torch::ScalarType dtype : {torch::kFloat, torch::kBFloat16}) {
    SampleOutput sample;
    sample.draft_logits = torch::tensor({0.25F, 9.0F, 0.75F, 9.0F}, dtype)
                              .slice(/*dim=*/0,
                                     /*start=*/0,
                                     /*end=*/4,
                                     /*step=*/2);
    const torch::Tensor original = sample.draft_logits.clone();
    const torch::Tensor unused_logits = torch::empty(
        {2, 4},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kMeta));
    const torch::Tensor confidence =
        select_input_confidence(sample, std::nullopt, unused_logits, {});
    EXPECT_EQ(confidence.scalar_type(), torch::kFloat);
    EXPECT_TRUE(torch::equal(confidence, torch::tensor({0.25F, 0.75F})));
    EXPECT_TRUE(torch::equal(sample.draft_logits, original));
    EXPECT_FALSE(sample.next_tokens.defined());
    if (dtype == torch::kFloat) {
      EXPECT_TRUE(confidence.is_alias_of(sample.draft_logits));
      EXPECT_EQ(confidence.stride(0), 2);
    }
  }
  SampleOutput empty;
  empty.draft_logits = torch::empty({0}, torch::kFloat);
  EXPECT_EQ(select_input_confidence(empty, std::nullopt, {}, {}).numel(), 0);
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
      select_input_confidence(sample, std::nullopt, logits, {});

  EXPECT_TRUE(torch::allclose(confidence, expected));
  EXPECT_TRUE(torch::equal(logits, original_logits));
  EXPECT_FALSE(sample.draft_logits.defined());
  EXPECT_EQ(confidence.scalar_type(), torch::kFloat);
}

TEST(MtpSeedSamplingTest, ConfidenceSelectsSampleRowsBeforeReduction) {
  DraftHistoryBatch batch;
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
      select_input_confidence(sample, std::nullopt, logits, sample_idxes);

  EXPECT_TRUE(torch::allclose(confidence, expected));
  EXPECT_EQ(confidence.scalar_type(), torch::kFloat);
  EXPECT_TRUE(torch::equal(logits, original_logits));
  EXPECT_FALSE(sample.draft_logits.defined());
}

TEST(MtpSeedSamplingTest, ConfidenceKeepsSelectedProbabilityPayloadForRebatch) {
  DraftHistoryBatch batch;
  SampleOutput prefill;
  prefill.next_tokens = torch::tensor({1}, torch::kLong);
  prefill.draft_logits = torch::tensor({0.75F}, torch::kBFloat16);
  const torch::Tensor original_logits = prefill.draft_logits.clone();
  SampleOutput extend;
  extend.next_tokens = torch::tensor({0}, torch::kLong);
  const torch::Tensor prefill_confidence =
      select_input_confidence(prefill, std::nullopt, {}, {});
  const torch::Tensor extend_confidence =
      select_input_confidence(extend, std::nullopt, torch::zeros({1, 2}), {});

  const DraftInput combined = cat_draft_inputs(
      {{prefill.next_tokens, torch::zeros({1, 3}), prefill_confidence},
       {extend.next_tokens, torch::zeros({1, 3}), extend_confidence}});

  EXPECT_TRUE(
      torch::allclose(combined.selected_probs, torch::tensor({0.75F, 0.5F})));
  EXPECT_EQ(combined.selected_probs.scalar_type(), torch::kFloat);
  EXPECT_FALSE(combined.draft_logits.defined());
  EXPECT_TRUE(torch::equal(prefill.draft_logits, original_logits));
}

}  // namespace
}  // namespace xllm
