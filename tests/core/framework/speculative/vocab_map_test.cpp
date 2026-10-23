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

#include "core/framework/speculative/vocab_map.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "core/framework/sampling/logits_utils.h"

namespace xllm {
namespace {

TEST(VocabMapTest, RejectsDuplicateOrOutOfRangeMapping) {
  const torch::Tensor duplicate = torch::tensor({1, 1}, torch::kLong);
  const torch::Tensor outside = torch::tensor({1, 4}, torch::kLong);
  const torch::Tensor negative = torch::tensor({-1, 2}, torch::kLong);

  EXPECT_DEATH(VocabMap(duplicate, 4), "duplicate");
  EXPECT_DEATH(VocabMap(outside, 4), "outside the target");
  EXPECT_DEATH(VocabMap(negative, 4), "outside the target");
}

TEST(VocabMapTest, OwnsMappingSnapshotAcrossLayoutsAndDtypes) {
  DraftHistoryBatch batch;
  constexpr int32_t kHighBit = std::numeric_limits<int32_t>::min();
  for (torch::ScalarType dtype : {torch::kInt, torch::kLong}) {
    for (bool strided : {false, true}) {
      SCOPED_TRACE(dtype);
      SCOPED_TRACE(strided);
      torch::Tensor producer =
          strided ? torch::tensor({31, 99, 32, 99, 63, 99, 64, 99}, dtype)
                        .slice(/*dim=*/0, /*start=*/0, /*end=*/8, /*step=*/2)
                  : torch::tensor({31, 32, 63, 64}, dtype);
      const VocabMap sampling_map(producer, /*target_vocab_size=*/70);
      producer.fill_(-1);
      EXPECT_EQ(sampling_map.draft_to_target().scalar_type(), torch::kLong);
      EXPECT_TRUE(sampling_map.draft_to_target().is_contiguous());
      EXPECT_FALSE(sampling_map.draft_to_target().is_alias_of(producer));
      EXPECT_TRUE(torch::equal(sampling_map.draft_to_target(),
                               torch::tensor({31, 32, 63, 64}, torch::kLong)));
      EXPECT_TRUE(torch::equal(sampling_map.target_to_draft().index_select(
                                   /*dim=*/0, sampling_map.draft_to_target()),
                               torch::arange(4, torch::kLong)));
      const torch::Tensor original =
          torch::tensor({{kHighBit, 0, 1}, {0, kHighBit + 1, 0}}, torch::kInt);
      for (int32_t step = 0; step < 3; ++step) {
        SamplingParameters params;
        params.filter_bitmask = original;
        EXPECT_TRUE(sampling_map.remap(params, batch).empty());
        EXPECT_TRUE(torch::equal(params.filter_bitmask,
                                 torch::tensor({{9}, {6}}, torch::kInt)));
        EXPECT_FALSE(params.filter_bitmask.is_alias_of(original));
      }
    }
  }
}

TEST(VocabMapTest, MissingMappingDoesNotChangeParameters) {
  DraftHistoryBatch batch;
  SamplingParameters params;
  params.filter_mask = torch::arange(6, torch::kFloat32).view({1, 6});
  const torch::Tensor original_mask = params.filter_mask;

  VocabMap().remap(params, batch);

  EXPECT_EQ(params.filter_mask.data_ptr(), original_mask.data_ptr());
}

TEST(VocabMapTest, UnconstrainedPathDoesNotReadHistoryData) {
  DraftHistoryBatch batch;
  SamplingParameters params;
  params.temperatures = torch::tensor({0.7F});
  const torch::Tensor original_temperatures = params.temperatures;
  const VocabMap sampling_map(torch::tensor({2, 5, 8}, torch::kLong),
                              /*target_vocab_size=*/10);
  params.unique_token_ids = torch::empty(
      {1, 3}, torch::TensorOptions().dtype(torch::kLong).device(torch::kMeta));

  sampling_map.remap(params, batch);

  EXPECT_EQ(params.temperatures.data_ptr(), original_temperatures.data_ptr());
}

TEST(VocabMapTest, SelectsDenseMaskColumnsWithoutMutatingOriginal) {
  DraftHistoryBatch batch;
  const torch::Tensor mapping = torch::tensor({4, 1, 5}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/6);
  SamplingParameters params;
  params.filter_mask = torch::arange(12, torch::kFloat32).view({2, 6});
  params.return_logits = true;
  const torch::Tensor original = params.filter_mask;

  sampling_map.remap(params, batch);

  EXPECT_TRUE(
      torch::equal(params.filter_mask,
                   torch::tensor({{4.0F, 1.0F, 5.0F}, {10.0F, 7.0F, 11.0F}})));
  EXPECT_TRUE(
      torch::equal(original, torch::arange(12, torch::kFloat32).view({2, 6})));
  EXPECT_TRUE(params.return_logits);
}

TEST(VocabMapTest, RepackagesTargetBitsAcrossWordAndSignBoundaries) {
  DraftHistoryBatch batch;
  constexpr int32_t kHighBit = std::numeric_limits<int32_t>::min();
  const torch::Tensor mapping = torch::tensor({31, 32, 63, 64}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/70);
  SamplingParameters params;
  params.filter_bitmask =
      torch::tensor({{kHighBit, 0, 1}, {0, kHighBit + 1, 0}}, torch::kInt);
  const torch::Tensor original = params.filter_bitmask;

  sampling_map.remap(params, batch);

  EXPECT_TRUE(torch::equal(params.filter_bitmask,
                           torch::tensor({{9}, {6}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(
      original,
      torch::tensor({{kHighBit, 0, 1}, {0, kHighBit + 1, 0}}, torch::kInt)));
}

TEST(VocabMapTest, RepackagesStridedMasksWithoutMutatingSharedStorage) {
  DraftHistoryBatch batch;
  constexpr int32_t kHighBit = std::numeric_limits<int32_t>::min();
  const torch::Tensor mapping = torch::tensor({31, 32, 63, 64}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/70);
  const torch::Tensor storage = torch::tensor(
      {{kHighBit, 7, 0, 8, 1, 9}, {0, 7, kHighBit + 1, 8, 0, 9}}, torch::kInt);
  const torch::Tensor original = storage.clone();
  SamplingParameters params;
  params.filter_bitmask =
      storage.slice(/*dim=*/1, /*start=*/0, /*end=*/6, /*step=*/2);
  ASSERT_FALSE(params.filter_bitmask.is_contiguous());

  EXPECT_TRUE(sampling_map.remap(params, batch).empty());

  EXPECT_TRUE(torch::equal(params.filter_bitmask,
                           torch::tensor({{9}, {6}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(storage, original));
  EXPECT_FALSE(params.filter_bitmask.is_alias_of(storage));
}

TEST(VocabMapTest, RepackagesMoreThanOneDraftWordAndClearsPaddingBits) {
  DraftHistoryBatch batch;
  constexpr int32_t kHighBit = std::numeric_limits<int32_t>::min();
  const torch::Tensor mapping = torch::arange(5, 39, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/80);
  SamplingParameters params;
  params.filter_bitmask = torch::tensor({{0, 48, 0}}, torch::kInt);

  sampling_map.remap(params, batch);

  EXPECT_TRUE(torch::equal(params.filter_bitmask,
                           torch::tensor({{kHighBit, 1}}, torch::kInt)));
}

TEST(VocabMapTest, RepackagesFullSignedWordsWithPermutedVocabulary) {
  DraftHistoryBatch batch;
  constexpr int32_t kHighBit = std::numeric_limits<int32_t>::min();
  constexpr int64_t kTargetVocabSize = 97;
  const torch::Tensor original =
      torch::tensor({{-1, -1, -1, -1},
                     {kHighBit, kHighBit, kHighBit, 0},
                     {kHighBit + 1, 0x55555555, -1, 1},
                     {0, 0, 0, 0}},
                    torch::kInt);
  const int32_t* source = original.const_data_ptr<int32_t>();
  for (int64_t width : {31, 32, 33, 63, 64, 65}) {
    SCOPED_TRACE(width);
    const torch::Tensor mapping =
        (torch::arange(width, torch::kLong) * 37).remainder(kTargetVocabSize);
    const VocabMap sampling_map(mapping, kTargetVocabSize);
    const int64_t words = (width + 31) / 32;
    torch::Tensor expected = torch::zeros({4, words}, torch::kLong);
    int64_t* packed = expected.data_ptr<int64_t>();
    std::vector<uint8_t> expected_invalid(4, 1);
    const int64_t* target_ids = mapping.const_data_ptr<int64_t>();
    for (int64_t row = 0; row < 4; ++row) {
      for (int64_t token = 0; token < width; ++token) {
        const int64_t target = target_ids[token];
        const uint32_t word =
            static_cast<uint32_t>(source[row * 4 + target / 32]);
        const int64_t allowed = (word >> (target % 32)) & 1;
        packed[row * words + token / 32] |= allowed << (token % 32);
        expected_invalid[row] &= static_cast<uint8_t>(!allowed);
      }
      packed[row * words] |= expected_invalid[row];
    }
    SamplingParameters params;
    params.filter_bitmask = original;

    const std::vector<uint8_t> invalid = sampling_map.remap(params, batch);

    EXPECT_EQ(invalid, expected_invalid);
    EXPECT_EQ(params.filter_bitmask.scalar_type(), torch::kInt);
    EXPECT_TRUE(torch::equal(params.filter_bitmask, expected.to(torch::kInt)));
    EXPECT_FALSE(params.filter_bitmask.is_alias_of(original));
  }
}

TEST(VocabMapTest, EmptyHotMaskReturnsFailureWithPrivatePlaceholder) {
  DraftHistoryBatch batch;
  const torch::Tensor mapping = torch::tensor({1, 3}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/5);
  SamplingParameters params;
  params.filter_bitmask = torch::tensor({{1}, {8}}, torch::kInt);
  const torch::Tensor original = params.filter_bitmask;

  const std::vector<uint8_t> invalid = sampling_map.remap(params, batch);

  EXPECT_EQ(invalid, std::vector<uint8_t>({1, 0}));
  EXPECT_TRUE(torch::equal(params.filter_bitmask,
                           torch::tensor({{1}, {2}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(original, torch::tensor({{1}, {8}}, torch::kInt)));
  // Row 0 is a placeholder; the runtime must reject it or fail the request.
}

TEST(VocabMapTest, EmptyDenseHotMaskReturnsFailureInsteadOfSuccess) {
  DraftHistoryBatch batch;
  const float forbidden = -std::numeric_limits<float>::infinity();
  const torch::Tensor mapping = torch::tensor({1, 3}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/4);
  SamplingParameters params;
  params.filter_mask = torch::tensor({{0.0F, forbidden, 0.0F, forbidden}});
  const torch::Tensor original = params.filter_mask;

  const std::vector<uint8_t> invalid = sampling_map.remap(params, batch);

  EXPECT_EQ(invalid, std::vector<uint8_t>({1}));
  EXPECT_TRUE(
      torch::equal(params.filter_mask, torch::tensor({{0.0F, forbidden}})));
  EXPECT_TRUE(torch::equal(
      original, torch::tensor({{0.0F, forbidden, 0.0F, forbidden}})));
}

TEST(VocabMapTest, DensePlaceholdersPreserveValidRowsAcrossDtypes) {
  DraftHistoryBatch batch;
  const float forbidden = -std::numeric_limits<float>::infinity();
  const VocabMap sampling_map(torch::tensor({2, 5, 8}, torch::kLong),
                              /*target_vocab_size=*/10);
  for (torch::ScalarType dtype :
       {torch::kFloat, torch::kHalf, torch::kBFloat16, torch::kDouble}) {
    SCOPED_TRACE(dtype);
    torch::Tensor storage = torch::full({4, 20}, forbidden, dtype);
    torch::Tensor input =
        storage.slice(/*dim=*/1, /*start=*/0, /*end=*/20, /*step=*/2);
    input.select(/*dim=*/0, /*index=*/2).fill_(-3);
    input.select(/*dim=*/0, /*index=*/1)
        .select(/*dim=*/0, /*index=*/2)
        .fill_(-3);
    input.select(/*dim=*/0, /*index=*/1)
        .select(/*dim=*/0, /*index=*/5)
        .fill_(0);
    input.select(/*dim=*/0, /*index=*/3)
        .select(/*dim=*/0, /*index=*/2)
        .fill_(2);
    input.select(/*dim=*/0, /*index=*/3)
        .select(/*dim=*/0, /*index=*/8)
        .fill_(1);
    const torch::Tensor original = storage.clone();
    SamplingParameters params;
    params.filter_mask = input;

    EXPECT_EQ(sampling_map.remap(params, batch),
              std::vector<uint8_t>({1, 0, 1, 0}));
    EXPECT_EQ(params.filter_mask.scalar_type(), dtype);
    EXPECT_TRUE(torch::equal(params.filter_mask,
                             torch::tensor({{0.0F, forbidden, forbidden},
                                            {-3.0F, 0.0F, forbidden},
                                            {0.0F, forbidden, forbidden},
                                            {2.0F, forbidden, 1.0F}},
                                           dtype)));
    EXPECT_FALSE(params.filter_mask.is_alias_of(storage));
    params.filter_mask.fill_(7);
    EXPECT_TRUE(torch::equal(storage, original));
  }
}

TEST(VocabMapTest, PackedMaskKeepsPrecedenceOverDenseMask) {
  DraftHistoryBatch batch;
  const torch::Tensor mapping = torch::tensor({1, 3}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/4);
  SamplingParameters params;
  params.filter_mask =
      torch::full({1, 4}, -std::numeric_limits<float>::infinity());
  params.filter_bitmask = torch::tensor({{2}}, torch::kInt);

  const std::vector<uint8_t> invalid = sampling_map.remap(params, batch);

  EXPECT_TRUE(invalid.empty());
  EXPECT_FALSE(params.filter_mask.defined());
  EXPECT_TRUE(
      torch::equal(params.filter_bitmask, torch::tensor({{1}}, torch::kInt)));
}

TEST(VocabMapTest, PackedMaskDoesNotReadOrRetainShadowedDenseData) {
  DraftHistoryBatch batch;
  const VocabMap sampling_map(torch::tensor({1, 3}, torch::kLong),
                              /*target_vocab_size=*/4);
  SamplingParameters source;
  // Meta data cannot be copied to a real device. A shadowed dense mask must
  // never be materialized, even when its shape and dtype are inapplicable.
  source.filter_mask = torch::empty(
      {7}, torch::TensorOptions().dtype(torch::kInt).device(torch::kMeta));
  source.filter_bitmask = torch::tensor({{2}, {1}}, torch::kInt);
  SamplingParameters params = source;

  EXPECT_EQ(sampling_map.remap(params, batch), std::vector<uint8_t>({0, 1}));
  EXPECT_FALSE(params.filter_mask.defined());
  EXPECT_TRUE(torch::equal(params.filter_bitmask,
                           torch::tensor({{1}, {1}}, torch::kInt)));
  EXPECT_TRUE(source.filter_mask.defined());
  EXPECT_EQ(source.filter_mask.device().type(), torch::kMeta);
  EXPECT_TRUE(torch::equal(source.filter_bitmask,
                           torch::tensor({{2}, {1}}, torch::kInt)));
}

TEST(VocabMapTest, RemapsPenaltyHistoryAndNeutralizesAbsentRows) {
  DraftHistoryBatch batch;
  const torch::Tensor mapping = torch::tensor({2, 5, 8}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/10);
  SamplingParameters params;
  params.frequency_penalties = torch::ones({3});
  params.presence_penalties = torch::full({3}, 0.5F);
  params.repetition_penalties = torch::full({3}, 2.0F);
  params.unique_token_ids =
      torch::tensor({{5, 9, 2, 0}, {9, 7, 0, 0}, {8, 2, 5, 9}}, torch::kLong);
  params.unique_token_counts =
      torch::tensor({{3, 4, 0, 0}, {2, 1, 0, 0}, {1, 2, 4, 0}}, torch::kInt);
  params.unique_token_ids_lens = torch::tensor({3, 2, 3}, torch::kInt);
  const SamplingParameters original = params;

  sampling_map.remap(params, batch);

  EXPECT_TRUE(torch::equal(
      params.unique_token_ids,
      torch::tensor({{1, 0, 0}, {0, 0, 0}, {2, 0, 1}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(
      params.unique_token_counts,
      torch::tensor({{3, 0, 0}, {0, 0, 0}, {1, 2, 4}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.unique_token_ids_lens,
                           torch::tensor({2, 0, 3}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.repetition_penalties,
                           torch::tensor({2.0F, 1.0F, 2.0F})));

  torch::Tensor logits = torch::full({3, 3}, 4.0F);
  apply_frequency_presence_penalties(logits,
                                     params.unique_token_ids,
                                     params.unique_token_counts,
                                     params.frequency_penalties,
                                     params.presence_penalties);
  apply_repetition_penalties(logits,
                             params.unique_token_ids,
                             params.repetition_penalties,
                             params.unique_token_ids_lens);
  EXPECT_TRUE(torch::equal(
      logits,
      torch::tensor(
          {{2.0F, 0.25F, 4.0F}, {4.0F, 4.0F, 4.0F}, {0.75F, -1.0F, 1.25F}})));
  EXPECT_TRUE(torch::equal(
      original.unique_token_ids,
      torch::tensor({{5, 9, 2, 0}, {9, 7, 0, 0}, {8, 2, 5, 9}}, torch::kLong)));
  // Remapping must not change penalty tensors held by the target sampler.
  EXPECT_TRUE(
      torch::equal(original.repetition_penalties, torch::full({3}, 2.0F)));
}

TEST(VocabMapTest, AllMissingHotTokensDisableOnlyPenaltyFields) {
  DraftHistoryBatch batch;
  const torch::Tensor mapping = torch::tensor({2, 5, 8}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/10);
  SamplingParameters params;
  params.frequency_penalties = torch::ones({1});
  params.presence_penalties = torch::ones({1});
  params.repetition_penalties = torch::full({1}, 2.0F);
  params.unique_token_ids = torch::tensor({{1, 9}}, torch::kLong);
  params.unique_token_counts = torch::tensor({{2, 3}}, torch::kInt);
  params.unique_token_ids_lens = torch::tensor({2}, torch::kInt);
  params.temperatures = torch::tensor({0.7F});
  const torch::Tensor original_temperatures = params.temperatures;

  sampling_map.remap(params, batch);

  EXPECT_FALSE(params.frequency_penalties.defined());
  EXPECT_FALSE(params.presence_penalties.defined());
  EXPECT_FALSE(params.repetition_penalties.defined());
  EXPECT_FALSE(params.unique_token_ids.defined());
  EXPECT_EQ(params.temperatures.data_ptr(), original_temperatures.data_ptr());
}

TEST(VocabMapTest, RemappingPreservesSharedTargetHistoriesAndRowOrder) {
  const torch::Tensor mapping = torch::tensor({2, 5, 8}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/10);
  auto hot = std::make_shared<DraftTokenHistory>();
  auto cold = std::make_shared<DraftTokenHistory>();
  hot->append(
      std::span<const int64_t>(std::vector<int64_t>{5, 2, 5}.data(), 3));
  cold->append(std::span<const int64_t>(std::vector<int64_t>{9}.data(), 1));
  const DraftTokenHistories histories{hot, cold, hot};
  SamplingParameters source;
  source.frequency_penalties = torch::tensor({0.5F, 1.0F, 1.5F});
  source.presence_penalties = torch::full({3}, 0.25F);
  source.repetition_penalties = torch::full({3}, 2.0F);
  source.unique_token_ids = torch::empty(
      {3, 1}, torch::TensorOptions().dtype(torch::kLong).device(torch::kMeta));
  SamplingParameters params = source;
  DraftHistoryBatch batch;
  sampling_map.remap(params, batch, histories);
  EXPECT_EQ(histories[0], hot);
  EXPECT_EQ(histories[1], cold);
  EXPECT_EQ(histories[2], hot);
  EXPECT_EQ(hot->ids(), std::vector<int64_t>({5, 2}));
  EXPECT_EQ(hot->counts(), std::vector<int32_t>({2, 1}));
  EXPECT_TRUE(torch::equal(params.unique_token_ids_lens,
                           torch::tensor({2, 0, 2}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(params.unique_token_counts,
                   torch::tensor({{2, 1}, {0, 0}, {2, 1}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.frequency_penalties,
                           torch::tensor({0.5F, 0.0F, 1.5F})));
  EXPECT_TRUE(torch::equal(params.repetition_penalties,
                           torch::tensor({2.0F, 1.0F, 2.0F})));
  EXPECT_TRUE(
      torch::equal(source.repetition_penalties, torch::full({3}, 2.0F)));

  cold->append(std::span<const int64_t>(std::vector<int64_t>{8}.data(), 1));
  params = source;
  sampling_map.remap(params, batch, histories);
  EXPECT_TRUE(torch::equal(params.unique_token_ids_lens,
                           torch::tensor({2, 1, 2}, torch::kInt)));
  EXPECT_EQ(params.frequency_penalties.data_ptr(),
            source.frequency_penalties.data_ptr());
  EXPECT_EQ(cold->ids(), std::vector<int64_t>({9, 8}));
}

TEST(VocabMapTest, ClearedHistoriesLeaveOwnedSamplingTensors) {
  const torch::Tensor mapping = torch::tensor({2, 5}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/10);
  DraftTokenHistories histories{std::make_shared<DraftTokenHistory>()};
  histories[0]->append(
      std::span<const int64_t>(std::vector<int64_t>{5, 2, 5}.data(), 3));
  const std::weak_ptr<DraftTokenHistory> weak = histories[0];
  SamplingParameters params;
  params.frequency_penalties = torch::ones({1});
  params.presence_penalties = torch::zeros({1});
  params.unique_token_ids = torch::empty(
      {1, 1}, torch::TensorOptions().dtype(torch::kLong).device(torch::kMeta));
  DraftHistoryBatch batch;
  sampling_map.remap(params, batch, histories);
  // remap borrows histories; release the caller's ownership before verifying
  // that the sampling tensors retain their own storage.
  histories.clear();
  EXPECT_TRUE(weak.expired());
  EXPECT_TRUE(torch::equal(params.unique_token_ids,
                           torch::tensor({{1, 0}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(params.unique_token_counts,
                           torch::tensor({{2, 1}}, torch::kInt)));
  torch::Tensor logits = torch::zeros({1, 2});
  apply_frequency_presence_penalties(logits,
                                     params.unique_token_ids,
                                     params.unique_token_counts,
                                     params.frequency_penalties,
                                     params.presence_penalties);
  EXPECT_TRUE(torch::equal(logits, torch::tensor({{-1.0F, -2.0F}})));
}

TEST(VocabMapTest, ReusesMappedHistoryAcrossRecursiveDraftsAndCommits) {
  const torch::Tensor mapping = torch::tensor({2, 5, 8}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/10);
  SamplingParameters original;
  original.frequency_penalties = torch::ones({1});
  original.presence_penalties = torch::zeros({1});
  original.unique_token_ids = torch::tensor({{2, 7, 5}}, torch::kLong);
  original.unique_token_counts = torch::tensor({{0, 2, 1}}, torch::kInt);
  original.unique_token_ids_lens = torch::tensor({3}, torch::kInt);
  DraftTokenHistories histories;
  import_sampling_histories(original, 1, histories);
  original.unique_token_ids = original.unique_token_ids.to(torch::kMeta);
  original.unique_token_counts = original.unique_token_counts.to(torch::kMeta);
  original.unique_token_ids_lens =
      original.unique_token_ids_lens.to(torch::kMeta);
  DraftHistoryBatch batch;
  SamplingParameters params = original;
  sampling_map.remap(params, batch, histories);
  const void* ids_storage = params.unique_token_ids.data_ptr();
  const void* counts_storage = params.unique_token_counts.data_ptr();
  EXPECT_TRUE(torch::equal(params.unique_token_ids,
                           torch::tensor({{0, 1}}, torch::kLong)));

  for (int32_t draft = 0; draft < 3; ++draft) {
    params = original;
    sampling_map.remap(params, batch, histories);
    EXPECT_EQ(params.unique_token_ids.data_ptr(), ids_storage);
    EXPECT_EQ(params.unique_token_counts.data_ptr(), counts_storage);
  }
  histories[0]->append(
      std::span<const int64_t>(std::vector<int64_t>{5, 5, 7}.data(), 3));
  params = original;
  sampling_map.remap(params, batch, histories);
  EXPECT_EQ(params.unique_token_counts.data_ptr(), counts_storage);
  EXPECT_TRUE(torch::equal(params.unique_token_counts,
                           torch::tensor({{0, 3}}, torch::kInt)));
  histories[0]->append(
      std::span<const int64_t>(std::vector<int64_t>{8}.data(), 1));
  params = original;
  sampling_map.remap(params, batch, histories);
  torch::Tensor logits = torch::zeros({1, 3});
  apply_frequency_presence_penalties(logits,
                                     params.unique_token_ids,
                                     params.unique_token_counts,
                                     params.frequency_penalties,
                                     params.presence_penalties);
  EXPECT_TRUE(torch::equal(logits, torch::tensor({{0.0F, -3.0F, -1.0F}})));
}

TEST(VocabMapTest, ColdUpdatesPreservePendingHotChangesAndRequestLifetime) {
  const torch::Tensor mapping = torch::tensor({2, 5}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/10);
  auto history = std::make_shared<DraftTokenHistory>();
  history->append(std::span<const int64_t>(std::vector<int64_t>{2}.data(), 1));
  auto mapped = history->mapped(sampling_map.target_to_draft());
  DraftHistoryBatch batch;
  SamplingParameters params;
  batch.apply(params, {mapped}, torch::kCPU);
  history->append(
      std::span<const int64_t>(std::vector<int64_t>{2, 2}.data(), 2));
  history->mapped(sampling_map.target_to_draft());
  // A cold-only update does not advance the mapped revision or erase its delta.
  history->append(std::span<const int64_t>(std::vector<int64_t>{7}.data(), 1));
  history->mapped(sampling_map.target_to_draft());
  batch.apply(params, {mapped}, torch::kCPU);
  EXPECT_TRUE(torch::equal(params.unique_token_counts,
                           torch::tensor({{3}}, torch::kInt)));
  std::weak_ptr<DraftTokenHistory> weak = mapped;
  mapped.reset();
  history.reset();
  EXPECT_TRUE(weak.expired());
}

TEST(VocabMapTest, RevisionGapsPreserveAndReplaceMappedDeltas) {
  const torch::Tensor mapping = torch::tensor({2, 5, 8}, torch::kLong);
  const VocabMap sampling_map(mapping, /*target_vocab_size=*/10);
  auto history = std::make_shared<DraftTokenHistory>();
  history->append(
      std::span<const int64_t>(std::vector<int64_t>{9, 2, 5, 8}.data(), 4));
  auto mapped = history->mapped(sampling_map.target_to_draft());
  EXPECT_EQ(mapped->changed_columns(), std::vector<int64_t>({0, 1, 2}));
  EXPECT_EQ(mapped->revision(), 1);
  DraftHistoryBatch batch;
  SamplingParameters params;
  batch.apply(params, {mapped}, torch::kCPU);

  history->append(std::span<const int64_t>(std::vector<int64_t>{5}.data(), 1));
  EXPECT_EQ(history->mapped(sampling_map.target_to_draft()), mapped);
  EXPECT_EQ(mapped->changed_columns(), std::vector<int64_t>({1}));
  EXPECT_EQ(mapped->revision(), 2);
  history->append(std::span<const int64_t>(std::vector<int64_t>{9}.data(), 1));
  history->append(std::span<const int64_t>(std::vector<int64_t>{9}.data(), 1));
  EXPECT_EQ(history->mapped(sampling_map.target_to_draft()), mapped);
  EXPECT_EQ(mapped->changed_columns(), std::vector<int64_t>({1}));
  EXPECT_EQ(mapped->revision(), 2);
  batch.apply(params, {mapped}, torch::kCPU);
  EXPECT_TRUE(torch::equal(params.unique_token_counts,
                           torch::tensor({{1, 2, 1}}, torch::kInt)));

  history->append(
      std::span<const int64_t>(std::vector<int64_t>{9, 2}.data(), 2));
  history->append(std::span<const int64_t>(std::vector<int64_t>{8}.data(), 1));
  EXPECT_EQ(history->mapped(sampling_map.target_to_draft()), mapped);
  EXPECT_EQ(mapped->changed_columns(), std::vector<int64_t>({0, 2}));
  EXPECT_EQ(mapped->counts(), std::vector<int32_t>({2, 2, 2}));
  EXPECT_EQ(mapped->revision(), 3);
  batch.apply(params, {mapped}, torch::kCPU);
  EXPECT_TRUE(torch::equal(params.unique_token_counts,
                           torch::tensor({{2, 2, 2}}, torch::kInt)));

  const VocabMap replacement(torch::tensor({8, 2, 5}, torch::kLong),
                             /*target_vocab_size=*/10);
  auto remapped = history->mapped(replacement.target_to_draft());
  EXPECT_NE(remapped, mapped);
  EXPECT_EQ(remapped->ids(), std::vector<int64_t>({1, 2, 0}));
  EXPECT_EQ(remapped->changed_columns(), std::vector<int64_t>({0, 1, 2}));
  EXPECT_EQ(remapped->counts(), std::vector<int32_t>({2, 2, 2}));
  EXPECT_EQ(remapped->revision(), 1);
}

}  // namespace
}  // namespace xllm
