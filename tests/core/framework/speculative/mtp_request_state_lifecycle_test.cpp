/* Copyright 2025-2026 The xLLM Authors.

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

#include <ATen/record_function.h>
#include <gtest/gtest.h>

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "core/framework/speculative/draft_input_sampling.h"
#include "core/framework/speculative/draft_input_state_store.h"
#include "core/framework/speculative/mtp_draft_input_test_utils.h"
#include "core/framework/speculative/target_decode_state_store.h"

namespace xllm {

namespace {

bool tensor_equal(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  return lhs.defined() && rhs.defined() && torch::equal(lhs, rhs);
}

class SeedConcatenationObserver final {
 public:
  SeedConcatenationObserver() {
    CHECK(active_ == nullptr);
    active_ = this;
    input_counts_.reserve(kDraftInputTensors.size());
    callback_ = at::addThreadLocalCallback(
        at::RecordFunctionCallback(record).needsInputs(true));
  }
  ~SeedConcatenationObserver() {
    at::removeCallback(callback_);
    active_ = nullptr;
  }
  SeedConcatenationObserver(const SeedConcatenationObserver&) = delete;
  SeedConcatenationObserver& operator=(const SeedConcatenationObserver&) =
      delete;

  const std::vector<size_t>& input_counts() const { return input_counts_; }

 private:
  static std::unique_ptr<at::ObserverContext> record(
      const at::RecordFunction& function) {
    if (active_ == nullptr ||
        std::string_view(function.name()) != "aten::cat") {
      return nullptr;
    }
    const auto inputs = function.inputs();
    if (!inputs.empty() && inputs[0].isTensorList()) {
      active_->input_counts_.emplace_back(inputs[0].toTensorList().size());
    }
    return nullptr;
  }

  static thread_local SeedConcatenationObserver* active_;
  at::CallbackHandle callback_ = 0;
  std::vector<size_t> input_counts_;
};

thread_local SeedConcatenationObserver* SeedConcatenationObserver::active_ =
    nullptr;

// Test-only bridge that exercises the two explicit production stores together
// where a lifecycle test intentionally spans both domains.
class MtpRequestStateLifecycle final {
 public:
  using DecodeState = TargetDecodeStateStore::DecodeState;

  explicit MtpRequestStateLifecycle(int32_t total_nums)
      : target_(total_nums), sampling_(total_nums) {}

  void write_prefill_target_context(const std::vector<int32_t>& ids,
                                    const std::vector<std::string>& requests,
                                    const torch::Tensor& tokens) {
    target_.write_prefill_target_context(ids, requests, tokens);
  }
  void write_target_bootstrap_context(int32_t id,
                                      const std::string& request,
                                      int32_t token) {
    target_.write_prefill_target_context(
        {id}, {request}, torch::tensor({token}, torch::kInt));
  }
  void write_target_context(const std::vector<int32_t>& ids,
                            const std::vector<std::string>& requests,
                            const torch::Tensor& tokens) {
    target_.write_target_context(ids, requests, tokens);
  }
  std::vector<DecodeState> read_decode_states(
      const std::vector<int32_t>& ids,
      const std::vector<std::string>& requests) const {
    return target_.read_decode_states(ids, requests);
  }
  std::vector<int32_t> read_accepted_prefix_lengths(
      const std::vector<int32_t>& ids,
      const std::vector<std::string>& requests) const {
    return target_.resolve_accepted_prefix_lengths(
        ids,
        requests,
        /*num_sequences=*/ids.size());
  }
  void write_draft_input(const std::vector<int32_t>& ids,
                         const std::vector<std::string>& requests,
                         const DraftInput& draft_input) {
    sampling_.write_draft_input(ids, requests, draft_input);
  }
  std::optional<DraftInput> collect_draft_inputs(
      const std::vector<int32_t>& ids,
      const std::vector<std::string>& requests) const {
    return sampling_.collect_draft_inputs(ids, requests);
  }
  void write_draft_input(const std::vector<int32_t>& ids,
                         const std::vector<std::string>& requests,
                         const DraftInput& draft_input,
                         DraftInputSamplingState sampling,
                         const std::vector<size_t>& failed_rows) {
    sampling_.write_draft_input(ids,
                                requests,
                                draft_input,
                                std::move(sampling.histories),
                                std::move(sampling.json_object_states),
                                failed_rows);
  }
  DraftInputSamplingState prepare_seed_sampling(
      const std::vector<int32_t>& ids,
      const std::vector<std::string>& requests,
      SamplingParameters params,
      std::vector<JsonObjectGrammarState> json_states,
      const torch::Tensor& tokens,
      const torch::Device& device) {
    return sampling_.prepare(ids,
                             requests,
                             std::move(params),
                             std::move(json_states),
                             tokens,
                             device);
  }
  std::optional<DraftInput> take_draft_input(
      const std::vector<int32_t>& ids,
      const std::vector<std::string>& requests,
      std::vector<uint8_t>* failed_mask = nullptr) {
    return sampling_.take_draft_input(ids, requests, failed_mask);
  }
  std::vector<uint8_t> failed_draft_input_mask(
      const std::vector<int32_t>& ids,
      const std::vector<std::string>& requests) const {
    return sampling_.failed_draft_input_mask(ids, requests);
  }
  DraftTokenHistories restore_draft_sampling(
      const std::vector<int32_t>& ids,
      const std::vector<std::string>& requests,
      SamplingParameters& params,
      std::vector<JsonObjectGrammarState>& json_states,
      const torch::Device& device) {
    return sampling_.restore(ids, requests, params, json_states, device);
  }
  void clear(const std::vector<int32_t>& ids) {
    target_.clear(ids);
    // clear() also reclaims the device projection the released rows kept
    // alive, in one call.
    sampling_.clear(ids);
  }

 private:
  TargetDecodeStateStore target_;
  DraftInputStateStore sampling_;
};

DraftInput make_numbered_draft_input(int64_t rows) {
  torch::Tensor draft_logits =
      torch::arange(1, rows * 8 + 1, torch::kFloat).view({rows, 8});
  draft_logits = draft_logits / draft_logits.sum(/*dim=*/1, /*keepdim=*/true);
  return {torch::arange(rows, torch::kLong),
          torch::arange(rows * 2, torch::kFloat).view({rows, 2}),
          torch::full({rows}, 0.5F),
          draft_logits};
}

}  // namespace

TEST(MtpRequestStateLifecycleTest, WritePrefillTargetContextAndClear) {
  MtpRequestStateLifecycle cache(/*total_nums=*/4);

  std::vector<int32_t> ids = {3, 2};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor target_tokens = torch::tensor({31, 41}, torch::kInt);
  cache.write_prefill_target_context(ids, request_ids, target_tokens);

  std::vector<MtpRequestStateLifecycle::DecodeState> states =
      cache.read_decode_states(ids, request_ids);
  ASSERT_EQ(states.size(), ids.size());
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].request_id, "req_0");
  EXPECT_EQ(states[0].token_id, 31);
  EXPECT_EQ(states[0].position_offset, 0);
  EXPECT_TRUE(states[1].valid);
  EXPECT_EQ(states[1].token_id, 41);
  EXPECT_EQ(states[1].position_offset, 0);

  cache.clear(ids);
  states = cache.read_decode_states(ids, request_ids);
  EXPECT_FALSE(states[0].valid);
  EXPECT_EQ(states[0].token_id, 0);
  EXPECT_EQ(states[0].position_offset, 0);
  EXPECT_FALSE(states[1].valid);
  EXPECT_EQ(states[1].token_id, 0);
  EXPECT_EQ(states[1].position_offset, 0);
}

TEST(MtpRequestStateLifecycleTest,
     PrefillTargetContextOwnsTokenValueAfterInputMutation) {
  MtpRequestStateLifecycle cache(/*total_nums=*/4);

  std::vector<int32_t> ids = {1, 2};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor target_tokens = torch::tensor({51, 61}, torch::kInt);
  cache.write_prefill_target_context(ids, request_ids, target_tokens);
  target_tokens.zero_();

  std::vector<MtpRequestStateLifecycle::DecodeState> states =
      cache.read_decode_states(ids, request_ids);
  ASSERT_EQ(states.size(), ids.size());
  EXPECT_EQ(states[0].token_id, 51);
  EXPECT_EQ(states[0].position_offset, 0);
  EXPECT_EQ(states[1].token_id, 61);
  EXPECT_EQ(states[1].position_offset, 0);
}

TEST(MtpRequestStateLifecycleTest, WriteValidateTargetContext) {
  MtpRequestStateLifecycle cache(/*total_nums=*/2);
  std::vector<int32_t> ids = {0, 1};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor accepted_tokens =
      torch::tensor({{11, 12, 13}, {21, -1, -1}}, torch::kInt);
  cache.write_target_context(ids, request_ids, accepted_tokens);

  std::vector<MtpRequestStateLifecycle::DecodeState> states =
      cache.read_decode_states(ids, request_ids);
  EXPECT_EQ(states[0].token_id, 13);
  EXPECT_EQ(states[0].position_offset, 2);

  EXPECT_EQ(states[1].token_id, 21);
  EXPECT_EQ(states[1].position_offset, 0);

  accepted_tokens.zero_();
  states = cache.read_decode_states(ids, request_ids);
  EXPECT_EQ(states[0].token_id, 13);
  EXPECT_EQ(states[0].position_offset, 2);
  EXPECT_EQ(states[1].token_id, 21);
  EXPECT_EQ(states[1].position_offset, 0);
}

TEST(MtpRequestStateLifecycleTest, RejectsEmptyAcceptedTokenRow) {
  MtpRequestStateLifecycle cache(/*total_nums=*/1);
  const torch::Tensor accepted_tokens = torch::tensor({{-1, -1}}, torch::kInt);

  EXPECT_DEATH(cache.write_target_context({0}, {"request"}, accepted_tokens),
               "Accepted token row 0 is empty");

  const torch::Tensor zero_width = torch::empty({1, 0}, torch::kInt);
  EXPECT_DEATH(cache.write_target_context({0}, {"request"}, zero_width),
               "Accepted token rows must not be empty");

  const torch::Tensor malformed_tail = torch::tensor({{1, -1, 2}}, torch::kInt);
  EXPECT_DEATH(cache.write_target_context({0}, {"request"}, malformed_tail),
               "must have only -1 padding after the valid prefix");
}

TEST(MtpRequestStateLifecycleTest, RequestMismatchMaterializesMissingState) {
  MtpRequestStateLifecycle cache(/*total_nums=*/2);
  std::vector<int32_t> ids = {0};
  std::vector<std::string> request_ids = {"old_req"};
  torch::Tensor target_tokens = torch::tensor({31}, torch::kInt);
  cache.write_prefill_target_context(ids, request_ids, target_tokens);

  std::vector<MtpRequestStateLifecycle::DecodeState> states =
      cache.read_decode_states(ids, {"new_req"});
  ASSERT_EQ(states.size(), ids.size());
  EXPECT_FALSE(states[0].valid);
  EXPECT_EQ(states[0].token_id, 0);
}

TEST(MtpRequestStateLifecycleTest,
     ReadAcceptedPrefixLengthsRejectsStaleRequest) {
  MtpRequestStateLifecycle cache(/*total_nums=*/2);
  std::vector<int32_t> ids = {0, 1};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor accepted_tokens =
      torch::tensor({{11, 12, 13}, {21, -1, -1}}, torch::kInt);
  // Only embedding_id 0 receives target output; id 1 stays uninitialized.
  cache.write_target_context(
      {0}, {"req_0"}, accepted_tokens.slice(/*dim=*/0, 0, 1));

  // Matching request_id returns position_offset + 1; an
  // uninitialized entry returns 1.
  std::vector<int32_t> lengths =
      cache.read_accepted_prefix_lengths(ids, request_ids);
  ASSERT_EQ(lengths.size(), ids.size());
  EXPECT_EQ(lengths[0], 3);
  EXPECT_EQ(lengths[1], 1);

  // A preempted-and-reused embedding_id now owned by a fresh request must not
  // leak the previous request's correction offset.
  std::vector<int32_t> reused =
      cache.read_accepted_prefix_lengths({0}, {"new_req"});
  ASSERT_EQ(reused.size(), 1u);
  EXPECT_EQ(reused[0], 1);
}

TEST(MtpRequestStateLifecycleTest,
     WriteMtpBootstrapContextStoresExactDecodeState) {
  MtpRequestStateLifecycle cache(/*total_nums=*/2);
  cache.write_target_bootstrap_context(
      /*embedding_id=*/1, "req_bootstrap", /*token_id=*/17);

  std::vector<MtpRequestStateLifecycle::DecodeState> states =
      cache.read_decode_states({1}, {"req_bootstrap"});
  ASSERT_EQ(states.size(), 1u);
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].request_id, "req_bootstrap");
  EXPECT_EQ(states[0].token_id, 17);
  EXPECT_EQ(states[0].position_offset, 0);

  cache.clear({1});
  states = cache.read_decode_states({1}, {"req_bootstrap"});
  EXPECT_FALSE(states[0].valid);
}

TEST(MtpRequestStateLifecycleTest,
     DraftInputCloneDetachesNoncontiguousStorage) {
  DraftInput draft_input{
      torch::tensor({1, 9, 0, 9}, torch::kLong)
          .slice(/*dim=*/0, /*start=*/0, /*end=*/4, /*step=*/2),
      torch::arange(8, torch::kFloat)
          .view({2, 4})
          .slice(/*dim=*/1, /*start=*/0, /*end=*/4, /*step=*/2),
      torch::Tensor(),
      torch::tensor({{0.25f, 9.0f, 0.75f, 9.0f}, {0.5f, 9.0f, 0.5f, 9.0f}})
          .slice(/*dim=*/1, /*start=*/0, /*end=*/4, /*step=*/2)};
  draft_input.selected_probs =
      torch::tensor({0.75f, 9.0f, 0.5f, 9.0f})
          .slice(/*dim=*/0, /*start=*/0, /*end=*/4, /*step=*/2);
  draft_input.hidden_states.set_requires_grad(/*requires_grad=*/true);
  const DraftInput cloned = clone_draft_input(draft_input);
  EXPECT_TRUE(cloned.token_ids.is_contiguous());
  EXPECT_TRUE(cloned.hidden_states.is_contiguous());
  EXPECT_TRUE(cloned.draft_logits.is_contiguous());
  EXPECT_TRUE(cloned.selected_probs.is_contiguous());
  EXPECT_FALSE(cloned.hidden_states.requires_grad());

  draft_input.token_ids.zero_();
  draft_input.hidden_states.detach().zero_();
  draft_input.draft_logits.zero_();
  draft_input.selected_probs.zero_();
  EXPECT_TRUE(
      tensor_equal(cloned.token_ids, torch::tensor({1, 0}, torch::kLong)));
  EXPECT_TRUE(tensor_equal(cloned.hidden_states,
                           torch::tensor({{0.0f, 2.0f}, {4.0f, 6.0f}})));
  EXPECT_TRUE(tensor_equal(cloned.draft_logits,
                           torch::tensor({{0.25f, 0.75f}, {0.5f, 0.5f}})));
  EXPECT_TRUE(
      tensor_equal(cloned.selected_probs, torch::tensor({0.75f, 0.5f})));
}

TEST(MtpRequestStateLifecycleTest,
     PointMassSeedsStaySparseUntilMixedBatchAssembly) {
  const DraftInput greedy{torch::tensor({1}, torch::kLong),
                          torch::ones({1, 2}),
                          torch::Tensor(),
                          torch::Tensor(),
                          /*point_mass_q=*/true};
  const DraftInput random{torch::tensor({2}, torch::kLong),
                          torch::ones({1, 2}),
                          torch::Tensor(),
                          torch::tensor({{0.1f, 0.2f, 0.7f}})};
  const DraftInput all_greedy = cat_draft_inputs({greedy, greedy});
  EXPECT_FALSE(all_greedy.draft_logits.defined());
  EXPECT_TRUE(all_greedy.point_mass_q);

  // The sparse representation forbids mixing point-mass rows with rows that
  // carry draft logits: a batch either agrees on the marker or all rows carry
  // logits (the sampler derives greedy rows per-row from the target's own
  // greedy path instead).
  DraftInput unmarked = greedy;
  unmarked.point_mass_q = false;
  // Mixing point-mass rows with logits rows is rejected by the
  // optional-payload agreement check (the marker check never sees them).
  EXPECT_DEATH((void)cat_draft_inputs({unmarked, random}),
               "agree on optional probability payloads");
  EXPECT_DEATH((void)cat_draft_inputs({greedy, random}),
               "agree on q semantics");

  DraftInputStateStore store(/*total_nums=*/2);
  store.write_draft_input({0}, {"random"}, random);
  store.write_draft_input({1}, {"random"}, random);
  const std::optional<DraftInput> restored =
      store.collect_draft_inputs({0, 1}, {"random", "random"});
  ASSERT_TRUE(restored.has_value());
  EXPECT_TRUE(
      torch::allclose(restored->draft_logits,
                      torch::tensor({{0.1f, 0.2f, 0.7f}, {0.1f, 0.2f, 0.7f}})));
}

TEST(MtpRequestStateLifecycleTest,
     DraftInputsOwnStorageAndFollowRequestIdentity) {
  MtpRequestStateLifecycle cache(/*total_nums=*/4);
  DraftInput draft_input{torch::tensor({1, 0}, torch::kLong),
                         torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}),
                         torch::tensor({0.75f, 0.5f}),
                         torch::tensor({{0.25f, 0.75f}, {0.5f, 0.5f}})};
  cache.write_draft_input({2, 1}, {"first", "second"}, draft_input);
  draft_input.token_ids.fill_(0);
  draft_input.hidden_states.fill_(0);
  draft_input.draft_logits.fill_(0);
  draft_input.selected_probs.fill_(0);

  const auto reordered =
      cache.collect_draft_inputs({1, 2}, {"second", "first"});
  ASSERT_TRUE(reordered.has_value());
  EXPECT_TRUE(
      torch::equal(reordered->token_ids, torch::tensor({0, 1}, torch::kLong)));
  EXPECT_TRUE(torch::equal(reordered->hidden_states,
                           torch::tensor({{3.0f, 4.0f}, {1.0f, 2.0f}})));
  EXPECT_TRUE(torch::equal(reordered->draft_logits,
                           torch::tensor({{0.5f, 0.5f}, {0.25f, 0.75f}})));
  EXPECT_TRUE(
      torch::equal(reordered->selected_probs, torch::tensor({0.5f, 0.75f})));
  EXPECT_FALSE(cache.collect_draft_inputs({1}, {"recycled"}).has_value());
  EXPECT_FALSE(
      cache.collect_draft_inputs({1, 3}, {"second", "missing"}).has_value());

  cache.clear({1});
  EXPECT_FALSE(cache.collect_draft_inputs({1}, {"second"}).has_value());
  EXPECT_TRUE(cache.collect_draft_inputs({2}, {"first"}).has_value());
  cache.write_draft_input(
      {1}, {"recycled"}, slice_draft_input(*reordered, 0, 1));
  EXPECT_TRUE(cache.collect_draft_inputs({1}, {"recycled"}).has_value());
  EXPECT_FALSE(cache.collect_draft_inputs({1}, {"second"}).has_value());
}

TEST(DraftInputStateStoreDeathTest,
     RejectsDuplicateSlotsWithinEachPublication) {
  DraftInputStateStore store(/*total_nums=*/4);
  const DraftInput draft_input{torch::tensor({1, 2}, torch::kLong),
                               torch::ones({2, 2}),
                               torch::Tensor(),
                               torch::Tensor()};
  store.write_draft_input({1, 2}, {"a", "b"}, draft_input);
  // Reusing a slot in a later publication is valid; duplication within one
  // publication would corrupt snapshot row accounting.
  store.write_draft_input({2, 1}, {"b", "a"}, draft_input);
  EXPECT_DEATH(store.write_draft_input({1, 1}, {"a", "a"}, draft_input),
               "duplicate draft input slot");
}

TEST(MtpRequestStateLifecycleTest, TakingDraftInputDoesNotClearTargetContext) {
  MtpRequestStateLifecycle cache(/*total_nums=*/2);
  cache.write_target_bootstrap_context(1, "request", 4);
  DraftInput draft_input{torch::tensor({5}, torch::kLong),
                         torch::tensor({{1.0f, 2.0f}}),
                         torch::tensor({0.9f}),
                         torch::Tensor()};
  cache.write_draft_input({1}, {"request"}, draft_input);

  EXPECT_FALSE(cache.take_draft_input({1}, {"other"}).has_value());
  const auto taken = cache.take_draft_input({1}, {"request"});
  ASSERT_TRUE(taken.has_value());
  EXPECT_TRUE(torch::equal(taken->hidden_states, draft_input.hidden_states));
  EXPECT_FALSE(cache.collect_draft_inputs({1}, {"request"}).has_value());
  const auto states = cache.read_decode_states({1}, {"request"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].token_id, 4);
}

TEST(MtpRequestStateLifecycleTest,
     StableBatchConsumesOneOwnedSnapshotWithoutCopy) {
  MtpRequestStateLifecycle cache(/*total_nums=*/4);
  DraftInput producer{torch::tensor({1, 2, 3}, torch::kLong),
                      torch::ones({3, 2}),
                      torch::ones({3}),
                      torch::full({3, 8}, 0.125F)};
  cache.write_draft_input({0, 2, 1}, {"a", "b", "c"}, producer, {}, {});
  const auto snapshot = cache.collect_draft_inputs({0, 2, 1}, {"a", "b", "c"});
  ASSERT_TRUE(snapshot.has_value());
  EXPECT_NE(snapshot->draft_logits.data_ptr(),
            producer.draft_logits.data_ptr());
  producer.draft_logits.zero_();
  const auto consumed = cache.take_draft_input({0, 2, 1}, {"a", "b", "c"});
  ASSERT_TRUE(consumed.has_value());
  EXPECT_EQ(consumed->draft_logits.data_ptr(),
            snapshot->draft_logits.data_ptr());
  EXPECT_EQ(consumed->hidden_states.data_ptr(),
            snapshot->hidden_states.data_ptr());
  EXPECT_TRUE(
      torch::equal(consumed->draft_logits, torch::full({3, 8}, 0.125F)));
  EXPECT_FALSE(cache.take_draft_input({0, 2, 1}, {"a", "b", "c"}).has_value());
}

TEST(MtpRequestStateLifecycleTest,
     PartialLifecycleChangesCompactSurvivingRows) {
  const DraftInput producer{torch::tensor({1, 2, 3, 4}, torch::kLong),
                            torch::ones({4, 2}),
                            torch::ones({4}),
                            torch::full({4, 1024}, 1.0F / 1024)};
  for (int32_t transition = 0; transition < 3; ++transition) {
    SCOPED_TRACE(transition);
    MtpRequestStateLifecycle cache(/*total_nums=*/4);
    cache.write_draft_input({0, 1, 2, 3}, {"a", "b", "c", "d"}, producer);
    if (transition == 0) {
      cache.clear({1, 2, 3});
    } else if (transition == 1) {
      ASSERT_TRUE(
          cache.take_draft_input({3, 1, 2}, {"d", "b", "c"}).has_value());
    } else {
      cache.write_draft_input({1, 2, 3},
                              {"new-b", "new-c", "new-d"},
                              slice_draft_input(producer, 1, 3));
    }
    const auto survivor = cache.collect_draft_inputs({0}, {"a"});
    ASSERT_TRUE(survivor.has_value());
    EXPECT_EQ(survivor->draft_logits.storage().nbytes(), 1024 * sizeof(float));
    EXPECT_EQ(survivor->hidden_states.storage().nbytes(), 2 * sizeof(float));
    EXPECT_TRUE(
        torch::equal(survivor->token_ids, torch::tensor({1}, torch::kLong)));
  }
}

TEST(MtpRequestStateLifecycleTest, MixedBatchReordersOnlyRequestedRows) {
  MtpRequestStateLifecycle cache(/*total_nums=*/4);
  const DraftInput first{torch::tensor({1, 2}, torch::kLong),
                         torch::tensor({{1.0F}, {2.0F}}),
                         torch::ones({2}),
                         torch::tensor({{0.25F, 0.75F}, {0.5F, 0.5F}})};
  const DraftInput second{torch::tensor({3}, torch::kLong),
                          torch::tensor({{3.0F}}),
                          torch::Tensor(),
                          torch::tensor({{0.75F, 0.25F}})};
  cache.write_draft_input({0, 1}, {"a", "b"}, first);
  cache.write_draft_input({2}, {"c"}, second);
  EXPECT_FALSE(cache.take_draft_input({2, 0}, {"c", "wrong"}).has_value());
  const auto consumed = cache.take_draft_input({2, 0}, {"c", "a"});
  ASSERT_TRUE(consumed.has_value());
  EXPECT_TRUE(torch::equal(consumed->draft_logits,
                           torch::tensor({{0.75F, 0.25F}, {0.25F, 0.75F}})));
  EXPECT_FALSE(consumed->selected_probs.defined());
  const auto survivor = cache.take_draft_input({1}, {"b"});
  ASSERT_TRUE(survivor.has_value());
  EXPECT_LE(survivor->draft_logits.storage().nbytes(), 2 * 2 * sizeof(float));
}

TEST(MtpRequestStateLifecycleTest, MixedBatchAssemblyGroupsAdjacentRows) {
  MtpRequestStateLifecycle cache(/*total_nums=*/12);
  const DraftInput producer = make_numbered_draft_input(/*rows=*/12);
  cache.write_draft_input({0, 1, 2, 3, 4, 5, 6, 7},
                          {"0", "1", "2", "3", "4", "5", "6", "7"},
                          slice_draft_input(producer, 0, 8));
  cache.write_draft_input({8, 9, 10, 11},
                          {"8", "9", "10", "11"},
                          slice_draft_input(producer, 8, 4));
  const std::vector<int32_t> ids = {3, 4, 5, 9, 10, 0, 1, 1};
  const std::vector<std::string> requests = {
      "3", "4", "5", "9", "10", "0", "1", "1"};
  std::optional<DraftInput> assembled;
  {
    SeedConcatenationObserver observer;
    assembled = cache.collect_draft_inputs(ids, requests);
    // Four runs preserve the requested order, including the duplicate row.
    EXPECT_EQ(observer.input_counts(), std::vector<size_t>({4, 4, 4, 4}));
  }
  ASSERT_TRUE(assembled.has_value());
  const torch::Tensor indices =
      torch::tensor({3, 4, 5, 9, 10, 0, 1, 1}, torch::kLong);
  for (const auto member : kDraftInputTensors) {
    EXPECT_TRUE(torch::equal((*assembled).*member,
                             (producer.*member).index_select(0, indices)));
  }
  auto stale_requests = requests;
  stale_requests.back() = "stale";
  {
    SeedConcatenationObserver observer;
    EXPECT_FALSE(cache.take_draft_input(ids, stale_requests).has_value());
    EXPECT_TRUE(observer.input_counts().empty());
  }
  EXPECT_TRUE(cache.collect_draft_inputs(ids, requests).has_value());
}

TEST(MtpRequestStateLifecycleTest,
     ContiguousSurvivorsCompactWithoutConcatenation) {
  MtpRequestStateLifecycle cache(/*total_nums=*/8);
  const DraftInput producer = make_numbered_draft_input(/*rows=*/8);
  cache.write_draft_input({0, 1, 2, 3, 4, 5, 6, 7},
                          {"0", "1", "2", "3", "4", "5", "6", "7"},
                          producer);
  const auto old_snapshot = cache.collect_draft_inputs(
      {0, 1, 2, 3, 4, 5, 6, 7}, {"0", "1", "2", "3", "4", "5", "6", "7"});
  ASSERT_TRUE(old_snapshot.has_value());
  {
    SeedConcatenationObserver observer;
    cache.clear({0, 1, 2, 3, 7});
    EXPECT_TRUE(observer.input_counts().empty());
  }
  const auto survivors = cache.collect_draft_inputs({4, 5, 6}, {"4", "5", "6"});
  ASSERT_TRUE(survivors.has_value());
  for (const auto member : kDraftInputTensors) {
    const torch::Tensor& tensor = (*survivors).*member;
    EXPECT_TRUE(torch::equal(tensor, (producer.*member).narrow(0, 4, 3)));
    EXPECT_EQ(tensor.storage().nbytes(), tensor.nbytes());
    EXPECT_FALSE(
        tensor.storage().is_alias_of(((*old_snapshot).*member).storage()));
  }
  EXPECT_TRUE(torch::equal(old_snapshot->token_ids, producer.token_ids));
}

TEST(MtpRequestStateLifecycleTest, FragmentedSurvivorsCompactAdjacentRows) {
  MtpRequestStateLifecycle cache(/*total_nums=*/12);
  const DraftInput producer = make_numbered_draft_input(/*rows=*/12);
  cache.write_draft_input(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
      {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"},
      producer);
  {
    SeedConcatenationObserver observer;
    cache.clear({0, 3, 4, 5, 6, 7, 8});
    EXPECT_EQ(observer.input_counts(), std::vector<size_t>({2, 2, 2, 2}));
  }
  const auto survivors = cache.collect_draft_inputs(
      {1, 2, 9, 10, 11}, {"1", "2", "9", "10", "11"});
  ASSERT_TRUE(survivors.has_value());
  const torch::Tensor indices = torch::tensor({1, 2, 9, 10, 11}, torch::kLong);
  for (const auto member : kDraftInputTensors) {
    const torch::Tensor& tensor = (*survivors).*member;
    EXPECT_TRUE(
        torch::equal(tensor, (producer.*member).index_select(0, indices)));
    EXPECT_EQ(tensor.storage().nbytes(), tensor.nbytes());
  }
}

TEST(MtpRequestStateLifecycleTest, FailedDrainRetainsOnlyItsOwnProbabilityRow) {
  MtpRequestStateLifecycle cache(/*total_nums=*/3);
  const DraftInput producer{torch::tensor({1, 2, 3}, torch::kLong),
                            torch::ones({3, 2}),
                            torch::Tensor(),
                            torch::full({3, 1024}, 1.0F / 1024)};
  cache.write_draft_input({0, 1, 2}, {"a", "b", "c"}, producer, {}, {1});
  ASSERT_TRUE(cache.take_draft_input({0, 1, 2}, {"a", "b", "c"}).has_value());
  EXPECT_FALSE(cache.collect_draft_inputs({1}, {"b"}).has_value());
  for (int32_t queued = 0; queued < 2; ++queued) {
    const auto drain = cache.take_draft_input({1}, {"b"});
    ASSERT_TRUE(drain.has_value());
    EXPECT_EQ(drain->draft_logits.storage().nbytes(), 1024 * sizeof(float));
  }
}

TEST(MtpRequestStateLifecycleTest,
     RepeatedPartialRetirementBoundsMemoryAndCopyVolume) {
  constexpr int32_t kBatch = 64;
  constexpr int32_t kVocab = 128;
  constexpr size_t kRowBytes = kVocab * sizeof(float);
  std::vector<int32_t> ids;
  std::vector<std::string> requests;
  ids.reserve(kBatch);
  requests.reserve(kBatch);
  for (int32_t row = 0; row < kBatch; ++row) {
    ids.emplace_back(row);
    requests.emplace_back(std::to_string(row));
  }
  const DraftInput producer{torch::arange(kBatch, torch::kLong),
                            torch::ones({kBatch, 2}),
                            torch::Tensor(),
                            torch::full({kBatch, kVocab}, 1.0F / kVocab)};
  for (int32_t transition = 0; transition < 3; ++transition) {
    SCOPED_TRACE(transition);
    MtpRequestStateLifecycle cache(kBatch);
    cache.write_draft_input(ids, requests, producer);
    size_t copied_rows = 0;
    for (int32_t row = 0; row < kBatch - 1; ++row) {
      const auto before =
          cache.collect_draft_inputs({row + 1}, {requests[row + 1]});
      ASSERT_TRUE(before.has_value());
      if (transition == 0) {
        ASSERT_TRUE(cache.take_draft_input({row}, {requests[row]}).has_value());
      } else if (transition == 1) {
        cache.clear({row});
      } else {
        cache.write_draft_input({row},
                                {"replacement-" + requests[row]},
                                slice_draft_input(producer, row, 1));
      }
      const auto after =
          cache.collect_draft_inputs({row + 1}, {requests[row + 1]});
      ASSERT_TRUE(after.has_value());
      const size_t retained_rows =
          after->draft_logits.storage().nbytes() / kRowBytes;
      const size_t live_rows = kBatch - row - 1;
      EXPECT_GE(retained_rows, live_rows);
      EXPECT_LE(retained_rows, 2 * live_rows);
      if (before->draft_logits.storage().data_ptr().get() !=
          after->draft_logits.storage().data_ptr().get()) {
        copied_rows += retained_rows;
      }
      EXPECT_TRUE(torch::equal(after->token_ids,
                               torch::tensor({row + 1}, torch::kLong)));
    }
    EXPECT_LT(copied_rows, kBatch);
    RecordProperty("copied_survivor_rows_" + std::to_string(transition),
                   static_cast<int32_t>(copied_rows));
  }
}

TEST(MtpRequestStateLifecycleTest,
     DuplicateReleasesPreserveSurvivorAccounting) {
  MtpRequestStateLifecycle cache(/*total_nums=*/3);
  const DraftInput draft_input{torch::tensor({1, 2, 3}, torch::kLong),
                               torch::ones({3, 2}),
                               torch::Tensor(),
                               torch::full({3, 8}, 0.125F)};
  cache.write_draft_input({0, 1, 2}, {"a", "b", "c"}, draft_input);
  cache.clear({0, 0});
  cache.clear({0});
  ASSERT_TRUE(cache.take_draft_input({1}, {"b"}).has_value());
  const auto survivor = cache.take_draft_input({2}, {"c"});
  ASSERT_TRUE(survivor.has_value());
  EXPECT_TRUE(
      torch::equal(survivor->token_ids, torch::tensor({3}, torch::kLong)));
  EXPECT_EQ(survivor->draft_logits.storage().nbytes(), 8 * sizeof(float));
}

TEST(MtpRequestStateLifecycleTest,
     DraftInputProbabilityFieldsHaveDistinctShapes) {
  DraftInput draft_input{torch::tensor({1}, torch::kLong),
                         torch::tensor({{1.0f, 2.0f}}),
                         torch::tensor({0.9f}),
                         torch::Tensor()};
  EXPECT_TRUE(is_valid_draft_input(draft_input));
  draft_input.draft_logits = torch::tensor({0.9f});
  EXPECT_FALSE(is_valid_draft_input(draft_input));
  draft_input.draft_logits = torch::tensor({{0.1f, 0.9f}});
  EXPECT_TRUE(is_valid_draft_input(draft_input));
  draft_input.selected_probs = torch::tensor({{0.9f}});
  EXPECT_FALSE(is_valid_draft_input(draft_input));
}

TEST(MtpRequestStateLifecycleTest,
     FailedSeedsDrainWithoutBecomingPublishedSeeds) {
  MtpRequestStateLifecycle cache(/*total_nums=*/4);
  const DraftInput draft_input{torch::tensor({5, 6}, torch::kLong),
                               torch::ones({2, 3}),
                               torch::ones({2}),
                               torch::Tensor()};
  SamplingParameters sampling;
  sampling.unique_token_ids = torch::tensor({{3}, {4}}, torch::kLong);
  sampling.unique_token_counts = torch::ones({2, 1}, torch::kInt);
  sampling.unique_token_ids_lens = torch::ones({2}, torch::kInt);
  DraftInputSamplingState sampling_state{sampling, {}, {}, {}};
  import_sampling_histories(sampling, 2, sampling_state.histories);
  cache.write_draft_input({1, 2}, {"a", "b"}, draft_input, sampling_state, {1});
  EXPECT_TRUE(cache.collect_draft_inputs({1}, {"a"}).has_value());
  EXPECT_FALSE(cache.collect_draft_inputs({2}, {"b"}).has_value());
  EXPECT_EQ(cache.failed_draft_input_mask({1, 2}, {"a", "b"}),
            std::vector<uint8_t>({0, 1}));

  // Failed batch acquisition must not consume another request's valid
  // draft_input.
  std::vector<uint8_t> failed_mask;
  EXPECT_FALSE(cache.take_draft_input({1, 3}, {"a", "missing"}, &failed_mask)
                   .has_value());
  EXPECT_TRUE(failed_mask.empty());
  ASSERT_TRUE(
      cache.take_draft_input({2, 1}, {"b", "a"}, &failed_mask).has_value());
  EXPECT_EQ(failed_mask, std::vector<uint8_t>({1, 0}));
  EXPECT_FALSE(cache.collect_draft_inputs({1}, {"a"}).has_value());
  EXPECT_TRUE(cache.take_draft_input({2}, {"b"}).has_value());
  EXPECT_TRUE(cache.take_draft_input({2}, {"b"}).has_value());

  // A later successful model phase cannot revive a failed request.
  cache.write_draft_input({1, 2}, {"a", "b"}, draft_input, sampling_state, {});
  EXPECT_FALSE(cache.collect_draft_inputs({2}, {"b"}).has_value());
  cache.clear({2});
  EXPECT_TRUE(cache.failed_draft_input_mask({2}, {"b"}).empty());
  cache.write_draft_input({2}, {"new"}, slice_draft_input(draft_input, 1, 1));
  EXPECT_TRUE(cache.collect_draft_inputs({2}, {"new"}).has_value());
  EXPECT_FALSE(cache.take_draft_input({2}, {"b"}).has_value());
  EXPECT_TRUE(cache.take_draft_input({2}, {"new"}).has_value());
}

TEST(MtpRequestStateLifecycleTest,
     SamplingContextFollowsSeedIdentityAndLifetime) {
  MtpRequestStateLifecycle cache(/*total_nums=*/3);
  const DraftInput draft_input{torch::tensor({5}, torch::kLong),
                               torch::ones({1, 2}),
                               torch::Tensor(),
                               torch::Tensor()};
  SamplingParameters committed;
  committed.unique_token_ids = torch::tensor({{3, 4}}, torch::kLong);
  committed.unique_token_counts = torch::tensor({{1, 2}}, torch::kInt);
  committed.unique_token_ids_lens = torch::tensor({2}, torch::kInt);
  DraftInputSamplingState sampling_state{committed, {}, {}, {}};
  import_sampling_histories(committed, 1, sampling_state.histories);
  cache.write_draft_input({1}, {"request"}, draft_input, sampling_state, {});
  committed.unique_token_counts.zero_();
  ASSERT_TRUE(cache.take_draft_input({1}, {"request"}).has_value());

  SamplingParameters restored;
  restored.repetition_penalties = torch::tensor({1.1F});
  std::vector<JsonObjectGrammarState> grammar;
  cache.restore_draft_sampling(
      {1}, {"request"}, restored, grammar, torch::Device(torch::kCPU));
  EXPECT_TRUE(torch::equal(restored.unique_token_ids,
                           torch::tensor({{3, 4}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(restored.unique_token_counts,
                           torch::tensor({{1, 2}}, torch::kInt)));
  cache.clear({1});
  cache.write_draft_input({1}, {"replacement"}, draft_input);
  EXPECT_FALSE(cache.collect_draft_inputs({1}, {"request"}).has_value());
}

TEST(MtpRequestStateLifecycleTest, PublishedGrammarStateCanBeMovedFromOwner) {
  MtpRequestStateLifecycle cache(/*total_nums=*/2);
  const DraftInput draft_input{torch::tensor({5}, torch::kLong),
                               torch::ones({1, 2}),
                               torch::Tensor(),
                               torch::Tensor()};
  JsonObjectGrammar grammar({"{", "\"a\"", ":", "1", "}"});
  JsonObjectGrammarState committed = grammar.initial_state();
  ASSERT_TRUE(committed.accept_token(/*open_object=*/0));
  DraftInputSamplingState sampling{{}, {committed}, {}, {}};

  cache.write_draft_input({0}, {"copy"}, draft_input, sampling, {});
  EXPECT_EQ(sampling.json_object_states[0].snapshot().token_ids,
            std::vector<int32_t>({0}));
  cache.write_draft_input({1}, {"move"}, draft_input, std::move(sampling), {});

  for (const auto& [slot, request] :
       {std::pair<int32_t, const char*>{0, "copy"}, {1, "move"}}) {
    SamplingParameters restored;
    std::vector<JsonObjectGrammarState> states(1);
    cache.restore_draft_sampling(
        {slot}, {request}, restored, states, torch::Device(torch::kCPU));
    EXPECT_EQ(states[0].snapshot().token_ids, std::vector<int32_t>({0}));
  }
}

}  // namespace xllm
