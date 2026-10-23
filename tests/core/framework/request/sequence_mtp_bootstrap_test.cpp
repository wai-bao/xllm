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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/framework/config/disagg_pd_config.h"
#include "framework/request/incremental_decoder.h"
#include "framework/request/sequence.h"

namespace xllm {
namespace {

class ScopedDisaggPDConfig final {
 public:
  ScopedDisaggPDConfig()
      : enabled_(DisaggPDConfig::get_instance().enable_disagg_pd()) {
    DisaggPDConfig::get_instance().enable_disagg_pd(true);
  }

  ~ScopedDisaggPDConfig() {
    DisaggPDConfig::get_instance().enable_disagg_pd(enabled_);
  }

 private:
  bool enabled_;
};

Sequence make_sequence(bool enable_overlap = false) {
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = 8;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = enable_overlap;
  params.rec_type = RecType::kNone;
  params.bos_token_id = 0;
  params.request_id = "mtp_bootstrap_req";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;

  std::vector<int32_t> prompt_token_ids = {1, 2, 3};
  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/prompt_token_ids.size(),
      /*echo=*/params.echo,
      /*skip_special_tokens=*/params.skip_special_tokens);
  return Sequence(/*index=*/0,
                  prompt_token_ids,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  params);
}

}  // namespace

TEST(SequenceMtpBootstrapTest, DraftInputDetachesFromProducerAndKeepsStorage) {
  Sequence sequence = make_sequence();
  DraftInput draft_input{torch::tensor({1}, torch::kLong),
                         torch::tensor({{1.0f, 2.0f}}),
                         torch::tensor({{0.25f, 0.75f}}),
                         torch::tensor({0.75f})};
  sequence.update_draft_input(draft_input);
  // Views share storage with the producer batch, so mutating the producer
  // is visible; what must NOT carry over is autograd history.
  draft_input.token_ids.fill_(0);
  draft_input.hidden_states.fill_(0);
  draft_input.probs.fill_(0);
  draft_input.selected_probs.fill_(0);

  const auto& stored = sequence.get_draft_input();
  ASSERT_TRUE(stored.has_value());
  EXPECT_EQ(stored->token_ids.dim(), 1);
  EXPECT_EQ(stored->hidden_states.dim(), 2);
  // Shared storage: the producer fill above is visible through the view.
  EXPECT_TRUE(
      torch::equal(stored->token_ids, torch::tensor({0}, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(stored->hidden_states, torch::tensor({{0.0f, 0.0f}})));
  EXPECT_TRUE(torch::equal(stored->probs, torch::tensor({{0.0f, 0.0f}})));
  EXPECT_TRUE(torch::equal(stored->selected_probs, torch::tensor({0.0f})));
  EXPECT_FALSE(stored->token_ids.requires_grad());
  sequence.clear_draft_input();
  EXPECT_FALSE(sequence.get_draft_input().has_value());
}

TEST(SequenceMtpBootstrapTest, ResetClearsUnconsumedPdHandoffBeforeRecompute) {
  for (bool enable_overlap : {false, true}) {
    SCOPED_TRACE(enable_overlap);
    Sequence sequence = make_sequence(enable_overlap);
    sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
    if (enable_overlap) {
      sequence.append_token(Token(-1));
      sequence.update_last_step_token(Token(4));
    } else {
      sequence.append_token(Token(4));
    }
    sequence.mark_target_decode_bootstrap();
    sequence.update_draft_input({torch::tensor({5}, torch::kLong),
                                 torch::tensor({{4.0f, 4.0f}}),
                                 torch::Tensor(),
                                 torch::tensor({0.9f})});

    sequence.reset();

    EXPECT_EQ(sequence.kv_cache_tokens_num(), 0);
    EXPECT_EQ(sequence.num_tokens(), 4);
    EXPECT_EQ(sequence.tokens()[3], 4);
    EXPECT_TRUE(sequence.is_prefill_stage());
    EXPECT_FALSE(sequence.has_target_decode_bootstrap());
    EXPECT_FALSE(sequence.get_draft_input().has_value());

    // Explicit clear also returns the flag to its default.
    sequence.mark_target_decode_bootstrap();
    EXPECT_TRUE(sequence.has_target_decode_bootstrap());
    sequence.clear_target_decode_bootstrap();
    EXPECT_FALSE(sequence.has_target_decode_bootstrap());
  }
}

TEST(SequenceMtpBootstrapTest, HostRestoreMetadataKeepsCurrentPdHandoff) {
  Sequence sequence = make_sequence();
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.append_token(Token(4));
  const DraftInput draft_input{torch::tensor({5}, torch::kLong),
                               torch::tensor({{4.0f, 4.0f}}),
                               torch::Tensor(),
                               torch::tensor({0.9f})};
  sequence.mark_target_decode_bootstrap();
  sequence.update_draft_input(draft_input);

  sequence.set_host_cache_match(/*restore_tokens=*/3, /*copy_units=*/1);
  sequence.set_host_cache_restore(/*restore_tokens=*/3, /*copy_units=*/1);
  sequence.clear_host_cache_match();

  EXPECT_EQ(sequence.kv_cache_tokens_num(), 3);
  EXPECT_FALSE(sequence.is_prefill_stage());
  EXPECT_TRUE(sequence.has_target_decode_bootstrap());
  ASSERT_TRUE(sequence.get_draft_input().has_value());
  EXPECT_TRUE(torch::equal(sequence.get_draft_input()->token_ids,
                           draft_input.token_ids));
  EXPECT_TRUE(torch::equal(sequence.get_draft_input()->hidden_states,
                           draft_input.hidden_states));

  sequence.reset();
  EXPECT_FALSE(sequence.has_target_decode_bootstrap());
  EXPECT_FALSE(sequence.get_draft_input().has_value());
}

TEST(SequenceMtpBootstrapTest, ResetDoesNotInvalidateCopiedHandoffState) {
  Sequence sequence = make_sequence();
  const DraftInput draft_input{torch::tensor({5}, torch::kLong),
                               torch::tensor({{4.0f, 4.0f}}),
                               torch::Tensor(),
                               torch::tensor({0.9f})};
  sequence.mark_target_decode_bootstrap();
  sequence.update_draft_input(draft_input);
  Sequence copied(sequence);
  auto forked = sequence.fork(/*index=*/1);

  sequence.reset();

  for (const Sequence* retained : {&copied, forked.get()}) {
    EXPECT_TRUE(retained->has_target_decode_bootstrap());
    ASSERT_TRUE(retained->get_draft_input().has_value());
    EXPECT_TRUE(torch::equal(retained->get_draft_input()->token_ids,
                             draft_input.token_ids));
    EXPECT_TRUE(torch::equal(retained->get_draft_input()->hidden_states,
                             draft_input.hidden_states));
  }
}

TEST(SequenceMtpBootstrapTest,
     PdOverlapWritesRealRootBeforeDraftInputConsumption) {
  // First-token metadata is recorded only in PD mode; also model the state
  // where the decode side already has the prompt KV.
  ScopedDisaggPDConfig pd_config;
  Sequence sequence = make_sequence(/*enable_overlap=*/true);
  EXPECT_FALSE(sequence.first_token().has_value());
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  ASSERT_EQ(sequence.stage(), SequenceStage::DECODE);
  sequence.append_token(Token(-1));
  EXPECT_EQ(sequence.tokens()[sequence.num_prompt_tokens()], -1);
  sequence.update_last_step_token(Token(4));
  EXPECT_TRUE(sequence.is_first_token());
  sequence.mark_target_decode_bootstrap();
  sequence.update_draft_input({torch::tensor({5}, torch::kLong),
                               torch::tensor({{4.0f, 4.0f}}),
                               torch::Tensor(),
                               torch::tensor({0.9f})});

  EXPECT_EQ(sequence.tokens()[sequence.num_prompt_tokens()], 4);
  ASSERT_TRUE(sequence.first_token().has_value());
  EXPECT_EQ(sequence.first_token()->token_id, 4);
  ASSERT_TRUE(sequence.get_draft_input().has_value());
  EXPECT_EQ(sequence.get_draft_input()->token_ids.item<int64_t>(), 5);
  EXPECT_TRUE(sequence.has_target_decode_bootstrap());
}

}  // namespace xllm
