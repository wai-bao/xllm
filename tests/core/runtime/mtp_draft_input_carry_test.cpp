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

#include <ATen/record_function.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/sampling/logits_utils.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/speculative/draft_input.h"
#include "core/framework/speculative/draft_input_state_store.h"
#include "core/framework/speculative/vocab_map.h"
#include "core/platform/platform.h"
#include "core/runtime/executor.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/mtp_target_context_handoff.h"
#include "core/runtime/mtp_worker_impl.h"
#include "core/runtime/options.h"
#include "core/runtime/spec_verify.h"
#include "core/runtime/speculative_input_builder.h"
#include "core/transport/draft_input_proto.h"
#include "core/util/tensor_helper.h"
#include "models/llm/npu/mtp_topk_state.h"

namespace xllm {

using proto_adapter::draft_input_from_proto;
using proto_adapter::draft_input_to_proto;
namespace {

TEST(DraftInputStateStoreTest, TopkSnapshotFollowsRequestStateLifecycle) {
  DraftInputStateStore store(/*total_nums=*/16);
  const MtpTopkStatePtr state = std::make_shared<npu::model::NpuMtpTopkState>(
      torch::tensor({{1, 2}, {3, 4}}, torch::kInt32));
  // Publication follows the seed write of the same round, so the slots are
  // already owned by these requests.
  const DraftInput seed{torch::tensor({1, 2}, torch::kLong),
                        torch::ones({2, 2}),
                        torch::Tensor(),
                        torch::Tensor()};
  store.write_draft_input({10, 11}, {"first", "second"}, seed);
  store.publish_topk_state({10, 11}, {"first", "second"}, state);

  EXPECT_EQ(store.read_topk_state({10, 11}, {"first", "second"}), state);
  // The snapshot is one shared object per batch; a full re-order of the same
  // owned rows still finds it. A row whose slot is owned by another request
  // rejects the batch.
  EXPECT_EQ(store.read_topk_state({11, 10}, {"second", "first"}), state);
  EXPECT_EQ(store.read_topk_state({10, 11}, {"replacement", "second"}),
            nullptr);

  store.clear({12});
  EXPECT_EQ(store.read_topk_state({10, 11}, {"first", "second"}), state);
  store.clear({11});
  EXPECT_EQ(store.read_topk_state({10, 11}, {"first", "second"}), nullptr);
}

TEST(MtpTargetContextHandoffTest, FlushWritesTargetStateExactlyOnce) {
  MtpTargetContextHandoff handoff;
  TargetDecodeStateStore store(/*total_nums=*/2);
  LlmForwardInput input;
  input.input_params.embedding.embedding_ids = {0, 1};
  input.input_params.embedding.request_ids = {"first", "second"};
  SampleOutput output;
  output.next_tokens = torch::tensor({{3, 4, -1}, {5, -1, -1}}, torch::kInt);
  torch::Tensor host = handoff.acquire_host_buffer(output.next_tokens,
                                                   /*configured_capacity=*/8);
  host.copy_(output.next_tokens);
  const int64_t* buffer_address = host.const_data_ptr<int64_t>();
  handoff.stage(input, nullptr, std::move(host));
  EXPECT_TRUE(handoff.has_pending());

  EXPECT_TRUE(handoff.flush(store, /*num_speculative_tokens=*/2));
  EXPECT_FALSE(handoff.has_pending());
  EXPECT_FALSE(handoff.flush(store, /*num_speculative_tokens=*/2));

  const auto states = store.read_decode_states({0, 1}, {"first", "second"});
  EXPECT_EQ(states[0].token_id, 4);
  EXPECT_EQ(states[0].position_offset, 1);
  EXPECT_EQ(states[1].token_id, 5);
  EXPECT_EQ(states[1].position_offset, 0);
  torch::Tensor reused = handoff.acquire_host_buffer(output.next_tokens,
                                                     /*configured_capacity=*/8);
  EXPECT_EQ(reused.const_data_ptr<int64_t>(), buffer_address);
}

constexpr int32_t kBlockSize = 4;
constexpr int32_t kEmbeddingId = 1;
constexpr int32_t kHiddenSize = 3;
constexpr int32_t kVocabSize = 64;
constexpr int32_t kRootToken = 13;
constexpr int32_t kRootPosition = 3;
constexpr int32_t kCorrectionToken = 40;
constexpr char kRequestId[] = "draft-input-carry-request";

enum class ModelRole : int32_t { TARGET = 1, DRAFT = 2 };

torch::Tensor snapshot(const torch::Tensor& tensor) {
  return tensor.defined() ? tensor.to(torch::kCPU).clone() : torch::Tensor();
}

// Hidden provenance deliberately records token and forward position separately:
// a_i belongs to the processed token_i, which need not have position i.
torch::Tensor hidden_rows(ModelRole role,
                          const torch::Tensor& tokens,
                          const torch::Tensor& positions) {
  return torch::stack(
      {tokens.to(torch::kFloat),
       positions.to(torch::kFloat),
       torch::full_like(tokens, static_cast<int32_t>(role), torch::kFloat)},
      /*dim=*/1);
}

struct ModelCall {
  ModelRole role;
  std::vector<int64_t> tokens;
  std::vector<int64_t> positions;
  std::vector<int64_t> slots;
  std::vector<int32_t> q_seq_lens;
  std::vector<int32_t> kv_seq_lens;
  std::vector<int64_t> selected_rows;
  torch::Tensor hidden_input;
  torch::Tensor hidden_output;
  bool is_spec_verify = false;
};

// Only the model is fake. Input preparation, eager Executor, leaf sampling,
// MTP rejection sampling, seed publication and draft_extend remain production.
class RecordingModel final : public CausalLM {
 public:
  RecordingModel(ModelRole role,
                 torch::TensorOptions options,
                 std::shared_ptr<std::vector<ModelCall>> calls)
      : role_(role), options_(options), calls_(std::move(calls)) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelInputParams& params) override {
    torch::Tensor hidden = hidden_rows(role_, tokens, positions);
    ModelCall call;
    call.role = role_;
    call.tokens = tensor_to_vector<int64_t>(tokens);
    call.positions = tensor_to_vector<int64_t>(positions);
    call.slots =
        tensor_to_vector<int64_t>(params.attention.device.new_cache_slots);
    call.q_seq_lens = params.attention.host.q_seq_lens;
    call.kv_seq_lens = params.attention.host.kv_seq_lens;
    call.hidden_input = snapshot(params.embedding.input_embedding);
    call.hidden_output = snapshot(hidden);
    call.is_spec_verify = params.is_spec_verify;
    calls_->emplace_back(std::move(call));
    last_call_ = calls_->size() - 1;
    return ModelOutput(hidden);
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected_rows) override {
    calls_->at(last_call_).selected_rows =
        tensor_to_vector<int64_t>(selected_rows);
    const torch::Tensor selected =
        hidden.index_select(/*dim=*/0, selected_rows.to(torch::kLong));
    torch::Tensor predictions =
        (selected.select(/*dim=*/1, /*index=*/0).to(torch::kLong) + 1)
            .remainder(kVocabSize);
    if (next_predictions_.has_value()) {
      CHECK_EQ(next_predictions_->size(), selected.size(0));
      predictions =
          torch::tensor(*next_predictions_, options_.dtype(torch::kLong));
      next_predictions_.reset();
    }
    torch::Tensor result = torch::full(
        {selected.size(0), kVocabSize}, /*fill_value=*/-1000.0, options_);
    result.scatter_(/*dim=*/1, predictions.view({-1, 1}), /*value=*/1000.0);
    return result;
  }

  void set_next_predictions(std::vector<int64_t> predictions) {
    next_predictions_ = std::move(predictions);
  }

  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}
  void update_expert_weight(int32_t /*layer_id*/) override {}

 private:
  ModelRole role_;
  torch::TensorOptions options_;
  std::shared_ptr<std::vector<ModelCall>> calls_;
  size_t last_call_ = 0;
  std::optional<std::vector<int64_t>> next_predictions_;
};

ModelArgs make_model_args(ModelRole role) {
  ModelArgs args;
  // MiMo selects causal chunked verification without graph-specific kernels.
  args.model_type(role == ModelRole::TARGET ? "mimo" : "mimo_mtp")
      .n_layers(1)
      .n_heads(1)
      .n_kv_heads(1)
      .head_dim(kHiddenSize)
      .hidden_size(kHiddenSize)
      .vocab_size(kVocabSize)
      .max_position_embeddings(64);
  return args;
}

runtime::Options make_options(int32_t speculative_tokens,
                              InstanceRole role = InstanceRole::DEFAULT) {
  runtime::Options options;
  options.model_id("recording_mtp")
      .backend("llm")
      .npu_kernel_backend("TORCH")
      .block_size(kBlockSize)
      .enable_graph(false)
      .enable_speculative_decode(true)
      .num_decoding_tokens(speculative_tokens + 1)
      .num_speculative_tokens(speculative_tokens)
      .enable_schedule_overlap(false)
      .world_size(1)
      .dp_size(1)
      .cp_size(1)
      .instance_role(role);
  return options;
}

class RecordingWorker final : public LLMWorkerImpl {
 public:
  RecordingWorker(const ParallelArgs& parallel_args,
                  const torch::Device& device,
                  const runtime::Options& options,
                  ModelRole role,
                  std::shared_ptr<std::vector<ModelCall>> calls)
      : LLMWorkerImpl(parallel_args, device, options) {
    dtype_ = torch::kFloat;
    const torch::TensorOptions tensor_options =
        torch::TensorOptions().dtype(dtype_).device(device);
    const ModelArgs args = make_model_args(role);
    context_ = ModelContext(parallel_args, args, QuantArgs(), tensor_options);
    model_ = std::make_unique<RecordingModel>(
        role, tensor_options, std::move(calls));
    model_executor_ =
        std::make_unique<Executor>(model_.get(), args, device, options);
    status_ = Status::READY;
  }

  RecordingModel& recording_model() {
    return static_cast<RecordingModel&>(*model_);
  }
};

class DraftInputCarryWorker : public MTPWorkerImpl<LlmForwardInput> {
 public:
  using MTPWorkerImpl<LlmForwardInput>::step;
  using MTPWorkerImpl<LlmForwardInput>::release_request_state;
  DraftInputCarryWorker(const torch::Device& device,
                        const runtime::Options& options,
                        std::shared_ptr<std::vector<ModelCall>> calls)
      : MTPWorkerImpl<LlmForwardInput>(ParallelArgs(/*rank=*/0,
                                                    /*world_size=*/1,
                                                    /*process_group=*/nullptr),
                                       device,
                                       options,
                                       WorkerType::LLM) {
    dtype_ = torch::kFloat;
    context_ =
        ModelContext(parallel_args_,
                     make_model_args(ModelRole::TARGET),
                     QuantArgs(),
                     torch::TensorOptions().dtype(dtype_).device(device));
    target_verify_capabilities_ =
        spec_verify::resolve_target_spec_verify_capabilities(
            TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL,
            /*is_python_model_impl=*/false);
    adaptive_spec_controller_.reset();
    target_state_store_ =
        std::make_unique<TargetDecodeStateStore>(/*total_nums=*/8);
    draft_input_state_store_ =
        std::make_unique<DraftInputStateStore>(/*total_nums=*/8);

    runtime::Options target_options = speculative_target_options(options);
    auto target = std::make_unique<RecordingWorker>(
        parallel_args_, device, target_options, ModelRole::TARGET, calls);
    target_model_ = &target->recording_model();
    impl_ = std::move(target);
    runtime::Options draft_options = speculative_draft_options(options);
    draft_impl_ = std::make_unique<RecordingWorker>(parallel_args_,
                                                    device,
                                                    draft_options,
                                                    ModelRole::DRAFT,
                                                    std::move(calls));
  }

  void set_target_predictions(std::vector<int64_t> predictions) {
    target_model_->set_next_predictions(std::move(predictions));
  }

  const DraftInputStateStore& cache() const {
    return *draft_input_state_store_;
  }
  const TargetDecodeStateStore& target_state() const {
    return *target_state_store_;
  }

  void bootstrap_target_state(int32_t embedding_id,
                              const std::string& request_id,
                              int32_t token_id) {
    target_state_store_->write_prefill_target_context(
        {embedding_id}, {request_id}, torch::tensor({token_id}, torch::kInt));
  }

  void import_draft_input(LlmForwardInput& input) {
    import_carried_draft_input(input);
    CHECK_EQ(compute_stream_->synchronize(), 0);
  }

  std::weak_ptr<DraftTokenHistory> sampling_history() {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    SamplingParameters params;
    params.repetition_penalties = torch::ones({1});
    std::vector<JsonObjectGrammarState> grammar;
    auto histories = draft_input_state_store_->restore(
        {kEmbeddingId}, {kRequestId}, params, grammar, device_);
    return histories.front();
  }

  void fail_vocabulary_on_call(int32_t call_index) {
    failed_vocabulary_call_ = call_index;
  }

 protected:
  std::vector<uint8_t> prepare_draft_vocabulary_sampling(
      SamplingParameters& sampling_params,
      const DraftTokenHistories& /*histories*/) override {
    ++vocabulary_call_count_;
    if (vocabulary_call_count_ != failed_vocabulary_call_) {
      return {};
    }
    // Match the remapper's private placeholder for an empty hot vocabulary.
    // Token 0 is outside the JSON grammar used by these tests.
    sampling_params.filter_mask = torch::Tensor();
    sampling_params.filter_bitmask =
        torch::zeros({sampling_params.sample_idxes.numel(), kVocabSize / 32},
                     torch::TensorOptions().dtype(torch::kInt).device(device_));
    sampling_params.filter_bitmask.select(/*dim=*/1, /*index=*/0).fill_(1);
    return std::vector<uint8_t>(sampling_params.sample_idxes.numel(),
                                /*value=*/1);
  }

 private:
  // Owned by impl_; the target worker is not replaced after construction.
  RecordingModel* target_model_ = nullptr;
  int32_t failed_vocabulary_call_ = -1;
  mutable int32_t vocabulary_call_count_ = 0;
};

// A vocabulary-only override deliberately does not call any base hook.
class RemappingDraftInputWorker final : public DraftInputCarryWorker {
 public:
  using DraftInputCarryWorker::DraftInputCarryWorker;

  DraftInput finish_draft_input(ForwardOutput& output,
                                const SamplingParameters& params,
                                DraftSamplingMode mode,
                                bool require_confidence) {
    draft_sampling_mode_ = mode;
    auto seed = assemble_draft_input(output, params, require_confidence);
    CHECK_EQ(compute_stream_->synchronize(), 0);
    return seed;
  }

  int remap_calls() const { return remap_calls_; }
  bool remapped_probs() const { return remapped_probs_; }

 protected:
  void remap_draft_sample_output(SampleOutput& sample) override {
    ++remap_calls_;
    // Hook contract: implementations consuming q drop proposals that cannot
    // use it before remapping; finalize_mixed_proposal_q is the single
    // authoritative drop for the all-greedy case (unknown here without the
    // sampling params).
    if (draft_sampling_mode_ != DraftSamplingMode::PROBABILISTIC) {
      sample.probs = torch::Tensor();
    }
    remapped_probs_ = sample.probs.defined();
    // Two draft tokens map to target tokens 5 and 6.
    sample.next_tokens = sample.next_tokens + 5;
    if (sample.probs.defined()) {
      auto target_probs = torch::zeros({sample.probs.size(0), kVocabSize},
                                       sample.probs.options());
      target_probs.narrow(/*dim=*/1, /*start=*/5, /*length=*/2)
          .copy_(sample.probs);
      sample.probs = target_probs;
    }
  }

 private:
  int remap_calls_ = 0;
  bool remapped_probs_ = false;
};

LlmForwardInput make_input(const torch::Device& device,
                           const std::vector<int32_t>& tokens,
                           const std::vector<int32_t>& positions,
                           BatchForwardType forward_type,
                           int32_t embedding_id = kEmbeddingId,
                           const std::string& request_id = kRequestId) {
  LlmForwardInput input;
  input.token_ids_host = torch::tensor(tokens, torch::kInt);
  input.positions_host = torch::tensor(positions, torch::kInt);
  input.token_ids = input.token_ids_host.to(device);
  input.positions = input.positions_host.to(device);
  auto& params = input.input_params;
  const int32_t q_len = static_cast<int32_t>(tokens.size());
  const int32_t kv_len = positions.back() + 1;
  params.meta.batch_forward_type = forward_type;
  params.meta.num_sequences = 1;
  params.meta.actual_num_sequences = 1;
  params.meta.q_max_seq_len = q_len;
  params.meta.kv_max_seq_len = kv_len;
  params.attention.host.q_seq_lens = {q_len};
  params.attention.host.q_cu_seq_lens = {0, q_len};
  params.attention.host.kv_seq_lens = {kv_len};
  // Non-identity physical blocks catch confusion between positions and slots.
  params.attention.host.block_tables =
      torch::tensor({{2, 5, 7, 9}}, torch::kInt);
  params.attention.host.new_cache_slots.reserve(positions.size());
  const std::vector<int32_t> block_ids = {2, 5, 7, 9};
  for (int32_t position : positions) {
    params.attention.host.new_cache_slots.emplace_back(
        block_ids[position / kBlockSize] * kBlockSize + position % kBlockSize);
  }
  // NPU views must wrap device storage, as in the real verifier builder.
  // Building a CPU buffer here would label host addresses as NPU pointers.
  CHECK(params.attention.rebuild_device_buffer(device));
  input.input_params.embedding.embedding_ids = {embedding_id};
  input.input_params.embedding.request_ids = {request_id};
  input.input_params.embedding.extra_token_ids = {-1};
  // The eager worker reads identity from the model params' embedding section;
  // the transport request_state mirror stays for the seed/bootstrap path.
  params.embedding.embedding_ids = {embedding_id};
  params.embedding.request_ids = {request_id};
  input.sample_sequence_ids = {request_id};
  input.sampling_params.selected_token_idxes =
      torch::tensor({q_len - 1}, torch::kInt);
  input.sampling_params.sample_idxes = torch::tensor({0}, torch::kInt);
  input.sampling_params.do_sample = torch::tensor({false}, torch::kBool);
  input.sampling_params.all_greedy_sample = true;
  input.sampling_params.all_random_sample = false;
  return input;
}

LlmForwardInput make_prefill(const torch::Device& device) {
  return make_input(device,
                    /*tokens=*/{10, 11, 12},
                    /*positions=*/{0, 1, 2},
                    BatchForwardType::PREFILL);
}

LlmForwardInput make_decode(const torch::Device& device,
                            int32_t token = kRootToken,
                            int32_t position = kRootPosition) {
  return make_input(device, {token}, {position}, BatchForwardType::DECODE);
}

void expect_hidden(const torch::Tensor& actual,
                   ModelRole role,
                   const std::vector<int64_t>& tokens,
                   const std::vector<int64_t>& positions) {
  ASSERT_TRUE(actual.defined());
  EXPECT_TRUE(
      torch::equal(snapshot(actual),
                   hidden_rows(role,
                               torch::tensor(tokens, torch::kLong),
                               torch::tensor(positions, torch::kLong))));
}

std::vector<ModelRole> call_roles(const std::vector<ModelCall>& calls) {
  std::vector<ModelRole> roles;
  roles.reserve(calls.size());
  for (const ModelCall& call : calls) {
    roles.emplace_back(call.role);
  }
  return roles;
}

class MtpDraftInputCarryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // WorkerImpl initializes accelerator streams even for torch::kCPU. No
    // weights/KV tensors are allocated, but a real runtime device is required.
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "MTP runtime construction requires an accelerator.";
    }
  }

  torch::Device device() const {
    return torch::Device(Platform::type_torch(), /*index=*/0);
  }
};

TEST_F(MtpDraftInputCarryTest, MixedEagleHistoriesNeutralizeDevicePenalties) {
  const torch::Tensor mapping_cpu = torch::tensor({2, 5, 8}, torch::kLong);
  const VocabMap sampling_map(mapping_cpu.to(device()),
                              /*target_vocab_size=*/10);
  DraftTokenHistories histories(3);
  for (auto& history : histories) {
    history = std::make_shared<DraftTokenHistory>();
  }
  histories[0]->append({5, 2, 5});
  histories[1]->append({9});
  histories[2]->append({8});
  SamplingParameters params;
  const torch::Tensor frequencies = torch::ones({3}).to(device());
  params.frequency_penalties = frequencies;
  params.presence_penalties = torch::full({3}, 0.5F).to(device());
  params.repetition_penalties = torch::full({3}, 2.0F).to(device());
  params.unique_token_ids = torch::empty(
      {3, 1}, torch::TensorOptions().dtype(torch::kLong).device(torch::kMeta));
  DraftHistoryBatch batch;
  EXPECT_TRUE(sampling_map.remap(params, batch, histories).empty());
  EXPECT_EQ(params.unique_token_ids.device(), device());
  EXPECT_EQ(params.frequency_penalties.device(), device());
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                           torch::tensor({2, 0, 1}, torch::kInt)));
  EXPECT_TRUE(torch::equal(snapshot(params.frequency_penalties),
                           torch::tensor({1.0F, 0.0F, 1.0F})));
  EXPECT_TRUE(torch::equal(snapshot(params.presence_penalties),
                           torch::tensor({0.5F, 0.0F, 0.5F})));
  EXPECT_TRUE(torch::equal(snapshot(params.repetition_penalties),
                           torch::tensor({2.0F, 1.0F, 2.0F})));
  // Remapping must leave the target sampler's shared penalty tensor intact.
  EXPECT_TRUE(torch::equal(snapshot(frequencies), torch::ones({3})));
  EXPECT_EQ(histories[0]->ids(), std::vector<int64_t>({5, 2}));

  torch::Tensor logits = torch::full({3, 3}, 4.0F).to(device());
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
      snapshot(logits),
      torch::tensor(
          {{1.25F, 0.75F, 4.0F}, {4.0F, 4.0F, 4.0F}, {4.0F, 4.0F, 1.25F}})));
}

struct AcceptanceCase {
  int32_t speculative_tokens;
  int32_t accepted_drafts;
};

class MtpDraftInputCarryAcceptanceTest
    : public MtpDraftInputCarryTest,
      public ::testing::WithParamInterface<AcceptanceCase> {};

class TensorCopyObserver final {
 public:
  explicit TensorCopyObserver(int64_t token_count, torch::Tensor lengths = {})
      : token_count_(token_count), lengths_(std::move(lengths)) {
    CHECK(active_ == nullptr);
    active_ = this;
    callback_ = at::addThreadLocalCallback(
        at::RecordFunctionCallback(record).needsInputs(true));
  }
  ~TensorCopyObserver() {
    at::removeCallback(callback_);
    active_ = nullptr;
  }
  size_t transfers() const { return transfers_; }
  size_t device_clones() const { return device_clones_; }
  size_t device_to_host_transfers() const { return device_to_host_transfers_; }
  int64_t device_to_host_elements() const { return device_to_host_elements_; }
  size_t device_concatenations() const { return device_concatenations_; }
  size_t device_fills() const { return device_fills_; }
  size_t host_to_device_transfers() const { return host_to_device_transfers_; }
  int64_t host_to_device_bytes() const { return host_to_device_bytes_; }
  size_t device_id_updates() const { return device_id_updates_; }
  size_t device_count_updates() const { return device_count_updates_; }
  size_t device_count_casts() const { return device_count_casts_; }
  size_t host_narrows() const { return host_narrows_; }
  int64_t host_widened_elements() const { return host_widened_elements_; }
  int64_t device_widened_elements() const { return device_widened_elements_; }
  size_t device_length_updates() const { return device_length_updates_; }
  size_t device_length_fills() const { return device_length_fills_; }
  size_t device_index_selections() const { return device_index_selections_; }
  size_t device_map_arithmetic() const { return device_map_arithmetic_; }
  size_t device_mask_placeholders() const { return device_mask_placeholders_; }
  size_t device_where_outputs() const { return device_where_outputs_; }

 private:
  static std::unique_ptr<at::ObserverContext> record(
      const at::RecordFunction& function) {
    if (active_ == nullptr) {
      return nullptr;
    }
    const auto inputs = function.inputs();
    if (!inputs.empty() && inputs[0].isTensor() &&
        !inputs[0].toTensor().device().is_cpu()) {
      const std::string_view name(function.name());
      if (name == "aten::index_select") {
        ++active_->device_index_selections_;
      } else if (name == "aten::floor_divide" || name == "aten::remainder") {
        ++active_->device_map_arithmetic_;
      } else if (name == "aten::full_like" || name == "aten::zeros_like") {
        ++active_->device_mask_placeholders_;
      } else if (name == "aten::where") {
        ++active_->device_where_outputs_;
      }
    }
    if (std::string_view(function.name()) == "aten::narrow" &&
        !inputs.empty() && inputs[0].isTensor() &&
        inputs[0].toTensor().device().is_cpu()) {
      ++active_->host_narrows_;
    }
    if (std::string_view(function.name()) == "aten::index_copy_" &&
        !inputs.empty() && inputs[0].isTensor() &&
        !inputs[0].toTensor().device().is_cpu()) {
      const torch::ScalarType dtype = inputs[0].toTensor().scalar_type();
      if (dtype == torch::kLong) {
        ++active_->device_id_updates_;
      } else if (dtype == torch::kInt) {
        if (active_->is_length_view(inputs[0].toTensor())) {
          ++active_->device_length_updates_;
        } else {
          ++active_->device_count_updates_;
        }
      }
    }
    if (std::string_view(function.name()) == "aten::fill_" && !inputs.empty() &&
        inputs[0].isTensor() && !inputs[0].toTensor().device().is_cpu()) {
      ++active_->device_fills_;
      if (active_->is_length_view(inputs[0].toTensor())) {
        ++active_->device_length_fills_;
      }
    }
    if (std::string_view(function.name()) == "aten::cat" && !inputs.empty() &&
        inputs[0].isTensorList()) {
      const auto tensors = inputs[0].toTensorList();
      if (!tensors.empty() && !tensors.get(0).device().is_cpu()) {
        ++active_->device_concatenations_;
      }
    }
    if (std::string_view(function.name()) == "aten::clone" && !inputs.empty() &&
        inputs[0].isTensor() && !inputs[0].toTensor().device().is_cpu()) {
      ++active_->device_clones_;
    }
    if (std::string_view(function.name()) != "aten::copy_") {
      return nullptr;
    }
    if (inputs.size() < 2 || !inputs[0].isTensor() || !inputs[1].isTensor()) {
      return nullptr;
    }
    const auto destination = inputs[0].toTensor();
    const auto source = inputs[1].toTensor();
    if (!destination.defined() || !source.defined()) {
      return nullptr;
    }
    const bool dst_cpu = destination.device().is_cpu();
    const bool src_cpu = source.device().is_cpu();
    // Matches a copy_ whose endpoints and dtypes are all fixed, e.g. the
    // int->long widen that the draft-input path avoids on device.
    const auto typed_copy = [&](bool dst_is_cpu,
                                bool src_is_cpu,
                                torch::ScalarType dst_dtype,
                                torch::ScalarType src_dtype) {
      return dst_cpu == dst_is_cpu && src_cpu == src_is_cpu &&
             destination.scalar_type() == dst_dtype &&
             source.scalar_type() == src_dtype;
    };
    if (typed_copy(/*dst_is_cpu=*/true,
                   /*src_is_cpu=*/true,
                   torch::kLong,
                   torch::kInt)) {
      active_->host_widened_elements_ += source.numel();
    }
    if (typed_copy(/*dst_is_cpu=*/false,
                   /*src_is_cpu=*/false,
                   torch::kLong,
                   torch::kInt)) {
      active_->device_widened_elements_ += source.numel();
    }
    if (typed_copy(/*dst_is_cpu=*/false,
                   /*src_is_cpu=*/false,
                   torch::kInt,
                   torch::kLong)) {
      ++active_->device_count_casts_;
    }
    if (!dst_cpu && src_cpu && source.numel() > 0) {
      ++active_->host_to_device_transfers_;
      active_->host_to_device_bytes_ += source.numel() * source.element_size();
    }
    if (dst_cpu && !src_cpu && source.numel() > 0) {
      ++active_->device_to_host_transfers_;
      active_->device_to_host_elements_ += source.numel();
    }
    // Decode handoff is [sequence, token]; ignore flat metadata/tracing copies.
    // Prefill root consumers may use either [sequence] or [sequence, 1].
    const bool prefill_root_copy =
        active_->token_count_ == 1 && source.scalar_type() == torch::kLong;
    const bool decode_handoff_copy = active_->token_count_ > 1 &&
                                     source.dim() == 2 && source.size(0) == 1 &&
                                     (source.scalar_type() == torch::kLong ||
                                      source.scalar_type() == torch::kInt);
    if (dst_cpu && !src_cpu && source.numel() == active_->token_count_ &&
        (prefill_root_copy || decode_handoff_copy)) {
      ++active_->transfers_;
    }
    return nullptr;
  }

  bool is_length_view(const torch::Tensor& tensor) const {
    if (!lengths_.defined() || tensor.scalar_type() != lengths_.scalar_type() ||
        !tensor.is_alias_of(lengths_)) {
      return false;
    }
    // Counts and lengths can share an allocation without sharing elements.
    const int64_t offset = tensor.storage_offset() - lengths_.storage_offset();
    return offset >= 0 && offset + tensor.numel() <= lengths_.numel();
  }

  static thread_local TensorCopyObserver* active_;
  at::CallbackHandle callback_ = 0;
  int64_t token_count_;
  torch::Tensor lengths_;
  size_t transfers_ = 0;
  size_t device_clones_ = 0;
  size_t device_to_host_transfers_ = 0;
  int64_t device_to_host_elements_ = 0;
  size_t device_concatenations_ = 0;
  size_t device_fills_ = 0;
  size_t host_to_device_transfers_ = 0;
  int64_t host_to_device_bytes_ = 0;
  size_t device_id_updates_ = 0;
  size_t device_count_updates_ = 0;
  size_t device_count_casts_ = 0;
  size_t host_narrows_ = 0;
  int64_t host_widened_elements_ = 0;
  int64_t device_widened_elements_ = 0;
  size_t device_length_updates_ = 0;
  size_t device_length_fills_ = 0;
  size_t device_index_selections_ = 0;
  size_t device_map_arithmetic_ = 0;
  size_t device_mask_placeholders_ = 0;
  size_t device_where_outputs_ = 0;
};

thread_local TensorCopyObserver* TensorCopyObserver::active_ = nullptr;

TEST_F(MtpDraftInputCarryTest,
       RebuiltHistoriesUploadOnceAndKeepIndependentFields) {
  constexpr int64_t kWideToken = int64_t{1} << 40;
  auto first = DraftTokenHistory::from_host_values(
      std::array<int64_t, 2>{kWideToken, 7},
      std::array<int32_t, 2>{0, std::numeric_limits<int32_t>::max()});
  auto empty = std::make_shared<DraftTokenHistory>();
  auto last = std::make_shared<DraftTokenHistory>();
  last->append({9, 9});
  DraftTokenHistories histories{first, empty, last};
  DraftHistoryBatch batch;
  SamplingParameters params;
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.host_to_device_transfers(), 1);
    // Six int64 IDs, six int32 counts, three int32 lengths and one pad word.
    EXPECT_EQ(operations.host_to_device_bytes(), 88);
    EXPECT_EQ(operations.device_to_host_transfers(), 0);
    EXPECT_EQ(operations.device_count_casts(), 0);
  }
  EXPECT_EQ(params.unique_token_ids.scalar_type(), torch::kLong);
  EXPECT_EQ(params.unique_token_counts.scalar_type(), torch::kInt);
  EXPECT_EQ(params.unique_token_ids_lens.scalar_type(), torch::kInt);
  EXPECT_TRUE(params.unique_token_ids.is_contiguous());
  EXPECT_TRUE(params.unique_token_counts.is_contiguous());
  EXPECT_TRUE(params.unique_token_ids_lens.is_contiguous());
  EXPECT_TRUE(torch::equal(
      snapshot(params.unique_token_ids),
      torch::tensor(std::vector<int64_t>{kWideToken, 7, 0, 0, 9, 9},
                    torch::kLong)
          .view({3, 2})));
  EXPECT_TRUE(torch::equal(
      snapshot(params.unique_token_counts),
      torch::tensor({{0, std::numeric_limits<int32_t>::max()}, {0, 0}, {2, 0}},
                    torch::kInt)));
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                           torch::tensor({2, 0, 1}, torch::kInt)));

  const SamplingParameters original = params;
  first->append({11});
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.host_to_device_transfers(), 1);
  }
  EXPECT_EQ(params.unique_token_ids.size(1), 4);
  EXPECT_FALSE(params.unique_token_ids.is_alias_of(original.unique_token_ids));
  EXPECT_TRUE(torch::equal(snapshot(original.unique_token_ids_lens),
                           torch::tensor({2, 0, 1}, torch::kInt)));

  std::reverse(histories.begin(), histories.end());
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.host_to_device_transfers(), 1);
  }
  EXPECT_EQ(params.unique_token_ids.size(1), 3);
  batch.clear();
  const torch::Tensor saved_ids = snapshot(params.unique_token_ids);
  params.unique_token_counts.zero_();
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids), saved_ids));
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                           torch::tensor({1, 0, 3}, torch::kInt)));
  params.unique_token_ids_lens.fill_(1);
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids), saved_ids));
  EXPECT_EQ(params.unique_token_counts.sum().item<int64_t>(), 0);

  // An all-empty batch still uploads lengths, including odd half-word padding.
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, {empty, empty, empty}, device());
    EXPECT_EQ(operations.host_to_device_transfers(), 1);
    EXPECT_EQ(operations.host_to_device_bytes(), 16);
  }
  EXPECT_EQ(params.unique_token_ids.numel(), 0);
  EXPECT_EQ(params.unique_token_counts.numel(), 0);
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                           torch::zeros({3}, torch::kInt)));
}

TEST_F(MtpDraftInputCarryTest, EaglePackedMasksReuseImmutableDeviceMap) {
  DraftHistoryBatch batch;
  constexpr int32_t kHighBit = std::numeric_limits<int32_t>::min();
  const torch::Tensor mapping_cpu = torch::arange(31, 96, torch::kLong);
  const VocabMap sampling_map(mapping_cpu.to(device()),
                              /*target_vocab_size=*/96);
  const torch::Tensor original =
      torch::tensor({{-1, -1, -1},
                     {kHighBit, kHighBit, kHighBit},
                     {0x7fffffff, 0x7fffffff, 0x7fffffff},
                     {0, 0, 0}},
                    torch::kInt);
  const torch::Tensor expected = torch::tensor(
      {{-1, -1, 1}, {1, 1, 1}, {-2, -2, 0}, {1, 0, 0}}, torch::kInt);
  for (bool host_mask : {false, true}) {
    SCOPED_TRACE(host_mask);
    const torch::Tensor input = host_mask ? original : original.to(device());
    SamplingParameters params;
    params.filter_bitmask = input;
    {
      TensorCopyObserver operations(/*token_count=*/0);
      EXPECT_EQ(sampling_map.remap(params, batch),
                std::vector<uint8_t>({0, 0, 0, 1}));
      EXPECT_EQ(operations.device_widened_elements(), 0);
      EXPECT_LE(operations.device_to_host_elements(), 4);
      EXPECT_EQ(operations.device_map_arithmetic(), 0);
      EXPECT_EQ(operations.device_index_selections(), 1);
      EXPECT_EQ(operations.host_to_device_transfers(), host_mask ? 1 : 0);
      EXPECT_EQ(operations.host_to_device_bytes(),
                host_mask ? original.nbytes() : 0);
    }
    EXPECT_EQ(params.filter_bitmask.device(), device());
    EXPECT_EQ(params.filter_bitmask.scalar_type(), torch::kInt);
    EXPECT_TRUE(torch::equal(snapshot(params.filter_bitmask), expected));
    EXPECT_TRUE(torch::equal(snapshot(input), original));
    EXPECT_FALSE(params.filter_bitmask.is_alias_of(input));
  }
}

TEST_F(MtpDraftInputCarryTest, PackedEagleMasksAvoidShadowedDenseTransfers) {
  DraftHistoryBatch batch;
  const VocabMap sampling_map(torch::arange(31, 95, torch::kLong).to(device()),
                              /*target_vocab_size=*/96);
  SamplingParameters source;
  source.filter_mask =
      torch::full({2, 96}, -std::numeric_limits<float>::infinity());
  source.filter_bitmask =
      torch::tensor({{-1, -1, -1}, {0, 0, 0}}, torch::kInt).to(device());
  SamplingParameters params = source;
  {
    TensorCopyObserver operations(/*token_count=*/0);
    EXPECT_EQ(sampling_map.remap(params, batch), std::vector<uint8_t>({0, 1}));
    EXPECT_EQ(operations.host_to_device_transfers(), 0);
    EXPECT_EQ(operations.device_index_selections(), 1);
    EXPECT_EQ(operations.device_to_host_elements(), 2);
  }
  EXPECT_FALSE(params.filter_mask.defined());
  EXPECT_TRUE(torch::equal(snapshot(params.filter_bitmask),
                           torch::tensor({{-1, -1}, {1, 0}}, torch::kInt)));
  EXPECT_TRUE(source.filter_mask.defined());
  EXPECT_TRUE(torch::equal(
      source.filter_mask,
      torch::full({2, 96}, -std::numeric_limits<float>::infinity())));
  EXPECT_TRUE(
      torch::equal(snapshot(source.filter_bitmask),
                   torch::tensor({{-1, -1, -1}, {0, 0, 0}}, torch::kInt)));

  params.selected_token_idxes = torch::arange(
      2, torch::TensorOptions().dtype(torch::kInt).device(device()));
  params.sample_idxes = params.selected_token_idxes;
  torch::Tensor logits =
      torch::arange(
          64, torch::TensorOptions().dtype(torch::kFloat).device(device()))
          .unsqueeze(/*dim=*/0)
          .repeat({2, 1});
  const SampleOutput sampled = Sampler().forward(logits, params);
  EXPECT_EQ(tensor_to_vector<int64_t>(sampled.next_tokens),
            std::vector<int64_t>({63, 0}));
}

TEST_F(MtpDraftInputCarryTest, InvalidEagleMasksReusePrivateOutputStorage) {
  DraftHistoryBatch batch;
  const float forbidden = -std::numeric_limits<float>::infinity();
  const VocabMap sampling_map(torch::arange(31, 96, torch::kLong).to(device()),
                              /*target_vocab_size=*/96);
  for (bool dense : {false, true}) {
    for (bool all_invalid : {false, true}) {
      SCOPED_TRACE(dense);
      SCOPED_TRACE(all_invalid);
      torch::Tensor host_mask = dense ? torch::full({4, 96}, forbidden)
                                      : torch::zeros({4, 3}, torch::kInt);
      torch::Tensor expected = dense ? torch::full({4, 65}, forbidden)
                                     : torch::zeros({4, 3}, torch::kInt);
      expected.select(/*dim=*/1, /*index=*/0).fill_(dense ? 0 : 1);
      if (dense) {
        host_mask.select(/*dim=*/0, /*index=*/2).fill_(-3);
      }
      if (!all_invalid) {
        for (int64_t row : {1, 3}) {
          host_mask.select(/*dim=*/0, row).fill_(dense ? 0 : -1);
          expected.select(/*dim=*/0, row).fill_(dense ? 0 : -1);
        }
        if (dense) {
          host_mask.select(/*dim=*/0, /*index=*/3)
              .select(/*dim=*/0, /*index=*/31)
              .fill_(2);
          expected.select(/*dim=*/0, /*index=*/3)
              .select(/*dim=*/0, /*index=*/0)
              .fill_(2);
        } else {
          expected.select(/*dim=*/0, /*index=*/1)
              .select(/*dim=*/0, /*index=*/2)
              .fill_(1);
          expected.select(/*dim=*/0, /*index=*/3)
              .select(/*dim=*/0, /*index=*/2)
              .fill_(1);
        }
      }
      const torch::Tensor input = host_mask.to(device());
      SamplingParameters params;
      const auto member = dense ? &SamplingParameters::filter_mask
                                : &SamplingParameters::filter_bitmask;
      params.*member = input;
      {
        TensorCopyObserver operations(/*token_count=*/0);
        EXPECT_EQ(sampling_map.remap(params, batch),
                  all_invalid ? std::vector<uint8_t>({1, 1, 1, 1})
                              : std::vector<uint8_t>({1, 0, 1, 0}));
        EXPECT_EQ(operations.device_mask_placeholders(), 0);
        EXPECT_EQ(operations.device_where_outputs(), 0);
        EXPECT_EQ(operations.host_to_device_transfers(), 0);
        EXPECT_EQ(operations.device_to_host_elements(), 4);
      }
      EXPECT_TRUE(torch::equal(snapshot(params.*member), expected));
      EXPECT_FALSE((params.*member).is_alias_of(input));
      EXPECT_TRUE(torch::equal(snapshot(input), host_mask));
    }
  }
}

TEST_F(MtpDraftInputCarryTest, SelectedConfidenceAvoidsDeviceTokenCasts) {
  SampleOutput sample;
  sample.next_tokens = torch::tensor({1, 0}, torch::kInt).to(device());
  sample.probs = torch::tensor({0.25F, 0.75F}).to(device());
  torch::Tensor confidence;
  {
    TensorCopyObserver operations(/*token_count=*/0);
    confidence = select_input_confidence(sample, {}, {});
    EXPECT_EQ(operations.device_widened_elements(), 0);
    EXPECT_EQ(operations.host_to_device_transfers(), 0);
    EXPECT_EQ(operations.device_to_host_transfers(), 0);
    EXPECT_TRUE(confidence.is_alias_of(sample.probs));
  }
  EXPECT_TRUE(
      torch::equal(snapshot(confidence), torch::tensor({0.25F, 0.75F})));
  EXPECT_EQ(tensor_to_vector<int64_t>(sample.next_tokens),
            std::vector<int64_t>({1, 0}));

  sample.probs = torch::tensor({{0.25F, 0.75F}, {0.5F, 0.5F}}).to(device());
  {
    TensorCopyObserver operations(/*token_count=*/0);
    confidence = select_input_confidence(sample, {}, {});
    EXPECT_EQ(operations.device_widened_elements(), 2);
  }
  EXPECT_TRUE(torch::equal(snapshot(confidence), torch::tensor({0.75F, 0.5F})));
}

TEST_F(MtpDraftInputCarryTest, SeedPolicySurroundsVocabularyOnlyOverride) {
  auto calls = std::make_shared<std::vector<ModelCall>>();
  RemappingDraftInputWorker worker(
      device(), make_options(/*speculative_tokens=*/2), calls);
  for (auto mode :
       {DraftSamplingMode::GREEDY, DraftSamplingMode::PROBABILISTIC}) {
    for (bool all_greedy : {false, true}) {
      for (bool require_confidence : {false, true}) {
        SCOPED_TRACE(static_cast<int>(mode));
        SCOPED_TRACE(all_greedy);
        SCOPED_TRACE(require_confidence);
        ForwardOutput output;
        auto& sample = output.sample_output;
        sample.next_tokens = torch::tensor({1}, torch::kLong).to(device());
        sample.embeddings = torch::zeros({1, kHiddenSize}).to(device());
        sample.probs = torch::tensor({{0.25F, 0.75F}}).to(device());
        SamplingParameters params;
        params.all_greedy_sample = all_greedy;
        params.all_random_sample = !all_greedy;

        const int32_t prior_remap_calls = worker.remap_calls();
        const auto seed =
            worker.finish_draft_input(output, params, mode, require_confidence);
        EXPECT_EQ(worker.remap_calls(), prior_remap_calls + 1);
        // The hook sees q only in probabilistic mode; the all-greedy drop
        // stays with finalize_mixed_proposal_q.
        EXPECT_EQ(worker.remapped_probs(),
                  mode == DraftSamplingMode::PROBABILISTIC);
        EXPECT_EQ(tensor_to_vector<int64_t>(seed.token_ids),
                  std::vector<int64_t>({6}));
        EXPECT_EQ(seed.selected_probs.defined(), require_confidence);
        if (require_confidence) {
          // Confidence is gathered at draft token 1 before q is
          // dropped/remapped.
          EXPECT_TRUE(torch::equal(snapshot(seed.selected_probs),
                                   torch::tensor({0.75F})));
        }
        if (mode == DraftSamplingMode::GREEDY) {
          EXPECT_FALSE(seed.probs.defined());
          EXPECT_FALSE(seed.point_mass_q);
        } else if (all_greedy) {
          EXPECT_FALSE(seed.probs.defined());
          EXPECT_TRUE(seed.point_mass_q);
        } else {
          auto expected = torch::zeros({1, kVocabSize});
          expected[0][5] = 0.25F;
          expected[0][6] = 0.75F;
          EXPECT_TRUE(torch::equal(snapshot(seed.probs), expected));
          EXPECT_FALSE(seed.point_mass_q);
        }
      }
    }
  }
}

TEST_F(MtpDraftInputCarryTest,
       MixedSeedNormalizesGreedyProposalAfterRemapping) {
  auto calls = std::make_shared<std::vector<ModelCall>>();
  RemappingDraftInputWorker worker(
      device(), make_options(/*speculative_tokens=*/2), calls);
  ForwardOutput output;
  auto& sample = output.sample_output;
  sample.next_tokens = torch::tensor({1, 0}, torch::kLong).to(device());
  sample.embeddings = torch::zeros({2, kHiddenSize}).to(device());
  sample.probs = torch::tensor({{0.25F, 0.75F}, {0.6F, 0.4F}}).to(device());
  SamplingParameters params;
  params.all_greedy_sample = false;
  params.all_random_sample = false;
  params.do_sample = torch::tensor({false, true}, torch::kBool).to(device());

  const DraftInput seed =
      worker.finish_draft_input(output,
                                params,
                                DraftSamplingMode::PROBABILISTIC,
                                /*require_confidence=*/false);
  EXPECT_EQ(worker.remap_calls(), 1);
  EXPECT_TRUE(worker.remapped_probs());
  EXPECT_FALSE(seed.point_mass_q);
  EXPECT_EQ(tensor_to_vector<int64_t>(seed.token_ids),
            std::vector<int64_t>({6, 5}));
  torch::Tensor expected = torch::zeros({2, kVocabSize});
  expected[0][6] = 1.0F;
  expected[1][5] = 0.6F;
  expected[1][6] = 0.4F;
  EXPECT_TRUE(torch::allclose(snapshot(seed.probs), expected));
}

TEST_F(MtpDraftInputCarryTest,
       SelectedGreedyProbabilityDoesNotBecomeProposalQ) {
  auto calls = std::make_shared<std::vector<ModelCall>>();
  RemappingDraftInputWorker worker(
      device(), make_options(/*speculative_tokens=*/2), calls);
  ForwardOutput output;
  SampleOutput& sample = output.sample_output;
  sample.next_tokens = torch::tensor({1}, torch::kLong).to(device());
  sample.embeddings = torch::zeros({1, kHiddenSize}).to(device());
  // Sampler::forward returns a 1-D selected confidence for all-greedy input.
  sample.probs = torch::tensor({0.75F}).to(device());
  SamplingParameters params;
  params.all_greedy_sample = true;

  const DraftInput seed =
      worker.finish_draft_input(output,
                                params,
                                DraftSamplingMode::PROBABILISTIC,
                                /*require_confidence=*/true);
  // Probabilistic mode keeps the 1-D greedy confidence through the remap
  // hook; finalize_mixed_proposal_q drops it as the all-greedy proposal q.
  EXPECT_TRUE(worker.remapped_probs());
  EXPECT_TRUE(
      torch::equal(snapshot(seed.selected_probs), torch::tensor({0.75F})));
  EXPECT_FALSE(seed.probs.defined());
  EXPECT_TRUE(seed.point_mass_q);
}

TEST_F(MtpDraftInputCarryTest, SeedPolicyHandlesMissingProbsBeforeRemapping) {
  auto calls = std::make_shared<std::vector<ModelCall>>();
  RemappingDraftInputWorker worker(
      device(), make_options(/*speculative_tokens=*/2), calls);
  for (auto mode :
       {DraftSamplingMode::GREEDY, DraftSamplingMode::PROBABILISTIC}) {
    for (bool all_greedy : {false, true}) {
      // Random proposals require q; only greedy sampling may omit it.
      if (mode == DraftSamplingMode::PROBABILISTIC && !all_greedy) {
        continue;
      }
      for (bool require_confidence : {false, true}) {
        SCOPED_TRACE(static_cast<int>(mode));
        SCOPED_TRACE(all_greedy);
        SCOPED_TRACE(require_confidence);
        ForwardOutput output;
        auto& sample = output.sample_output;
        sample.next_tokens = torch::tensor({1, 0}, torch::kLong).to(device());
        sample.embeddings = torch::zeros({2, kHiddenSize}).to(device());
        SamplingParameters params;
        params.all_greedy_sample = all_greedy;
        params.all_random_sample = !all_greedy;
        if (require_confidence) {
          // Sampling selects rows 2 and 0, not the first two logits rows.
          output.logits =
              torch::tensor({{2.0F, 0.0F}, {8.0F, -8.0F}, {0.0F, 1.0F}})
                  .to(device());
          params.sample_idxes = torch::tensor({2, 0}, torch::kInt).to(device());
        }
        const int32_t prior_remap_calls = worker.remap_calls();
        const auto seed =
            worker.finish_draft_input(output, params, mode, require_confidence);
        EXPECT_EQ(worker.remap_calls(), prior_remap_calls + 1);
        EXPECT_FALSE(worker.remapped_probs());
        EXPECT_EQ(tensor_to_vector<int64_t>(seed.token_ids),
                  std::vector<int64_t>({6, 5}));
        EXPECT_TRUE(seed.hidden_states.is_alias_of(sample.embeddings));
        EXPECT_EQ(seed.selected_probs.defined(), require_confidence);
        if (require_confidence) {
          // Confidence uses draft ids before their non-identity remapping.
          EXPECT_TRUE(torch::allclose(snapshot(seed.selected_probs),
                                      torch::tensor({0.7310586F, 0.8807971F})));
          EXPECT_TRUE(torch::equal(
              snapshot(output.logits),
              torch::tensor({{2.0F, 0.0F}, {8.0F, -8.0F}, {0.0F, 1.0F}})));
          EXPECT_EQ(tensor_to_vector<int64_t>(params.sample_idxes),
                    std::vector<int64_t>({2, 0}));
        } else {
          // Local fixed-width seeds must not depend on logits or row indices.
          EXPECT_FALSE(output.logits.defined());
          EXPECT_FALSE(params.sample_idxes.defined());
        }
        if (mode == DraftSamplingMode::GREEDY) {
          EXPECT_FALSE(seed.probs.defined());
          EXPECT_FALSE(seed.point_mass_q);
        } else {
          EXPECT_FALSE(seed.probs.defined());
          EXPECT_TRUE(seed.point_mass_q);
        }
      }
    }
  }
}

TEST_F(MtpDraftInputCarryTest, SeedPenaltiesReuseTheAcceptedTokenHostCopy) {
  for (bool overlap : {false, true}) {
    for (bool penalty : {false, true}) {
      SCOPED_TRACE(overlap);
      SCOPED_TRACE(penalty);
      auto calls = std::make_shared<std::vector<ModelCall>>();
      calls->reserve(8);
      auto options = make_options(/*speculative_tokens=*/3);
      options.enable_schedule_overlap(overlap);
      DraftInputCarryWorker worker(device(), options, calls);
      auto prefill = make_prefill(device());
      auto decode = make_decode(device());
      if (penalty) {
        prefill.sampling_params.repetition_penalties = torch::tensor({1.2F});
        prefill.sampling_params.unique_token_ids =
            torch::tensor({{10, 11, 12}}, torch::kLong);
        prefill.sampling_params.unique_token_counts =
            torch::ones({1, 3}, torch::kInt);
        prefill.sampling_params.unique_token_ids_lens =
            torch::tensor({3}, torch::kInt);
        prefill.sampling_params =
            prefill.sampling_params.to(device(), torch::kFloat);
        decode.sampling_params.repetition_penalties =
            prefill.sampling_params.repetition_penalties;
      }
      {
        TensorCopyObserver copies(/*token_count=*/1);
        const auto output = worker.step(prefill);
        ASSERT_TRUE(output.has_value());
        EXPECT_TRUE(output->sample_output.next_tokens.device().is_cpu());
        EXPECT_EQ(copies.transfers(), 1);
      }
      worker.set_target_predictions({14, 15, 16, 17});
      {
        TensorCopyObserver copies(/*token_count=*/4);
        const auto output = worker.step(decode);
        ASSERT_TRUE(output.has_value());
        EXPECT_EQ(copies.transfers(), 1);
      }
      EXPECT_TRUE(
          worker.release_idle_request_state_async({kEmbeddingId}).get());
    }
  }
}

TEST_F(MtpDraftInputCarryTest, PackedDraftMetadataUsesOneDeviceSnapshot) {
  LlmForwardInput source;
  auto& attention = source.input_params.attention;
  attention.host.q_seq_lens = {1, 2};
  attention.host.q_cu_seq_lens = {0, 1, 3};
  attention.host.kv_seq_lens = {5, 6};
  attention.host.new_cache_slots = {7, 8, 9};
  attention.host.block_tables = torch::tensor({{1, 2}, {3, 4}}, torch::kInt);
  attention.reserve_device_buffer_capacity(/*capacity=*/4096, device());
  ASSERT_TRUE(attention.rebuild_device_buffer(device()));
  LlmForwardInput snapshot = source.clone();
  {
    TensorCopyObserver copies(/*token_count=*/0);
    specBuilder::clone_mtp_draft_metadata(snapshot);
    EXPECT_EQ(copies.device_clones(), 1);
  }
  attention.attention_device_buffer.zero_();
  attention.host.q_seq_lens = {3, 4};
  attention.host.q_cu_seq_lens = {0, 3, 7};
  ASSERT_TRUE(attention.rebuild_device_buffer(
      device(), {}, LlmAttentionInput::BufferReusePolicy::GROWABLE));
  source = LlmForwardInput();

  const auto& result = snapshot.input_params.attention.device;
  EXPECT_EQ(tensor_to_vector<int64_t>(result.q_seq_lens),
            (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(tensor_to_vector<int64_t>(result.q_cu_seq_lens),
            (std::vector<int64_t>{0, 1, 3}));
  EXPECT_EQ(tensor_to_vector<int64_t>(result.kv_seq_lens),
            (std::vector<int64_t>{5, 6}));
  EXPECT_EQ(tensor_to_vector<int64_t>(result.new_cache_slots),
            (std::vector<int64_t>{7, 8, 9}));
  EXPECT_EQ(tensor_to_vector<int64_t>(result.block_tables),
            (std::vector<int64_t>{1, 2, 3, 4}));
  EXPECT_LT(result.q_seq_lens.storage().nbytes(), 4096);
}

TEST_F(MtpDraftInputCarryTest, IncrementalPenaltyProjectionRunsOnDevice) {
  SamplingParameters params;
  params.repetition_penalties = torch::full({2}, 1.2F).to(device());
  DraftHistoryBatch batch;
  auto sampling = prepare_draft_input_sampling(
      params,
      {},
      torch::tensor({{0, 2, -1}, {1, 3, 4}}, torch::kLong).to(device()),
      device(),
      batch);
  const void* storage = sampling.sampling_params.unique_token_ids.data_ptr();
  sampling = prepare_draft_input_sampling(
      params,
      {},
      torch::tensor({{2, 2}, {1, 4}}, torch::kLong).to(device()),
      device(),
      batch,
      sampling.histories);
  EXPECT_EQ(sampling.sampling_params.unique_token_ids.data_ptr(), storage);
  EXPECT_TRUE(
      torch::equal(snapshot(sampling.sampling_params.unique_token_counts),
                   torch::tensor({{1, 3, 0}, {2, 1, 2}}, torch::kInt)));
  sampling.histories[0]->append({5, 6});
  batch.apply(params, sampling.histories, device());
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                           torch::tensor({4, 3}, torch::kInt)));
  EXPECT_TRUE(torch::equal(
      snapshot(params.unique_token_ids),
      torch::tensor({{0, 2, 5, 6, 6, 6}, {1, 3, 4, 4, 4, 4}}, torch::kLong)));
}

TEST_F(MtpDraftInputCarryTest, ProjectionWritesOnlyChangedLengthsAndPadding) {
  DraftTokenHistories histories{std::make_shared<DraftTokenHistory>(),
                                std::make_shared<DraftTokenHistory>(),
                                std::make_shared<DraftTokenHistory>()};
  histories[0]->append({1, 2});
  histories[1]->append({3, 4, 5, 6});
  SamplingParameters params;
  DraftHistoryBatch batch;
  batch.apply(params, histories, device());
  const void* storage = params.unique_token_ids.data_ptr();

  histories[0]->append({1});
  histories[1]->append({3});
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_fills(), 0);
    EXPECT_EQ(operations.device_to_host_transfers(), 0);
  }
  EXPECT_EQ(params.unique_token_ids.data_ptr(), storage);
  EXPECT_TRUE(torch::equal(
      snapshot(params.unique_token_counts),
      torch::tensor({{2, 1, 0, 0}, {2, 1, 1, 1}, {0, 0, 0, 0}}, torch::kInt)));

  // Repeating the last token keeps the padding suffix at zero counts.
  histories[0]->append({2});
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_fills(), 1);
  }
  EXPECT_TRUE(torch::equal(
      snapshot(params.unique_token_counts),
      torch::tensor({{2, 2, 0, 0}, {2, 1, 1, 1}, {0, 0, 0, 0}}, torch::kInt)));

  // Skipped revisions still compare against the last applied projection.
  histories[0]->append({1});
  histories[0]->append({1});
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_fills(), 0);
  }
  histories[0]->append({2});
  histories[0]->append({1});
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_fills(), 1);
  }
  EXPECT_TRUE(torch::equal(
      snapshot(params.unique_token_counts),
      torch::tensor({{5, 3, 0, 0}, {2, 1, 1, 1}, {0, 0, 0, 0}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(
      snapshot(params.unique_token_ids),
      torch::tensor({{1, 2, 2, 2}, {3, 4, 5, 6}, {0, 0, 0, 0}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                           torch::tensor({2, 4, 0}, torch::kInt)));
}

TEST_F(MtpDraftInputCarryTest, ProjectionCoalescesMatchingPaddingRuns) {
  DraftTokenHistories histories;
  histories.reserve(5);
  for (int64_t row = 0; row < 5; ++row) {
    auto history = std::make_shared<DraftTokenHistory>();
    history->append({10 * row, 10 * row + 1});
    histories.emplace_back(std::move(history));
  }
  histories.back()->append({42, 43, 44, 45, 46, 47});
  SamplingParameters params;
  DraftHistoryBatch batch;
  batch.apply(params, histories, device());
  const auto expect_projection = [&]() {
    DraftHistoryBatch rebuilt;
    SamplingParameters expected;
    rebuilt.apply(expected, histories, torch::kCPU);
    EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids),
                             expected.unique_token_ids));
    EXPECT_TRUE(torch::equal(snapshot(params.unique_token_counts),
                             expected.unique_token_counts));
    EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                             expected.unique_token_ids_lens));
  };

  for (int64_t row = 0; row < 4; ++row) {
    histories[row]->append({10 * row + 1});
  }
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_fills(), 1);
    EXPECT_EQ(operations.host_to_device_transfers(), 1);
    EXPECT_EQ(operations.host_to_device_bytes(), 6 * sizeof(int64_t));
    EXPECT_EQ(operations.device_to_host_transfers(), 0);
  }
  expect_projection();

  // Matching new tail IDs and counts form independent padding runs.
  for (int64_t row = 0; row < 4; ++row) {
    histories[row]->append({99});
  }
  {
    TensorCopyObserver operations(/*token_count=*/0,
                                  params.unique_token_ids_lens);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_fills(), 2);
    EXPECT_EQ(operations.device_length_updates(), 1);
    EXPECT_EQ(operations.device_length_fills(), 0);
  }
  expect_projection();

  // Zero-count padding never changes with the tail count, so only the ID
  // run re-fills here.
  histories[0]->append({99, 99});
  for (int64_t row = 1; row < 4; ++row) {
    histories[row]->append({99});
  }
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_fills(), 1);
  }
  expect_projection();

  // Two matching rows retain direct fills instead of a strided rectangle.
  histories[0]->append({99});
  histories[1]->append({99, 99});
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_fills(), 2);
  }
  expect_projection();
}

TEST_F(MtpDraftInputCarryTest, PaddingRunsRespectRowAndColumnBoundaries) {
  DraftTokenHistories histories;
  histories.reserve(7);
  for (int64_t row = 0; row < 7; ++row) {
    auto history = std::make_shared<DraftTokenHistory>();
    history->append({10 * row, 10 * row + 1});
    if (row >= 3) {
      history->append({10 * row + 2, 10 * row + 3});
    }
    histories.emplace_back(std::move(history));
  }
  histories.back()->append({64, 65, 66, 67});
  SamplingParameters params;
  DraftHistoryBatch batch;
  batch.apply(params, histories, device());
  const auto expect_projection = [&]() {
    DraftHistoryBatch rebuilt;
    SamplingParameters expected;
    rebuilt.apply(expected, histories, torch::kCPU);
    EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids),
                             expected.unique_token_ids));
    EXPECT_TRUE(torch::equal(snapshot(params.unique_token_counts),
                             expected.unique_token_counts));
    EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                             expected.unique_token_ids_lens));
  };

  // Row 1 is untouched; rows 2 and 3 have different valid prefix lengths.
  for (int64_t row : {0, 2, 3, 4, 5}) {
    histories[row]->append({histories[row]->ids().back()});
  }
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_fills(), 3);
    EXPECT_EQ(operations.device_to_host_transfers(), 0);
  }
  expect_projection();
  EXPECT_TRUE(torch::equal(
      snapshot(params.unique_token_counts.select(/*dim=*/0, /*index=*/1)),
      torch::tensor({1, 1, 0, 0, 0, 0, 0, 0}, torch::kInt)));

  std::swap(histories[1], histories[3]);
  batch.apply(params, histories, device());
  for (int64_t row = 0; row < 6; ++row) {
    histories[row]->append({histories[row]->ids().back()});
  }
  batch.apply(params, histories, device());
  expect_projection();
}

TEST_F(MtpDraftInputCarryTest,
       ProjectionRefreshesPaddingAfterGrowthAndRebatch) {
  DraftTokenHistories histories{std::make_shared<DraftTokenHistory>(),
                                std::make_shared<DraftTokenHistory>()};
  histories[1]->append({3, 4, 5, 6});
  SamplingParameters params;
  DraftHistoryBatch batch;
  batch.apply(params, histories, device());
  const void* storage = params.unique_token_ids.data_ptr();

  // An empty row becomes nonempty without growing the batch allocation.
  histories[0]->append({1, 2});
  batch.apply(params, histories, device());
  EXPECT_EQ(params.unique_token_ids.data_ptr(), storage);
  EXPECT_TRUE(
      torch::equal(snapshot(params.unique_token_ids),
                   torch::tensor({{1, 2, 2, 2}, {3, 4, 5, 6}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                           torch::tensor({2, 4}, torch::kInt)));

  histories[0]->append({7});
  batch.apply(params, histories, device());
  EXPECT_TRUE(
      torch::equal(snapshot(params.unique_token_ids),
                   torch::tensor({{1, 2, 7, 7}, {3, 4, 5, 6}}, torch::kLong)));

  // Reordering invalidates row metadata even when the allocation shape fits.
  std::swap(histories[0], histories[1]);
  batch.apply(params, histories, device());
  histories[1]->append({7, 7});
  batch.apply(params, histories, device());
  EXPECT_TRUE(
      torch::equal(snapshot(params.unique_token_counts),
                   torch::tensor({{1, 1, 1, 1}, {1, 1, 3, 0}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                           torch::tensor({4, 3}, torch::kInt)));

  // Capacity growth creates a new padded suffix with the current tail count.
  histories[1]->append({8, 9});
  batch.apply(params, histories, device());
  EXPECT_EQ(params.unique_token_ids.size(1), 8);
  EXPECT_TRUE(torch::equal(
      snapshot(params.unique_token_ids),
      torch::tensor({{3, 4, 5, 6, 6, 6, 6, 6}, {1, 2, 7, 8, 9, 9, 9, 9}},
                    torch::kLong)));
  histories[1]->append({9});
  batch.apply(params, histories, device());
  EXPECT_TRUE(torch::equal(
      snapshot(params.unique_token_counts),
      torch::tensor({{1, 1, 1, 1, 0, 0, 0, 0}, {1, 1, 3, 1, 2, 0, 0, 0}},
                    torch::kInt)));
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                           torch::tensor({4, 5}, torch::kInt)));

  batch.clear();
  batch.apply(params, {}, device());
  EXPECT_FALSE(params.unique_token_ids.defined());
  EXPECT_FALSE(params.unique_token_counts.defined());
  EXPECT_FALSE(params.unique_token_ids_lens.defined());
  batch.apply(params, histories, device());
  EXPECT_EQ(params.unique_token_ids.size(1), 5);
  EXPECT_TRUE(torch::equal(
      snapshot(params.unique_token_counts),
      torch::tensor({{1, 1, 1, 1, 0}, {1, 1, 3, 1, 2}}, torch::kInt)));
}

TEST_F(MtpDraftInputCarryTest, RepeatedHistoryCountsDoNotUploadTokenIds) {
  DraftTokenHistories histories{std::make_shared<DraftTokenHistory>(),
                                std::make_shared<DraftTokenHistory>()};
  histories[0]->append({1, 2});
  histories[1]->append({3, 4});
  SamplingParameters params;
  DraftHistoryBatch batch;
  batch.apply(params, histories, device());
  const void* storage = params.unique_token_ids.data_ptr();

  histories[0]->append({1, 1, 2});
  histories[1]->append({3});
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_id_updates(), 0);
    EXPECT_EQ(operations.device_count_updates(), 1);
    EXPECT_EQ(operations.host_to_device_transfers(), 1);
    // Three changed counts need only their indices and values.
    EXPECT_EQ(operations.host_to_device_bytes(),
              3 * sizeof(int64_t) + 4 * sizeof(int32_t));
    EXPECT_EQ(operations.device_count_casts(), 0);
    EXPECT_EQ(operations.device_to_host_transfers(), 0);
  }
  EXPECT_EQ(params.unique_token_ids.data_ptr(), storage);
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids),
                           torch::tensor({{1, 2}, {3, 4}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_counts),
                           torch::tensor({{3, 2}, {2, 1}}, torch::kInt)));
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_id_updates(), 0);
    EXPECT_EQ(operations.device_count_updates(), 0);
    EXPECT_EQ(operations.host_to_device_transfers(), 0);
  }

  histories[0]->append({1});
  histories[0]->append({2});
  histories[1]->append({4});
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_id_updates(), 0);
    EXPECT_EQ(operations.host_to_device_bytes(),
              3 * sizeof(int64_t) + 4 * sizeof(int32_t));
    EXPECT_EQ(operations.device_count_casts(), 0);
  }
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_counts),
                           torch::tensor({{4, 3}, {2, 2}}, torch::kInt)));
}

TEST_F(MtpDraftInputCarryTest, MixedHistoryDeltasKeepNewIdsAlignedWithCounts) {
  DraftTokenHistories histories{std::make_shared<DraftTokenHistory>(),
                                std::make_shared<DraftTokenHistory>(),
                                std::make_shared<DraftTokenHistory>()};
  histories[0]->append({10, 20});
  histories[1]->append({30, 40});
  histories[2]->append({50, 60, 70, 80, 90, 100});
  SamplingParameters params;
  DraftHistoryBatch batch;
  batch.apply(params, histories, device());

  histories[0]->append({10, 11, 12, 11});
  histories[1]->append({41});
  histories[1]->append({30, 42});
  {
    TensorCopyObserver operations(/*token_count=*/0,
                                  params.unique_token_ids_lens);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_id_updates(), 1);
    EXPECT_EQ(operations.device_count_updates(), 1);
    EXPECT_EQ(operations.device_length_updates(), 0);
    EXPECT_EQ(operations.device_length_fills(), 2);
    EXPECT_EQ(operations.host_to_device_transfers(), 1);
    // A small group keeps direct length writes without additional payload.
    EXPECT_EQ(operations.host_to_device_bytes(),
              (7 + 4) * sizeof(int64_t) + 8 * sizeof(int32_t));
    EXPECT_EQ(operations.device_count_casts(), 0);
    EXPECT_EQ(operations.device_to_host_transfers(), 0);
  }
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids),
                           torch::tensor({{10, 20, 11, 12, 12, 12},
                                          {30, 40, 41, 42, 42, 42},
                                          {50, 60, 70, 80, 90, 100}},
                                         torch::kLong)));
  EXPECT_TRUE(torch::equal(
      snapshot(params.unique_token_counts),
      torch::tensor(
          {{2, 1, 2, 1, 0, 0}, {2, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 1, 1}},
          torch::kInt)));
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                           torch::tensor({4, 4, 6}, torch::kInt)));

  histories[0]->append({13});
  histories[1]->append({43});
  {
    TensorCopyObserver operations(/*token_count=*/0,
                                  params.unique_token_ids_lens);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_id_updates(), 1);
    EXPECT_EQ(operations.device_length_updates(), 0);
    EXPECT_EQ(operations.device_length_fills(), 2);
    EXPECT_EQ(operations.host_to_device_bytes(),
              2 * (2 * sizeof(int64_t) + sizeof(int32_t)));
    EXPECT_EQ(operations.device_count_casts(), 0);
  }
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids),
                           torch::tensor({{10, 20, 11, 12, 13, 13},
                                          {30, 40, 41, 42, 43, 43},
                                          {50, 60, 70, 80, 90, 100}},
                                         torch::kLong)));
}

TEST_F(MtpDraftInputCarryTest, ProjectionCoalescesGrowingRowLengths) {
  DraftTokenHistories histories{std::make_shared<DraftTokenHistory>(),
                                std::make_shared<DraftTokenHistory>(),
                                std::make_shared<DraftTokenHistory>(),
                                std::make_shared<DraftTokenHistory>(),
                                std::make_shared<DraftTokenHistory>(),
                                std::make_shared<DraftTokenHistory>()};
  histories[1]->append({20});
  histories[2]->append({30, 31});
  histories[3]->append({40});
  histories[4]->append({50, 51});
  histories[5]->append({60, 61, 62, 63, 64, 65, 66, 67});
  SamplingParameters params;
  DraftHistoryBatch batch;
  batch.apply(params, histories, device());
  const void* lengths_storage = params.unique_token_ids_lens.data_ptr();
  const auto expect_projection = [&]() {
    DraftHistoryBatch rebuilt;
    SamplingParameters expected;
    rebuilt.apply(expected, histories, torch::kCPU);
    EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids),
                             expected.unique_token_ids));
    EXPECT_TRUE(torch::equal(snapshot(params.unique_token_counts),
                             expected.unique_token_counts));
    EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                             expected.unique_token_ids_lens));
  };

  // Grow four rows alongside an unchanged-length row and a capacity anchor.
  histories[0]->append({10});
  histories[1]->append({20, 21, 22, 21});
  histories[2]->append({30});
  histories[3]->append({41});
  histories[4]->append({52});
  {
    TensorCopyObserver operations(/*token_count=*/0,
                                  params.unique_token_ids_lens);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_length_updates(), 1);
    EXPECT_EQ(operations.device_length_fills(), 0);
    EXPECT_EQ(operations.host_to_device_transfers(), 1);
    EXPECT_EQ(operations.host_to_device_bytes(),
              16 * sizeof(int64_t) + 12 * sizeof(int32_t));
    EXPECT_EQ(operations.device_to_host_transfers(), 0);
  }
  EXPECT_EQ(params.unique_token_ids_lens.data_ptr(), lengths_storage);
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                           torch::tensor({1, 3, 2, 2, 3, 8}, torch::kInt)));
  expect_projection();

  // Sparse row IDs and accumulated revisions must use the final lengths.
  histories[0]->append({11});
  histories[2]->append({32});
  histories[2]->append({33});
  histories[3]->append({42});
  histories[4]->append({53});
  {
    TensorCopyObserver operations(/*token_count=*/0,
                                  params.unique_token_ids_lens);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_length_updates(), 1);
    EXPECT_EQ(operations.device_length_fills(), 0);
  }
  expect_projection();

  // A single growing row retains the direct write and adds no length payload.
  histories[1]->append({23});
  {
    TensorCopyObserver operations(/*token_count=*/0,
                                  params.unique_token_ids_lens);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_length_updates(), 0);
    EXPECT_EQ(operations.device_length_fills(), 1);
    EXPECT_EQ(operations.host_to_device_bytes(), 3 * sizeof(int64_t));
  }
  expect_projection();
  {
    TensorCopyObserver operations(/*token_count=*/0,
                                  params.unique_token_ids_lens);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_length_updates(), 0);
    EXPECT_EQ(operations.device_length_fills(), 0);
    EXPECT_EQ(operations.host_to_device_transfers(), 0);
  }

  std::swap(histories[0], histories[2]);
  batch.apply(params, histories, device());
  histories[0]->append({34});
  histories[1]->append({24});
  histories[3]->append({43});
  histories[4]->append({54});
  {
    TensorCopyObserver operations(/*token_count=*/0,
                                  params.unique_token_ids_lens);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_length_updates(), 1);
    EXPECT_EQ(operations.device_length_fills(), 0);
  }
  expect_projection();
}

TEST_F(MtpDraftInputCarryTest,
       NativeCountPayloadPreservesOddSizesAndIntegerLimits) {
  constexpr int64_t kHighId = int64_t{1} << 40;
  constexpr int32_t kMaxCount = std::numeric_limits<int32_t>::max();
  auto history = DraftTokenHistory::from_host_values(
      std::array<int64_t, 3>{kHighId, kHighId + 1, kHighId + 2},
      std::array<int64_t, 3>{kMaxCount - 1, 0, 5});
  DraftTokenHistories histories{history};
  SamplingParameters params;
  DraftHistoryBatch batch;
  batch.apply(params, histories, device());

  history->append({kHighId});
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_count_casts(), 0);
    EXPECT_EQ(operations.host_to_device_transfers(), 1);
    EXPECT_EQ(operations.host_to_device_bytes(),
              sizeof(int64_t) + 2 * sizeof(int32_t));
  }
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_counts),
                           torch::tensor({{kMaxCount, 0, 5}}, torch::kInt)));

  history->append({kHighId + 1, kHighId + 1, kHighId + 2});
  batch.apply(params, histories, device());
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_counts),
                           torch::tensor({{kMaxCount, 2, 6}}, torch::kInt)));

  // Grow capacity, then mix an existing count with a new 64-bit token ID.
  history->append({kHighId + 3, kHighId + 3});
  batch.apply(params, histories, device());
  history->append({kHighId + 3});
  batch.apply(params, histories, device());
  history->append({kHighId + 4, kHighId + 1});
  {
    TensorCopyObserver operations(/*token_count=*/0);
    batch.apply(params, histories, device());
    EXPECT_EQ(operations.device_count_casts(), 0);
    EXPECT_EQ(operations.host_to_device_bytes(),
              3 * sizeof(int64_t) + 2 * sizeof(int32_t));
  }
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids),
                           torch::tensor({{kHighId,
                                           kHighId + 1,
                                           kHighId + 2,
                                           kHighId + 3,
                                           kHighId + 4,
                                           kHighId + 4}},
                                         torch::kLong)));
  EXPECT_TRUE(
      torch::equal(snapshot(params.unique_token_counts),
                   torch::tensor({{kMaxCount, 3, 6, 3, 1, 0}}, torch::kInt)));
  EXPECT_TRUE(torch::equal(snapshot(params.unique_token_ids_lens),
                           torch::tensor({5}, torch::kInt)));
}

TEST_F(MtpDraftInputCarryTest,
       BatchedHistoryImportCopiesOnlyMissingValidPrefixes) {
  SamplingParameters params;
  params.unique_token_ids = torch::tensor({{0, 2, 4, 6, -1, -1},
                                           {-1, -1, -1, -1, -1, -1},
                                           {9, 10, -1, -1, -1, -1},
                                           {15, -1, -1, -1, -1, -1}},
                                          torch::kLong)
                                .to(device());
  params.unique_token_counts = torch::tensor({{1, 2, 3, 4, -1, -1},
                                              {-1, -1, -1, -1, -1, -1},
                                              {5, 6, -1, -1, -1, -1},
                                              {7, -1, -1, -1, -1, -1}},
                                             torch::kInt)
                                   .to(device());
  params.unique_token_ids_lens =
      torch::tensor({4, 1000, 2, 1}, torch::kInt).to(device());
  DraftTokenHistories histories(4);
  histories[1] = DraftTokenHistory::from_host_values(std::array<int64_t, 1>{42},
                                                     std::array<int64_t, 1>{7});
  const auto existing = histories[1];
  {
    TensorCopyObserver copies(/*token_count=*/0);
    import_sampling_histories(params, /*batch_size=*/4, histories);
    EXPECT_EQ(copies.device_to_host_transfers(), 3);
    // Four lengths plus seven IDs/counts. No padding or existing row payload.
    EXPECT_LE(copies.device_to_host_elements(), 4 + 2 * 7);
  }
  EXPECT_EQ(histories[0]->ids(), std::vector<int64_t>({0, 2, 4, 6}));
  EXPECT_EQ(histories[0]->counts(), std::vector<int32_t>({1, 2, 3, 4}));
  EXPECT_EQ(histories[1], existing);
  EXPECT_EQ(existing->ids(), std::vector<int64_t>({42}));
  EXPECT_EQ(existing->counts(), std::vector<int32_t>({7}));
  EXPECT_EQ(histories[2]->ids(), std::vector<int64_t>({9, 10}));
  EXPECT_EQ(histories[2]->counts(), std::vector<int32_t>({5, 6}));
  EXPECT_EQ(histories[3]->ids(), std::vector<int64_t>({15}));
  EXPECT_EQ(histories[3]->counts(), std::vector<int32_t>({7}));

  // A single missing row keeps the direct, prefix-sized transfer; the
  // batched import still reads the whole lengths vector in one coalesced
  // copy.
  histories[2].reset();
  {
    TensorCopyObserver copies(/*token_count=*/0);
    import_sampling_histories(params, /*batch_size=*/4, histories);
    EXPECT_EQ(copies.device_to_host_transfers(), 3);
    EXPECT_LE(copies.device_to_host_elements(), 4 + 2 * 2);
  }
  EXPECT_EQ(histories[2]->ids(), std::vector<int64_t>({9, 10}));

  // Once every history exists, even unreadable device statistics are ignored.
  const auto meta = torch::TensorOptions().device(torch::kMeta);
  params.unique_token_ids = torch::empty({4, 6}, meta.dtype(torch::kLong));
  params.unique_token_counts = torch::empty({4, 6}, meta.dtype(torch::kInt));
  params.unique_token_ids_lens = torch::empty({4}, meta.dtype(torch::kInt));
  {
    TensorCopyObserver copies(/*token_count=*/0);
    import_sampling_histories(params, /*batch_size=*/4, histories);
    EXPECT_EQ(copies.device_to_host_transfers(), 0);
  }
}

TEST_F(MtpDraftInputCarryTest, BatchedHistoryImportHandlesEmptyRows) {
  SamplingParameters params;
  params.unique_token_ids =
      torch::tensor({{7, -1, -1}, {-1, -1, -1}, {9, 10, -1}}, torch::kLong)
          .to(device());
  params.unique_token_counts =
      torch::tensor({{2, -1, -1}, {-1, -1, -1}, {3, 4, -1}}, torch::kInt)
          .to(device());
  for (bool all_empty : {true, false}) {
    SCOPED_TRACE(all_empty);
    params.unique_token_ids_lens =
        (all_empty ? torch::zeros({3}, torch::kInt)
                   : torch::tensor({1, 0, 2}, torch::kInt))
            .to(device());
    DraftTokenHistories histories;
    {
      TensorCopyObserver copies(/*token_count=*/0);
      import_sampling_histories(params, /*batch_size=*/3, histories);
      EXPECT_EQ(copies.device_to_host_transfers(), all_empty ? 1 : 3);
      EXPECT_LE(copies.device_to_host_elements(), all_empty ? 3 : 9);
    }
    ASSERT_EQ(histories.size(), 3);
    for (const auto& history : histories) {
      ASSERT_NE(history, nullptr);
      EXPECT_EQ(history->revision(), 0);
    }
    EXPECT_TRUE(histories[1]->ids().empty());
    EXPECT_TRUE(histories[1]->counts().empty());
    if (all_empty) {
      EXPECT_TRUE(histories[0]->ids().empty());
      EXPECT_TRUE(histories[2]->ids().empty());
    } else {
      EXPECT_EQ(histories[0]->ids(), std::vector<int64_t>({7}));
      EXPECT_EQ(histories[0]->counts(), std::vector<int32_t>({2}));
      EXPECT_EQ(histories[2]->ids(), std::vector<int64_t>({9, 10}));
      EXPECT_EQ(histories[2]->counts(), std::vector<int32_t>({3, 4}));
    }
  }
}

TEST_F(MtpDraftInputCarryTest, CpuHistoryImportReadsOnlyNativeRowPrefixes) {
  for (torch::ScalarType count_type : {torch::kInt, torch::kLong}) {
    SCOPED_TRACE(static_cast<int32_t>(count_type));
    torch::Tensor ids = torch::full({6, 4}, -1, torch::kLong);
    torch::Tensor counts = torch::full({6, 5}, -1, count_type);
    ids.select(/*dim=*/0, /*index=*/0)
        .narrow(/*dim=*/0, /*start=*/0, /*length=*/2)
        .copy_(torch::tensor({10, 11}, torch::kLong));
    ids.select(/*dim=*/0, /*index=*/4)
        .narrow(/*dim=*/0, /*start=*/0, /*length=*/3)
        .copy_(torch::tensor({20, 21, 22}, torch::kLong));
    counts.select(/*dim=*/0, /*index=*/0)
        .narrow(/*dim=*/0, /*start=*/0, /*length=*/2)
        .copy_(torch::tensor({2, 3}, torch::kInt));
    counts.select(/*dim=*/0, /*index=*/4)
        .narrow(/*dim=*/0, /*start=*/0, /*length=*/3)
        .copy_(torch::tensor({0, 4, 5}, torch::kInt));
    SamplingParameters params;
    params.unique_token_ids =
        ids.slice(/*dim=*/0, /*start=*/0, /*end=*/6, /*step=*/2);
    params.unique_token_counts =
        counts.slice(/*dim=*/0, /*start=*/0, /*end=*/6, /*step=*/2)
            .narrow(/*dim=*/1, /*start=*/0, /*length=*/4);
    params.unique_token_ids_lens = torch::tensor({2, 1000, 3}, torch::kInt);
    ASSERT_FALSE(params.unique_token_ids.is_contiguous());
    ASSERT_FALSE(params.unique_token_counts.is_contiguous());
    ASSERT_EQ(params.unique_token_ids.stride(1), 1);
    ASSERT_EQ(params.unique_token_counts.stride(1), 1);
    DraftTokenHistories histories(3);
    histories[1] = DraftTokenHistory::from_host_values(
        std::array<int64_t, 1>{42}, std::array<int64_t, 1>{7});
    const auto existing = histories[1];
    {
      TensorCopyObserver operations(/*token_count=*/0);
      import_sampling_histories(params, /*batch_size=*/3, histories);
      EXPECT_EQ(operations.host_narrows(), 0);
      // Only the length vector may be widened; count prefixes stay native.
      EXPECT_LE(operations.host_widened_elements(), 3);
      EXPECT_EQ(operations.device_to_host_transfers(), 0);
      EXPECT_EQ(operations.host_to_device_transfers(), 0);
    }
    ids.zero_();
    counts.zero_();
    EXPECT_EQ(histories[0]->ids(), std::vector<int64_t>({10, 11}));
    EXPECT_EQ(histories[0]->counts(), std::vector<int32_t>({2, 3}));
    EXPECT_EQ(histories[1], existing);
    EXPECT_EQ(existing->ids(), std::vector<int64_t>({42}));
    EXPECT_EQ(existing->counts(), std::vector<int32_t>({7}));
    EXPECT_EQ(histories[2]->ids(), std::vector<int64_t>({20, 21, 22}));
    EXPECT_EQ(histories[2]->counts(), std::vector<int32_t>({0, 4, 5}));
  }
}

TEST_F(MtpDraftInputCarryTest, CpuHistoryImportReadsDeviceLengthsOnce) {
  SamplingParameters params;
  params.unique_token_ids =
      torch::tensor({{1, -1}, {2, 3}, {-1, -1}, {4, -1}}, torch::kLong);
  params.unique_token_counts =
      torch::tensor({{2, -1}, {0, 5}, {-1, -1}, {7, -1}}, torch::kInt);
  params.unique_token_ids_lens =
      torch::tensor({1, 2, 0, 1}, torch::kInt).to(device());
  DraftTokenHistories histories;
  {
    TensorCopyObserver operations(/*token_count=*/0);
    import_sampling_histories(params, /*batch_size=*/4, histories);
    EXPECT_EQ(operations.device_to_host_transfers(), 1);
    EXPECT_EQ(operations.device_to_host_elements(), 4);
    EXPECT_EQ(operations.host_narrows(), 0);
  }
  ASSERT_EQ(histories.size(), 4);
  EXPECT_EQ(histories[0]->ids(), std::vector<int64_t>({1}));
  EXPECT_EQ(histories[0]->counts(), std::vector<int32_t>({2}));
  EXPECT_EQ(histories[1]->ids(), std::vector<int64_t>({2, 3}));
  EXPECT_EQ(histories[1]->counts(), std::vector<int32_t>({0, 5}));
  EXPECT_TRUE(histories[2]->ids().empty());
  EXPECT_TRUE(histories[2]->counts().empty());
  EXPECT_EQ(histories[3]->ids(), std::vector<int64_t>({4}));
  EXPECT_EQ(histories[3]->counts(), std::vector<int32_t>({7}));
}

TEST_F(MtpDraftInputCarryTest, CompleteHistoryMatricesAvoidDevicePacking) {
  constexpr int64_t kBatch = 3;
  constexpr int64_t kWidth = 4;
  for (bool strided : {false, true}) {
    SCOPED_TRACE(strided);
    const torch::Tensor ids =
        torch::arange(kBatch * kWidth, torch::kLong).to(device());
    const torch::Tensor counts =
        torch::arange(1, kBatch * kWidth + 1, torch::kInt).to(device());
    SamplingParameters params;
    params.unique_token_ids = strided
                                  ? ids.view({kWidth, kBatch}).transpose(0, 1)
                                  : ids.view({kBatch, kWidth});
    params.unique_token_counts =
        strided ? counts.view({kWidth, kBatch}).transpose(0, 1)
                : counts.view({kBatch, kWidth});
    params.unique_token_ids_lens =
        torch::full({kBatch}, kWidth, torch::kInt).to(device());
    EXPECT_EQ(params.unique_token_ids.is_contiguous(), !strided);
    DraftTokenHistories histories;
    {
      TensorCopyObserver copies(/*token_count=*/0);
      import_sampling_histories(params, kBatch, histories);
      EXPECT_EQ(copies.device_concatenations(), 0);
      EXPECT_EQ(copies.host_narrows(), 0);
      EXPECT_EQ(copies.device_to_host_transfers(), 3);
      EXPECT_EQ(copies.device_to_host_elements(), kBatch + 2 * kBatch * kWidth);
    }
    ids.zero_();
    counts.zero_();
    ASSERT_EQ(histories.size(), kBatch);
    for (int64_t row = 0; row < kBatch; ++row) {
      std::vector<int64_t> expected_ids;
      std::vector<int32_t> expected_counts;
      expected_ids.reserve(kWidth);
      expected_counts.reserve(kWidth);
      for (int64_t column = 0; column < kWidth; ++column) {
        const int64_t value =
            strided ? column * kBatch + row : row * kWidth + column;
        expected_ids.emplace_back(value);
        expected_counts.emplace_back(static_cast<int32_t>(value + 1));
      }
      EXPECT_EQ(histories[row]->ids(), expected_ids);
      EXPECT_EQ(histories[row]->counts(), expected_counts);
    }
  }
}

TEST_F(MtpDraftInputCarryTest,
       FullWidthMissingRowsDoNotExposeEstablishedHistory) {
  SamplingParameters params;
  params.unique_token_ids =
      torch::tensor({{1, 2}, {-1, -1}, {5, 6}}, torch::kLong).to(device());
  params.unique_token_counts =
      torch::tensor({{2, 3}, {-1, -1}, {4, 5}}, torch::kInt).to(device());
  params.unique_token_ids_lens =
      torch::tensor({2, 1000, 2}, torch::kInt).to(device());
  DraftTokenHistories histories(3);
  histories[1] = DraftTokenHistory::from_host_values(std::array<int64_t, 1>{42},
                                                     std::array<int64_t, 1>{7});
  const auto existing = histories[1];
  {
    TensorCopyObserver copies(/*token_count=*/0);
    import_sampling_histories(params, /*batch_size=*/3, histories);
    EXPECT_EQ(copies.device_concatenations(), 2);
    EXPECT_EQ(copies.device_to_host_transfers(), 3);
    EXPECT_LE(copies.device_to_host_elements(), 3 + 2 * 4);
  }
  EXPECT_EQ(histories[0]->ids(), std::vector<int64_t>({1, 2}));
  EXPECT_EQ(histories[0]->counts(), std::vector<int32_t>({2, 3}));
  EXPECT_EQ(histories[1], existing);
  EXPECT_EQ(existing->ids(), std::vector<int64_t>({42}));
  EXPECT_EQ(existing->counts(), std::vector<int32_t>({7}));
  EXPECT_EQ(histories[2]->ids(), std::vector<int64_t>({5, 6}));
  EXPECT_EQ(histories[2]->counts(), std::vector<int32_t>({4, 5}));
}

TEST_F(MtpDraftInputCarryTest, NoncontiguousSeedSnapshotCopiesEachPayloadOnce) {
  DraftInput producer{
      torch::tensor({1, 9, 0, 9}, torch::kLong)
          .to(device())
          .slice(/*dim=*/0, /*start=*/0, /*end=*/4, /*step=*/2),
      torch::arange(8, torch::kFloat)
          .to(device())
          .view({2, 4})
          .slice(/*dim=*/1, /*start=*/0, /*end=*/4, /*step=*/2),
      torch::tensor({{0.25F, 9.0F, 0.75F, 9.0F}, {0.5F, 9.0F, 0.5F, 9.0F}})
          .to(device())
          .slice(/*dim=*/1, /*start=*/0, /*end=*/4, /*step=*/2),
      torch::tensor({0.75F, 9.0F, 0.5F, 9.0F})
          .to(device())
          .slice(/*dim=*/0, /*start=*/0, /*end=*/4, /*step=*/2)};
  producer.hidden_states.set_requires_grad(/*requires_grad=*/true);
  DraftInput cloned;
  {
    TensorCopyObserver copies(/*token_count=*/0);
    cloned = clone_draft_input(producer);
    EXPECT_EQ(copies.device_clones(), kDraftInputTensors.size());
    EXPECT_EQ(copies.host_to_device_transfers(), 0);
    EXPECT_EQ(copies.device_to_host_transfers(), 0);
  }
  for (torch::Tensor DraftInput::* member : kDraftInputTensors) {
    const torch::Tensor& original = producer.*member;
    const torch::Tensor& copied = cloned.*member;
    ASSERT_FALSE(original.is_contiguous());
    EXPECT_TRUE(copied.is_contiguous());
    EXPECT_FALSE(copied.requires_grad());
    EXPECT_EQ(copied.scalar_type(), original.scalar_type());
    EXPECT_EQ(copied.device(), original.device());
    EXPECT_FALSE(copied.is_alias_of(original));
    const torch::Tensor expected = snapshot(original);
    original.detach().zero_();
    EXPECT_TRUE(torch::equal(snapshot(copied), expected));
  }
}

TEST_F(MtpDraftInputCarryTest, BatchSeedSnapshotsBoundRetentionOnDevice) {
  DraftInputStateStore cache(/*total_nums=*/3);
  DraftInput producer{torch::tensor({1, 2, 3}, torch::kLong).to(device()),
                      torch::ones({3, 2}).to(device()),
                      torch::full({3, 64}, 1.0F / 64).to(device()),
                      torch::Tensor()};
  cache.write_draft_input({0, 1, 2}, {"a", "b", "c"}, producer);
  producer.probs.zero_();
  const auto partial = cache.take_draft_input({1}, {"b"});
  ASSERT_TRUE(partial.has_value());
  EXPECT_TRUE(
      torch::equal(snapshot(partial->probs), torch::full({1, 64}, 1.0F / 64)));
  const auto survivor = cache.read_draft_input({0}, {"a"});
  ASSERT_TRUE(survivor.has_value());
  EXPECT_LE(survivor->probs.storage().nbytes(), 2 * 2 * 64 * sizeof(float));
  const auto consumed = cache.take_draft_input({0, 2}, {"a", "c"});
  ASSERT_TRUE(consumed.has_value());
  EXPECT_TRUE(
      torch::equal(snapshot(consumed->probs), torch::full({2, 64}, 1.0F / 64)));
  EXPECT_TRUE(torch::equal(snapshot(consumed->token_ids),
                           torch::tensor({1, 3}, torch::kLong)));
}

TEST_P(MtpDraftInputCarryAcceptanceTest,
       CarriesPrefillAndExtendSeedsAcrossRounds) {
  const AcceptanceCase test_case = GetParam();
  const int32_t width = test_case.speculative_tokens + 1;
  const int32_t accepted_drafts = test_case.accepted_drafts;
  auto calls = std::make_shared<std::vector<ModelCall>>();
  calls->reserve(8);
  DraftInputCarryWorker worker(
      device(), make_options(test_case.speculative_tokens), calls);

  const auto prefill_output = worker.step(make_prefill(device()));
  ASSERT_TRUE(prefill_output.has_value());
  EXPECT_EQ(
      tensor_to_vector<int64_t>(prefill_output->sample_output.next_tokens),
      std::vector<int64_t>({kRootToken}));
  ASSERT_EQ(calls->size(), 2);
  EXPECT_EQ(call_roles(*calls),
            (std::vector<ModelRole>{ModelRole::TARGET, ModelRole::DRAFT}));
  EXPECT_EQ(calls->at(0).tokens, (std::vector<int64_t>{10, 11, 12}));
  EXPECT_FALSE(calls->at(0).hidden_input.defined());
  const ModelCall& draft_prefill = calls->at(1);
  EXPECT_EQ(draft_prefill.tokens, (std::vector<int64_t>{11, 12, 13}));
  EXPECT_EQ(draft_prefill.positions, (std::vector<int64_t>{0, 1, 2}));
  EXPECT_EQ(draft_prefill.slots, (std::vector<int64_t>{8, 9, 10}));
  EXPECT_EQ(draft_prefill.selected_rows, (std::vector<int64_t>{2}));
  expect_hidden(draft_prefill.hidden_input,
                ModelRole::TARGET,
                /*tokens=*/{10, 11, 12},
                /*positions=*/{0, 1, 2});
  const auto prefill_draft_input =
      worker.cache().read_draft_input({kEmbeddingId}, {kRequestId});
  ASSERT_TRUE(prefill_draft_input.has_value());
  EXPECT_FALSE(prefill_draft_input->selected_probs.defined());
  EXPECT_EQ(tensor_to_vector<int64_t>(prefill_draft_input->token_ids),
            (std::vector<int64_t>{14}));
  expect_hidden(prefill_draft_input->hidden_states,
                ModelRole::DRAFT,
                /*tokens=*/{13},
                /*positions=*/{2});

  std::vector<int64_t> target_predictions;
  std::vector<int64_t> expected_accepted(width, -1);
  std::vector<int64_t> expected_extend_tokens(width, 0);
  target_predictions.reserve(width);
  for (int32_t row = 0; row < width; ++row) {
    target_predictions.emplace_back(kRootToken + row + 1);
  }
  target_predictions[accepted_drafts] = kCorrectionToken;
  for (int32_t row = 0; row < accepted_drafts; ++row) {
    expected_accepted[row] = kRootToken + row + 1;
    expected_extend_tokens[row] = kRootToken + row + 1;
  }
  expected_accepted[accepted_drafts] = kCorrectionToken;
  expected_extend_tokens[accepted_drafts] = kCorrectionToken;
  worker.set_target_predictions(std::move(target_predictions));
  const auto decode_output = worker.step(make_decode(device()));
  ASSERT_TRUE(decode_output.has_value());
  EXPECT_EQ(tensor_to_vector<int64_t>(decode_output->sample_output.next_tokens),
            expected_accepted);
  EXPECT_FALSE(decode_output->sample_output.embeddings.defined());
  ASSERT_EQ(calls->size(), static_cast<size_t>(width + 2));

  std::vector<ModelRole> expected_roles = {ModelRole::TARGET, ModelRole::DRAFT};
  expected_roles.reserve(8);
  for (int32_t draft = 1; draft < test_case.speculative_tokens; ++draft) {
    expected_roles.emplace_back(ModelRole::DRAFT);
  }
  expected_roles.emplace_back(ModelRole::TARGET);
  expected_roles.emplace_back(ModelRole::DRAFT);
  EXPECT_EQ(call_roles(*calls), expected_roles);
  if (test_case.speculative_tokens == 2) {
    const ModelCall& recursive_draft = calls->at(2);
    EXPECT_EQ(recursive_draft.tokens, (std::vector<int64_t>{14}));
    EXPECT_EQ(recursive_draft.positions, (std::vector<int64_t>{3}));
    EXPECT_EQ(recursive_draft.slots, (std::vector<int64_t>{11}));
    expect_hidden(recursive_draft.hidden_input,
                  ModelRole::DRAFT,
                  /*tokens=*/{13},
                  /*positions=*/{2});
  }

  std::vector<int64_t> verify_tokens;
  std::vector<int64_t> verify_positions;
  verify_tokens.reserve(width);
  verify_positions.reserve(width);
  for (int32_t row = 0; row < width; ++row) {
    verify_tokens.emplace_back(kRootToken + row);
    verify_positions.emplace_back(kRootPosition + row);
  }
  const size_t verify_index = calls->size() - 2;
  const ModelCall& verify = calls->at(verify_index);
  const ModelCall& extend = calls->back();
  EXPECT_EQ(verify.tokens, verify_tokens);
  EXPECT_EQ(verify.positions, verify_positions);
  EXPECT_EQ(verify.slots,
            width == 2 ? std::vector<int64_t>({11, 20})
                       : std::vector<int64_t>({11, 20, 21}));
  EXPECT_TRUE(verify.is_spec_verify);
  EXPECT_FALSE(verify.hidden_input.defined());
  EXPECT_EQ(extend.tokens, expected_extend_tokens);
  EXPECT_EQ(extend.positions, verify_positions);
  EXPECT_EQ(extend.slots, verify.slots);
  EXPECT_EQ(extend.q_seq_lens, std::vector<int32_t>({width}));
  EXPECT_EQ(extend.kv_seq_lens, std::vector<int32_t>({kRootPosition + width}));
  EXPECT_EQ(extend.selected_rows, std::vector<int64_t>({accepted_drafts}));
  EXPECT_FALSE(extend.is_spec_verify);
  EXPECT_TRUE(torch::equal(extend.hidden_input, verify.hidden_output));

  const auto next_draft_input =
      worker.cache().read_draft_input({kEmbeddingId}, {kRequestId});
  ASSERT_TRUE(next_draft_input.has_value());
  EXPECT_EQ(tensor_to_vector<int64_t>(next_draft_input->token_ids),
            std::vector<int64_t>({kCorrectionToken + 1}));
  expect_hidden(next_draft_input->hidden_states,
                ModelRole::DRAFT,
                /*tokens=*/{kCorrectionToken},
                /*positions=*/{kRootPosition + accepted_drafts});

  // Consume the published seed, rather than merely inspecting cache contents.
  const size_t next_round_begin = calls->size();
  const int32_t next_root_position = kRootPosition + accepted_drafts + 1;
  const auto next_output =
      worker.step(make_decode(device(), kCorrectionToken, next_root_position));
  ASSERT_TRUE(next_output.has_value());
  ASSERT_EQ(calls->size(), next_round_begin + width);
  if (test_case.speculative_tokens == 2) {
    const ModelCall& recursive_draft = calls->at(next_round_begin);
    EXPECT_EQ(recursive_draft.role, ModelRole::DRAFT);
    EXPECT_EQ(recursive_draft.tokens,
              std::vector<int64_t>({kCorrectionToken + 1}));
    EXPECT_EQ(recursive_draft.positions,
              std::vector<int64_t>({next_root_position}));
    expect_hidden(recursive_draft.hidden_input,
                  ModelRole::DRAFT,
                  /*tokens=*/{kCorrectionToken},
                  /*positions=*/{next_root_position - 1});
  }
  const ModelCall& next_verify = calls->at(calls->size() - 2);
  EXPECT_EQ(next_verify.role, ModelRole::TARGET);
  EXPECT_EQ(next_verify.tokens.front(), kCorrectionToken);
  EXPECT_EQ(next_verify.tokens.at(1), kCorrectionToken + 1);
  EXPECT_EQ(next_verify.positions.front(), next_root_position);
  EXPECT_EQ(calls->back().role, ModelRole::DRAFT);
}

INSTANTIATE_TEST_SUITE_P(
    K1AndK2,
    MtpDraftInputCarryAcceptanceTest,
    ::testing::Values(AcceptanceCase{1, 0},
                      AcceptanceCase{1, 1},
                      AcceptanceCase{2, 0},
                      AcceptanceCase{2, 1},
                      AcceptanceCase{2, 2}),
    [](const ::testing::TestParamInfo<AcceptanceCase>& info) {
      return "K" + std::to_string(info.param.speculative_tokens) + "Accept" +
             std::to_string(info.param.accepted_drafts);
    });

TEST_F(MtpDraftInputCarryTest,
       ColdPdDecodeImportsSeedWithoutDraftPrefillReplay) {
  auto prefill_calls = std::make_shared<std::vector<ModelCall>>();
  prefill_calls->reserve(2);
  DraftInputCarryWorker prefill_worker(
      device(),
      make_options(/*speculative_tokens=*/2, InstanceRole::PREFILL),
      prefill_calls);
  const auto prefill = prefill_worker.step(make_prefill(device()));
  ASSERT_TRUE(prefill.has_value());
  ASSERT_TRUE(prefill->sample_output.draft_input.has_value());
  EXPECT_TRUE(prefill->sample_output.draft_input->selected_probs.defined());
  EXPECT_FALSE(prefill->sample_output.embeddings.defined());

  auto decode_calls = std::make_shared<std::vector<ModelCall>>();
  decode_calls->reserve(3);
  DraftInputCarryWorker decode_worker(
      device(),
      make_options(/*speculative_tokens=*/2, InstanceRole::DECODE),
      decode_calls);
  EXPECT_FALSE(decode_worker.cache()
                   .read_draft_input({kEmbeddingId}, {kRequestId})
                   .has_value());
  LlmForwardInput input = make_decode(device());
  input.request_state.draft_input_row_idxes = {0};
  proto::DraftInput wire_draft_input;
  ASSERT_TRUE(
      draft_input_to_proto(draft_input_to(*prefill->sample_output.draft_input,
                                          torch::Device(torch::kCPU)),
                           &wire_draft_input));
  input.request_state.draft_input = draft_input_from_proto(wire_draft_input);
  ASSERT_TRUE(input.request_state.draft_input.has_value());

  const auto output = decode_worker.step(input);
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(tensor_to_vector<int64_t>(output->sample_output.next_tokens),
            (std::vector<int64_t>{14, 15, 16}));
  ASSERT_EQ(decode_calls->size(), 3);
  EXPECT_EQ(call_roles(*decode_calls),
            (std::vector<ModelRole>{
                ModelRole::DRAFT, ModelRole::TARGET, ModelRole::DRAFT}));
  EXPECT_EQ(decode_calls->at(0).tokens, (std::vector<int64_t>{14}));
  EXPECT_EQ(decode_calls->at(0).positions, (std::vector<int64_t>{3}));
  expect_hidden(decode_calls->at(0).hidden_input,
                ModelRole::DRAFT,
                /*tokens=*/{13},
                /*positions=*/{2});
  EXPECT_EQ(decode_calls->at(1).tokens, (std::vector<int64_t>{13, 14, 15}));
  EXPECT_EQ(decode_calls->at(2).tokens, (std::vector<int64_t>{14, 15, 16}));
  const auto next_draft_input =
      decode_worker.cache().read_draft_input({kEmbeddingId}, {kRequestId});
  ASSERT_TRUE(next_draft_input.has_value());
  EXPECT_EQ(tensor_to_vector<int64_t>(next_draft_input->token_ids),
            (std::vector<int64_t>{17}));
}

TEST_F(MtpDraftInputCarryTest,
       BatchedPdSeedImportKeepsRowMappingAndOneSnapshot) {
  auto calls = std::make_shared<std::vector<ModelCall>>();
  DraftInputCarryWorker worker(
      device(),
      make_options(/*speculative_tokens=*/2, InstanceRole::DECODE),
      calls);
  LlmForwardInput initial;
  initial.token_ids_host = torch::tensor({10}, torch::kInt);
  initial.input_params.embedding.embedding_ids = {4};
  initial.input_params.embedding.request_ids = {"existing"};
  initial.request_state.draft_input_row_idxes = {0};
  initial.request_state.draft_input =
      DraftInput{torch::tensor({11}, torch::kLong),
                 torch::zeros({1, kHiddenSize}),
                 torch::Tensor(),
                 torch::Tensor()};
  worker.import_draft_input(initial);

  LlmForwardInput input;
  input.token_ids_host = torch::tensor({10, 20, 30, 40}, torch::kInt);
  input.input_params.embedding.embedding_ids = {4, 1, 6, 3};
  input.input_params.embedding.request_ids = {
      "existing", "second", "third", "fourth"};
  input.request_state.draft_input_row_idxes = {3, 1, 2};
  const torch::Tensor tokens = torch::tensor({41, 21, 31}, torch::kLong);
  const torch::Tensor hidden =
      torch::arange(3 * kHiddenSize, torch::kFloat).view({3, kHiddenSize});
  const torch::Tensor probs =
      make_point_mass_probs(tokens, kVocabSize, torch::dtype(torch::kFloat));
  const torch::Tensor confidence = torch::tensor({0.25F, 0.5F, 0.75F});
  input.request_state.draft_input =
      DraftInput{tokens, hidden, probs, confidence};
  {
    TensorCopyObserver copies(/*token_count=*/0);
    worker.import_draft_input(input);
    // draft_input_to stages the transported CPU tensors to the device; the
    // imported snapshot now moves the payload without a second clone.
    EXPECT_EQ(copies.device_clones(), 0);
  }
  EXPECT_FALSE(input.request_state.draft_input.has_value());
  EXPECT_TRUE(input.request_state.draft_input_row_idxes.empty());
  EXPECT_TRUE(calls->empty());
  const std::vector<int32_t> ids = {3, 1, 6};
  const std::vector<std::string> request_ids = {"fourth", "second", "third"};
  const auto imported = worker.cache().read_draft_input(ids, request_ids);
  ASSERT_TRUE(imported.has_value());
  EXPECT_TRUE(torch::equal(imported->hidden_states.cpu(), hidden));
  EXPECT_TRUE(torch::equal(imported->probs.cpu(), probs));
  EXPECT_TRUE(torch::equal(imported->selected_probs.cpu(), confidence));
  tokens.zero_();
  hidden.zero_();
  probs.zero_();
  confidence.zero_();
  EXPECT_EQ(tensor_to_vector<int64_t>(imported->token_ids),
            (std::vector<int64_t>{41, 21, 31}));
  const auto states =
      worker.target_state().read_decode_states(ids, request_ids);
  ASSERT_EQ(states.size(), 3);
  EXPECT_EQ(states[0].token_id, 40);
  EXPECT_EQ(states[1].token_id, 20);
  EXPECT_EQ(states[2].token_id, 30);
  const auto existing = worker.cache().read_draft_input({4}, {"existing"});
  ASSERT_TRUE(existing.has_value());
  EXPECT_EQ(tensor_to_vector<int64_t>(existing->token_ids),
            (std::vector<int64_t>{11}));
  EXPECT_FALSE(worker.cache().read_draft_input({3}, {"another-request"}));
  EXPECT_TRUE(worker.release_idle_request_state_async({1, 3}).get());
  const auto remaining = worker.cache().read_draft_input({6}, {"third"});
  ASSERT_TRUE(remaining.has_value());
  EXPECT_EQ(tensor_to_vector<int64_t>(remaining->token_ids),
            (std::vector<int64_t>{31}));
  EXPECT_EQ(remaining->probs.storage().nbytes(), kVocabSize * sizeof(float));
}

TEST_F(MtpDraftInputCarryTest, DraftExtendUsesAcceptedTokensInAclExecutor) {
  auto calls = std::make_shared<std::vector<ModelCall>>();
  calls->reserve(1);
  runtime::Options options = make_options(/*speculative_tokens=*/2);
  options.enable_graph(true).is_draft_engine(true).num_decoding_tokens(1);
  ModelArgs args = make_model_args(ModelRole::DRAFT);
  args.dtype("float32");
  RecordingModel model(
      ModelRole::DRAFT,
      torch::TensorOptions().dtype(torch::kFloat).device(device()),
      calls);
  Executor executor(&model, args, device(), options);
  const LlmForwardInput host_verify = make_input(
      device(), {13, 14, 15}, {3, 4, 5}, BatchForwardType::CHUNKED_PREFILL);
  LlmForwardInput verify = host_verify.to(device(), torch::kFloat);
  verify.input_params.graph.input_tokens_override = verify.token_ids;
  verify.input_params.graph.spec_verify_draft_token_sources = {
      torch::tensor({14}, verify.token_ids.options()),
      torch::tensor({15}, verify.token_ids.options())};
  verify.input_params.graph.spec_verify_source_addresses_stable = true;
  verify.input_params.is_spec_verify = true;
  const torch::Tensor accepted = torch::tensor(
      {{14, 40, -1}}, verify.token_ids.options().dtype(torch::kLong));
  const torch::Tensor hidden =
      hidden_rows(ModelRole::TARGET, verify.token_ids, verify.positions)
          .view({1, 3, kHiddenSize});
  const auto rows = specBuilder::build_mtp_draft_extend_rows(
      accepted, hidden, {3}, /*step_major_layout=*/false);
  const LlmForwardInput draft =
      specBuilder::build_mtp_draft_extend_input(verify, rows, {});
  std::vector<KVCache> caches;
  // A one-layer model exercises the ACL executor's eager branch, which also
  // honors graph token overrides. Clearing only is_spec_verify is insufficient.
  const ModelOutput output =
      executor.forward(draft.token_ids,
                       draft.positions,
                       caches,
                       ModelInputParams(draft.input_params));
  ASSERT_EQ(calls->size(), 1);
  EXPECT_EQ(calls->front().tokens, std::vector<int64_t>({14, 40, 0}));
  const torch::Tensor logits = model.logits(
      output.hidden_states, draft.sampling_params.selected_token_idxes);
  EXPECT_EQ(tensor_to_vector<int64_t>(logits.argmax(/*dim=*/1)),
            std::vector<int64_t>({41}));
}

TEST_F(MtpDraftInputCarryTest,
       ReleasedEmbeddingClearsTargetSeedAndSamplingHistory) {
  auto calls = std::make_shared<std::vector<ModelCall>>();
  calls->reserve(4);
  DraftInputCarryWorker worker(
      device(), make_options(/*speculative_tokens=*/2), calls);
  LlmForwardInput prefill = make_prefill(device());
  prefill.sampling_params.unique_token_ids =
      torch::tensor({{10, 11, 12}}, torch::kLong);
  prefill.sampling_params.unique_token_counts =
      torch::tensor({{1, 1, 1}}, torch::kInt);
  prefill.sampling_params.unique_token_ids_lens =
      torch::tensor({3}, torch::kInt);
  ASSERT_TRUE(worker.step(prefill).has_value());
  ASSERT_TRUE(worker.cache()
                  .read_draft_input({kEmbeddingId}, {kRequestId})
                  .has_value());
  ASSERT_TRUE(worker.target_state()
                  .read_decode_states({kEmbeddingId}, {kRequestId})
                  .front()
                  .valid);

  LlmForwardInput next_prefill = make_input(device(),
                                            /*tokens=*/{20, 21, 22},
                                            /*positions=*/{0, 1, 2},
                                            BatchForwardType::PREFILL,
                                            /*embedding_id=*/2,
                                            /*request_id=*/"new-request");
  next_prefill.request_state.released_embedding_ids = {kEmbeddingId};
  const LlmForwardInput read_only_prefill = std::move(next_prefill);
  ASSERT_TRUE(worker.step(read_only_prefill).has_value());
  EXPECT_EQ(read_only_prefill.request_state.released_embedding_ids,
            std::vector<int32_t>({kEmbeddingId}));
  EXPECT_FALSE(worker.cache()
                   .read_draft_input({kEmbeddingId}, {kRequestId})
                   .has_value());
  EXPECT_FALSE(worker.target_state()
                   .read_decode_states({kEmbeddingId}, {kRequestId})
                   .front()
                   .valid);
  EXPECT_TRUE(
      worker.cache().read_draft_input({2}, {"new-request"}).has_value());
  ASSERT_EQ(calls->size(), 4);
}

TEST_F(MtpDraftInputCarryTest,
       IdleReleaseRetiresPendingStateWithoutModelExecution) {
  auto calls = std::make_shared<std::vector<ModelCall>>();
  calls->reserve(10);
  auto options = make_options(/*speculative_tokens=*/2);
  options.enable_schedule_overlap(true);
  DraftInputCarryWorker worker(device(), options, calls);
  LlmForwardInput prefill = make_prefill(device());
  prefill.sampling_params.repetition_penalties = torch::tensor({1.1F});
  prefill.sampling_params.unique_token_ids =
      torch::tensor({{10, 11, 12}}, torch::kLong);
  prefill.sampling_params.unique_token_counts =
      torch::zeros({1, 3}, torch::kInt);
  prefill.sampling_params.unique_token_ids_lens =
      torch::tensor({3}, torch::kInt);
  prefill.sampling_params = prefill.sampling_params.to(device(), torch::kFloat);
  ASSERT_TRUE(worker.step(prefill).has_value());
  LlmForwardInput decode = make_decode(device());
  decode.sampling_params.repetition_penalties = torch::tensor({1.1F});
  decode.sampling_params = decode.sampling_params.to(device(), torch::kFloat);
  ASSERT_TRUE(worker.step(decode).has_value());
  auto history = worker.sampling_history();
  ASSERT_FALSE(history.expired());
  const size_t model_calls = calls->size();

  auto released = worker.release_idle_request_state_async({kEmbeddingId});
  ASSERT_TRUE(std::move(released).get());
  EXPECT_EQ(calls->size(), model_calls);
  EXPECT_TRUE(history.expired());
  EXPECT_FALSE(worker.cache().read_draft_input({kEmbeddingId}, {kRequestId}));
  EXPECT_FALSE(worker.target_state()
                   .read_decode_states({kEmbeddingId}, {kRequestId})
                   .front()
                   .valid);
  auto duplicate = worker.release_idle_request_state_async({kEmbeddingId});
  ASSERT_TRUE(std::move(duplicate).get());
  EXPECT_EQ(calls->size(), model_calls);

  const auto next = make_input(device(),
                               /*tokens=*/{20, 21, 22},
                               /*positions=*/{0, 1, 2},
                               BatchForwardType::PREFILL,
                               /*embedding_id=*/2,
                               /*request_id=*/"replacement");
  ASSERT_TRUE(worker.step(next).has_value());
  EXPECT_FALSE(worker.target_state()
                   .read_decode_states({kEmbeddingId}, {kRequestId})
                   .front()
                   .valid);
}

void expect_failed_request(const ForwardOutput& output) {
  EXPECT_TRUE(std::any_of(output.json_object_errors.begin(),
                          output.json_object_errors.end(),
                          [](const JsonObjectOutputError& error) {
                            return error.sample_sequence_id == kRequestId;
                          }));
}

void expect_k4_decode_phases(const std::vector<ModelCall>& calls,
                             size_t begin) {
  ASSERT_EQ(calls.size(), begin + 5);
  EXPECT_EQ(calls[begin].role, ModelRole::DRAFT);
  EXPECT_EQ(calls[begin + 1].role, ModelRole::DRAFT);
  EXPECT_EQ(calls[begin + 2].role, ModelRole::DRAFT);
  EXPECT_EQ(calls[begin + 3].role, ModelRole::TARGET);
  EXPECT_EQ(calls[begin + 4].role, ModelRole::DRAFT);
  EXPECT_EQ(calls[begin].tokens.size(), 1);
  EXPECT_EQ(calls[begin + 1].tokens.size(), 1);
  EXPECT_EQ(calls[begin + 2].tokens.size(), 1);
  const ModelCall& verify = calls[begin + 3];
  const ModelCall& extend = calls[begin + 4];
  EXPECT_EQ(verify.tokens.size(), 5);
  EXPECT_EQ(extend.tokens.size(), 5);
  EXPECT_EQ(extend.positions, verify.positions);
  EXPECT_EQ(extend.slots, verify.slots);
  EXPECT_FALSE(verify.hidden_input.defined());
  EXPECT_TRUE(torch::equal(extend.hidden_input, verify.hidden_output));
}

class MtpDraftInputCarryFailureTest
    : public MtpDraftInputCarryTest,
      public ::testing::WithParamInterface<int32_t> {};

TEST_F(MtpDraftInputCarryTest, PrefillPenaltyAndVocabularyFailureRejectsSeed) {
  auto calls = std::make_shared<std::vector<ModelCall>>();
  DraftInputCarryWorker worker(
      device(), make_options(/*speculative_tokens=*/2), calls);
  worker.fail_vocabulary_on_call(/*call_index=*/1);

  LlmForwardInput prefill = make_prefill(device());
  prefill.sampling_params.repetition_penalties = torch::tensor({1.2F});
  prefill.sampling_params.unique_token_ids =
      torch::tensor({{10, 11, 12}}, torch::kLong);
  prefill.sampling_params.unique_token_counts =
      torch::ones({1, 3}, torch::kInt);
  prefill.sampling_params.unique_token_ids_lens =
      torch::tensor({3}, torch::kInt);
  prefill.sampling_params = prefill.sampling_params.to(device(), torch::kFloat);

  const auto output = worker.step(prefill);
  ASSERT_TRUE(output.has_value());
  ASSERT_TRUE(output->ready_event == nullptr ||
              output->ready_event->synchronize());
  expect_failed_request(*output);
}

TEST_P(MtpDraftInputCarryFailureTest, FailedSeedKeepsQueuedOverlapTurnsAlive) {
  auto calls = std::make_shared<std::vector<ModelCall>>();
  calls->reserve(16);
  runtime::Options options = make_options(/*speculative_tokens=*/4);
  options.enable_schedule_overlap(true);
  DraftInputCarryWorker worker(device(), options, calls);
  worker.fail_vocabulary_on_call(GetParam());

  // These scheduler inputs already exist when the earlier turn fails. They
  // deliberately contain no copy of that turn's newly reported request error.
  const LlmForwardInput first_decode = make_decode(device());
  const LlmForwardInput queued_decode =
      make_decode(device(), /*token=*/-1, /*position=*/kRootPosition + 1);
  const auto prefill = worker.step(make_prefill(device()));
  ASSERT_TRUE(prefill.has_value());
  ASSERT_TRUE(prefill->ready_event == nullptr ||
              prefill->ready_event->synchronize());
  if (GetParam() == 1) {
    expect_failed_request(*prefill);
    EXPECT_FALSE(worker.cache()
                     .read_draft_input({kEmbeddingId}, {kRequestId})
                     .has_value());

    // A queued mixed-PD turn can route this same failed request through prefill
    // again. Its tombstone must survive even though the hook no longer fails.
    const auto repeated_prefill = worker.step(make_prefill(device()));
    ASSERT_TRUE(repeated_prefill.has_value());
    ASSERT_TRUE(repeated_prefill->ready_event == nullptr ||
                repeated_prefill->ready_event->synchronize());
    expect_failed_request(*repeated_prefill);
    EXPECT_FALSE(worker.cache()
                     .read_draft_input({kEmbeddingId}, {kRequestId})
                     .has_value());
  } else {
    ASSERT_TRUE(worker.cache()
                    .read_draft_input({kEmbeddingId}, {kRequestId})
                    .has_value());
  }

  const size_t first_decode_begin = calls->size();
  const auto failed = worker.step(first_decode);
  ASSERT_TRUE(failed.has_value());
  ASSERT_TRUE(failed->ready_event == nullptr ||
              failed->ready_event->synchronize());
  expect_failed_request(*failed);
  expect_k4_decode_phases(*calls, first_decode_begin);
  EXPECT_FALSE(worker.cache()
                   .read_draft_input({kEmbeddingId}, {kRequestId})
                   .has_value());

  // The hook fails only once. A tombstone, not a fresh hook failure or a fake
  // published seed, must protect the already-queued turn from missing-seed
  // CHECK.
  const size_t queued_decode_begin = calls->size();
  const auto queued = worker.step(queued_decode);
  ASSERT_TRUE(queued.has_value());
  ASSERT_TRUE(queued->ready_event == nullptr ||
              queued->ready_event->synchronize());
  expect_failed_request(*queued);
  expect_k4_decode_phases(*calls, queued_decode_begin);
  EXPECT_FALSE(worker.cache()
                   .read_draft_input({kEmbeddingId}, {kRequestId})
                   .has_value());

  // Releasing the slot must also remove its failure identity. A replacement
  // request can publish a real prefill seed rather than inheriting the failure.
  LlmForwardInput replacement =
      make_input(device(),
                 /*tokens=*/{20, 21, 22},
                 /*positions=*/{0, 1, 2},
                 BatchForwardType::PREFILL,
                 kEmbeddingId,
                 /*request_id=*/"replacement-request");
  replacement.request_state.released_embedding_ids = {kEmbeddingId};
  const auto replacement_output = worker.step(replacement);
  ASSERT_TRUE(replacement_output.has_value());
  ASSERT_TRUE(replacement_output->ready_event == nullptr ||
              replacement_output->ready_event->synchronize());
  EXPECT_TRUE(replacement_output->json_object_errors.empty());
  const auto replacement_draft_input =
      worker.cache().read_draft_input({kEmbeddingId}, {"replacement-request"});
  ASSERT_TRUE(replacement_draft_input.has_value());
  EXPECT_EQ(tensor_to_vector<int64_t>(replacement_draft_input->token_ids),
            std::vector<int64_t>({24}));
}

INSTANTIATE_TEST_SUITE_P(
    PrefillMiddleDraftAndExtend,
    MtpDraftInputCarryFailureTest,
    // K4 hook order: prefill, recursive 1/2/3, extend. Call 3 fails the middle
    // recursive forward, leaving another real draft forward to execute.
    ::testing::Values(1, 3, 5),
    [](const ::testing::TestParamInfo<int32_t>& info) {
      return "FailVocabularyCall" + std::to_string(info.param);
    });

TEST_F(MtpDraftInputCarryTest, EmptyJsonDraftVocabularyRejectsOnlyProposals) {
  std::vector<std::string> pieces(kVocabSize, "?");
  pieces[kRootToken] = "{";
  pieces[14] = "}";
  JsonObjectGrammar grammar(std::move(pieces), /*stop_token_ids=*/{63});
  JsonObjectGrammarState root_state = grammar.initial_state();
  ASSERT_TRUE(root_state.accept_token(kRootToken));

  for (bool probabilistic : {false, true}) {
    SCOPED_TRACE(probabilistic);
    for (int32_t empty_vocabulary_call : {1, 3, 5}) {
      SCOPED_TRACE(empty_vocabulary_call);
      auto calls = std::make_shared<std::vector<ModelCall>>();
      calls->reserve(12);
      runtime::Options options = make_options(/*speculative_tokens=*/4);
      options.enable_schedule_overlap(true);
      options.draft_sampling_mode(probabilistic ? "probabilistic" : "greedy");
      DraftInputCarryWorker worker(device(), options, calls);
      worker.fail_vocabulary_on_call(empty_vocabulary_call);
      LlmForwardInput prefill_input = make_prefill(device());
      prefill_input.json_object_states = {grammar.initial_state()};
      prefill_input.sampling_params.filter_bitmask =
          grammar.build_filter_bitmask(grammar.initial_state(), device())
              .unsqueeze(/*dim=*/0);
      LlmForwardInput decode = make_decode(device());
      decode.json_object_states = {root_state};
      decode.sampling_params.filter_bitmask =
          grammar.build_filter_bitmask(root_state, device()).unsqueeze(0);
      LlmForwardInput queued_decode =
          make_decode(device(), /*token=*/-1, /*position=*/kRootPosition + 1);
      queued_decode.json_object_states = {root_state};
      queued_decode.sampling_params.filter_bitmask =
          grammar.build_filter_bitmask(root_state, device()).unsqueeze(0);
      for (LlmForwardInput* input : {&prefill_input, &decode, &queued_decode}) {
        input->sampling_params.do_sample =
            torch::tensor({probabilistic}, torch::kBool);
        input->sampling_params.all_greedy_sample = !probabilistic;
        input->sampling_params.all_random_sample = probabilistic;
      }

      const auto prefill = worker.step(prefill_input);
      ASSERT_TRUE(prefill.has_value());
      ASSERT_TRUE(prefill->ready_event == nullptr ||
                  prefill->ready_event->synchronize());
      ASSERT_TRUE(prefill->json_object_errors.empty());
      ASSERT_TRUE(worker.cache()
                      .read_draft_input({kEmbeddingId}, {kRequestId})
                      .has_value());

      worker.set_target_predictions({14, 63, 63, 63, 63});
      const auto output = worker.step(decode);
      ASSERT_TRUE(output.has_value());
      ASSERT_TRUE(output->ready_event == nullptr ||
                  output->ready_event->synchronize());
      ASSERT_TRUE(output->json_object_errors.empty());
      expect_k4_decode_phases(*calls, /*begin=*/2);
      const std::vector<int64_t> accepted =
          tensor_to_vector<int64_t>(output->sample_output.next_tokens);
      EXPECT_EQ(accepted.front(), 14);
      EXPECT_EQ(std::count(accepted.begin(), accepted.end(), 0), 0);
      ASSERT_TRUE(worker.cache()
                      .read_draft_input({kEmbeddingId}, {kRequestId})
                      .has_value());

      // A turn queued before completion must reject any placeholder seed too.
      worker.set_target_predictions({63, 63, 63, 63, 63});
      const auto queued = worker.step(queued_decode);
      ASSERT_TRUE(queued.has_value());
      ASSERT_TRUE(queued->ready_event == nullptr ||
                  queued->ready_event->synchronize());
      ASSERT_TRUE(queued->json_object_errors.empty());
      expect_k4_decode_phases(*calls, /*begin=*/7);
      const std::vector<int64_t> stop_tokens =
          tensor_to_vector<int64_t>(queued->sample_output.next_tokens);
      EXPECT_EQ(stop_tokens.front(), 63);
      EXPECT_EQ(std::count(stop_tokens.begin(), stop_tokens.end(), 0), 0);
      EXPECT_TRUE(worker.cache()
                      .read_draft_input({kEmbeddingId}, {kRequestId})
                      .has_value());
    }
  }
}

class MtpDraftInputCarryDeathTest : public MtpDraftInputCarryTest {};

TEST_F(MtpDraftInputCarryDeathTest,
       MissingDraftInputFailsInsteadOfReplayingDraft) {
  // Re-exec, rather than forking an initialized accelerator runtime.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        auto calls = std::make_shared<std::vector<ModelCall>>();
        DraftInputCarryWorker worker(
            device(), make_options(/*speculative_tokens=*/2), calls);
        LlmForwardInput input = make_decode(device());
        worker.bootstrap_target_state(kEmbeddingId, kRequestId, kRootToken);
        worker.step(input);
      },
      "MTP decode requires a matching prefill/extend draft input");
}

}  // namespace
}  // namespace xllm
