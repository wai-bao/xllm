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

#include "runtime/dspark_worker_impl.h"

#include <c10/core/StreamGuard.h>
#include <glog/logging.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/common/types.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/model_type.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/sampling/sampling_params.h"
#include "runtime/llm_worker_impl.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
namespace {

// The DSpark Markov head: [rows] tokens in, [rows, vocab] bias out. Both
// callers hold the draft model privately, so the head enters as a callable.
using DSparkMarkovBias =
    std::function<torch::Tensor(const torch::Tensor& previous_token_ids)>;

// The sampled tokens must agree across TP ranks before the next step;
// greedy argmax is rank-local, so the walker never invokes this on greedy
// batches. The caller owns which process groups to broadcast over.
using SyncSampledTokens = std::function<void(torch::Tensor& sampled_tokens)>;

// Walks the block logits [rows, steps, vocab] step by step through
// `markov_bias` and the shared sample(); `anchors` seed the first step.
DraftProposal sample_dspark_markov_block(
    const DSparkMarkovBias& markov_bias,
    const torch::Tensor& base_logits,
    const torch::Tensor& anchors,
    const SamplingParameters& sampling,
    const SyncSampledTokens& sync_tokens,
    bool is_probabilistic_draft,
    const DraftVocabMapping& vocab_mapping) {
  const int64_t rows = base_logits.size(0);
  const int64_t steps = base_logits.size(1);
  const int64_t draft_model_vocab = base_logits.size(2);
  torch::Tensor token_ids = torch::empty({rows, steps}, anchors.options());
  const auto logits_options = is_probabilistic_draft
                                  ? base_logits.options().dtype(torch::kFloat32)
                                  : base_logits.options();
  torch::Tensor draft_logits;
  if (sampling.sampling_mode() != SamplingMode::GREEDY) {
    draft_logits = torch::empty(
        {rows, steps, vocab_mapping.mapped_vocab_size(draft_model_vocab)},
        logits_options);
  }
  const torch::Tensor writable_draft_logits =
      vocab_mapping.draft_logits_view(draft_logits);
  const torch::Tensor workspace =
      writable_draft_logits.defined()
          ? torch::Tensor()
          : torch::empty({rows, draft_model_vocab}, logits_options);
  auto previous = anchors;

  for (int64_t step = 0; step < steps; ++step) {
    auto bias = markov_bias(previous);
    CHECK_EQ(bias.sizes(), torch::IntArrayRef({rows, draft_model_vocab}))
        << "DSpark markov bias must be [rows, draft_model_vocab].";
    torch::Tensor step_logits =
        workspace.defined() ? workspace
                            : writable_draft_logits.select(/*dim=*/1, step);
    torch::add_out(step_logits, base_logits.select(/*dim=*/1, step), bias);

    auto sampled = sample(step_logits, sampling);
    if (sampling.sampling_mode() != SamplingMode::GREEDY) {
      sync_tokens(sampled.next_tokens);
    }
    torch::Tensor target_logits;
    if (draft_logits.defined() && workspace.defined()) {
      sampled.draft_logits = step_logits;
      target_logits = draft_logits.select(/*dim=*/1, step);
    }
    vocab_mapping.remap(sampled, std::move(target_logits));
    token_ids.select(/*dim=*/1, step).copy_(sampled.next_tokens);
    previous = sampled.next_tokens;
  }
  return DraftProposal(std::move(token_ids), std::move(draft_logits));
}

}  // namespace

DSparkWorkerImpl::DSparkWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : DFlashWorkerImpl(parallel_args, device, options) {
  sample_from_anchor_ = true;
}

bool DSparkWorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  MasterStatus master_status) {
  const bool result = DFlashWorkerImpl::init_model(
      model_weights_path, random_seed, master_status);
  if (draft_impl_->get_status() != WorkerImpl::Status::LOADED) {
    return result;
  }

  const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
  const ModelArgs& target_args = impl_->context_.get_model_args();
  const int64_t draft_vocab_size = draft_args.effective_draft_vocab_size();
  CHECK(draft_vocab_size == target_args.vocab_size() ||
        draft_vocab_.mapped_vocab_size(draft_vocab_size) != draft_vocab_size)
      << "A reduced draft vocabulary requires a DSpark d2t mapping.";

  if (draft_args.model_type() == "deepseek_v4_dspark") {
    draft_input_per_request_ = draft_args.dspark_use_native_sas();
    draft_forward_type_ = BatchForwardType::DECODE;
  }
  draft_needs_dummy_tokens_ =
      draft_input_per_request_ &&
      ModelConfig::is_python_model_impl(context_.get_model_impl());
  return result;
}

std::pair<torch::Tensor, torch::Tensor> DSparkWorkerImpl::run_decode_draft(
    const LlmForwardInput& input,
    LlmForwardInput& validate_input,
    LlmForwardInput& draft_forward_input) {
  Timer draft_timer;

  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();

  CHECK_GT(num_speculative_tokens, 0)
      << "DSpark requires num_speculative_tokens > 0.";
  torch::Tensor anchors;
  std::optional<ForwardOutput> draft_output_storage =
      launch_query_forward(input, anchors);
  CHECK(draft_output_storage.has_value())
      << "DSpark draft forward must return an output.";
  CHECK(draft_output_storage->logits.defined())
      << "DSpark draft forward must return logits.";
  prepare_validate_inputs(input, validate_input);

  ForwardOutput& draft_output = *draft_output_storage;
  const int64_t num_rows = draft_output.logits.size(/*dim=*/0);
  CHECK_EQ(num_rows % num_speculative_tokens, 0)
      << "DSpark draft logits rows must be divisible by "
         "num_speculative_tokens.";
  CHECK_EQ(num_rows / num_speculative_tokens, batch_size)
      << "DSpark draft logits batch must match the decode batch.";
  torch::Tensor base_logits = draft_output.logits.view(
      {batch_size, num_speculative_tokens, draft_output.logits.size(-1)});
  DraftProposal draft_proposal;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    SamplingParameters sampling_params_on_device = input.sampling_params;
    prepare_draft_sampling(is_probabilistic_draft_, sampling_params_on_device);
    sampling_params_on_device = sampling_params_on_device.to(
        base_logits.device(), base_logits.scalar_type());
    draft_proposal = sample_dspark_markov_block(
        [this](const torch::Tensor& tokens) {
          return draft_impl_->dspark_markov_bias(tokens);
        },
        base_logits,
        anchors,
        sampling_params_on_device,
        [this](torch::Tensor& tokens) { broadcast_spec_tokens(tokens); },
        is_probabilistic_draft_,
        draft_vocab_);
  }

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());
  return {draft_proposal.token_ids(),
          draft_proposal.logits().has_value() ? *draft_proposal.logits()
                                              : torch::Tensor()};
}

}  // namespace xllm
