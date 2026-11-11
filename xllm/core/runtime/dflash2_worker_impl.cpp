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

// DFlash2 draft selection: walks the candidate tree step by step, argmaxing
// each row over the predecessor's edge distribution with Gumbel noise.
// Model-specific knowledge lives here; the sampling directory holds only
// the model-agnostic gumbel primitives it consumes.

#include "runtime/dflash2_worker_impl.h"

#include <c10/core/StreamGuard.h>
#include <glog/logging.h>

#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "common/metrics.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/logits_processor.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/util/tensor_helper.h"
#include "util/timer.h"

namespace xllm {

bool DFlash2WorkerImpl::init_model(const std::string& model_weights_path,
                                   int32_t random_seed,
                                   MasterStatus master_status) {
  const bool result = DFlashWorkerImpl::init_model(
      model_weights_path, random_seed, master_status);
  if (result && draft_impl_->get_status() == WorkerImpl::Status::LOADED &&
      is_probabilistic_draft_) {
    const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
    const ModelArgs& target_args = impl_->context_.get_model_args();
    CHECK_EQ(draft_args.effective_draft_vocab_size(), target_args.vocab_size())
        << "DFlash2 probabilistic drafts require draft and target vocabulary "
           "sizes to match.";
  }
  return result;
}

// The selector takes only temperature — truncation, penalties, and grammar
// constraints belong to verification.
std::pair<torch::Tensor, torch::Tensor> sample_dflash2_path(
    const DFlash2CandidateOutput& candidates,
    const SamplingParameters& sampling_params,
    torch::Tensor gumbel_noise,
    int64_t vocab_size) {
  CHECK_EQ(candidates.candidate_ids.dim(), 3);
  CHECK_EQ(candidates.edge_logits.dim(), 4);
  const int64_t batch_size = candidates.candidate_ids.size(0);
  const int64_t num_steps = candidates.candidate_ids.size(1);
  const int64_t top_k = candidates.candidate_ids.size(2);
  CHECK_EQ(candidates.edge_logits.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k, top_k}));
  const bool is_probabilistic_draft = gumbel_noise.defined();
  if (is_probabilistic_draft) {
    CHECK_EQ(gumbel_noise.sizes(),
             torch::IntArrayRef({batch_size, num_steps, top_k}));
  }
  const torch::Device device = candidates.edge_logits.device();
  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  torch::Tensor edge_logits = candidates.edge_logits.to(torch::kFloat32);
  if (sampling_params.temperatures().defined()) {
    apply_temperatures(edge_logits, sampling_params.temperatures());
  }
  // log_softmax is monotone per row, so argmax(log_probs + noise) matches
  // argmax(logits + noise); normalize only when q is reported.
  torch::Tensor edge_log_probs =
      is_probabilistic_draft ? torch::log_softmax(edge_logits, /*dim=*/-1)
                             : edge_logits;

  torch::Tensor token_ids =
      torch::empty({batch_size, num_steps}, candidates.candidate_ids.options());
  torch::Tensor draft_logits;
  if (is_probabilistic_draft) {
    draft_logits = torch::full({batch_size, num_steps, vocab_size},
                               -std::numeric_limits<float>::infinity(),
                               float_options);
  }
  torch::Tensor previous_indices =
      torch::zeros({batch_size}, candidates.candidate_ids.options());

  torch::Tensor greedy_rows;
  if (is_probabilistic_draft &&
      sampling_params.sampling_mode() == SamplingMode::MIXED) {
    greedy_rows = greedy_mask(sampling_params, device);
  }

  for (int64_t step = 0; step < num_steps; ++step) {
    torch::Tensor gather_indices = previous_indices.view({batch_size, 1, 1})
                                       .expand({batch_size, 1, top_k});
    torch::Tensor row_log_probs =
        edge_log_probs.select(/*dim=*/1, /*index=*/step)
            .gather(/*dim=*/1, gather_indices)
            .squeeze(/*dim=*/1);
    // gumbel_argmax consumes its noise buffer in place; each step's slice
    // has no later reader, so no clone.
    torch::Tensor sampled_indices =
        is_probabilistic_draft
            ? gumbel_argmax(row_log_probs,
                            gumbel_noise.select(/*dim=*/1, /*index=*/step))
            : greedy_sample(row_log_probs);

    torch::Tensor step_candidates =
        candidates.candidate_ids.select(/*dim=*/1, /*index=*/step);
    torch::Tensor sampled_tokens =
        index_select_2d(step_candidates, /*dim=*/1, sampled_indices);
    if (is_probabilistic_draft) {
      torch::Tensor step_log_probs = row_log_probs;
      if (sampling_params.sampling_mode() == SamplingMode::MIXED) {
        // Greedy rows embed a log-space one-hot (q = 1).
        step_log_probs = torch::where(
            greedy_rows,
            point_mass_logits(sampled_indices, top_k, float_options),
            step_log_probs);
      }
      draft_logits.select(/*dim=*/1, step)
          .scatter_(/*dim=*/1, step_candidates, step_log_probs);
    }
    token_ids.select(/*dim=*/1, step).copy_(sampled_tokens);
    previous_indices = sampled_indices;
  }

  return {std::move(token_ids), std::move(draft_logits)};
}

namespace {
torch::Tensor broadcast_gumbel_noise(
    int64_t batch_size,
    const SamplingParameters& sampling,
    int64_t num_steps,
    int64_t num_classes,
    const torch::Device& device,
    const std::function<void(torch::Tensor&)>& sync_noise) {
  if (sampling.sampling_mode() == SamplingMode::GREEDY) {
    return torch::Tensor();
  }
  CHECK_GT(batch_size, 0);
  torch::Tensor noise =
      gumbel_noise(sampling, {batch_size, num_steps, num_classes}, device);
  sync_noise(noise);
  return noise;
}
}  // namespace

DFlash2WorkerImpl::DFlash2WorkerImpl(const ParallelArgs& parallel_args,
                                     const torch::Device& device,
                                     const runtime::Options& options)
    : DFlashWorkerImpl(parallel_args, device, options) {}

std::pair<torch::Tensor, torch::Tensor> DFlash2WorkerImpl::run_decode_draft(
    const LlmForwardInput& input,
    LlmForwardInput& validate_input,
    LlmForwardInput& draft_forward_input) {
  Timer draft_timer;

  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_speculative_tokens, 0);

  const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
  const int64_t selector_top_k = draft_args.dflash2_selector_top_k();
  c10::StreamGuard noise_guard = compute_stream_->set_stream_guard();
  SamplingParameters draft_sampling_params = input.sampling_params;
  prepare_draft_sampling(is_probabilistic_draft_, draft_sampling_params);
  draft_sampling_params =
      draft_sampling_params.to(draft_impl_->device(), torch::kFloat32);
  torch::Tensor selector_noise = broadcast_gumbel_noise(
      batch_size,
      draft_sampling_params,
      num_speculative_tokens,
      selector_top_k,
      draft_impl_->device(),
      [this](torch::Tensor& noise) { broadcast_spec_tokens(noise); });

  torch::Tensor anchors_storage;
  std::optional<ForwardOutput> draft_output_storage =
      launch_query_forward(input, anchors_storage);
  CHECK(draft_output_storage.has_value());
  ForwardOutput& draft_output = *draft_output_storage;
  CHECK(draft_output.logits.defined());
  CHECK(draft_output.selected_hidden.defined())
      << "DFlash2 requires selected pre-lm-head hidden states.";
  prepare_validate_inputs(input, validate_input);

  const int64_t num_rows = draft_output.logits.size(0);
  CHECK_EQ(num_rows, static_cast<int64_t>(batch_size) * num_speculative_tokens);
  torch::Tensor unary_logits = draft_output.logits.view(
      {batch_size, num_speculative_tokens, draft_output.logits.size(-1)});
  torch::Tensor hidden_states = draft_output.selected_hidden.view(
      {batch_size,
       num_speculative_tokens,
       draft_output.selected_hidden.size(-1)});

  const DFlash2CandidateOutput candidates =
      draft_impl_->get_causal_lm()->dflash2_candidates(
          hidden_states, unary_logits, anchors_storage);
  auto [draft_token_ids, draft_logits] =
      sample_dflash2_path(candidates,
                          draft_sampling_params,
                          selector_noise,
                          unary_logits.size(/*dim=*/-1));
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());
  return {std::move(draft_token_ids), std::move(draft_logits)};
}

}  // namespace xllm
