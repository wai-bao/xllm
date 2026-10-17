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

#include "core/runtime/speculative_output.h"

#include <torch/torch.h>

#include "common/macros.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/target_decode_state_store.h"
#include "runtime/vlm_forward_params.h"

namespace xllm {

namespace {

void clear_selected_embeddings(ForwardOutput& output) {
  output.sample_output.selected_embeddings = torch::Tensor();
}

}  // namespace

void output_captured_hidden_states(
    SampleOutput& sample_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& aux_hidden_states,
    const torch::Tensor& logits_indices,
    const torch::Tensor& gathered_sample_hidden_states,
    CaptureHiddenMode mode,
    bool is_target_prefill,
    bool cp_enabled) {
  CHECK(mode != CaptureHiddenMode::CAPTURED_LAYERS ||
        aux_hidden_states.defined())
      << "speculative target requires captured auxiliary hidden states";
  CHECK(mode != CaptureHiddenMode::MODEL_HOOK || aux_hidden_states.defined())
      << "speculative target requires model-provided hidden states";
  // Every non-primary mode consumes aux rows; the CHECKs above already pin
  // the two modes that must carry them, so MODEL_PROVIDED is the only one
  // whose aux is genuinely optional.
  const bool use_aux =
      mode != CaptureHiddenMode::FINAL_OUTPUT && aux_hidden_states.defined();
  const torch::Tensor& target_hidden_states =
      use_aux ? aux_hidden_states : hidden_states;
  torch::Tensor sample_hidden_states;
  if (logits_indices.defined()) {
    // CP decode reuses pre-gathered primary rows only when the mode is the
    // primary output. Captured and model-provided aux rows are global-ordered
    // by their model before selecting the same logits indices.
    const bool must_use_gathered = cp_enabled && !is_target_prefill && !use_aux;
    if (must_use_gathered) {
      CHECK(gathered_sample_hidden_states.defined())
          << "gathered selected hidden is required on the CP decode path";
    }
    // must_use_gathered already implies !use_aux, so the pre-gathered rows are
    // exactly the primary-mode case.
    if (gathered_sample_hidden_states.defined() && !use_aux) {
      sample_hidden_states = gathered_sample_hidden_states;
    } else {
      CHECK(target_hidden_states.defined())
          << "speculative hidden states are undefined";
      sample_hidden_states = target_hidden_states.index_select(
          /*dim=*/0,
          logits_indices.to(torch::dtype(torch::kLong)
                                .device(target_hidden_states.device())));
    }
    CHECK_EQ(sample_hidden_states.dim(), 2);
    CHECK_EQ(sample_hidden_states.size(0), logits_indices.numel())
        << "selected speculative hidden must match logits_indices rows";
  }
  if (is_target_prefill) {
    sample_output.embeddings = target_hidden_states;
    sample_output.selected_embeddings = sample_hidden_states;
  } else {
    sample_output.embeddings = sample_hidden_states;
  }
}

void clear_all_output_embeddings(ForwardOutput& output) {
  output.sample_output.embeddings = torch::Tensor();
  clear_selected_embeddings(output);
}

void prepare_target_prefill_state(TargetDecodeStateStore& target_state_store,
                                  const LlmForwardInput& input,
                                  ForwardOutput& output) {
  const torch::Tensor& sample_hidden_states =
      output.sample_output.selected_embeddings;
  CHECK(sample_hidden_states.defined())
      << "target worker must populate "
         "selected_embeddings on the speculative "
         "prefill path";
  target_state_store.write_prefill_target_context(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      output.sample_output.next_tokens);
  output.sample_output.embeddings = input.sampling_params.is_embeddings
                                        ? sample_hidden_states.detach()
                                        : torch::Tensor();
  clear_selected_embeddings(output);
}

}  // namespace xllm
