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

#include "core/runtime/eagle3_worker_impl.h"

#include <glog/logging.h>

#include "core/framework/model/aux_hidden_capture.h"
#include "runtime/llm_worker_impl.h"

namespace xllm {

namespace {

runtime::Options eagle3_draft_options(const runtime::Options& options) {
  auto opts = speculative_draft_options(options);
  opts.enable_graph(false)
      .enable_graph_mode_decode_no_padding(false)
      .enable_prefill_piecewise_graph(false);
  return opts;
}

}  // namespace

template <typename TargetInput>
Eagle3WorkerImpl<TargetInput>::Eagle3WorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    WorkerType worker_type)
    : MTPWorkerImpl<TargetInput>(parallel_args,
                                 device,
                                 options,
                                 speculative_target_options(options),
                                 eagle3_draft_options(options),
                                 worker_type) {
  // Context parallelism does not expose the auxiliary hidden states.
  CHECK_LE(parallel_args.cp_size(), 1)
      << "EAGLE-3 speculative decoding does not support context parallelism "
         "(cp_size > 1).";
}

template <typename TargetInput>
Eagle3WorkerImpl<TargetInput>::~Eagle3WorkerImpl() {
  this->drain_pending_execution();
}

template <typename TargetInput>
bool Eagle3WorkerImpl<TargetInput>::init_model(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  bool result = MTPWorkerImpl<TargetInput>::init_model(
      model_weights_path, random_seed, master_status);

  // The base init already extracted the d2t offsets from the same state
  // dict and normalized them (draft_id + arange) into draft_vocab_.
  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    if (uses_embedded_eagle3_draft()) {
      vocab_map_ = VocabMap();
    } else if (draft_vocab_.draft_to_target_ids_.defined()) {
      vocab_map_ = VocabMap(draft_vocab_.draft_to_target_ids_,
                            context_.get_model_args().vocab_size());
    }
  }

  return result;
}

template <typename TargetInput>
void Eagle3WorkerImpl<TargetInput>::release_algorithm_state(
    const std::vector<int32_t>& embedding_ids) {
  MTPWorkerImpl<TargetInput>::release_algorithm_state(embedding_ids);
  if (!embedding_ids.empty()) {
    draft_history_batch_.reclaim_unused_projection();
  }
}

template <typename TargetInput>
std::vector<uint8_t>
Eagle3WorkerImpl<TargetInput>::prepare_draft_vocabulary_sampling(
    SamplingParameters& sampling_params,
    const DraftTokenHistories& histories) {
  if (!draft_token_mapping_active()) {
    return {};
  }
  return vocab_map_.remap(sampling_params, draft_history_batch_, histories);
}

template <typename TargetInput>
void Eagle3WorkerImpl<TargetInput>::remap_draft_sample_output(
    SampleOutput& sample_output) {
  // Same sparse-logits scatter and token remap as the DSpark proposal path.
  draft_vocab_.remap(sample_output);
}

template <typename TargetInput>
void Eagle3WorkerImpl<TargetInput>::check_draft_input_embedding(
    const torch::Tensor& embedding,
    const std::string& phase) const {
  if (!embedding.defined()) {
    CHECK_NE(phase, "prefill")
        << "Eagle3 prefill requires verifier aux hidden-state embeddings. "
        << "Check that the target model captures its aux hidden-state layers.";
    return;
  }

  const int64_t expected_hidden_size =
      AuxHiddenCapture::aux_hidden_dim(context_.get_model_args());
  const int64_t draft_hidden_size =
      draft_impl_ == nullptr ? 0 : draft_impl_->hidden_size();
  CHECK_EQ(embedding.dim(), 2)
      << "Eagle3 " << phase << " embedding must be a 2-D tensor, got dim "
      << embedding.dim();
  CHECK_GT(embedding.size(0), 0)
      << "Eagle3 " << phase << " embedding must contain at least one row.";
  if (phase == "decode") {
    CHECK(embedding.size(-1) == expected_hidden_size ||
          embedding.size(-1) == draft_hidden_size)
        << "Eagle3 " << phase
        << " embedding hidden size mismatch, expected target aux hidden size "
        << expected_hidden_size << " or draft hidden size " << draft_hidden_size
        << ", got " << embedding.size(-1);
    return;
  }

  CHECK_EQ(embedding.size(-1), expected_hidden_size)
      << "Eagle3 " << phase
      << " embedding hidden size mismatch, expected target aux hidden size "
      << expected_hidden_size << ", got " << embedding.size(-1)
      << ". Check that the target model captures its aux hidden-state layers "
         "for Eagle3.";
}

template class Eagle3WorkerImpl<LlmForwardInput>;
template class Eagle3WorkerImpl<VlmForwardInput>;

}  // namespace xllm
