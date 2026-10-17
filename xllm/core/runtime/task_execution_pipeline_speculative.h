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

#pragma once

#include <optional>

#include "core/framework/sampling/rejection_sampler.h"
#include "core/framework/speculative/draft_input_state_store.h"
#include "core/runtime/task_execution_pipeline.h"

namespace xllm {

// Admission occurs before any graph execution mutates the task state.
bool supports_unified_block_graph(const SpeculativeTaskCapacity& capacity,
                                  uint32_t rows,
                                  const LlmForwardInput& input);

struct SpecContextTensors {
  torch::Tensor positions;
  torch::Tensor kv_seq_lens;
};

// Tensor rows use the existing EMBEDDING block ids. The scheduler owns their
// allocation and reuse; this storage has no allocator or retirement protocol.
// Every gather/publication runs in FIFO order on the pipeline's task stream.
class SpecContextStorage {
 public:
  // Retires one embedding row: unpublishes it and drops the request guard.
  // The single seam every slot-release path routes through.
  void release_slot(int32_t id) {
    published_[id] = 0;
    request_ids_[id].clear();
  }

  // Each embedding row is guarded by its owning request id. Metadata promises
  // an earlier accepted Task will publish before a following decode reads it.
  torch::Tensor tokens_;
  SpecContextTensors spec_state_;
  std::vector<std::string> request_ids_;
  std::vector<uint8_t> published_;
};

// One Slot's fixed snapshot; embedding rows belong to the scheduler.
struct SpecContextView {
  SpecContextStorage* context_ = nullptr;
  uint32_t capacity_;
  bool prepared_ = false;
  bool read_published_state_ = false;
  std::vector<int32_t> sorted_ids_;
  std::vector<uint32_t> identity_rows_;
  std::vector<int32_t> model_to_embedding_;
  std::vector<uint8_t> publication_mask_;
  torch::Tensor host_rows_;
  torch::Tensor device_rows_;
  torch::Tensor rows_;
  torch::Tensor token_storage_;
  SpecContextTensors storage_;
  torch::Tensor tokens_;
  SpecContextTensors state_;
  torch::Tensor index_storage_;
  torch::Tensor last_indices_;
  torch::Tensor current_token_output_;
};

// Fixed storage for the forwards in one speculative round. No scheduler,
// streams or independently retiring execution object lives here.
struct TaskExecutionPipeline::SpeculativeSlot {
  struct Draft {
    std::vector<int64_t> graph_batch_sizes;
    std::vector<int64_t> captured_graph_batch_sizes;
    uint32_t padded_batch_size = 0;
    std::unique_ptr<SlotBuffer> input;
    std::optional<Sampling> sampling;
    torch::Tensor token_storage;
    torch::Tensor hidden_storage;
    torch::Tensor topk_storage;
    torch::Tensor next_hidden;
    torch::Tensor next_topk;
    ModelOutput output;
    torch::Tensor logits;
    SampleOutput sampled;
  };

  // Seed store calls run only on the FIFO launch thread. Slots own these keys
  // because Prepare may already admit a later batch while Launch runs.
  std::vector<int32_t> embedding_ids;
  std::vector<std::string> request_ids;
  std::vector<int32_t> released_embedding_ids;
  SamplingParameters host_sampling;
  std::optional<DraftInput> draft_input;
  // Pinned host copy of the last committed tokens, staged before the draft
  // forward so penalty imports can read a settled buffer (see
  // stage_draft_input_tokens).
  torch::Tensor staged_input_tokens;
  // Reused D2H-completion event for the staged tokens (see
  // stage_draft_input_tokens).
  StreamEventPtr penalty_copy_event;
  bool decode = false;
  bool run_models = false;
  torch::Tensor dummy_hidden;
  bool is_warmup = false;
  bool unified_graph = false;
  uint32_t rows = 0;
  uint32_t samples = 0;
  uint32_t model_tokens = 0;
  uint32_t target_padded_batch_size = 0;
  std::unique_ptr<SlotBuffer> target_prefill;
  std::unique_ptr<SlotBuffer> draft_prefill;
  std::unique_ptr<SlotBuffer> validate_input;
  std::unique_ptr<SpecContextView> state;
  std::optional<Sampling> target_sampling;
  std::unique_ptr<RejectionSampler> rejection;
  std::vector<std::unique_ptr<Draft>> drafts;
  std::vector<int32_t> upper_positions;
  std::vector<int32_t> upper_kv_lengths;
  SamplingParameters host_validate_sampling;
  torch::Tensor draft_block_indices;
  torch::Tensor draft_sample_indices;
  torch::Tensor greedy_do_sample;
  torch::Tensor proposal_storage;
  // Capacity-grown [rows, width-1, vocab] dense-q stack; each round narrows
  // the current shape out of it instead of reallocating (mirrors
  // proposal_storage).
  torch::Tensor draft_probs_storage;
  torch::Tensor accepted_mask_storage;
  torch::Tensor host_hidden_storage;
  torch::Tensor proposal;
  torch::Tensor accepted_mask;
  torch::Tensor context_slots_storage;
  torch::Tensor context_slots;
  torch::Tensor block_gumbel;
  torch::Tensor host_hidden;
  ModelOutput target_output;
  torch::Tensor target_logits;
};

}  // namespace xllm
