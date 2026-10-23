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

#include <array>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "core/framework/sampling/rejection_sampler.h"
#include "core/framework/speculative/adaptive/spec_cost_profile_store.h"
#include "core/framework/speculative/adaptive/spec_startup_calibration.h"
#include "core/framework/speculative/adaptive/verify_budget_planner.h"
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
class SpecContextStorage final {
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
class TaskExecutionPipeline::SpeculativeSlot {
 public:
  struct Adaptive {
    VerifyBudgetPlan plan;
    uint64_t generation = 0;
    uint32_t max_kv_length = 0;
    uint32_t selected_draft_tokens = 0;
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    std::vector<double> lagged_confidence;
    std::vector<int32_t> draft_counts;
    // Complete raw-B/G bindings are captured at drained startup for this Slot.
    // Admission checks this set before mutating persistent request context.
    std::unordered_set<uint64_t> unified_graph_bindings;
    // Proposal IDs, confidence and widths share an owned arena. Proposal +
    // confidence consensus is one broadcast; confidence + width feedback is
    // one tiny D2H copy covered by the ordinary result-ready fence.
    torch::Tensor arena_storage;
    // Control aliases otherwise-unused adaptive rejection-mask storage. The
    // pinned source borrows retired feedback until the next result D2H starts.
    torch::Tensor control;
    torch::Tensor host_control;
    torch::Tensor consensus;
    torch::Tensor confidence;
    torch::Tensor widths;
    torch::Tensor feedback;
    torch::Tensor host_feedback_storage;
    torch::Tensor host_feedback;
    torch::Tensor host_confidence;
    torch::Tensor host_widths;
    torch::Tensor previous_storage;
    torch::Tensor previous_tokens;
    torch::Tensor offsets_storage;
    torch::Tensor offsets;
    torch::Tensor owners_storage;
    torch::Tensor owners;
    torch::Tensor columns_storage;
    torch::Tensor columns;
  };
  struct Draft {
    std::vector<int64_t> graph_batch_sizes;
    std::vector<int64_t> captured_graph_batch_sizes;
    // First warmed compact FIA table bound for each physical draft bucket.
    std::unordered_map<uint32_t, uint64_t> captured_graph_kv_capacities;
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
  // True while host_sampling reflects the current row set (decode rounds of a
  // stable batch skip the per-round tensor rebuild; see prepare_speculative).
  bool host_sampling_valid = false;
  std::optional<DraftInput> draft_input;
  // Pinned host copy of the last committed tokens, staged before the draft
  // forward so penalty imports can read a settled buffer (see
  // stage_mtp_draft_input_tokens).
  torch::Tensor staged_input_tokens;
  // Reused D2H-completion event for the staged tokens (see
  // stage_mtp_draft_input_tokens).
  StreamEventPtr penalty_copy_event;
  bool decode = false;
  // Immutable until this Slot retires; capacity remains the maximum depth.
  uint32_t draft_steps = 0;
  std::unique_ptr<Adaptive> adaptive;
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
  // Layout contract of `drafts`, established once at construction (see
  // prepare_speculative): block-draft holds a single BLOCK_DRAFT entry,
  // otherwise the recursive DRAFT steps are followed by one trailing EXTEND
  // invocation that only produces the next round's seed.
  bool has_extend_step() const {
    return !drafts.empty() &&
           drafts.back()->input->input_scratch_->spec_.kind ==
               SpecInvocationKind::EXTEND;
  }
  uint32_t recursive_draft_calls() const {
    return static_cast<uint32_t>(drafts.size()) - (has_extend_step() ? 1 : 0);
  }
  Draft& extend_step() {
    CHECK(has_extend_step());
    return *drafts.back();
  }
  std::unique_ptr<Draft> prefix_refresh;
  std::vector<int32_t> upper_positions;
  std::vector<int32_t> upper_kv_lengths;
  SamplingParameters host_validate_sampling;
  torch::Tensor draft_block_indices;
  torch::Tensor target_token_storage;
  torch::Tensor target_tokens;
  torch::Tensor proposal_storage;
  // Capacity-grown rejection-mask storage; each round narrows the current
  // shape out of it instead of reallocating (mirrors proposal_storage).
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

struct TaskExecutionPipeline::DSparkAdaptiveState {
  struct Feedback {
    uint64_t generation = 0;
    uint64_t rounds_since_full_width = 0;
    // Host-only history uses the largest compiled width. Only the configured
    // prefix is copied/consumed; Device and pinned buffers use the exact width.
    std::array<double, 7> confidence{};
  };
  struct History {
    std::array<Feedback, 2> versions;
  };
  struct RetiredPlan {
    uint64_t generation = 0;
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    VerifyBudgetPlan plan;
  };
  uint64_t retired_generation = 0;
  std::vector<History> histories;
  std::array<RetiredPlan, 2> retired_plans;
  std::shared_ptr<SpecCostProfileStore> owned_costs;
  std::optional<SpecCalibrationPoint> calibration_point;
  std::unique_ptr<SpecCalibrationSession> calibration_session;
  bool startup_calibration_complete = false;
};

}  // namespace xllm
