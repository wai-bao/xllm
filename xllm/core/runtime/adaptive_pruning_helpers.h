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

#include <glog/logging.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "core/framework/speculative/target_decode_state_store.h"
#include "core/runtime/forward_params.h"

// Runtime-only post-processing for speculative validation outputs.

namespace xllm {
namespace adaptive_pruning {

// Extract per-step selected token probabilities from draft outputs into
// a [batch, num_speculative_tokens] matrix for the pruning controller.
torch::Tensor selected_probs_by_step(
    std::span<const ForwardOutput> draft_outputs);

// Check whether all draft outputs have defined probs (required for pruning).
bool has_selected_probs_by_step(std::span<const ForwardOutput> draft_outputs);

// Clamp pruning prefix lengths to valid range [0, num_speculative_tokens].
void clamp_prefix_lengths(std::vector<int32_t>& prefix_lengths,
                          int32_t batch_size,
                          int32_t num_speculative_tokens);

// Get the maximum prefix length in the batch (determines padded validate
// width).
int32_t max_pruned_prefix_length(const std::vector<int32_t>& prefix_lengths,
                                 int32_t num_speculative_tokens);

// Whether the selected prefixes actually reduce target validation work. A
// uniform-width backend only benefits when the batch-wide width shrinks.
bool reduces_validate_work(const std::vector<int32_t>& prefix_lengths,
                           int32_t num_speculative_tokens,
                           int32_t effective_speculative_tokens,
                           bool requires_uniform_validate_width);

struct MtpValidatePlan {
  bool use_static_tail = true;
  int32_t effective_speculative_tokens = 0;
  std::vector<int32_t> prefix_lengths;
  std::vector<int32_t> per_seq_val_tokens;
};

// Per-sequence KV lengths for the adaptive controller's cost model, derived
// from the input's host kv_seq_lens. Rows without a host entry stay 0.
// Defined in the .cpp to keep speculative_input_builder.h out of this header.
template <typename TargetInput>
std::vector<double> controller_kv_lens(const TargetInput& input,
                                       int32_t batch_size);

// The replay checkpoint can force full uniform width before host probability
// selection. The callback is invoked only when selection can reduce work.
template <typename SelectPrefixes>
MtpValidatePlan plan_mtp_validate(
    const std::vector<TargetDecodeStateStore::DecodeState>& last_states,
    int32_t num_speculative_tokens,
    bool supports_replay_update,
    bool requires_uniform_width,
    SelectPrefixes&& select_prefixes) {
  CHECK_GT(num_speculative_tokens, 0);
  int32_t max_accepted = 0;
  if (supports_replay_update) {
    for (const auto& state : last_states) {
      max_accepted = std::max(
          max_accepted, TargetDecodeStateStore::accepted_prefix_length(state));
    }
    if (requires_uniform_width && max_accepted >= num_speculative_tokens) {
      return {true, num_speculative_tokens, {}, {}};
    }
  }

  std::vector<int32_t> prefixes =
      std::forward<SelectPrefixes>(select_prefixes)();
  clamp_prefix_lengths(prefixes,
                       static_cast<int32_t>(last_states.size()),
                       num_speculative_tokens);
  // Replay and uniform-width support are independent capabilities. If a
  // replay backend supports variable widths, each row must still contain its
  // own committed checkpoint even when the controller selects a shorter
  // rejection prefix. Keep the selected prefixes for the sampler.
  std::vector<int32_t> work_prefixes;
  const std::vector<int32_t>* validation_prefixes = &prefixes;
  if (supports_replay_update && !requires_uniform_width) {
    work_prefixes = prefixes;
    for (size_t row = 0; row < last_states.size(); ++row) {
      work_prefixes[row] = std::max(
          work_prefixes[row],
          std::min(
              TargetDecodeStateStore::accepted_prefix_length(last_states[row]),
              num_speculative_tokens));
    }
    validation_prefixes = &work_prefixes;
  }
  int32_t effective =
      max_pruned_prefix_length(*validation_prefixes, num_speculative_tokens);
  effective = std::max(effective, 1);
  if (supports_replay_update) {
    effective =
        std::min(std::max(effective, max_accepted), num_speculative_tokens);
  }
  if (!reduces_validate_work(*validation_prefixes,
                             num_speculative_tokens,
                             effective,
                             requires_uniform_width)) {
    return {true, num_speculative_tokens, {}, {}};
  }

  std::vector<int32_t> widths;
  widths.reserve(prefixes.size());
  for (int32_t prefix : *validation_prefixes) {
    widths.emplace_back(requires_uniform_width ? effective + 1
                                               : std::max(prefix, 1) + 1);
  }
  return {false, effective, std::move(prefixes), std::move(widths)};
}

// Truncate draft outputs to only the first num_speculative_tokens entries.
// Borrows the first num_speculative_tokens entries of draft_outputs. Callers
// already hold draft_outputs alive, so a view avoids deep-copying ForwardOutput
// (logits, [batch, vocab] probs, error strings) once per decode step.
std::span<const ForwardOutput> truncate_draft_outputs(
    const std::vector<ForwardOutput>& draft_outputs,
    int32_t num_speculative_tokens);

// Build the drop mask [B, W] (positions > prefix_len) on `device`.
torch::Tensor build_pruned_prefix_masks(
    const std::vector<int32_t>& pruned_prefix_lengths,
    int32_t num_speculative_tokens,
    const torch::Device& device);

// Apply per-seq pruning to rejection sampling output: mask positions beyond
// each seq's prefix_len to -1.
void apply_pruned_prefix_lengths(SampleOutput& sample_output,
                                 int32_t num_speculative_tokens,
                                 const torch::Tensor& drop_mask);

// Scatter a per-seq variable-length target output (produced by a pruned
// validate forward) back into the padded dense [batch * max_val_tokens, ...]
// layout the rejection sampler and the apply pruning helper expect.
// seq i's cu-packed rows [cu_offset, cu_offset + per_seq_val_tokens[i]) map
// 1:1 onto dense cols [0, per_seq_val_tokens[i]) of row i, so the per-seq
// bonus lands at col (per_seq_val_tokens[i] - 1), not the fixed last column.
// Verify inputs are logits-only, so only logits and embeddings are padded;
// logits pad with -inf (strictly-rejecting), embeddings with 0. Rewrites
// `target_output` in place; no-op when its rows already equal
// batch_size * max_val_tokens (uniform width).
void scatter_varlen_target_output_to_dense(
    ForwardOutput& target_output,
    const std::vector<int32_t>& per_seq_val_tokens,
    int32_t batch_size,
    int32_t max_val_tokens);

}  // namespace adaptive_pruning
}  // namespace xllm
