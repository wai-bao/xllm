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

#include <cstdint>
#include <optional>
#include <span>
#include <utility>

namespace xllm {

class SampleOutput;

// The sampled tokens and their actual processed sampling logits. An absent
// logits tensor denotes a deterministic proposal. Mapped vocabularies keep
// their compact support until first-reject recovery.
class DraftProposal final {
 public:
  DraftProposal() = default;

  explicit DraftProposal(torch::Tensor token_ids,
                         std::optional<torch::Tensor> logits = std::nullopt,
                         torch::Tensor vocab_ids = {},
                         torch::Tensor sampled_indices = {})
      : token_ids_(std::move(token_ids)),
        logits_(std::move(logits)),
        vocab_ids_(normalize_vocab_ids(std::move(vocab_ids))),
        sampled_indices_(std::move(sampled_indices)) {}

  const torch::Tensor& token_ids() const { return token_ids_; }
  const std::optional<torch::Tensor>& logits() const { return logits_; }
  const torch::Tensor& vocab_ids() const { return vocab_ids_; }
  const torch::Tensor& sampled_indices() const { return sampled_indices_; }
  bool is_greedy() const { return !logits_.has_value(); }
  // True when the compact support is a [1, 1, K] view shared by every batch
  // row and position, as opposed to per-position [B, N, K] candidates.
  bool shared_support() const {
    return vocab_ids_.defined() && vocab_ids_.size(1) == 1;
  }
  // Compact-support ids for the first-reject recovery: a shared [1, 1, K]
  // support expands to the gathered row layout; per-position [B, N, K]
  // support gathers along dim 1 at the index slot. The single home for the
  // two-support-layout convention, so consumers never branch on raw shape.
  torch::Tensor support_at(const torch::Tensor& index_slot,
                           const torch::Tensor& gathered) const {
    return shared_support() ? vocab_ids_.expand_as(gathered)
                            : vocab_ids_.take_along_dim(index_slot, 1);
  }

  void validate(int64_t batch_size,
                int64_t vocab_size,
                int64_t num_steps) const {
    CHECK(token_ids_.defined());
    CHECK_EQ(token_ids_.dim(), 2);
    CHECK_EQ(token_ids_.size(0), batch_size);
    CHECK_EQ(token_ids_.size(1), num_steps);
    if (is_greedy()) {
      CHECK(!vocab_ids_.defined());
      CHECK(!sampled_indices_.defined());
      return;
    }
    CHECK(logits_->defined());
    CHECK_EQ(logits_->dim(), 3);
    CHECK_EQ(logits_->size(0), batch_size);
    CHECK_EQ(logits_->size(1), num_steps);
    if (!vocab_ids_.defined()) {
      CHECK_EQ(logits_->size(-1), vocab_size);
      CHECK(!sampled_indices_.defined());
      return;
    }
    CHECK(sampled_indices_.defined());
    CHECK_EQ(sampled_indices_.sizes(), token_ids_.sizes());
    CHECK_EQ(vocab_ids_.size(-1), logits_->size(-1));
    // Shared [1, 1, K] support broadcasts over batch and positions.
    CHECK(vocab_ids_.sizes() == logits_->sizes() ||
          vocab_ids_.sizes() == torch::IntArrayRef({1, 1, logits_->size(-1)}));
  }

  DraftProposal coerce_to(const torch::Tensor& reference_token_ids) const {
    const auto device = reference_token_ids.device();
    return DraftProposal(
        token_ids_.to(reference_token_ids),
        logits_ ? std::make_optional(logits_->to(device)) : std::nullopt,
        vocab_ids_.defined() ? vocab_ids_.to(device) : torch::Tensor(),
        sampled_indices_.defined() ? sampled_indices_.to(device)
                                   : torch::Tensor());
  }

  DraftProposal slice_speculative(int64_t start, int64_t end) const {
    return DraftProposal(
        token_ids_.slice(1, start, end),
        logits_ ? std::make_optional(logits_->slice(1, start, end))
                : std::nullopt,
        // A shared [1, 1, K] support broadcasts across the sliced positions.
        vocab_ids_.defined() && !shared_support()
            ? vocab_ids_.slice(1, start, end)
            : vocab_ids_,
        sampled_indices_.defined() ? sampled_indices_.slice(1, start, end)
                                   : torch::Tensor());
  }

  // Stack per-step sampled outputs into one proposal. Callers whose
  // SampleOutputs live inside larger structs (ForwardOutput, pipeline
  // invocations) collect pointers and pass the span-of-pointers form.
  // stacked_tokens (when given) reuses token storage the caller already
  // filled (e.g. a preallocated per-slot buffer); only the logits stack.
  static DraftProposal from_steps(std::span<const SampleOutput* const> steps,
                                  bool probabilistic,
                                  torch::Tensor stacked_tokens = {});

 private:
  // Shared compact support arrives as [K]; store it as a [1, 1, K] broadcast
  // view so every consumer handles a single 3-D layout. Zero-copy.
  static torch::Tensor normalize_vocab_ids(torch::Tensor vocab_ids) {
    if (vocab_ids.defined() && vocab_ids.dim() == 1) {
      return vocab_ids.view({1, 1, -1});
    }
    return vocab_ids;
  }

  torch::Tensor token_ids_;
  std::optional<torch::Tensor> logits_;
  // Per-position [B, N, K] candidates, or a shared [1, 1, K] broadcast view.
  torch::Tensor vocab_ids_;
  torch::Tensor sampled_indices_;
};

}  // namespace xllm
