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

#include "core/framework/speculative/adaptive_pruning_helpers.h"

#include <glog/logging.h>

#include <algorithm>

#include "core/framework/sampling/logits_utils.h"
#include "core/util/tensor_helper.h"

namespace xllm {
namespace adaptive_pruning {

torch::Tensor selected_probs_by_step(
    const std::vector<ForwardOutput>& draft_outputs) {
  CHECK(!draft_outputs.empty()) << "draft outputs must not be empty";
  std::vector<torch::Tensor> probs_steps;
  probs_steps.reserve(draft_outputs.size());
  int64_t batch_size = -1;
  for (const ForwardOutput& draft_output : draft_outputs) {
    torch::Tensor probs = draft_output.sample_output.probs;
    if (!probs.defined()) {
      // Normalize only the selected score when the controller needs it.
      const auto& sampled = draft_output.sample_output;
      const torch::Tensor& logits = sampled.logits;
      const torch::Tensor& next_tokens = sampled.next_tokens;
      CHECK(logits.defined() && next_tokens.defined())
          << "adaptive pruning fallback needs draft logits and next_tokens";
      CHECK_EQ(logits.dim(), 2)
          << "adaptive pruning expects draft logits [batch,vocab], got "
          << logits.sizes();
      const auto& indices = sampled.sampled_indices.defined()
                                ? sampled.sampled_indices
                                : next_tokens;
      probs = compute_selected_logprobs(logits, indices).exp();
    }
    CHECK(probs.dim() == 1 || probs.dim() == 2)
        << "adaptive pruning expects draft probs [batch] or [batch,1], got "
        << probs.sizes();
    if (probs.dim() == 2) {
      CHECK_EQ(probs.size(1), 1)
          << "adaptive pruning expects draft probs [batch,1], got "
          << probs.sizes();
      probs = probs.squeeze(/*dim=*/1);
    }
    if (batch_size < 0) {
      batch_size = probs.size(0);
    }
    CHECK_EQ(probs.size(0), batch_size)
        << "adaptive pruning draft prob batch mismatch";
    probs_steps.emplace_back(probs.view({-1, 1}));
  }
  return torch::cat(probs_steps, /*dim=*/1);
}

bool has_selected_probs_by_step(
    const std::vector<ForwardOutput>& draft_outputs) {
  if (draft_outputs.empty()) {
    return false;
  }
  for (const ForwardOutput& draft_output : draft_outputs) {
    const torch::Tensor& probs = draft_output.sample_output.probs;
    if (probs.defined()) {
      continue;  // ok
    }
    // Fallback: selected_probs_by_step will compute from logits+next_tokens.
    const torch::Tensor& logits = draft_output.sample_output.logits;
    const torch::Tensor& next_tokens = draft_output.sample_output.next_tokens;
    if (!logits.defined() || !next_tokens.defined()) {
      return false;
    }
  }
  return true;
}

void clamp_prefix_lengths(std::vector<int32_t>& prefix_lengths,
                          int32_t batch_size,
                          int32_t num_speculative_tokens) {
  CHECK_EQ(prefix_lengths.size(), static_cast<size_t>(batch_size))
      << "adaptive pruning prefix length batch mismatch";
  for (int32_t& prefix_len : prefix_lengths) {
    prefix_len = std::clamp(prefix_len, 0, num_speculative_tokens);
  }
}

int32_t max_pruned_prefix_length(const std::vector<int32_t>& prefix_lengths,
                                 int32_t num_speculative_tokens) {
  if (prefix_lengths.empty()) {
    return num_speculative_tokens;
  }
  const int32_t max_prefix_len =
      *std::max_element(prefix_lengths.begin(), prefix_lengths.end());
  return std::clamp(max_prefix_len, 0, num_speculative_tokens);
}

std::vector<ForwardOutput> truncate_draft_outputs(
    const std::vector<ForwardOutput>& draft_outputs,
    int32_t num_speculative_tokens) {
  CHECK_GE(num_speculative_tokens, 0)
      << "num_speculative_tokens must be non-negative";
  CHECK_GE(draft_outputs.size(), static_cast<size_t>(num_speculative_tokens))
      << "draft outputs are fewer than the requested validation width";
  std::vector<ForwardOutput> truncated_outputs;
  truncated_outputs.reserve(static_cast<size_t>(num_speculative_tokens));
  for (int32_t i = 0; i < num_speculative_tokens; ++i) {
    truncated_outputs.emplace_back(draft_outputs[static_cast<size_t>(i)]);
  }
  return truncated_outputs;
}

void scatter_varlen_target_output_to_dense(
    ForwardOutput& target_output,
    const std::vector<int32_t>& per_seq_val_tokens,
    int32_t batch_size,
    int32_t max_val_tokens) {
  CHECK(target_output.logits.defined())
      << "target logits must be defined for varlen->dense scatter";
  const int32_t total_tokens =
      static_cast<int32_t>(target_output.logits.size(0));
  const int64_t padded_total =
      static_cast<int64_t>(batch_size) * max_val_tokens;
  if (total_tokens == static_cast<int32_t>(padded_total)) {
    // Already uniform max width; no scatter needed.
    return;
  }
  const int32_t vocab_size =
      static_cast<int32_t>(target_output.logits.size(-1));

  // Pure order-preserving scatter: seq i's cu-packed rows [cu_offset,
  // cu_offset + seq_tokens) map 1:1 onto dense cols [0, seq_tokens) of row i.
  // The source rows are exactly [0, total_tokens) in order, so index_copy_
  // reads target_output.logits directly with no source gather. Row j is the
  // target's prediction after seeing anchor + draft[0..j-1]; col 0 predicts
  // draft[0] (anchor row) and col (seq_tokens - 1) is the per-seq bonus.
  // Trailing cols [seq_tokens, max_val_tokens) stay padded, so the bonus lands
  // at col (seq_tokens - 1), NOT the fixed last column; validate() and
  // verification therefore reads the bonus/cut position per-seq.
  std::vector<int64_t> dst_indices_vec;
  dst_indices_vec.reserve(static_cast<size_t>(total_tokens));
  for (int32_t i = 0; i < batch_size; ++i) {
    const int32_t seq_tokens = per_seq_val_tokens[static_cast<size_t>(i)];
    for (int32_t j = 0; j < seq_tokens; ++j) {
      dst_indices_vec.push_back(static_cast<int64_t>(i) * max_val_tokens + j);
    }
  }
  torch::Tensor dst_indices =
      async_h2d_tensor(dst_indices_vec, target_output.logits.device());

  // Padded rows are excluded by each request's verification length; using
  // empty + a targeted index_fill_ over the complement of dst_indices avoids
  // paying a full padded_total row write when most rows get overwritten by
  // index_copy_.
  torch::Tensor pad_indices;
  if (dst_indices_vec.size() < static_cast<size_t>(padded_total)) {
    std::vector<uint8_t> valid(static_cast<size_t>(padded_total), 0);
    for (int64_t idx : dst_indices_vec) {
      valid[static_cast<size_t>(idx)] = true;
    }
    std::vector<int64_t> pad_indices_vec;
    pad_indices_vec.reserve(static_cast<size_t>(padded_total) -
                            dst_indices_vec.size());
    for (int64_t i = 0; i < padded_total; ++i) {
      if (!valid[static_cast<size_t>(i)]) {
        pad_indices_vec.push_back(i);
      }
    }
    pad_indices =
        async_h2d_tensor(pad_indices_vec, target_output.logits.device());
  }
  // Scatter one [total_tokens, ...] tensor into a padded
  // [padded_total, ...] buffer: fill only the padded rows, then copy the rest.
  // The row axis is replaced by padded_total; trailing axes are preserved.
  auto pad_rows = [&](const torch::Tensor& src,
                      double pad_value) -> torch::Tensor {
    std::vector<int64_t> sizes{padded_total};
    sizes.insert(sizes.end(), src.sizes().begin() + 1, src.sizes().end());
    torch::Tensor padded = torch::empty(sizes, src.options());
    if (pad_indices.defined()) {
      padded.index_fill_(/*dim=*/0, pad_indices, pad_value);
    }
    padded.index_copy_(/*dim=*/0, dst_indices, src);
    return padded;
  };
  torch::Tensor padded_logits =
      pad_rows(target_output.logits, /*pad_value=*/-1e9);
  const auto& processed = target_output.sample_output.logits;
  if (processed.defined()) {
    target_output.sample_output.logits =
        processed.is_same(target_output.logits)
            ? padded_logits
            : pad_rows(processed, /*pad_value=*/0);
  }
  target_output.logits = padded_logits;

  // Keep the sampled outputs aligned with the dense token layout; every
  // one of them pads with 0.
  auto& sample = target_output.sample_output;
  if (sample.next_tokens.defined()) {
    sample.next_tokens = pad_rows(sample.next_tokens, /*pad_value=*/0);
  }
  if (sample.embeddings.defined()) {
    sample.embeddings = pad_rows(sample.embeddings, /*pad_value=*/0);
  }
  if (sample.logprobs.defined()) {
    sample.logprobs = pad_rows(sample.logprobs, /*pad_value=*/0);
  }
  if (sample.top_tokens.defined()) {
    sample.top_tokens = pad_rows(sample.top_tokens, /*pad_value=*/0);
  }
  if (sample.top_logprobs.defined()) {
    sample.top_logprobs = pad_rows(sample.top_logprobs, /*pad_value=*/0);
  }
}

}  // namespace adaptive_pruning
}  // namespace xllm
