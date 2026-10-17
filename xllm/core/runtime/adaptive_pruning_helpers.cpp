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

#include "core/runtime/adaptive_pruning_helpers.h"

#include <glog/logging.h>

#include "core/framework/speculative/draft_input_sampling.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/speculative_input_builder.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/tensor_helper.h"

namespace xllm {
namespace adaptive_pruning {

torch::Tensor build_pruned_prefix_masks(
    const std::vector<int32_t>& pruned_prefix_lengths,
    int32_t num_speculative_tokens,
    const torch::Device& device) {
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  torch::Tensor prefix_lengths =
      async_h2d_tensor(
          pruned_prefix_lengths,
          torch::TensorOptions().dtype(torch::kLong).device(device))
          .clamp(0, num_speculative_tokens);
  torch::Tensor positions =
      torch::arange(num_val_tokens, prefix_lengths.options());
  return positions.unsqueeze(/*dim=*/0).gt(prefix_lengths.unsqueeze(/*dim=*/1));
}

template <typename TargetInput>
std::vector<double> controller_kv_lens(const TargetInput& input,
                                       int32_t batch_size) {
  std::vector<double> kv_lens(static_cast<size_t>(batch_size), 0.0);
  const Slice<int32_t> kv_seq_lens =
      input.input_params.attention.host.kv_seq_lens;
  for (int32_t i = 0; i < batch_size; ++i) {
    if (static_cast<size_t>(i) < kv_seq_lens.size()) {
      kv_lens[static_cast<size_t>(i)] = static_cast<double>(
          specBuilder::calc_kv_len(kv_seq_lens, i, /*offset=*/0));
    }
  }
  return kv_lens;
}

torch::Tensor selected_probs_by_step(
    std::span<const ForwardOutput> draft_outputs) {
  CHECK(!draft_outputs.empty()) << "draft outputs must not be empty";
  std::vector<torch::Tensor> probs_steps;
  probs_steps.reserve(draft_outputs.size());
  int64_t batch_size = -1;
  for (const ForwardOutput& draft_output : draft_outputs) {
    // Single shared per-row confidence extractor: draft-input-selected probs
    // take priority, then the sampler's probs (1-D passthrough or 2-D gather),
    // then the logits+next_tokens softmax fallback. Keeps the adaptive
    // controller in lockstep with the carried-draft-input confidence semantics.
    torch::Tensor probs = select_input_confidence(
        draft_output.sample_output, draft_output.logits, /*sample_idxes=*/{});
    CHECK(probs.dim() == 1)
        << "adaptive pruning expects per-row confidence [batch], got "
        << probs.sizes();
    if (batch_size < 0) {
      batch_size = probs.size(0);
    }
    CHECK_EQ(probs.size(0), batch_size)
        << "adaptive pruning draft prob batch mismatch";
    probs_steps.emplace_back(probs.view({-1, 1}));
  }
  return torch::cat(probs_steps, /*dim=*/1);
}

bool has_selected_probs_by_step(std::span<const ForwardOutput> draft_outputs) {
  if (draft_outputs.empty()) {
    return false;
  }
  for (const ForwardOutput& draft_output : draft_outputs) {
    // Shared confidence ladder: selected_probs_by_step will take the same
    // rung that select_input_confidence would, so the availability check
    // only has to reject a missing source.
    if (!has_input_confidence(draft_output.sample_output,
                              draft_output.logits)) {
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

bool reduces_validate_work(const std::vector<int32_t>& prefix_lengths,
                           int32_t num_speculative_tokens,
                           int32_t effective_speculative_tokens,
                           bool requires_uniform_validate_width) {
  CHECK_GT(num_speculative_tokens, 0);
  CHECK_GE(effective_speculative_tokens, 1);
  CHECK_LE(effective_speculative_tokens, num_speculative_tokens);
  const bool has_pruned_prefix =
      std::any_of(prefix_lengths.begin(),
                  prefix_lengths.end(),
                  [num_speculative_tokens](int32_t prefix_len) {
                    return prefix_len < num_speculative_tokens;
                  });
  return has_pruned_prefix &&
         (!requires_uniform_validate_width ||
          effective_speculative_tokens < num_speculative_tokens);
}

std::span<const ForwardOutput> truncate_draft_outputs(
    const std::vector<ForwardOutput>& draft_outputs,
    int32_t num_speculative_tokens) {
  CHECK_GE(num_speculative_tokens, 0)
      << "num_speculative_tokens must be non-negative";
  CHECK_GE(draft_outputs.size(), static_cast<size_t>(num_speculative_tokens))
      << "draft outputs are fewer than the requested validation width";
  return std::span<const ForwardOutput>(draft_outputs)
      .first(static_cast<size_t>(num_speculative_tokens));
}

void apply_pruned_prefix_lengths(SampleOutput& sample_output,
                                 int32_t num_speculative_tokens,
                                 const torch::Tensor& drop_mask) {
  CHECK(sample_output.next_tokens.defined())
      << "validate output tokens are undefined";
  CHECK_EQ(sample_output.next_tokens.dim(), 2)
      << "validate output tokens should be [batch, width]";
  const int32_t batch_size =
      static_cast<int32_t>(sample_output.next_tokens.size(0));
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  CHECK_EQ(sample_output.next_tokens.size(1), num_val_tokens)
      << "validate output width mismatch";
  sample_output.next_tokens.masked_fill_(drop_mask, -1);

  if (sample_output.logprobs.defined()) {
    CHECK_EQ(sample_output.logprobs.dim(), 2)
        << "validate output logprobs should be [batch, width]";
    CHECK_EQ(sample_output.logprobs.size(0), batch_size)
        << "validate logprob batch mismatch";
    CHECK_EQ(sample_output.logprobs.size(1), num_val_tokens)
        << "validate logprob width mismatch";
    sample_output.logprobs.masked_fill_(
        drop_mask.to(sample_output.logprobs.device()), 0);
  }

  if (sample_output.top_tokens.defined()) {
    CHECK(sample_output.top_logprobs.defined())
        << "top_logprobs must be defined when top_tokens are defined";
    CHECK_EQ(sample_output.top_tokens.dim(), 3)
        << "validate top_tokens should be [batch, width, top_k]";
    CHECK_EQ(sample_output.top_logprobs.dim(), 3)
        << "validate top_logprobs should be [batch, width, top_k]";
    CHECK_EQ(sample_output.top_tokens.size(0), batch_size)
        << "validate top_tokens batch mismatch";
    CHECK_EQ(sample_output.top_tokens.size(1), num_val_tokens)
        << "validate top_tokens width mismatch";
    CHECK_EQ(sample_output.top_logprobs.size(0), batch_size)
        << "validate top_logprobs batch mismatch";
    CHECK_EQ(sample_output.top_logprobs.size(1), num_val_tokens)
        << "validate top_logprobs width mismatch";
    const torch::Tensor top_drop_mask =
        drop_mask.to(sample_output.top_tokens.device()).unsqueeze(/*dim=*/-1);
    sample_output.top_tokens.masked_fill_(top_drop_mask, 0);
    sample_output.top_logprobs.masked_fill_(
        top_drop_mask.to(sample_output.top_logprobs.device()), 0);
  }

  if (!sample_output.embeddings.defined()) {
    return;
  }
  CHECK_EQ(sample_output.embeddings.dim(), 3)
      << "validate output embeddings should be [batch, width, hidden]";
  CHECK_EQ(sample_output.embeddings.size(0), batch_size)
      << "validate embedding batch mismatch";
  CHECK_EQ(sample_output.embeddings.size(1), num_val_tokens)
      << "validate embedding width mismatch";

  const torch::Tensor embedding_drop_mask =
      drop_mask.to(sample_output.embeddings.device()).unsqueeze(/*dim=*/-1);
  sample_output.embeddings.masked_fill_(embedding_drop_mask, 0);
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
  // apply_pruned_prefix_lengths therefore read the bonus/cut position per-seq.
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

  // Logits pad to [B*max_val_tokens, V]. Only the padding rows need the -inf
  // sentinel (strictly-rejecting distribution); using empty + a targeted
  // index_fill_ over the complement of dst_indices avoids paying vocab_size ×
  // padded_total writes when most rows get overwritten by index_copy_.
  torch::Tensor padded_logits =
      torch::empty({padded_total, vocab_size}, target_output.logits.options());
  if (dst_indices_vec.size() < static_cast<size_t>(padded_total)) {
    std::vector<bool> valid(static_cast<size_t>(padded_total), false);
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
    torch::Tensor pad_indices =
        async_h2d_tensor(pad_indices_vec, target_output.logits.device());
    padded_logits.index_fill_(/*dim=*/0, pad_indices, -1e9);
  }
  padded_logits.index_copy_(/*dim=*/0, dst_indices, target_output.logits);
  target_output.logits = padded_logits;

  if (target_output.sample_output.embeddings.defined()) {
    const int32_t hidden_size =
        static_cast<int32_t>(target_output.sample_output.embeddings.size(-1));
    torch::Tensor padded_embeddings =
        torch::zeros({padded_total, hidden_size},
                     target_output.sample_output.embeddings.options());
    padded_embeddings.index_copy_(
        /*dim=*/0, dst_indices, target_output.sample_output.embeddings);
    target_output.sample_output.embeddings = padded_embeddings;
  }
}

template std::vector<double> controller_kv_lens(const LlmForwardInput&,
                                                int32_t);
template std::vector<double> controller_kv_lens(const VlmForwardInput&,
                                                int32_t);

}  // namespace adaptive_pruning
}  // namespace xllm
