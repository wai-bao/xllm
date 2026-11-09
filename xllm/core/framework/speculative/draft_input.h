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
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "core/framework/speculative/draft_input_model_spec.h"
#include "core/util/tensor_helper.h"

namespace xllm {

// Gathers logits at `token_ids` along the last dim. -1 sentinels clamp to
// row 0 and are the caller's to re-mask.
inline torch::Tensor gather_from_logits(const torch::Tensor& logits,
                                        const torch::Tensor& token_ids) {
  return index_select_2d(
      logits, /*dim=*/-1, token_ids.clamp_min(0).to(torch::kLong));
}

// log P(index) over a pre-clamped int64 index; lets multi-gather callers
// share one index materialization.
inline torch::Tensor gather_log_probs_from_index(const torch::Tensor& logits,
                                                 const torch::Tensor& lse,
                                                 const torch::Tensor& index) {
  return index_select_2d(logits, /*dim=*/-1, index)
      .to(torch::kFloat32)
      .sub(lse);
}

// P(token_ids) without materializing the dense softmax; -1 sentinels read 0.
inline torch::Tensor gather_probs_from_logits(const torch::Tensor& logits,
                                              const torch::Tensor& token_ids) {
  const torch::Tensor lse = torch::logsumexp(
      logits.to(torch::kFloat32), /*dim=*/-1, /*keepdim=*/false);
  return gather_log_probs_from_index(
             logits, lse, token_ids.clamp_min(0).to(torch::kLong))
      .exp()
      .masked_fill_(token_ids.lt(0), 0);
}

inline void embed_sparse_logits_inplace(torch::Tensor& dense,
                                        const torch::Tensor& token_ids,
                                        const torch::Tensor& values) {
  dense.fill_(-std::numeric_limits<float>::infinity());
  dense.scatter_(/*dim=*/1, token_ids, values);
}

// Allocating variant over a fresh [rows, width] buffer.
inline torch::Tensor embed_sparse_logits(const torch::Tensor& token_ids,
                                         const torch::Tensor& values,
                                         int64_t width) {
  torch::Tensor dense =
      torch::empty({token_ids.size(0), width}, values.options());
  embed_sparse_logits_inplace(dense, token_ids, values);
  return dense;
}

// One-hot q at the proposed token (log q(token) = 0): the point-mass row a
// greedy proposal materializes as a dense logits row.
inline torch::Tensor point_mass_logits(
    const torch::Tensor& token_ids,
    int64_t vocab_size,
    const torch::TensorOptions& options =
        torch::TensorOptions().dtype(torch::kFloat32)) {
  const int64_t rows = token_ids.size(0);
  return embed_sparse_logits(token_ids.to(torch::kLong).view({rows, 1}),
                             torch::zeros({rows, 1}, options),
                             vocab_size);
}

// A sampled first proposal and its recurrent draft hidden. Every tensor keeps
// its leading batch dimension, including a draft_input owned by a single
// sequence.
struct DraftInput {
  torch::Tensor token_ids;       // [batch], already in the target vocabulary.
  torch::Tensor hidden_states;   // [batch, draft_hidden].
  torch::Tensor selected_probs;  // Optional [batch], adaptive confidence only.
  // Sparse draft logits [batch, draft_vocab] when the sampler exported them;
  // the rejection sampler reconstructs q from these. Carried instead of a
  // dense probs materialization.
  torch::Tensor draft_logits;
  // All rows have point-mass q at token_ids: the rejection sampler treats
  // every greedy proposal as deterministically accepted.
  bool point_mass_q = false;
};

// Canonical shared-memory and transform order. Proto fields have an independent
// numbered wire schema and must be added explicitly to that adapter.
inline constexpr std::array<torch::Tensor DraftInput::*, 4> kDraftInputTensors =
    {&DraftInput::token_ids,
     &DraftInput::hidden_states,
     &DraftInput::selected_probs,
     &DraftInput::draft_logits};

// token_ids must be int32/int64; every other payload must be floating point.
// The proto decoder checks the wire datatypes before allocation, then this
// predicate validates the decoded draft_input.
inline bool is_valid_draft_input_tensor_dtype(torch::ScalarType dtype,
                                              bool is_token_ids) {
  if (is_token_ids) {
    return dtype == torch::kInt || dtype == torch::kLong;
  }
  return torch::isFloatingType(dtype);
}

namespace detail {

// First violated shape/dtype constraint, or nullptr when valid.
inline const char* draft_input_invalid_reason(const DraftInput& draft_input) {
  if (!draft_input.token_ids.defined()) {
    return "token ids are undefined";
  }
  if (!draft_input.hidden_states.defined()) {
    return "hidden states are undefined";
  }
  if (draft_input.token_ids.dim() != 1) {
    return "token ids must be 1-D [batch]";
  }
  if (!xllm::is_valid_draft_input_tensor_dtype(
          draft_input.token_ids.scalar_type(),
          /*is_token_ids=*/true)) {
    return "token ids must be int32 or int64";
  }
  if (draft_input.hidden_states.dim() != 2) {
    return "hidden states must be 2-D [batch, hidden]";
  }
  if (draft_input.hidden_states.size(0) != draft_input.token_ids.size(0)) {
    return "hidden states batch does not match token ids";
  }
  if (!xllm::is_valid_draft_input_tensor_dtype(
          draft_input.hidden_states.scalar_type(),
          /*is_token_ids=*/false)) {
    return "hidden states must be floating point";
  }
  const int64_t batch_size = draft_input.token_ids.size(0);
  if (draft_input.draft_logits.defined() &&
      (draft_input.draft_logits.dim() != 2 ||
       draft_input.draft_logits.size(0) != batch_size ||
       draft_input.draft_logits.size(1) <= 0 ||
       !xllm::is_valid_draft_input_tensor_dtype(
           draft_input.draft_logits.scalar_type(),
           /*is_token_ids=*/false))) {
    return "draft logits must be floating point [batch, draft_vocab > 0]";
  }
  if (draft_input.selected_probs.defined() &&
      (draft_input.selected_probs.dim() != 1 ||
       draft_input.selected_probs.size(0) != batch_size ||
       !xllm::is_valid_draft_input_tensor_dtype(
           draft_input.selected_probs.scalar_type(),
           /*is_token_ids=*/false))) {
    return "selected probabilities must be floating point [batch]";
  }
  return nullptr;
}

}  // namespace detail

inline bool is_valid_draft_input(const DraftInput& draft_input) {
  return detail::draft_input_invalid_reason(draft_input) == nullptr;
}

inline void check_draft_input(const DraftInput& draft_input) {
  const char* reason = detail::draft_input_invalid_reason(draft_input);
  CHECK(reason == nullptr) << "invalid draft input: " << reason;
}

namespace detail {

// The hidden width and dtype must match the loaded draft model.
inline const char* draft_input_model_reason(const DraftInput& draft_input,
                                            const DraftInputModelSpec& spec) {
  const char* reason = draft_input_invalid_reason(draft_input);
  if (reason != nullptr) {
    return reason;
  }
  if (spec.hidden_size <= 0) {
    return "draft model hidden specification is unavailable";
  }
  if (draft_input.hidden_states.size(1) != spec.hidden_size) {
    return "draft_input hidden width does not match the loaded draft model";
  }
  if (draft_input.hidden_states.scalar_type() != spec.hidden_dtype) {
    return "draft_input hidden dtype does not match the loaded draft model";
  }
  return nullptr;
}

}  // namespace detail

inline void check_draft_input_model(const DraftInput& draft_input,
                                    const DraftInputModelSpec& spec) {
  const char* reason = detail::draft_input_model_reason(draft_input, spec);
  CHECK(reason == nullptr) << "draft input violates model contract: " << reason;
}

namespace detail {

// Probabilistic draft_inputs carry full-vocabulary q or an explicit point-mass
// marker; missing q alone never implies deterministic proposal semantics.
inline const char* draft_input_mode_reason(const DraftInput& draft_input,
                                           bool probabilistic) {
  if (probabilistic) {
    if (!draft_input.point_mass_q && !draft_input.draft_logits.defined()) {
      return "probabilistic draft input requires point-mass q or draft logits";
    }
  } else if (draft_input.point_mass_q) {
    return "non-probabilistic draft input must not carry proposal q";
  }
  return nullptr;
}

}  // namespace detail

inline void check_draft_input_mode(const DraftInput& draft_input,
                                   bool probabilistic) {
  const char* reason =
      detail::draft_input_mode_reason(draft_input, probabilistic);
  CHECK(reason == nullptr) << "draft input violates sampling-mode contract: "
                           << reason;
}

// The "PD-imported draft_input" contract on top of the model contract:
// exactly one row whose draft token is in the target vocabulary. Importers
// route through this so the receiver checks cannot drift from the producer
// contracts.
inline const char* draft_input_import_reason(const DraftInput& draft_input,
                                             bool probabilistic,
                                             int64_t vocab_size,
                                             const DraftInputModelSpec& spec) {
  const char* reason = detail::draft_input_model_reason(draft_input, spec);
  if (reason != nullptr) {
    return reason;
  }
  if (draft_input.token_ids.size(0) != 1) {
    return "imported draft_input must carry exactly one row";
  }
  reason = detail::draft_input_mode_reason(draft_input, probabilistic);
  if (reason != nullptr) {
    return reason;
  }
  const int64_t draft_token_id = draft_input.token_ids.item<int64_t>();
  if (draft_token_id < 0 || draft_token_id >= vocab_size) {
    return "imported draft_input draft token is outside the target vocabulary";
  }
  return nullptr;
}

// Placeholder draft_input for graph-warmup steps with no real draft input.
// Warmup is neither persisted nor PD-transferred, so the proposal
// distribution stays omitted.
inline DraftInput make_warmup_draft_input(
    int64_t batch_size,
    int64_t draft_hidden_size,
    const torch::TensorOptions& draft_options,
    bool probabilistic) {
  DraftInput draft_input;
  draft_input.token_ids =
      torch::zeros({batch_size}, draft_options.dtype(torch::kInt64));
  draft_input.hidden_states =
      torch::zeros({batch_size, draft_hidden_size}, draft_options);
  if (!probabilistic) {
    draft_input.selected_probs =
        torch::ones({batch_size}, draft_options.dtype(torch::kFloat));
  }
  draft_input.point_mass_q = probabilistic;
  return draft_input;
}

// Shared skeleton for the transforms below: copy the scalars, then apply
// Transform to every defined tensor.
template <typename Transform>
inline DraftInput transform_draft_input(const DraftInput& draft_input,
                                        Transform transform) {
  DraftInput out;
  out.point_mass_q = draft_input.point_mass_q;
  for (torch::Tensor DraftInput::* member : kDraftInputTensors) {
    const torch::Tensor& tensor = draft_input.*member;
    if (tensor.defined()) {
      out.*member = transform(tensor);
    }
  }
  return out;
}

// Deep copy into independent contiguous storage, detached from autograd.
inline DraftInput clone_draft_input(const DraftInput& draft_input) {
  DraftInput out;
  out.point_mass_q = draft_input.point_mass_q;
  for (torch::Tensor DraftInput::* member : kDraftInputTensors) {
    const torch::Tensor& tensor = draft_input.*member;
    if (tensor.defined()) {
      out.*member = clone_contiguous_detached_tensor(tensor);
    }
  }
  return out;
}

// Views sharing the source storage, no autograd history; keeps the
// producer's memory instead of copying [rows, vocab] payloads per request.
inline DraftInput detach_draft_input(const DraftInput& draft_input) {
  return transform_draft_input(
      draft_input, [](const torch::Tensor& tensor) { return tensor.detach(); });
}

inline DraftInput slice_draft_input(const DraftInput& draft_input,
                                    int64_t start,
                                    int64_t length) {
  return transform_draft_input(draft_input,
                               [start, length](const torch::Tensor& t) {
                                 return t.narrow(/*dim=*/0, start, length);
                               });
}

// A sequence's draft_input is its detached single-row view of the batch's.
inline DraftInput take_draft_input_row(const DraftInput& draft_input,
                                       int64_t row) {
  CHECK_GE(row, 0) << "draft input row must be non-negative";
  CHECK_LT(row, draft_input.token_ids.size(0));
  return slice_draft_input(draft_input, row, /*length=*/1);
}

inline DraftInput cat_draft_inputs(
    const std::vector<DraftInput>& draft_inputs) {
  CHECK(!draft_inputs.empty());
  for (const DraftInput& draft_input : draft_inputs) {
    check_draft_input(draft_input);
  }
  const auto cat = [&draft_inputs](torch::Tensor DraftInput::* member) {
    const bool defined = (draft_inputs.front().*member).defined();
    std::vector<torch::Tensor> tensors;
    tensors.reserve(draft_inputs.size());
    for (const DraftInput& draft_input : draft_inputs) {
      CHECK_EQ((draft_input.*member).defined(), defined)
          << "draft inputs must agree on optional probability payloads";
      if (defined) {
        tensors.emplace_back(draft_input.*member);
      }
    }
    return defined ? torch::cat(tensors, /*dim=*/0) : torch::Tensor();
  };
  const torch::Tensor token_ids = cat(&DraftInput::token_ids);
  const torch::Tensor hidden_states = cat(&DraftInput::hidden_states);

  // Uniform batches keep the point-mass marker. Mixed batches embed every
  // point-mass row's one-hot q (log-prob 0 at its token) into the merged
  // draft_logits so the rejection sampler derives per-row greedy semantics.
  const auto reference_logits =
      std::find_if(draft_inputs.begin(),
                   draft_inputs.end(),
                   [](const DraftInput& draft_input) {
                     return draft_input.draft_logits.defined();
                   });
  const bool any_logits = reference_logits != draft_inputs.end();
  const torch::TensorOptions row_options =
      token_ids.options().dtype(torch::kFloat32);
  // Materialize selected_probs only when some input carries confidence;
  // undefined otherwise, matching the undefined-member convention of `cat`
  // above.
  const bool any_selected_probs = std::any_of(
      draft_inputs.begin(), draft_inputs.end(), [](const DraftInput& other) {
        return other.selected_probs.defined();
      });
  torch::Tensor selected_probs;
  if (any_selected_probs) {
    selected_probs = torch::ones({token_ids.size(0)}, row_options);
    int64_t row_offset = 0;
    for (const DraftInput& draft_input : draft_inputs) {
      if (draft_input.selected_probs.defined()) {
        selected_probs
            .narrow(/*dim=*/0, row_offset, draft_input.token_ids.size(0))
            .copy_(draft_input.selected_probs.to(torch::kFloat32));
      }
      row_offset += draft_input.token_ids.size(0);
    }
  }
  if (!any_logits) {
    const bool point_mass_q = draft_inputs.front().point_mass_q;
    for (const DraftInput& draft_input : draft_inputs) {
      CHECK_EQ(draft_input.point_mass_q, point_mass_q)
          << "Sparse draft inputs must agree on q semantics";
    }
    return {token_ids,
            hidden_states,
            selected_probs,
            /*draft_logits=*/torch::Tensor(),
            point_mass_q};
  }

  std::vector<torch::Tensor> logits_rows;
  logits_rows.reserve(draft_inputs.size());
  const int64_t vocab_size = reference_logits->draft_logits.size(-1);
  for (const DraftInput& draft_input : draft_inputs) {
    if (draft_input.draft_logits.defined()) {
      CHECK_EQ(draft_input.draft_logits.size(-1), vocab_size)
          << "mixed draft inputs must share the draft vocab dimension";
      logits_rows.emplace_back(draft_input.draft_logits.to(torch::kFloat32));
      continue;
    }
    CHECK(draft_input.point_mass_q)
        << "probabilistic batch rows must carry draft logits";
    logits_rows.emplace_back(
        point_mass_logits(draft_input.token_ids, vocab_size));
  }
  return {token_ids,
          hidden_states,
          selected_probs,
          torch::cat(logits_rows, /*dim=*/0),
          /*point_mass_q=*/false};
}

inline DraftInput draft_input_to(const DraftInput& draft_input,
                                 const torch::Device& device,
                                 bool non_blocking = false) {
  check_draft_input(draft_input);
  // safe_to is defined for undefined tensors, so it runs on every member.
  return transform_draft_input(
      draft_input, [&device, non_blocking](const torch::Tensor& tensor) {
        return safe_to(tensor, device, non_blocking);
      });
}

inline std::optional<DraftInput> draft_input_to(
    const std::optional<DraftInput>& draft_input,
    const torch::Device& device,
    bool non_blocking = false) {
  if (!draft_input.has_value()) {
    return std::nullopt;
  }
  return draft_input_to(*draft_input, device, non_blocking);
}

}  // namespace xllm
