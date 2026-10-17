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

// A sampled first proposal and its recurrent draft hidden. Every tensor keeps
// its leading batch dimension, including a draft_input owned by a single
// sequence.
struct DraftInput {
  torch::Tensor token_ids;       // [batch], already in the target vocabulary.
  torch::Tensor hidden_states;   // [batch, draft_hidden].
  torch::Tensor probs;           // Optional [batch, target_vocab], full q.
  torch::Tensor selected_probs;  // Optional [batch], adaptive confidence only.
  // All rows have point-mass q at token_ids; probs is omitted until a mixed
  // batch needs a dense proposal distribution for random rejection sampling.
  bool point_mass_q = false;
};

// Canonical shared-memory and transform order. Proto fields have an independent
// numbered wire schema and must be added explicitly to that adapter.
inline constexpr std::array<torch::Tensor DraftInput::*, 4> kDraftInputTensors =
    {&DraftInput::token_ids,
     &DraftInput::hidden_states,
     &DraftInput::probs,
     &DraftInput::selected_probs};

namespace detail {

// Assembled-draft_input dtype rule: token_ids must be int32/int64, every other
// payload must be floating point. The proto decoder checks corresponding wire
// datatypes before allocation, then this predicate validates the decoded
// draft_input.
inline bool is_valid_draft_input_tensor_dtype(torch::ScalarType dtype,
                                              bool is_token_ids) {
  if (is_token_ids) {
    return dtype == torch::kInt || dtype == torch::kLong;
  }
  return c10::isFloatingType(dtype);
}

// Single source of truth for the DraftInput shape/dtype invariant. Returns a
// static reason string for the first violated constraint, or nullptr when the
// draft_input is valid. is_valid_draft_input and check_draft_input both route
// through this so the boolean guard and the fatal assertion cannot diverge.
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
  if (!is_valid_draft_input_tensor_dtype(draft_input.token_ids.scalar_type(),
                                         /*is_token_ids=*/true)) {
    return "token ids must be int32 or int64";
  }
  if (draft_input.hidden_states.dim() != 2) {
    return "hidden states must be 2-D [batch, hidden]";
  }
  if (draft_input.hidden_states.size(0) != draft_input.token_ids.size(0)) {
    return "hidden states batch does not match token ids";
  }
  if (!is_valid_draft_input_tensor_dtype(
          draft_input.hidden_states.scalar_type(),
          /*is_token_ids=*/false)) {
    return "hidden states must be floating point";
  }
  const int64_t batch_size = draft_input.token_ids.size(0);
  if (draft_input.probs.defined() &&
      (draft_input.probs.dim() != 2 ||
       draft_input.probs.size(0) != batch_size ||
       draft_input.probs.size(1) <= 0 ||
       !is_valid_draft_input_tensor_dtype(draft_input.probs.scalar_type(),
                                          /*is_token_ids=*/false))) {
    return "probabilities must be floating point [batch, target_vocab > 0]";
  }
  if (draft_input.selected_probs.defined() &&
      (draft_input.selected_probs.dim() != 1 ||
       draft_input.selected_probs.size(0) != batch_size ||
       !is_valid_draft_input_tensor_dtype(
           draft_input.selected_probs.scalar_type(),
           /*is_token_ids=*/false))) {
    return "selected probabilities must be floating point [batch]";
  }
  if (draft_input.point_mass_q && draft_input.probs.defined()) {
    return "point-mass draft_input must not carry dense probabilities";
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

// Model-contract validation on top of the shape/dtype invariant: the
// draft_input's hidden width and dtype must match the loaded draft model.
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

// Probabilistic draft_inputs carry either full-vocabulary q or an explicit
// point-mass marker. Missing q alone never implies deterministic proposal
// semantics.
inline const char* draft_input_mode_reason(const DraftInput& draft_input,
                                           bool probabilistic,
                                           int64_t vocab_size) {
  if (probabilistic) {
    if (!draft_input.probs.defined() && !draft_input.point_mass_q) {
      return "probabilistic draft input requires full-vocab probs or "
             "point-mass q";
    }
  } else if (draft_input.probs.defined() || draft_input.point_mass_q) {
    return "non-probabilistic draft input must not carry proposal q";
  }
  if (probabilistic && draft_input.probs.defined() &&
      (draft_input.probs.dim() != 2 ||
       draft_input.probs.size(1) != vocab_size)) {
    return "probs must span the full target vocabulary";
  }
  return nullptr;
}

}  // namespace detail

inline void check_draft_input_mode(const DraftInput& draft_input,
                                   bool probabilistic,
                                   int64_t vocab_size) {
  const char* reason =
      detail::draft_input_mode_reason(draft_input, probabilistic, vocab_size);
  CHECK(reason == nullptr) << "draft input violates sampling-mode contract: "
                           << reason;
}

// Single source of truth for the "PD-imported draft_input" contract: a
// shape/mode valid draft_input that carries exactly one row whose draft token
// is in vocabulary. Returns the first violated constraint's reason, or nullptr
// when importable. Any importer routes through this so the receiver checks
// cannot drift from the producer contract in draft_input_invalid_reason /
// draft_input_mode_reason.
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
  reason =
      detail::draft_input_mode_reason(draft_input, probabilistic, vocab_size);
  if (reason != nullptr) {
    return reason;
  }
  const int64_t draft_token_id = draft_input.token_ids.item<int64_t>();
  if (draft_token_id < 0 || draft_token_id >= vocab_size) {
    return "imported draft_input draft token is outside the target vocabulary";
  }
  return nullptr;
}

// Builds a [rows, vocab_size] point-mass q: each row is one-hot at its index.
// Represents a deterministic (greedy) proposal as a full distribution so it can
// be recombined with random-sampling rows in one probabilistic batch.
inline torch::Tensor make_point_mass_probs(
    const torch::Tensor& indices,
    int64_t vocab_size,
    const torch::TensorOptions& options) {
  torch::Tensor probs = torch::zeros({indices.numel(), vocab_size}, options);
  probs.scatter_(/*dim=*/1,
                 indices.to(probs.device()).to(torch::kLong).view({-1, 1}),
                 1.0);
  return probs;
}

// Single source of the "materialize a dense proposal q" rule: a draft_input
// that already carries q passes it through, a point-mass row materializes one
// at its token ids. cat_draft_inputs and the task pipeline's rejection sampling
// route through this so the recombination contract cannot drift.
inline torch::Tensor dense_draft_q(const torch::Tensor& token_ids,
                                   const torch::Tensor& probs,
                                   int64_t vocab_size,
                                   const torch::TensorOptions& options) {
  return probs.defined()
             ? probs
             : make_point_mass_probs(token_ids, vocab_size, options);
}

// Materializes a placeholder draft_input for graph-warmup steps that have no
// real prefill/extend draft input to consume. Warmup is neither persisted nor
// PD-transferred and does not reach probabilistic validation, so the proposal
// distribution stays omitted; greedy warmup rows carry a full confidence so
// downstream consumers never see an undefined selected_probs.
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

// Shared skeleton for the draft_input transforms below: validate, copy the
// scalars, then apply Transform to every defined tensor. `Transform` is a
// template parameter rather than std::function so the per-tensor call inlines
// and the captureless transforms stay allocation-free.
template <typename Transform>
inline DraftInput transform_draft_input(const DraftInput& draft_input,
                                        Transform transform) {
  check_draft_input(draft_input);
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

inline DraftInput clone_draft_input(const DraftInput& draft_input) {
  return transform_draft_input(draft_input, [](const torch::Tensor& tensor) {
    return clone_contiguous_detached_tensor(tensor);
  });
}

// Returns views that share the source storage but carry no autograd history.
// Storage stays alive through the source tensor's refcount, so callers keep
// the producer's memory instead of copying [rows, vocab] payloads per request.
inline DraftInput detach_draft_input(const DraftInput& draft_input) {
  return transform_draft_input(
      draft_input, [](const torch::Tensor& tensor) { return tensor.detach(); });
}

// Return views; callers retaining producer-owned storage must clone them.
inline DraftInput slice_draft_input(const DraftInput& draft_input,
                                    int64_t start,
                                    int64_t length) {
  return transform_draft_input(draft_input,
                               [start, length](const torch::Tensor& t) {
                                 return t.narrow(/*dim=*/0, start, length);
                               });
}

// Single source of the "one draft_input row per sequence" contract shared by
// every output consumer (batch handler, SHM serializer, proto adapter): the
// batch draft_input must carry exactly one row per sequence, and a sequence's
// draft_input is its detached single-row view.
inline DraftInput take_draft_input_row(const DraftInput& draft_input,
                                       int64_t row) {
  CHECK_EQ(draft_input.token_ids.size(0), draft_input.hidden_states.size(0));
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
  const auto dense_input = std::find_if(draft_inputs.begin(),
                                        draft_inputs.end(),
                                        [](const DraftInput& draft_input) {
                                          return draft_input.probs.defined();
                                        });
  const bool any_dense = dense_input != draft_inputs.end();
  const bool point_mass_q = !any_dense && draft_inputs.front().point_mass_q;
  torch::Tensor probs;
  if (any_dense) {
    const int64_t vocab_size = dense_input->probs.size(1);
    std::vector<torch::Tensor> rows;
    rows.reserve(draft_inputs.size());
    for (const DraftInput& draft_input : draft_inputs) {
      if (draft_input.probs.defined()) {
        CHECK_EQ(draft_input.probs.size(1), vocab_size);
        rows.emplace_back(draft_input.probs);
      } else {
        CHECK(draft_input.point_mass_q)
            << "Missing proposal q requires an explicit point-mass marker";
        rows.emplace_back(make_point_mass_probs(
            draft_input.token_ids, vocab_size, dense_input->probs.options()));
      }
    }
    probs = torch::cat(rows, /*dim=*/0);
  } else {
    for (const DraftInput& draft_input : draft_inputs) {
      CHECK_EQ(draft_input.point_mass_q, point_mass_q)
          << "Sparse draft inputs must agree on q semantics";
    }
  }
  // Missing confidence disables that optional batch payload.
  const bool has_confidence =
      std::all_of(draft_inputs.begin(),
                  draft_inputs.end(),
                  [](const DraftInput& draft_input) {
                    return draft_input.selected_probs.defined();
                  });
  return {cat(&DraftInput::token_ids),
          cat(&DraftInput::hidden_states),
          std::move(probs),
          has_confidence ? cat(&DraftInput::selected_probs) : torch::Tensor(),
          point_mass_q};
}

inline DraftInput draft_input_to(const DraftInput& draft_input,
                                 const torch::Device& device,
                                 bool non_blocking = false) {
  check_draft_input(draft_input);
  DraftInput out;
  out.point_mass_q = draft_input.point_mass_q;
  // Not routed through transform_draft_input: this transform runs on the
  // undefined tensors too (safe_to is defined for them), while the shared
  // skeleton deliberately skips them.
  for (torch::Tensor DraftInput::* member : kDraftInputTensors) {
    out.*member = safe_to(draft_input.*member, device, non_blocking);
  }
  return out;
}

// Optional-lifted variant: returns nullopt when the input has no draft_input,
// else moves the copy through the underlying draft_input_to. Keeps every RPC
// producer path from re-implementing the has_value / to(device) pair.
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
