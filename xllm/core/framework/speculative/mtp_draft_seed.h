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

#include "core/framework/speculative/draft_seed_model_spec.h"
#include "core/util/tensor_helper.h"

namespace xllm {

// A sampled first proposal and its recurrent draft hidden. Every tensor keeps
// its leading batch dimension, including a seed owned by a single sequence.
struct MtpDraftSeed {
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
inline constexpr std::array<torch::Tensor MtpDraftSeed::*, 4>
    kMtpDraftSeedTensors = {&MtpDraftSeed::token_ids,
                            &MtpDraftSeed::hidden_states,
                            &MtpDraftSeed::probs,
                            &MtpDraftSeed::selected_probs};

namespace detail {

// Assembled-seed dtype rule: token_ids must be int32/int64, every other
// payload must be floating point. The proto decoder checks corresponding wire
// datatypes before allocation, then this predicate validates the decoded seed.
inline bool is_valid_seed_tensor_dtype(torch::ScalarType dtype,
                                       bool is_token_ids) {
  if (is_token_ids) {
    return dtype == torch::kInt || dtype == torch::kLong;
  }
  return c10::isFloatingType(dtype);
}

// Single source of truth for the MtpDraftSeed shape/dtype invariant. Returns a
// static reason string for the first violated constraint, or nullptr when the
// seed is valid. is_valid_mtp_draft_seed and check_mtp_draft_seed both route
// through this so the boolean guard and the fatal assertion cannot diverge.
inline const char* mtp_draft_seed_invalid_reason(const MtpDraftSeed& seed) {
  if (!seed.token_ids.defined()) {
    return "token ids are undefined";
  }
  if (!seed.hidden_states.defined()) {
    return "hidden states are undefined";
  }
  if (seed.token_ids.dim() != 1) {
    return "token ids must be 1-D [batch]";
  }
  if (!is_valid_seed_tensor_dtype(seed.token_ids.scalar_type(),
                                  /*is_token_ids=*/true)) {
    return "token ids must be int32 or int64";
  }
  if (seed.hidden_states.dim() != 2) {
    return "hidden states must be 2-D [batch, hidden]";
  }
  if (seed.hidden_states.size(0) != seed.token_ids.size(0)) {
    return "hidden states batch does not match token ids";
  }
  if (!is_valid_seed_tensor_dtype(seed.hidden_states.scalar_type(),
                                  /*is_token_ids=*/false)) {
    return "hidden states must be floating point";
  }
  const int64_t batch_size = seed.token_ids.size(0);
  if (seed.probs.defined() &&
      (seed.probs.dim() != 2 || seed.probs.size(0) != batch_size ||
       seed.probs.size(1) <= 0 ||
       !is_valid_seed_tensor_dtype(seed.probs.scalar_type(),
                                   /*is_token_ids=*/false))) {
    return "probabilities must be floating point [batch, target_vocab > 0]";
  }
  if (seed.selected_probs.defined() &&
      (seed.selected_probs.dim() != 1 ||
       seed.selected_probs.size(0) != batch_size ||
       !is_valid_seed_tensor_dtype(seed.selected_probs.scalar_type(),
                                   /*is_token_ids=*/false))) {
    return "selected probabilities must be floating point [batch]";
  }
  if (seed.point_mass_q && seed.probs.defined()) {
    return "point-mass seed must not carry dense probabilities";
  }
  return nullptr;
}

}  // namespace detail

inline bool is_valid_mtp_draft_seed(const MtpDraftSeed& seed) {
  return detail::mtp_draft_seed_invalid_reason(seed) == nullptr;
}

inline void check_mtp_draft_seed(const MtpDraftSeed& seed) {
  const char* reason = detail::mtp_draft_seed_invalid_reason(seed);
  CHECK(reason == nullptr) << "invalid MTP draft seed: " << reason;
}

inline const char* mtp_draft_seed_model_reason(const MtpDraftSeed& seed,
                                               const DraftSeedModelSpec& spec) {
  const char* reason = detail::mtp_draft_seed_invalid_reason(seed);
  if (reason != nullptr) {
    return reason;
  }
  if (spec.hidden_size <= 0) {
    return "draft model hidden specification is unavailable";
  }
  if (seed.hidden_states.size(1) != spec.hidden_size) {
    return "seed hidden width does not match the loaded draft model";
  }
  if (seed.hidden_states.scalar_type() != spec.hidden_dtype) {
    return "seed hidden dtype does not match the loaded draft model";
  }
  return nullptr;
}

inline void check_mtp_draft_seed_model(const MtpDraftSeed& seed,
                                       const DraftSeedModelSpec& spec) {
  const char* reason = mtp_draft_seed_model_reason(seed, spec);
  CHECK(reason == nullptr) << "MTP draft seed violates model contract: "
                           << reason;
}

namespace detail {

// Probabilistic seeds carry either full-vocabulary q or an explicit point-mass
// marker. Missing q alone never implies deterministic proposal semantics.
inline const char* mtp_draft_seed_mode_reason(const MtpDraftSeed& seed,
                                              bool probabilistic,
                                              int64_t vocab_size) {
  if (probabilistic) {
    if (!seed.probs.defined() && !seed.point_mass_q) {
      return "probabilistic seed requires full-vocab probs or point-mass q";
    }
  } else if (seed.probs.defined() || seed.point_mass_q) {
    return "non-probabilistic seed must not carry proposal q";
  }
  if (probabilistic && seed.probs.defined() &&
      (seed.probs.dim() != 2 || seed.probs.size(1) != vocab_size)) {
    return "probs must span the full target vocabulary";
  }
  return nullptr;
}

}  // namespace detail

inline void check_mtp_draft_seed_mode(const MtpDraftSeed& seed,
                                      bool probabilistic,
                                      int64_t vocab_size) {
  const char* reason =
      detail::mtp_draft_seed_mode_reason(seed, probabilistic, vocab_size);
  CHECK(reason == nullptr) << "MTP draft seed violates sampling-mode contract: "
                           << reason;
}

// Single source of truth for the "PD-imported seed" contract: a shape/mode
// valid seed that carries exactly one row whose draft token is in vocabulary.
// Returns the first violated constraint's reason, or nullptr when importable.
// Any importer routes through this so the receiver checks cannot drift from the
// producer contract in mtp_draft_seed_invalid_reason /
// mtp_draft_seed_mode_reason.
inline const char* mtp_draft_seed_import_reason(
    const MtpDraftSeed& seed,
    bool probabilistic,
    int64_t vocab_size,
    const DraftSeedModelSpec& spec) {
  const char* reason = mtp_draft_seed_model_reason(seed, spec);
  if (reason != nullptr) {
    return reason;
  }
  if (seed.token_ids.size(0) != 1) {
    return "imported seed must carry exactly one row";
  }
  reason = detail::mtp_draft_seed_mode_reason(seed, probabilistic, vocab_size);
  if (reason != nullptr) {
    return reason;
  }
  const int64_t draft_token_id = seed.token_ids.item<int64_t>();
  if (draft_token_id < 0 || draft_token_id >= vocab_size) {
    return "imported seed draft token is outside the target vocabulary";
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

// In-place variant of make_point_mass_probs: zeroes the greedy rows of an
// existing q (rows where do_sample is false) and writes 1.0 at their sampled
// index, leaving random rows untouched.
inline void write_point_mass_rows(const torch::Tensor& token_ids,
                                  torch::Tensor& probs,
                                  const torch::Tensor& do_sample) {
  CHECK(token_ids.defined() && probs.defined() && do_sample.defined());
  CHECK_EQ(token_ids.dim(), 1);
  CHECK_EQ(probs.dim(), 2);
  CHECK_EQ(do_sample.dim(), 1);
  CHECK_EQ(probs.size(0), token_ids.size(0));
  CHECK_EQ(do_sample.size(0), token_ids.size(0));
  CHECK_EQ(do_sample.scalar_type(), torch::kBool);
  probs.mul_(do_sample.to(probs.device(), probs.scalar_type()).view({-1, 1}));
  probs.scatter_add_(
      /*dim=*/1,
      token_ids.view({-1, 1}),
      do_sample.logical_not()
          .to(probs.device(), probs.scalar_type())
          .view({-1, 1}));
}

// Copy every scalar in the same order used by shared-memory serialization.
inline void copy_mtp_draft_seed_metadata(MtpDraftSeed& out,
                                         const MtpDraftSeed& seed) {
  out.point_mass_q = seed.point_mass_q;
}

inline MtpDraftSeed clone_mtp_draft_seed(const MtpDraftSeed& seed) {
  check_mtp_draft_seed(seed);
  MtpDraftSeed out;
  copy_mtp_draft_seed_metadata(out, seed);
  for (torch::Tensor MtpDraftSeed::* member : kMtpDraftSeedTensors) {
    const torch::Tensor& tensor = seed.*member;
    if (tensor.defined()) {
      out.*member = clone_contiguous_detached_tensor(tensor);
    }
  }
  return out;
}

// Returns views that share the source storage but carry no autograd history.
// Storage stays alive through the source tensor's refcount, so callers keep
// the producer's memory instead of copying [rows, vocab] payloads per request.
inline MtpDraftSeed detach_mtp_draft_seed(const MtpDraftSeed& seed) {
  check_mtp_draft_seed(seed);
  MtpDraftSeed out;
  copy_mtp_draft_seed_metadata(out, seed);
  for (torch::Tensor MtpDraftSeed::* member : kMtpDraftSeedTensors) {
    const torch::Tensor& tensor = seed.*member;
    if (tensor.defined()) {
      out.*member = tensor.detach();
    }
  }
  return out;
}

// Return views; callers retaining producer-owned storage must clone them.
inline MtpDraftSeed slice_mtp_draft_seed(const MtpDraftSeed& seed,
                                         int64_t start,
                                         int64_t length) {
  check_mtp_draft_seed(seed);
  MtpDraftSeed out;
  copy_mtp_draft_seed_metadata(out, seed);
  for (torch::Tensor MtpDraftSeed::* member : kMtpDraftSeedTensors) {
    const torch::Tensor& tensor = seed.*member;
    if (tensor.defined()) {
      out.*member = tensor.narrow(/*dim=*/0, start, length);
    }
  }
  return out;
}

inline MtpDraftSeed cat_mtp_draft_seeds(
    const std::vector<MtpDraftSeed>& seeds) {
  CHECK(!seeds.empty());
  for (const MtpDraftSeed& seed : seeds) {
    check_mtp_draft_seed(seed);
  }
  const auto cat = [&seeds](torch::Tensor MtpDraftSeed::* member) {
    const bool defined = (seeds.front().*member).defined();
    std::vector<torch::Tensor> tensors;
    tensors.reserve(seeds.size());
    for (const MtpDraftSeed& seed : seeds) {
      CHECK_EQ((seed.*member).defined(), defined)
          << "MTP draft seeds must agree on optional probability payloads";
      if (defined) {
        tensors.emplace_back(seed.*member);
      }
    }
    return defined ? torch::cat(tensors, /*dim=*/0) : torch::Tensor();
  };
  const auto dense_seed =
      std::find_if(seeds.begin(), seeds.end(), [](const MtpDraftSeed& seed) {
        return seed.probs.defined();
      });
  const bool any_dense = dense_seed != seeds.end();
  const bool point_mass_q = !any_dense && seeds.front().point_mass_q;
  torch::Tensor probs;
  if (any_dense) {
    const int64_t vocab_size = dense_seed->probs.size(1);
    std::vector<torch::Tensor> rows;
    rows.reserve(seeds.size());
    for (const MtpDraftSeed& seed : seeds) {
      if (seed.probs.defined()) {
        CHECK_EQ(seed.probs.size(1), vocab_size);
        rows.emplace_back(seed.probs);
      } else {
        CHECK(seed.point_mass_q)
            << "Missing proposal q requires an explicit point-mass marker";
        rows.emplace_back(make_point_mass_probs(
            seed.token_ids, vocab_size, dense_seed->probs.options()));
      }
    }
    probs = torch::cat(rows, /*dim=*/0);
  } else {
    for (const MtpDraftSeed& seed : seeds) {
      CHECK_EQ(seed.point_mass_q, point_mass_q)
          << "Sparse MTP draft seeds must agree on q semantics";
    }
  }
  // Missing confidence disables that optional batch payload.
  const bool has_confidence =
      std::all_of(seeds.begin(), seeds.end(), [](const MtpDraftSeed& seed) {
        return seed.selected_probs.defined();
      });
  return {cat(&MtpDraftSeed::token_ids),
          cat(&MtpDraftSeed::hidden_states),
          std::move(probs),
          has_confidence ? cat(&MtpDraftSeed::selected_probs) : torch::Tensor(),
          point_mass_q};
}

inline MtpDraftSeed mtp_draft_seed_to(const MtpDraftSeed& seed,
                                      const torch::Device& device,
                                      bool non_blocking = false) {
  check_mtp_draft_seed(seed);
  MtpDraftSeed out;
  copy_mtp_draft_seed_metadata(out, seed);
  for (torch::Tensor MtpDraftSeed::* member : kMtpDraftSeedTensors) {
    out.*member = safe_to(seed.*member, device, non_blocking);
  }
  return out;
}

// Optional-lifted variant: returns nullopt when the input has no seed, else
// moves the copy through the underlying mtp_draft_seed_to. Keeps every RPC
// producer path from re-implementing the has_value / to(device) pair.
inline std::optional<MtpDraftSeed> mtp_draft_seed_to(
    const std::optional<MtpDraftSeed>& seed,
    const torch::Device& device,
    bool non_blocking = false) {
  if (!seed.has_value()) {
    return std::nullopt;
  }
  return mtp_draft_seed_to(*seed, device, non_blocking);
}

}  // namespace xllm
