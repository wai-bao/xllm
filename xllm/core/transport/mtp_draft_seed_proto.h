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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "core/framework/speculative/mtp_draft_seed.h"
#include "core/util/utils.h"
#include "disagg_pd.pb.h"

namespace xllm::proto_adapter {

inline bool mtp_draft_seed_to_proto(const MtpDraftSeed& seed,
                                    proto::MtpDraftSeed* output) {
  if (!is_valid_mtp_draft_seed(seed)) {
    return false;
  }
  output->Clear();
  output->set_point_mass_q(seed.point_mass_q);
  if (!util::torch_to_proto(seed.token_ids, output->mutable_token_ids()) ||
      !util::torch_to_proto(seed.hidden_states,
                            output->mutable_hidden_states())) {
    return false;
  }
  if (seed.probs.defined() &&
      !util::torch_to_proto(seed.probs, output->mutable_probs())) {
    return false;
  }
  return !seed.selected_probs.defined() ||
         util::torch_to_proto(seed.selected_probs,
                              output->mutable_selected_probs());
}

namespace detail {

// Reject wire dtypes outside the seed contract before the generic decoder
// allocates and copies their payloads. The assembled-seed validator remains
// authoritative for the decoded torch::ScalarType contract.
inline bool is_valid_mtp_seed_proto_dtype(std::string_view dtype,
                                          bool token_ids) {
  if (token_ids) {
    return dtype == "INT32" || dtype == "INT64";
  }
  return dtype == "FP16" || dtype == "BF16" || dtype == "FP32" ||
         dtype == "FP64";
}

// The generic decoder owns shape, payload, and overflow validation.
inline std::optional<torch::Tensor> decode_mtp_seed_tensor(
    const proto::Tensor& tensor,
    int32_t dimensions,
    int64_t batch_size,
    bool token_ids) {
  if (tensor.shape_size() != dimensions || tensor.shape(0) != batch_size ||
      !is_valid_mtp_seed_proto_dtype(tensor.datatype(), token_ids)) {
    return std::nullopt;
  }
  return util::try_proto_to_torch(tensor);
}

}  // namespace detail

inline std::optional<MtpDraftSeed> mtp_draft_seed_from_proto(
    const proto::MtpDraftSeed& input) {
  if (!input.has_token_ids() || !input.has_hidden_states() ||
      input.token_ids().shape_size() != 1) {
    return std::nullopt;
  }
  const int64_t batch_size = input.token_ids().shape(0);
  std::optional<torch::Tensor> token_ids = detail::decode_mtp_seed_tensor(
      input.token_ids(), /*dimensions=*/1, batch_size, /*token_ids=*/true);
  std::optional<torch::Tensor> hidden_states = detail::decode_mtp_seed_tensor(
      input.hidden_states(), /*dimensions=*/2, batch_size, /*token_ids=*/false);
  if (!token_ids.has_value() || !hidden_states.has_value()) {
    return std::nullopt;
  }
  std::optional<torch::Tensor> probs;
  if (input.has_probs()) {
    probs = detail::decode_mtp_seed_tensor(
        input.probs(), /*dimensions=*/2, batch_size, /*token_ids=*/false);
    if (!probs.has_value()) {
      return std::nullopt;
    }
  }
  std::optional<torch::Tensor> selected_probs;
  if (input.has_selected_probs()) {
    selected_probs = detail::decode_mtp_seed_tensor(input.selected_probs(),
                                                    /*dimensions=*/1,
                                                    batch_size,
                                                    /*token_ids=*/false);
    if (!selected_probs.has_value()) {
      return std::nullopt;
    }
  }
  MtpDraftSeed seed{std::move(*token_ids),
                    std::move(*hidden_states),
                    std::move(probs).value_or(torch::Tensor()),
                    std::move(selected_probs).value_or(torch::Tensor()),
                    input.point_mass_q()};
  if (!is_valid_mtp_draft_seed(seed)) {
    return std::nullopt;
  }
  // proto_to_torch already established independent storage; no clone needed.
  return seed;
}

}  // namespace xllm::proto_adapter
