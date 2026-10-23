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

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "core/framework/speculative/draft_input.h"
#include "core/framework/speculative/pd_speculative_handoff.h"
#include "core/util/utils.h"
#include "disagg_pd.pb.h"

namespace xllm::proto_adapter {

inline bool draft_input_to_proto(const DraftInput& draft_input,
                                 proto::DraftInput* output) {
  if (!is_valid_draft_input(draft_input)) {
    return false;
  }
  output->Clear();
  output->set_point_mass_q(draft_input.point_mass_q);
  if (!util::torch_to_proto(draft_input.token_ids,
                            output->mutable_token_ids()) ||
      !util::torch_to_proto(draft_input.hidden_states,
                            output->mutable_hidden_states())) {
    return false;
  }
  if (draft_input.draft_logits.defined() &&
      !util::torch_to_proto(draft_input.draft_logits,
                            output->mutable_draft_logits())) {
    return false;
  }
  return !draft_input.selected_probs.defined() ||
         util::torch_to_proto(draft_input.selected_probs,
                              output->mutable_selected_probs());
}

namespace detail {

// Reject wire dtypes outside the draft_input contract before the generic
// decoder allocates and copies their payloads; the dtype contract itself lives
// in is_valid_draft_input_tensor_dtype so the wire and torch forms cannot
// drift. The assembled-draft_input validator remains authoritative for the
// decoded tensors.
inline bool is_valid_draft_input_proto_dtype(const std::string& dtype,
                                             bool token_ids) {
  const auto scalar_type = util::try_datatype_proto_to_torch(dtype);
  return scalar_type.has_value() &&
         is_valid_draft_input_tensor_dtype(*scalar_type, token_ids);
}

// The generic decoder owns shape, payload, and overflow validation.
inline std::optional<torch::Tensor> decode_draft_input_tensor(
    const proto::Tensor& tensor,
    int32_t dimensions,
    int64_t batch_size,
    bool token_ids) {
  if (tensor.shape_size() != dimensions || tensor.shape(0) != batch_size ||
      !is_valid_draft_input_proto_dtype(tensor.datatype(), token_ids)) {
    return std::nullopt;
  }
  return util::try_proto_to_torch(tensor);
}

}  // namespace detail

inline std::optional<DraftInput> draft_input_from_proto(
    const proto::DraftInput& input) {
  if (!input.has_token_ids() || !input.has_hidden_states() ||
      input.token_ids().shape_size() != 1) {
    return std::nullopt;
  }
  const int64_t batch_size = input.token_ids().shape(0);
  std::optional<torch::Tensor> token_ids = detail::decode_draft_input_tensor(
      input.token_ids(), /*dimensions=*/1, batch_size, /*token_ids=*/true);
  std::optional<torch::Tensor> hidden_states =
      detail::decode_draft_input_tensor(input.hidden_states(),
                                        /*dimensions=*/2,
                                        batch_size,
                                        /*token_ids=*/false);
  if (!token_ids.has_value() || !hidden_states.has_value()) {
    return std::nullopt;
  }
  std::optional<torch::Tensor> selected_probs;
  if (input.has_selected_probs()) {
    selected_probs = detail::decode_draft_input_tensor(input.selected_probs(),
                                                       /*dimensions=*/1,
                                                       batch_size,
                                                       /*token_ids=*/false);
    if (!selected_probs.has_value()) {
      return std::nullopt;
    }
  }
  std::optional<torch::Tensor> draft_logits;
  if (input.has_draft_logits()) {
    draft_logits = detail::decode_draft_input_tensor(input.draft_logits(),
                                                     /*dimensions=*/2,
                                                     batch_size,
                                                     /*token_ids=*/false);
    if (!draft_logits.has_value()) {
      return std::nullopt;
    }
  }
  DraftInput draft_input{std::move(*token_ids),
                         std::move(*hidden_states),
                         std::move(selected_probs).value_or(torch::Tensor()),
                         std::move(draft_logits).value_or(torch::Tensor()),
                         input.point_mass_q()};
  if (!is_valid_draft_input(draft_input)) {
    return std::nullopt;
  }
  // proto_to_torch already established independent storage; no clone needed.
  return draft_input;
}

inline const char* encode_pd_speculative_handoff(
    SpeculativeHandoffKind kind,
    const std::optional<DraftInput>& draft_input,
    proto::DisaggGenerationsRequest* output) {
  switch (kind) {
    case SpeculativeHandoffKind::NONE:
    case SpeculativeHandoffKind::BOOTSTRAP_TOKEN:
      return nullptr;
    case SpeculativeHandoffKind::DRAFT_INPUT:
      return draft_input.has_value() &&
                     draft_input_to_proto(*draft_input,
                                          output->mutable_draft_input())
                 ? nullptr
                 : "Missing or invalid draft input";
    default:
      break;
  }
  CHECK(false) << "unhandled speculative PD handoff kind";
}

}  // namespace xllm::proto_adapter
