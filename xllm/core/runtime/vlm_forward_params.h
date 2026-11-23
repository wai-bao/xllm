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

#include "core/framework/model/vlm_model_params.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/runtime/forward_params.h"

namespace xllm {

class VlmForwardInput;
namespace detail {
bool unpack_from_input_host_buffer(const VlmForwardInput& input,
                                   const torch::Device& device,
                                   torch::ScalarType dtype,
                                   VlmForwardInput& output,
                                   bool materialize_device_buffer);
}  // namespace detail

class VlmForwardInput final {
 public:
  VlmForwardInput() = default;
  VlmForwardInput(const VlmForwardInput&) = delete;
  VlmForwardInput& operator=(const VlmForwardInput&) = delete;
  VlmForwardInput(VlmForwardInput&&) = default;
  VlmForwardInput& operator=(VlmForwardInput&&) = default;

  bool to_packed_proto(proto::PackedForwardInput* packed_forward_input) const;
  bool from_packed_proto(const proto::PackedForwardInput& packed_forward_input);
  // Large continuous CPU tensors are retained by the attachment. Callers must
  // keep their storage immutable until the attachment is released.
  bool to_packed_attachment(proto::PackedForwardInput* metadata,
                            butil::IOBuf& attachment) const;
  bool from_packed_attachment(const proto::PackedForwardInput& metadata,
                              const butil::IOBuf& attachment);

  VlmForwardInput clone() const;

  VlmForwardInput to(const torch::Device& device,
                     torch::ScalarType dtype) const;

  const torch::Tensor& host_token_ids() const {
    return token_ids_host.defined() ? token_ids_host : token_ids;
  }

  const torch::Tensor& host_positions() const {
    return positions_host.defined() ? positions_host : positions;
  }

  // flatten token ids
  torch::Tensor token_ids;
  // flatten positions
  torch::Tensor positions;
  torch::Tensor token_ids_host;
  torch::Tensor positions_host;
  mutable VlmModelParams input_params;
  SpeculativeInput speculative_;
  std::vector<std::string> request_ids;
  mutable MMBatchData mm_data_;
  SamplingParameters sampling_params;
  std::vector<std::string> sample_sequence_ids;
  std::vector<int32_t> sample_prior_output_rows;
  std::vector<JsonObjectGrammarState> json_object_states;
  std::vector<JsonObjectGrammarSnapshot> json_object_state_snapshots;
  // Errors detected while aligning prior overlap output with grammar rows.
  std::vector<JsonObjectOutputError> json_object_errors;

  // step-level decode metadata
  // If true, skip sampler forward and keep raw logits; the speculative
  // consumer applies the sampling transforms itself.
  bool skip_sampling_for_logits_only = false;
  // If true, populate ForwardOutput.selected_hidden with hidden states matching
  // the `logits` selection layout. Used by DSpark ConfidenceHead which needs
  // pre-lm_head hidden states of the draft tokens.
  bool return_selected_hidden = false;

  // kv info for disaggregated prefill/decode
  std::vector<TransferKVInfo> transfer_kv_infos;

  int32_t input_host_sample_count = kUnknownPackedSampleCount;
  ForwardRuntimeState runtime;
};

LlmForwardInput make_llm_draft_input(const LlmForwardInput& target,
                                     bool shared_full_cache = false);

LlmForwardInput make_llm_draft_input(const VlmForwardInput& target,
                                     bool shared_full_cache = false);

}  // namespace xllm
