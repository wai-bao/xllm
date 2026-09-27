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

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "common/macros.h"

namespace xllm {

// Per-request continuation state produced by the target model. This store is
// independent from any draft algorithm, proposal distribution, or grammar.
class TargetDecodeStateStore final {
 public:
  struct DecodeState {
    bool valid = false;
    std::string request_id;
    int32_t token_id = -1;
    int32_t position_offset = 0;
  };

  explicit TargetDecodeStateStore(int32_t total_nums);
  ~TargetDecodeStateStore() = default;
  DISALLOW_COPY_AND_ASSIGN(TargetDecodeStateStore);

  void write_prefill_target_context(const std::vector<int32_t>& embedding_ids,
                                    const std::vector<std::string>& request_ids,
                                    const torch::Tensor& next_tokens);
  void write_target_bootstrap_context(int32_t embedding_id,
                                      const std::string& request_id,
                                      int32_t token_id);
  // Accepted prefix length per row is computed inside write_target_context
  // anyway; passing a non-null out-param reuses it instead of re-reading the
  // store (which would repeat the per-row ownership string compares).
  void write_target_context(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids,
      const torch::Tensor& accepted_tokens,
      std::vector<int32_t>* accepted_prefix_lengths = nullptr);
  std::vector<DecodeState> read_decode_states(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  std::vector<int32_t> read_accepted_prefix_lengths(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  // Missing or invalid state has the same one-token fallback as a fresh slot.
  static int32_t accepted_prefix_length(const DecodeState& state);
  void clear(const std::vector<int32_t>& embedding_ids);

 private:
  std::vector<DecodeState> states_;

  DecodeState& mutable_state(int32_t embedding_id);
  const DecodeState& get_state(int32_t embedding_id) const;
  // A slot only answers to the request that last wrote it; a mismatched
  // request sees the fresh-slot fallback instead.
  const DecodeState* owned_state(int32_t embedding_id,
                                 const std::string& request_id) const;
  void reset_state(int32_t embedding_id,
                   std::string request_id,
                   int32_t token_id,
                   int32_t position_offset);
};

}  // namespace xllm
