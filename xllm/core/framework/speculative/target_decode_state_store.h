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
  // Validates every row index against all three inputs before indexing them,
  // then bootstraps each carried row's slot in one pass.
  void write_target_bootstrap_rows(const std::vector<int32_t>& rows,
                                   const std::vector<int32_t>& embedding_ids,
                                   const std::vector<std::string>& request_ids,
                                   const torch::Tensor& token_ids_host);
  // Writes one DecodeState per row: the last accepted token and its offset.
  void write_target_context(const std::vector<int32_t>& embedding_ids,
                            const std::vector<std::string>& request_ids,
                            const torch::Tensor& accepted_tokens);
  std::vector<DecodeState> read_decode_states(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids) const;
  // Read-or-default variant shared by the chunked-prefill verify tails: rows
  // without stored state keep the one-token fallback.
  std::vector<int32_t> resolve_accepted_prefix_lengths(
      const std::vector<int32_t>& embedding_ids,
      const std::vector<std::string>& request_ids,
      int32_t num_sequences) const;
  // Missing or invalid state has the same one-token fallback as a fresh slot.
  static int32_t accepted_prefix_length(const DecodeState& state);
  void clear(const std::vector<int32_t>& embedding_ids);

 private:
  std::vector<DecodeState> states_;

  const DecodeState* owned_state(int32_t embedding_id,
                                 const std::string* request_id) const;
  void reset_state(int32_t embedding_id,
                   std::string request_id,
                   int32_t token_id,
                   int32_t position_offset);
};

}  // namespace xllm
