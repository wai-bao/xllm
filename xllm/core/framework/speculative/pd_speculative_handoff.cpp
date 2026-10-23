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

#include "core/framework/speculative/pd_speculative_handoff.h"

#include <limits>

#include "core/framework/config/speculative_config.h"
#include "core/framework/request/sequence.h"

namespace xllm {
namespace {

const char* validate_pd_bootstrap_handoff(int32_t slot_id, int64_t token_id) {
  if (slot_id < 0) {
    return "Invalid target decode bootstrap slot";
  }
  if (token_id < 0 ||
      token_id > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return "Invalid target decode bootstrap token";
  }
  return nullptr;
}

const char* validate_pd_draft_input_handoff(
    int32_t slot_id,
    int64_t token_id,
    const std::optional<DraftInput>& draft_input,
    bool probabilistic,
    int64_t vocab_size,
    const DraftInputModelSpec* draft_spec) {
  if (draft_spec == nullptr) {
    return "draft model input specification is unavailable";
  }
  if (!draft_input.has_value()) {
    return "missing draft input";
  }
  if (slot_id < 0 || token_id < 0 || token_id >= vocab_size) {
    return "invalid decode context";
  }
  return draft_input_import_reason(
      *draft_input, probabilistic, vocab_size, *draft_spec);
}

}  // namespace

SpeculativeHandoffKind pd_speculative_handoff_kind(
    std::string_view algorithm,
    int32_t num_speculative_tokens) {
  if (num_speculative_tokens <= 0) {
    return SpeculativeHandoffKind::NONE;
  }
  if (SpeculativeConfig::produces_draft_input(algorithm)) {
    return SpeculativeHandoffKind::DRAFT_INPUT;
  }
  return SpeculativeConfig::is_block_diffusion_algorithm(algorithm)
             ? SpeculativeHandoffKind::BOOTSTRAP_TOKEN
             : SpeculativeHandoffKind::NONE;
}

const char* apply_pd_handoff(SpeculativeHandoffKind kind,
                             Sequence& sequence,
                             int64_t token_id,
                             const std::optional<DraftInput>& draft_input,
                             bool probabilistic,
                             int64_t vocab_size,
                             const DraftInputModelSpec* draft_spec) {
  const int32_t slot_id = sequence.get_embedding_block_id();
  switch (kind) {
    case SpeculativeHandoffKind::BOOTSTRAP_TOKEN: {
      // The DRAFT_INPUT validator re-checks slot/token with a tighter
      // vocabulary bound, so only the bootstrap-token branch needs the
      // standalone handoff validation.
      const char* bootstrap_reason =
          validate_pd_bootstrap_handoff(slot_id, token_id);
      if (bootstrap_reason != nullptr) {
        return bootstrap_reason;
      }
      sequence.mark_target_decode_bootstrap();
      return nullptr;
    }
    case SpeculativeHandoffKind::DRAFT_INPUT: {
      const char* import_reason = validate_pd_draft_input_handoff(slot_id,
                                                                  token_id,
                                                                  draft_input,
                                                                  probabilistic,
                                                                  vocab_size,
                                                                  draft_spec);
      if (import_reason != nullptr) {
        return import_reason;
      }
      sequence.update_draft_input(*draft_input);
      return nullptr;
    }
    case SpeculativeHandoffKind::NONE:
      return nullptr;
    default:
      break;
  }
  CHECK(false) << "unhandled speculative PD handoff kind";
}

}  // namespace xllm
