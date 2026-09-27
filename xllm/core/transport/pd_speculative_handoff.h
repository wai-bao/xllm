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
#include <string_view>

#include "core/framework/speculative/mtp_draft_seed.h"

namespace xllm::proto {
class DisaggGenerationsRequest;
}

namespace xllm::proto_adapter {

enum class SpeculativeHandoffKind {
  NONE,
  BOOTSTRAP_TOKEN,
  DRAFT_SEED,
};

SpeculativeHandoffKind pd_speculative_handoff_kind(
    std::string_view algorithm,
    int32_t num_speculative_tokens);

// Returns nullptr on success; the scheduler retains ownership of the source
// request until it has decided how to handle the RPC result.
const char* encode_pd_speculative_handoff(
    SpeculativeHandoffKind kind,
    const std::optional<MtpDraftSeed>& draft_seed,
    proto::DisaggGenerationsRequest* output);

// Validate decoded payloads before the scheduler mutates request state.
// Wire parsing remains at the RPC service boundary.
const char* validate_pd_bootstrap_handoff(int32_t slot_id, int64_t token_id);

const char* validate_pd_draft_seed_handoff(
    int32_t slot_id,
    int64_t token_id,
    const std::optional<MtpDraftSeed>& draft_seed,
    bool probabilistic,
    int64_t vocab_size,
    const DraftSeedModelSpec* draft_spec);

}  // namespace xllm::proto_adapter
