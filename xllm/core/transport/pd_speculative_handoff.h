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

#include <optional>

#include "core/framework/speculative/pd_speculative_handoff.h"

namespace xllm::proto {
class DisaggGenerationsRequest;
}

namespace xllm::proto_adapter {

// Returns nullptr on success; the scheduler retains ownership of the source
// request until it has decided how to handle the RPC result.
const char* encode_pd_speculative_handoff(
    SpeculativeHandoffKind kind,
    const std::optional<DraftInput>& draft_input,
    proto::DisaggGenerationsRequest* output);

}  // namespace xllm::proto_adapter
