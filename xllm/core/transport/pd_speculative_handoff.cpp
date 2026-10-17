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

#include "core/transport/pd_speculative_handoff.h"

#include <glog/logging.h>

#include "core/transport/draft_input_proto.h"
#include "disagg_pd.pb.h"

namespace xllm::proto_adapter {

const char* encode_pd_speculative_handoff(
    SpeculativeHandoffKind kind,
    const std::optional<DraftInput>& draft_input,
    proto::DisaggGenerationsRequest* output) {
  CHECK(output != nullptr);
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
  }
  return "Unknown speculative PD handoff kind";
}

}  // namespace xllm::proto_adapter
