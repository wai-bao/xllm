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

#include <limits>

#include "core/framework/config/speculative_config.h"
#include "core/transport/mtp_draft_seed_proto.h"
#include "disagg_pd.pb.h"

namespace xllm::proto_adapter {

SpeculativeHandoffKind pd_speculative_handoff_kind(
    std::string_view algorithm,
    int32_t num_speculative_tokens) {
  if (num_speculative_tokens <= 0) {
    return SpeculativeHandoffKind::NONE;
  }
  if (SpeculativeConfig::produces_mtp_draft_seed(algorithm)) {
    return SpeculativeHandoffKind::DRAFT_SEED;
  }
  return SpeculativeConfig::is_block_diffusion_algorithm(algorithm)
             ? SpeculativeHandoffKind::BOOTSTRAP_TOKEN
             : SpeculativeHandoffKind::NONE;
}

const char* encode_pd_speculative_handoff(
    SpeculativeHandoffKind kind,
    const std::optional<MtpDraftSeed>& draft_seed,
    proto::DisaggGenerationsRequest* output) {
  CHECK(output != nullptr);
  switch (kind) {
    case SpeculativeHandoffKind::NONE:
    case SpeculativeHandoffKind::BOOTSTRAP_TOKEN:
      return nullptr;
    case SpeculativeHandoffKind::DRAFT_SEED:
      return draft_seed.has_value() &&
                     mtp_draft_seed_to_proto(*draft_seed,
                                             output->mutable_mtp_draft_seed())
                 ? nullptr
                 : "Missing or invalid MTP draft seed";
  }
  return "Unknown speculative PD handoff kind";
}

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

const char* validate_pd_draft_seed_handoff(
    int32_t slot_id,
    int64_t token_id,
    const std::optional<MtpDraftSeed>& draft_seed,
    bool probabilistic,
    int64_t vocab_size,
    const DraftSeedModelSpec* draft_spec) {
  if (draft_spec == nullptr) {
    return "draft model seed specification is unavailable";
  }
  if (!draft_seed.has_value()) {
    return "missing MTP draft seed";
  }
  if (slot_id < 0 || token_id < 0 || token_id >= vocab_size) {
    return "invalid decode context";
  }
  return mtp_draft_seed_import_reason(
      *draft_seed, probabilistic, vocab_size, *draft_spec);
}

}  // namespace xllm::proto_adapter
