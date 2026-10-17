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

#include <torch/types.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/platform/stream_event.h"

namespace xllm {

class TargetDecodeStateStore;

// Owns the asynchronous target-result handoff across speculative rounds.
// The worker controls stream ordering; this object owns the pinned destination,
// pending event and request identity until the result reaches the target store.
class MtpTargetContextHandoff final {
 public:
  bool has_pending() const { return pending_.accepted_tokens_host.defined(); }

  torch::Tensor acquire_host_buffer(const torch::Tensor& accepted_tokens,
                                    int64_t configured_capacity);
  template <typename TargetInput>
  void stage(const TargetInput& input,
             StreamEventPtr ready_event,
             torch::Tensor accepted_tokens_host);
  // Writes the staged target context; true when a pending stage was flushed.
  bool flush(TargetDecodeStateStore& store, int32_t num_speculative_tokens);

 private:
  struct PendingContext {
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    torch::Tensor accepted_tokens_host;
    StreamEventPtr ready_event;
  };

  PendingContext pending_;
  torch::Tensor accepted_tokens_host_buffer_;
};

}  // namespace xllm
