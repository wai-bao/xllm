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

#include "core/common/batch_rows.h"
#include "core/platform/stream_event.h"

namespace xllm {

class TargetDecodeStateStore;

// Owns the asynchronous target-result handoff across speculative rounds.
// The worker controls stream ordering; this object owns the pinned destination,
// pending event and request identity until the result reaches the target store.
class TargetContextHandoff final {
 public:
  bool has_pending() const { return pending_.accepted_tokens_host.defined(); }

  torch::Tensor acquire_host_buffer(const torch::Tensor& accepted_tokens);
  // stage copies the row identities out of the borrowed BatchRows view; the
  // pending context must own them because the batch builder's vectors are
  // rewritten by the overlapped next round before flush runs.
  void stage(const BatchRows& rows,
             StreamEventPtr ready_event,
             torch::Tensor accepted_tokens_host);
  void flush(TargetDecodeStateStore& store);

 private:
  struct PendingContext {
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    torch::Tensor accepted_tokens_host;
    StreamEventPtr ready_event;

    BatchRows rows() const { return {embedding_ids, request_ids}; }
  };

  PendingContext pending_;
};

}  // namespace xllm
