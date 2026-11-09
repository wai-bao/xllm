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

#include "core/runtime/target_context_handoff.h"

#include <glog/logging.h>

#include <utility>

#include "core/framework/speculative/target_decode_state_store.h"
#include "core/util/tensor_helper.h"

namespace xllm {

torch::Tensor TargetContextHandoff::acquire_host_buffer(
    const torch::Tensor& accepted_tokens) {
  CHECK(accepted_tokens.defined()) << "accepted tokens must be defined";
  CHECK_GT(accepted_tokens.numel(), 0) << "accepted tokens must not be empty";

  // A fresh pinned allocation per round (not a reused buffer): under schedule
  // overlap the next round's D2H may run while the engine still serializes
  // this round's output. Reference counting plus the allocator's free-list
  // give every consumer an independent lease.
  return torch::empty(accepted_tokens.sizes(),
                      pinned_cpu_options(torch::kLong));
}

void TargetContextHandoff::stage(const BatchRows& rows,
                                 StreamEventPtr ready_event,
                                 torch::Tensor accepted_tokens_host) {
  CHECK(!pending_.accepted_tokens_host.defined())
      << "previous MTP target context must be flushed before staging another";
  check_batch_rows(rows);
  pending_.embedding_ids.assign(rows.embedding_ids.begin(),
                                rows.embedding_ids.end());
  pending_.request_ids.assign(rows.request_ids.begin(), rows.request_ids.end());
  pending_.accepted_tokens_host = std::move(accepted_tokens_host);
  pending_.ready_event = std::move(ready_event);
}

void TargetContextHandoff::flush(TargetDecodeStateStore& store) {
  if (!pending_.accepted_tokens_host.defined()) {
    return;
  }
  CHECK(pending_.ready_event == nullptr || pending_.ready_event->synchronize())
      << "failed to wait for pending MTP target context";
  const torch::Tensor& accepted_tokens_host = pending_.accepted_tokens_host;
  // Failed requests may still have a scheduler-overlapped turn in flight.
  // Their target bookkeeping stays until the embedding is released.
  store.write_target_context(pending_.rows(), accepted_tokens_host);
  // Only the guard is reset here. stage() overwrites the other fields, and
  // clearing the row identities would destroy every std::string element,
  // forcing the next stage() to re-allocate each request-id buffer; assigning
  // over live strings reuses them instead. The undefined accepted_tokens_host
  // is what both stage() and flush() test, so leaving stale rows is not
  // observable.
  pending_.accepted_tokens_host = torch::Tensor();
  pending_.ready_event.reset();
}

}  // namespace xllm
