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

#include "core/runtime/mtp_target_context_handoff.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "core/framework/speculative/target_decode_state_store.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/vlm_forward_params.h"

namespace xllm {

torch::Tensor MtpTargetContextHandoff::acquire_host_buffer(
    const torch::Tensor& accepted_tokens,
    int64_t configured_capacity) {
  CHECK(accepted_tokens.defined()) << "accepted tokens must be defined";
  CHECK_GT(accepted_tokens.numel(), 0) << "accepted tokens must not be empty";
  CHECK(!pending_.accepted_tokens_host.defined())
      << "accepted-token host buffer is still in use";

  const int64_t required_capacity = accepted_tokens.numel();
  // The buffer is always int64: the D2H copy_ below fuses any int32 source
  // widening, so every downstream consumer's to_cpu_contiguous(x, kLong)
  // becomes a dtype-preserving no-op instead of a per-round matrix copy.
  const bool needs_allocation =
      !accepted_tokens_host_buffer_.defined() ||
      accepted_tokens_host_buffer_.numel() < required_capacity;
  if (needs_allocation) {
    const int64_t capacity = std::max(required_capacity, configured_capacity);
    accepted_tokens_host_buffer_ =
        torch::empty({capacity}, pinned_cpu_options(torch::kLong));
  }
  return accepted_tokens_host_buffer_
      .narrow(/*dim=*/0, /*start=*/0, required_capacity)
      .view(accepted_tokens.sizes());
}

template <typename TargetInput>
void MtpTargetContextHandoff::stage(const TargetInput& input,
                                    StreamEventPtr ready_event,
                                    torch::Tensor accepted_tokens_host) {
  CHECK(!pending_.accepted_tokens_host.defined())
      << "previous MTP target context must be flushed before staging another";
  pending_.embedding_ids = input.input_params.embedding.embedding_ids;
  pending_.request_ids = input.input_params.embedding.request_ids;
  pending_.accepted_tokens_host = std::move(accepted_tokens_host);
  pending_.ready_event = std::move(ready_event);
}

bool MtpTargetContextHandoff::flush(TargetDecodeStateStore& store,
                                    int32_t num_speculative_tokens) {
  if (!pending_.accepted_tokens_host.defined()) {
    return false;
  }
  CHECK(pending_.ready_event == nullptr || pending_.ready_event->synchronize())
      << "failed to wait for pending MTP target context";
  const int32_t num_validation_tokens = num_speculative_tokens + 1;
  const torch::Tensor& accepted_tokens_host = pending_.accepted_tokens_host;
  CHECK_EQ(accepted_tokens_host.numel() % num_validation_tokens, 0)
      << "MTP validation output width mismatch";
  // Failed requests may still have a scheduler-overlapped turn in flight.
  // Their target bookkeeping stays until the embedding is released.
  store.write_target_context(
      pending_.embedding_ids, pending_.request_ids, accepted_tokens_host);
  // Only the guard is reset here. stage() overwrites the other fields, and
  // clearing request_ids would destroy every std::string element, forcing the
  // next stage() to re-allocate each request-id buffer; copy-assigning over
  // live strings reuses them instead. The undefined accepted_tokens_host is
  // what both stage() and flush() test, so leaving stale rows is not
  // observable.
  pending_.accepted_tokens_host = torch::Tensor();
  pending_.ready_event.reset();
  return true;
}

template void MtpTargetContextHandoff::stage(const LlmForwardInput&,
                                             StreamEventPtr,
                                             torch::Tensor);
template void MtpTargetContextHandoff::stage(const VlmForwardInput&,
                                             StreamEventPtr,
                                             torch::Tensor);

}  // namespace xllm
