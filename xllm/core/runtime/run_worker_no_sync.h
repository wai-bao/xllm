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
#include <type_traits>
#include <utility>

#include "core/runtime/forward_params.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/worker_impl.h"
#include "platform/stream.h"

namespace xllm {
namespace detail {

// Shared no-sync dispatch for the draft-model workers (MTP and block-diffusion
// families): prepare on the worker's prepare stream, execute on the compute
// stream, never synchronizing the host. The metadata-ready event is recorded
// only when the two streams differ; a same-stream dispatch has no ordering
// gap to publish. LLM workers additionally take the record-free fast path.
// `processed_output` optionally receives the prepared input for callers that
// reuse the prepared views.
template <typename TargetInput>
inline std::optional<ForwardOutput> run_worker_no_sync(
    WorkerImpl& worker,
    const TargetInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    TargetInput* processed_output = nullptr) {
  TargetInput processed_input;
  worker.prepare_work_before_execute_on_stream(
      input,
      processed_input,
      prepare_stream,
      /*record_ready_event=*/&prepare_stream != &compute_stream);
  std::optional<ForwardOutput> output;
  if constexpr (std::is_same_v<TargetInput, LlmForwardInput>) {
    if (auto* llm_worker = dynamic_cast<LLMWorkerImpl*>(&worker);
        llm_worker != nullptr) {
      output = llm_worker->execute_no_sync_on_stream(
          processed_input, compute_stream, /*record_ready_event=*/false);
    } else {
      output =
          worker.execute_no_sync_on_stream(processed_input, compute_stream);
    }
  } else {
    output = worker.execute_no_sync_on_stream(processed_input, compute_stream);
  }
  if (processed_output != nullptr) {
    *processed_output = std::move(processed_input);
  }
  return output;
}

}  // namespace detail
}  // namespace xllm
