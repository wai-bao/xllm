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

#include <glog/logging.h>
#include <torch/types.h>

#include <cstdint>
#include <vector>

#include "core/platform/stream.h"
#include "core/platform/stream_event.h"

namespace xllm {

// Worker-local KV slot layout for NPU CP (not transported).
enum class KvSlotLayout : int8_t {
  LOGICAL_REAL = 0,  // Builder slots; input to prepare_cache_slots.
  NPU_CP_RECOVERED_PHYSICAL = 1,  // Already CP-expanded; skip re-prepare.
};

// Buffer ownership and execution lifetime state shared by forward domains.
struct ForwardRuntimeState {
  // Own packed host staging storage and its corresponding device storage.
  // Model input tensors may reference regions within these buffers.
  torch::Tensor input_host_buffer;
  torch::Tensor device_input_buffer;
  bool input_host_buffer_has_layout = false;

  // True when all model input tensors already reference execution-device
  // views. Runtime preparation can reuse them without rebuilding or copying.
  bool device_tensors_ready = false;

  // KV slot layout; flip after one-shot context-parallel remapping.
  KvSlotLayout kv_slot_layout = KvSlotLayout::LOGICAL_REAL;

  // Device-side readiness dependencies for inputs prepared on a different
  // stream. These are local runtime handles and are intentionally not included
  // in proto or shared-memory transport.
  StreamEventPtr metadata_ready_event;

  // Keep cross-stream metadata sources alive through no-sync execution. These
  // handles are local runtime state and are not serialized.
  std::vector<torch::Tensor> retained_device_tensors;
};

// Marks the input's runtime state as ready on `stream` so consumers on other
// streams can wait on metadata_ready_event instead of a full synchronize.
template <typename Input>
inline void record_metadata_ready_event(Stream& stream, Input& input) {
  input.runtime.metadata_ready_event = stream.record_event_or_sync();
}

template <typename Input>
inline void wait_metadata_ready_event(const Input& input, Stream& stream) {
  CHECK(stream.wait_event(input.runtime.metadata_ready_event))
      << "failed to wait metadata ready event";
}

// Re-arms the ready event on `stream`: waits out the previous event first so
// the old handle is not replaced while still awaited elsewhere.
template <typename Input>
inline void record_current_metadata_ready_event(Input& input, Stream& stream) {
  wait_metadata_ready_event(input, stream);
  record_metadata_ready_event(stream, input);
}

template <typename Input>
inline void clear_ready_events(Input& input) {
  input.runtime.metadata_ready_event.reset();
}

}  // namespace xllm
