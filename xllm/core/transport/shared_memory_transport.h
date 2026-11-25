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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/runtime/forward_params.h"
#include "core/transport/packed_forward_codec.h"
#include "core/util/shared_memory_manager.h"

namespace xllm {
class VlmForwardInput;

class Stream;

constexpr int64_t kNumWaitNanoseconds = 1000;  // 1us

struct ControlMetadata {
  volatile uint64_t version;
};

enum class ForwardType : int8_t {
  RAW_INPUT = 3,
  RAW_OUTPUT = 4,
};

// Owns a shared segment and transports forward messages in its wire format.
class SharedMemoryTransport final {
 public:
  explicit SharedMemoryTransport(const std::string& name,
                                 size_t size,
                                 bool& is_creator);
  ~SharedMemoryTransport();

  void* base_address() const { return segment_.base_address(); }
  int64_t size() const { return segment_.size(); }
  static std::string create_unique_name(const std::string& prefix,
                                        int dp_group,
                                        ForwardType forward_type,
                                        int rank);

  bool input_write(const DiTForwardInput& input);
  bool input_read(DiTForwardInput& input);

  // Check transport support and capacity without copying tensor payloads.
  // Every rank in a DP group must select the same transport before execution.
  static bool can_write_input(const DiTForwardInput& input,
                              size_t segment_size);
  static bool can_write_input(const VlmForwardInput& input,
                              size_t segment_size);
  static bool can_write_input(const LlmForwardInput& input,
                              size_t segment_size);

  bool input_write(const VlmForwardInput& input);
  bool input_read(VlmForwardInput& input,
                  const torch::Device& device,
                  InputDeviceMaterializationPolicy policy =
                      InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ);

  bool input_write(const LlmForwardInput& input);
  bool input_read(LlmForwardInput& input,
                  const torch::Device& device,
                  InputDeviceMaterializationPolicy policy =
                      InputDeviceMaterializationPolicy::MATERIALIZE_ON_READ);

  bool llm_output_write(
      const ForwardOutput& output,
      const std::vector<SpeculativeTokenStats>& speculative_token_stats);
  bool vlm_output_write(
      const ForwardOutput& output,
      const std::vector<SpeculativeTokenStats>& speculative_token_stats);
  bool dit_output_write(const DiTForwardOutput& output);

  // Read into owned storage and replace the destination only on success.
  bool llm_output_read(RawForwardOutput& output);
  bool vlm_output_read(RawForwardOutput& output);
  bool dit_output_read(RawForwardOutput& output);

  // Permanently cancel input reads on this transport without publishing
  // a synthetic request or changing the shared segment for other readers.
  void stop_input_read();

  void clear();

 private:
  bool wait_for_input();
  void wait_for_output();
  void publish();
  PackedForwardCodec codec() const;

  SharedMemoryManager segment_;
  std::atomic<bool> input_read_stopped_{false};
  uint64_t last_version_ = 0;
  ControlMetadata* control_ptr_ = nullptr;
  std::unique_ptr<Stream> stream_ = nullptr;
};
}  // namespace xllm
