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

#if defined(USE_NPU)
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/metrics.h"
#include "core/framework/speculative/mtp_json_object_state.h"
#include "core/platform/platform.h"
#include "core/platform/stream.h"
#include "core/util/tensor_helper.h"
#include "util/timer.h"

namespace xllm {
namespace detail {

class NpuJsonDraftTokenHandoff final {
 public:
  explicit NpuJsonDraftTokenHandoff(const int64_t max_sequences_per_batch)
      : max_sequences_per_batch_(max_sequences_per_batch) {}

  ~NpuJsonDraftTokenHandoff() = default;

  std::vector<int32_t> read_tokens(const torch::Tensor& next_tokens,
                                   Stream& compute_stream,
                                   const int32_t draft_index) {
    Timer total_timer;
    const std::string metric_key = std::to_string(draft_index);
    const int64_t token_count = next_tokens.numel();
    std::vector<int32_t> token_ids;

    if (can_use_async_handoff(next_tokens, token_count)) {
      c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
      ensure_resources();
      Timer copy_submission_timer;
      torch::Tensor host_tokens =
          pinned_tokens_.narrow(/*dim=*/0, /*start=*/0, token_count);
      host_tokens.copy_(next_tokens.flatten(), /*non_blocking=*/true);
      MULTI_HISTOGRAM_OBSERVE(
          speculative_draft_token_copy_submission_latency_microseconds,
          metric_key,
          copy_submission_timer.elapsed_microseconds());

      // The caller needs the tokens now, so the host blocks on the recorded
      // copy directly instead of routing through a dedicated wait stream.
      compute_stream.record_event(*ready_event_);
      Timer ready_wait_timer;
      const bool ready = ready_event_->synchronize();
      MULTI_HISTOGRAM_OBSERVE(
          speculative_draft_token_ready_wait_latency_microseconds,
          metric_key,
          ready_wait_timer.elapsed_microseconds());
      CHECK(ready) << "Failed to wait for JSON draft token host copy";
      Timer bulk_read_timer;
      token_ids = copy_json_draft_token_ids(pinned_tokens_.data_ptr<int64_t>(),
                                            static_cast<size_t>(token_count));
      MULTI_HISTOGRAM_OBSERVE(
          speculative_draft_token_bulk_read_latency_microseconds,
          metric_key,
          bulk_read_timer.elapsed_microseconds());
    } else {
      COUNTER_INC(speculative_draft_token_handoff_fallback_total);
      token_ids = read_json_draft_tokens_host(next_tokens);
    }

    const int64_t total_microseconds =
        static_cast<int64_t>(total_timer.elapsed_microseconds());
    MULTI_HISTOGRAM_OBSERVE(
        speculative_draft_token_handoff_latency_microseconds,
        metric_key,
        total_microseconds);
    HISTOGRAM_OBSERVE(speculative_draft_token_d2h_latency_microseconds,
                      total_microseconds);
    return token_ids;
  }

 private:
  bool can_use_async_handoff(const torch::Tensor& next_tokens,
                             const int64_t token_count) const {
    return max_sequences_per_batch_ > 0 &&
           token_count <= max_sequences_per_batch_ &&
           !next_tokens.device().is_cpu() && next_tokens.is_contiguous() &&
           next_tokens.scalar_type() == torch::kLong;
  }

  void ensure_resources() {
    ensure_capacity(pinned_tokens_,
                    {max_sequences_per_batch_},
                    pinned_cpu_options(torch::kLong));

    if (ready_event_ == nullptr) {
      ready_event_ = std::make_unique<StreamEvent>(Platform::type_torch());
    }
  }

  int64_t max_sequences_per_batch_ = 0;
  torch::Tensor pinned_tokens_;
  std::unique_ptr<StreamEvent> ready_event_;
};

}  // namespace detail

}  // namespace xllm
#endif
