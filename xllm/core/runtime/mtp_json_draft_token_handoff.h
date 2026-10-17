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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/metrics.h"
#include "core/framework/speculative/mtp_json_object_state.h"
#include "core/platform/platform.h"
#include "core/platform/stream.h"
#include "util/timer.h"

namespace xllm {
namespace detail {

struct DraftTokenHandoffMetrics final {
  bvar::LatencyRecorder* copy_submission = nullptr;
  bvar::LatencyRecorder* ready_wait = nullptr;
  bvar::LatencyRecorder* bulk_read = nullptr;
  bvar::LatencyRecorder* total_handoff = nullptr;
};

class NpuJsonDraftTokenHandoff final {
 public:
  NpuJsonDraftTokenHandoff(const int64_t max_sequences_per_batch,
                           const int32_t num_speculative_tokens)
      : max_sequences_per_batch_(max_sequences_per_batch) {
    const int32_t metric_count = std::max(num_speculative_tokens, 0);
    metrics_.reserve(metric_count);
    for (int32_t draft_index = 0; draft_index < metric_count; ++draft_index) {
      const std::string metric_key = std::to_string(draft_index);
      DraftTokenHandoffMetrics metrics;
      metrics.copy_submission =
          MULTI_HISTOGRAM_speculative_draft_token_copy_submission_latency_microseconds
              .get_stats({metric_key});
      metrics.ready_wait =
          MULTI_HISTOGRAM_speculative_draft_token_ready_wait_latency_microseconds
              .get_stats({metric_key});
      metrics.bulk_read =
          MULTI_HISTOGRAM_speculative_draft_token_bulk_read_latency_microseconds
              .get_stats({metric_key});
      metrics.total_handoff =
          MULTI_HISTOGRAM_speculative_draft_token_handoff_latency_microseconds
              .get_stats({metric_key});
      metrics_.emplace_back(metrics);
    }
  }

  ~NpuJsonDraftTokenHandoff() = default;

  std::vector<int32_t> read_tokens(const torch::Tensor& next_tokens,
                                   Stream& compute_stream,
                                   const int32_t draft_index) {
    Timer total_timer;
    const DraftTokenHandoffMetrics& metrics = get_metrics(draft_index);
    const int64_t token_count = next_tokens.numel();
    std::vector<int32_t> token_ids;

    if (can_use_async_handoff(next_tokens, token_count)) {
      c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
      ensure_resources();
      Timer copy_submission_timer;
      torch::Tensor host_tokens =
          pinned_tokens_.narrow(/*dim=*/0, /*start=*/0, token_count);
      host_tokens.copy_(next_tokens.flatten(), /*non_blocking=*/true);
      observe_metric(metrics.copy_submission,
                     copy_submission_timer.elapsed_microseconds());

      // The caller needs the tokens now, so the host blocks on the recorded
      // copy directly instead of routing through a dedicated wait stream.
      compute_stream.record_event(*ready_event_);
      Timer ready_wait_timer;
      const bool ready = ready_event_->synchronize();
      observe_metric(metrics.ready_wait,
                     ready_wait_timer.elapsed_microseconds());
      CHECK(ready) << "Failed to wait for JSON draft token host copy";
      Timer bulk_read_timer;
      token_ids = copy_json_draft_token_ids(pinned_tokens_.data_ptr<int64_t>(),
                                            static_cast<size_t>(token_count));
      observe_metric(metrics.bulk_read, bulk_read_timer.elapsed_microseconds());
    } else {
      COUNTER_INC(speculative_draft_token_handoff_fallback_total);
      token_ids = read_json_draft_tokens_host(next_tokens);
    }

    const int64_t total_microseconds =
        static_cast<int64_t>(total_timer.elapsed_microseconds());
    observe_metric(metrics.total_handoff, total_microseconds);
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
    if (!pinned_tokens_.defined()) {
      pinned_tokens_ = torch::empty({max_sequences_per_batch_},
                                    pinned_cpu_options(torch::kLong));
    }

    if (ready_event_ == nullptr) {
      ready_event_ = std::make_unique<StreamEvent>(Platform::type_torch());
    }
  }

  const DraftTokenHandoffMetrics& get_metrics(const int32_t draft_index) const {
    CHECK_GE(draft_index, 0) << "JSON draft index is negative";
    CHECK_LT(draft_index, static_cast<int32_t>(metrics_.size()))
        << "JSON draft index exceeds configured speculative width";
    return metrics_[draft_index];
  }

  void observe_metric(bvar::LatencyRecorder* recorder,
                      const double elapsed_microseconds) const {
    if (recorder != nullptr) {
      *recorder << static_cast<int64_t>(elapsed_microseconds);
    }
  }

  int64_t max_sequences_per_batch_ = 0;
  torch::Tensor pinned_tokens_;
  std::unique_ptr<StreamEvent> ready_event_;
  std::vector<DraftTokenHandoffMetrics> metrics_;
};

}  // namespace detail

}  // namespace xllm
#endif
