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

#include "core/distributed_runtime/request_state_cleanup.h"

#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <vector>

#include "core/framework/block/kv_cache_manager.h"
#include "core/runtime/worker_client.h"

namespace xllm {

void flush_released_request_state(
    KVCacheManager& manager,
    const std::vector<std::shared_ptr<WorkerClient>>& workers,
    int32_t workers_per_dp) {
  if (workers.empty()) {
    return;
  }
  CHECK_GT(workers_per_dp, 0);
  CHECK_EQ(workers.size() % workers_per_dp, 0);
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(workers.size());
  const int32_t dp_size = static_cast<int32_t>(workers.size() / workers_per_dp);
  for (int32_t dp_rank = 0; dp_rank < dp_size; ++dp_rank) {
    const auto ids = manager.take_released_embedding_ids(dp_rank);
    if (ids.empty()) {
      continue;
    }
    const int32_t begin = dp_rank * workers_per_dp;
    for (int32_t rank = begin; rank < begin + workers_per_dp; ++rank) {
      futures.emplace_back(
          workers[rank]->release_idle_request_state_async(ids));
    }
  }
  auto results = folly::collectAll(futures).get();
  for (auto& result : results) {
    CHECK(result.hasValue() && result.value())
        << "Failed to release idle worker request state";
  }
}
}  // namespace xllm
