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

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/model/mtp_topk_state.h"

namespace xllm {

// Retains top-k state for one produced batch. Reuse requires the same slot and
// request identity in every row; releasing any slot invalidates the snapshot.
class MtpSeedTopkCache final {
 public:
  void publish(std::vector<int32_t> embedding_ids,
               std::vector<std::string> request_ids,
               MtpTopkStatePtr state) {
    embedding_ids_ = std::move(embedding_ids);
    request_ids_ = std::move(request_ids);
    state_ = std::move(state);
  }

  MtpTopkStatePtr lookup(const std::vector<int32_t>& embedding_ids,
                         const std::vector<std::string>& request_ids) const {
    return embedding_ids_ == embedding_ids && request_ids_ == request_ids
               ? state_
               : nullptr;
  }

  void clear(const std::vector<int32_t>& released) {
    if (embedding_ids_.empty() || released.empty()) {
      return;
    }
    const std::unordered_set<int32_t> released_ids(released.begin(),
                                                   released.end());
    if (std::any_of(embedding_ids_.begin(),
                    embedding_ids_.end(),
                    [&released_ids](int32_t id) {
                      return released_ids.count(id) != 0;
                    })) {
      embedding_ids_.clear();
      request_ids_.clear();
      state_.reset();
    }
  }

 private:
  std::vector<int32_t> embedding_ids_;
  std::vector<std::string> request_ids_;
  MtpTopkStatePtr state_;
};

}  // namespace xllm
