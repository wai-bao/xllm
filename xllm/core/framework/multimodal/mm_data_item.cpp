/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/util/tensor_helper.h"
#include "mm_data.h"

namespace xllm {

MMDataItem::MMDataItem(MMType type) : type_(type) {}

MMDataItem::MMDataItem(MMType type, MMDict data)
    : type_(type), data_(std::move(data)) {}

MMDataItem::MMDataItem(MMType type, MMDict data, MMMetadata metadata)
    : type_(type), data_(std::move(data)), metadata_(std::move(metadata)) {}

bool MMDataItem::has(const MMKey& key) const {
  if (!valid()) return false;

  const auto& itor = data_.find(key);
  return itor != data_.end();
}

void MMDataItem::get(const MMKey& key, std::vector<torch::Tensor>& vec) const {
  if (!valid()) return;

  const auto& itor = data_.find(key);
  if (itor == data_.end()) return;

  if (std::holds_alternative<torch::Tensor>(itor->second)) {
    vec.push_back(std::get<torch::Tensor>(itor->second));
  } else if (std::holds_alternative<std::vector<torch::Tensor>>(itor->second)) {
    const auto& data = std::get<std::vector<torch::Tensor>>(itor->second);
    vec.insert(vec.end(), data.begin(), data.end());
  }
}

}  // namespace xllm
