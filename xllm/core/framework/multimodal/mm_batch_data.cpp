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

#include "core/framework/multimodal/mm_batch_data.h"

#include <mutex>

#include "core/framework/multimodal/mm_visitor.h"
#include "core/util/tensor_helper.h"
#include "core/util/utils.h"

namespace xllm {

class MMBatchData::TensorAggregation final {
 public:
  explicit TensorAggregation(
      std::unordered_map<MMKey, std::vector<torch::Tensor>> tensors)
      : tensors_(std::move(tensors)) {}

  const MMDict& data() {
    // Batch copies can be read by independent worker dispatches.
    std::call_once(materialized_, [this] {
      for (auto& [key, tensors] : tensors_) {
        if (tensors.size() == 1 && tensors.front().defined() &&
            tensors.front().dim() > 0) {
          data_[key] = tensors.front().contiguous();
          continue;
        }
        torch::Tensor gathered;
        if (safe_concat(tensors, gathered)) {
          data_[key] = std::move(gathered);
        } else {
          data_[key] = std::move(tensors);
        }
      }
      tensors_.clear();
    });
    return data_;
  }

 private:
  std::once_flag materialized_;
  std::unordered_map<MMKey, std::vector<torch::Tensor>> tensors_;
  MMDict data_;
};

MMBatchData::MMBatchData(std::vector<MMData> data) {
  this->batch(std::move(data));
}

MMBatchData::MMBatchData(uint32_t type, MMDict items)
    : type_(type), data_(std::move(items)) {}

MultimodalEmbeddings MMBatchData::merge_visual_embeddings(
    torch::Tensor inputs_embeds,
    size_t num_deep_stacks) const {
  if (!valid()) {
    return {std::move(inputs_embeds), {}};
  }

  std::vector<torch::Tensor> deep_stacks;
  deep_stacks.reserve(num_deep_stacks);
  for (size_t layer = 0; layer < num_deep_stacks; ++layer) {
    deep_stacks.emplace_back(torch::zeros_like(inputs_embeds));
  }
  const auto merge = [&](const MMKey& embedding_key, const MMKey& mask_key) {
    const auto embedding = get<torch::Tensor>(embedding_key);
    if (!embedding) {
      return;
    }
    const auto mask = get<torch::Tensor>(mask_key);
    if (!mask) {
      return;
    }
    if (num_deep_stacks == 0) {
      inputs_embeds.index_put_({*mask}, *embedding);
      return;
    }
    const std::vector<int64_t> embedding_split_sizes(
        num_deep_stacks + 1, inputs_embeds.size(/*dim=*/-1));
    const auto slices =
        embedding->split_with_sizes(embedding_split_sizes, /*dim=*/-1);
    inputs_embeds.index_put_({*mask}, slices.front());
    for (size_t layer = 0; layer < num_deep_stacks; ++layer) {
      deep_stacks[layer].index_put_({*mask}, slices[layer + 1]);
    }
  };
  merge(/*embedding_key=*/"image|embedding", /*mask_key=*/"image|mask");
  merge(/*embedding_key=*/"video|embedding", /*mask_key=*/"video|mask");
  return {std::move(inputs_embeds), std::move(deep_stacks)};
}

namespace {
void move_dict_to_device(MMDict& dict, const torch::Device& device) {
  for (auto& [key, value] : dict) {
    if (std::holds_alternative<torch::Tensor>(value)) {
      value = safe_to(std::get<torch::Tensor>(value), device, true);
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(value)) {
      for (auto& tensor : std::get<std::vector<torch::Tensor>>(value)) {
        tensor = safe_to(tensor, device, true);
      }
    }
  }
}
}  // namespace

void MMBatchData::to(const torch::Device& device) {
  if (!aggregation_) {
    move_dict_to_device(data_, device);
  }

  // Item-backed dictionaries are derived again after moving their sources.
  for (auto& mm_data : mm_datas_) {
    if (mm_data.hold<MMItemVec>()) {
      for (auto& item : mm_data.items<MMItemVec>()) {
        move_dict_to_device(item.mutable_data(), device);
      }
    } else if (mm_data.hold<MMDict>()) {
      move_dict_to_device(mm_data.items<MMDict>(), device);
    }
  }
  if (aggregation_) {
    batch(std::move(mm_datas_));
  }
}

MMBatchData MMBatchData::copy_to(const torch::Device& device) const {
  MMBatchData new_mm_data = *this;
  new_mm_data.to(device);
  return new_mm_data;
}

void MMBatchData::batch(std::vector<MMData> mm_datas) {
  mm_datas_ = std::move(mm_datas);
  CollectMMDataTensorVisitor visitor;
  this->foreach (static_cast<MMData::IVisitor&>(visitor));

  type_ = visitor.type_;
  data_.clear();
  aggregation_ = std::make_shared<TensorAggregation>(std::move(visitor.datas_));
}

const MMDict& MMBatchData::data() const {
  return aggregation_ ? aggregation_->data() : data_;
}

void MMBatchData::replace(MMDict data) {
  aggregation_.reset();
  data_ = std::move(data);
}

}  // namespace xllm
