/* Copyright 2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/batch/batch_state.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <utility>

#include "core/framework/batch/forward_input_builder.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/model/model_args.h"

namespace xllm {

void BatchState::add(Sequence* sequence, uint32_t allowed_max_token) {
  CHECK(sequence != nullptr);
  CHECK(!sequence->finished());
  CHECK_GT(allowed_max_token, 0);

  set_batch_id();
  sequence_plan_.add(sequence, allowed_max_token);
  retain_host_restore_blocks(*sequence);

  if (sequence->has_input_embedding()) {
    input_embeddings_vec_.emplace_back(sequence->get_input_embedding());
  }

  update_forward_type(sequence);
}

void BatchState::retain_host_restore_blocks(Sequence& sequence) {
  auto retained = sequence.take_host_restore_blocks();
  if (retained != nullptr) {
    host_restore_leases_.reserve(host_restore_leases_.size() + 1);
    host_restore_leases_.emplace_back(std::move(retained));
  }
}

void BatchState::update_forward_type(Sequence* sequence) {
  if (batch_forward_type_.value() == BatchForwardType::MIXED) {
    return;
  }
  const SequenceStage stage = sequence->stage();
  switch (batch_forward_type_.value()) {
    case BatchForwardType::PREFILL:
      if (stage == SequenceStage::CHUNKED_PREFILL) {
        batch_forward_type_ = BatchForwardType::CHUNKED_PREFILL;
      } else if (stage == SequenceStage::DECODE) {
        batch_forward_type_ = BatchForwardType::MIXED;
      }
      break;
    case BatchForwardType::CHUNKED_PREFILL:
      if (stage == SequenceStage::DECODE) {
        batch_forward_type_ = BatchForwardType::MIXED;
      }
      break;
    case BatchForwardType::DECODE:
      if (stage != SequenceStage::DECODE) {
        batch_forward_type_ = BatchForwardType::MIXED;
      }
      break;
    case BatchForwardType::MIXED:
      break;
    case BatchForwardType::EMPTY:
      batch_forward_type_ = BatchForwardType(static_cast<int32_t>(stage));
      break;
  }
}

void BatchState::reserve(size_t sequence_count, size_t input_embedding_count) {
  sequence_plan_.reserve(sequence_count);
  host_restore_leases_.reserve(sequence_count);
  input_embeddings_vec_.reserve(input_embedding_count);
}

BatchInputData BatchState::prepare_batch_input() {
  if (KVCacheConfig::get_instance().enable_vmm_kvcache()) {
    std::vector<Block> retained_blocks;
    size_t block_count = 0;
    for (auto* sequence : sequence_plan_.sequences()) {
      block_count += sequence->kv_state().num_blocks(BlockType::KV);
    }
    retained_blocks.reserve(block_count);
    for (auto* sequence : sequence_plan_.sequences()) {
      const auto blocks = sequence->kv_state().blocks(BlockType::KV);
      retained_blocks.insert(
          retained_blocks.end(), blocks.begin(), blocks.end());
    }
    vmm_kv_blocks_ = std::move(retained_blocks);
  }
  return {sequence_plan_.sequences(),
          sequence_plan_.budgets(),
          input_embeddings_vec_,
          mm_data_vec_,
          batch_id_,
          batch_forward_type_};
}

template <typename Input>
Input BatchState::build_forward_input(const BatchInputData& data,
                                      const ModelArgs& args,
                                      ThreadPool* thread_pool,
                                      int32_t cp_size,
                                      SharedKVMappingTracker* mapping_tracker) {
  ForwardInputBuilder builder(
      data, &args, cp_size, thread_pool, mapping_tracker);
  auto input = builder.build_forward_input<Input>();

  input.input_params.meta.requires_host_restore =
      std::exchange(requires_host_restore_, false);
  return input;
}

template LlmForwardInput BatchState::build_forward_input<LlmForwardInput>(
    const BatchInputData&,
    const ModelArgs&,
    ThreadPool*,
    int32_t,
    SharedKVMappingTracker*);
template VlmForwardInput BatchState::build_forward_input<VlmForwardInput>(
    const BatchInputData&,
    const ModelArgs&,
    ThreadPool*,
    int32_t,
    SharedKVMappingTracker*);

}  // namespace xllm
