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

#include "speculative_engine.h"

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>

#include "core/common/types.h"
#include "core/distributed_runtime/kv_capacity_planner.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/eplb/eplb_controller.h"
#include "llm_engine.h"
#include "vlm_engine.h"

namespace xllm {

template <typename TargetEngine>
SpeculativeEngineBase<TargetEngine>::~SpeculativeEngineBase() = default;

template <typename TargetEngine>
SpeculativeEngineBase<TargetEngine>::SpeculativeEngineBase(
    const runtime::Options& options)
    : options_(options) {
  CHECK_GT(options.num_speculative_tokens(), 0)
      << "speculative tokens should not be zero";

  runtime::Options dist_options = options;
  dist_options.enable_speculative_decode(true);
  distributed_worker_manager_ =
      std::make_shared<DistributedWorkerManager>(dist_options);

  runtime::Options target_engine_options = options_;
  engine_ = std::make_unique<TargetEngine>(target_engine_options,
                                           distributed_worker_manager_);

  if (!SpeculativeConfig::is_suffix_algorithm(
          options.speculative_algorithm())) {
    // draft engine
    // In vlm mtp case, we'd like to use llm backend instead of vlm backend for
    // draft engine.
    runtime::Options draft_engine_options = options_;
    // The draft engine always runs on the target's device list.
    draft_engine_options.model_path(options_.draft_model_path().value_or(""))
        .devices(options.devices())
        .backend("llm")
        .enable_speculative_decode(/*enable_speculative_decode=*/false)
        .enable_graph(/*enable_graph=*/false)
        .is_draft_engine(true);
    if (KVCacheConfig::get_instance().enable_vmm_kvcache()) {
      draft_engine_options.model_id(options_.model_id() + "::mtp_draft");
    }
    draft_engine_ = std::make_unique<LLMEngine>(draft_engine_options,
                                                distributed_worker_manager_);
  }
}

template <typename TargetEngine>
bool SpeculativeEngineBase<TargetEngine>::init(MasterStatus master_status) {
  return init(master_status, /*prepare_model=*/{});
}

template <typename TargetEngine>
bool SpeculativeEngineBase<TargetEngine>::init(
    MasterStatus master_status,
    const LLMEngine::ModelInitCallback& prepare_model,
    std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator) {
  if (!init_model(master_status, prepare_model)) {
    return false;
  }

  return ::xllm::allocate_kv_cache(
      *engine_, std::move(transfer_coordinator), draft_engine_.get());
}

template <typename TargetEngine>
bool SpeculativeEngineBase<TargetEngine>::init_model(
    MasterStatus master_status,
    const LLMEngine::ModelInitCallback& prepare_model) {
  if constexpr (std::is_same_v<TargetEngine, LLMEngine>) {
    if (!engine_->init_model(master_status, prepare_model)) {
      return false;
    }
  } else if (!engine_->init_model(master_status)) {
    return false;
  }

  model_args_ = engine_->model_args();

  if (LoadConfig::get_instance().enable_vmm_weights() &&
      draft_engine_ != nullptr) {
    LOG(ERROR) << "Virtual weight memory does not support an external "
                  "speculative draft engine.";
    return false;
  }
  if (KVCacheConfig::get_instance().enable_vmm_kvcache() &&
      draft_engine_ != nullptr &&
      (!SpeculativeConfig::is_mtp_algorithm(options_.speculative_algorithm()) ||
       options_.enable_disagg_pd() ||
       options_.instance_role() != InstanceRole::DEFAULT)) {
    LOG(ERROR) << "Virtual KV cache supports co-located MTP target and draft "
                  "engines only; PD is unsupported.";
    return false;
  }

  if (draft_engine_ != nullptr) {
    if (!draft_engine_->init_model(MasterStatus::WAKEUP, prepare_model)) {
      return false;
    }

    // Draft tokens are detokenized by the target, so the draft engine needs no
    // tokenizer and no draft/target vocab-compatibility check (not universal;
    // see llm_engine.cpp).

    // check if the max context length are the same
    const auto& draft_model_args = draft_engine_->model_args();
    if (model_args_.max_position_embeddings() !=
        draft_model_args.max_position_embeddings()) {
      LOG(WARNING) << "draft and target models have different max context "
                      "lengths, draft max_position_embeddings: "
                   << draft_model_args.max_position_embeddings()
                   << ", target max_position_embeddings: "
                   << model_args_.max_position_embeddings()
                   << ", using the minimum between them";
      model_args_.max_position_embeddings() =
          std::min(model_args_.max_position_embeddings(),
                   draft_model_args.max_position_embeddings());
    }
  }

  if constexpr (std::is_same_v<TargetEngine, LLMEngine>) {
    engine_->eplb_controller_ = EplbController::create(
        engine_->model_args(),
        static_cast<int32_t>(engine_->worker_clients_num_),
        engine_->options_.ep_size(),
        engine_->options_.eplb_controller_config());
    engine_->forward_input_factory_->set_eplb_controller(
        engine_->eplb_controller_.get());
  }
  return true;
}

// TODO: support dp batches later
template <typename TargetEngine>
ForwardOutput SpeculativeEngineBase<TargetEngine>::step(BatchGroup& batches) {
  return engine_->step(batches);
}

template <typename TargetEngine>
std::vector<RawForwardOutput>
SpeculativeEngineBase<TargetEngine>::collect_step_outputs(
    const BatchGroup& batch) {
  return engine_->collect_step_outputs(batch);
}

template class SpeculativeEngineBase<LLMEngine>;
template class SpeculativeEngineBase<VLMEngine>;

}  // namespace xllm
