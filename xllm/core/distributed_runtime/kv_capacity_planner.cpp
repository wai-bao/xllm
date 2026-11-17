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

#include "core/distributed_runtime/kv_capacity_planner.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "core/common/metrics.h"
#include "core/distributed_runtime/distributed_vmm_coordinator.h"
#include "core/distributed_runtime/llm_engine.h"
#include "core/distributed_runtime/vlm_engine.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/allocator/vmm_arena_capacity.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/kv_cache/layout/vmm_kv_cache_layout.h"
#include "core/runtime/options.h"
#include "core/runtime/worker_client.h"
#include "models/model_registry.h"

namespace xllm {

template <typename Engine>
KVCapacityPlanner::KVCapacityPlanner(const Engine& engine)
    : model_args_(engine.model_args()),
      options_(engine.options()),
      worker_clients_(engine.worker_clients_),
      dtype_(engine.dtype_),
      world_size_(engine.dp_local_tp_size_),
      is_multimodal_(std::is_same_v<Engine, VLMEngine>),
      reserved_memory_bytes_(0) {
  if constexpr (std::is_same_v<Engine, LLMEngine>) {
    reserved_memory_bytes_ = engine.initial_weight_reservation_bytes_;
  }
}

template KVCapacityPlanner::KVCapacityPlanner(const LLMEngine&);
template KVCapacityPlanner::KVCapacityPlanner(const VLMEngine&);

KVCacheEstimateContext KVCapacityPlanner::observe_memory() const {
  KVCacheEstimateContext context;
  context.dtype = dtype_;
  context.world_size = world_size_;
  context.is_multimodal = is_multimodal_;
  context.reserved_memory_bytes = reserved_memory_bytes_;
  context.linear_state_cache_block_limit =
      get_npu_linear_state_cache_block_limit(model_args_.model_type());
  const KVCacheConfig& config = KVCacheConfig::get_instance();
  const auto& phy_pool = PhysicalPagePool::get_instance();
  if (config.enable_vmm_kvcache() && phy_pool.is_initialized()) {
    context.virtual_memory_cache_size =
        static_cast<int64_t>(phy_pool.num_total()) *
        config.phy_page_granularity_size();
  } else {
    CHECK(!worker_clients_.empty()) << "KV cache estimation requires workers";
    std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
    futures.reserve(worker_clients_.size());
    for (const auto& worker : worker_clients_) {
      futures.emplace_back(worker->estimate_kv_cache_capacity_async());
    }
    auto results = folly::collectAll(futures).get();
    context.worker_memory.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
      CHECK(results[i].hasValue())
          << "Failed to estimate KV cache capacity for worker: " << i;
      const auto& [available_memory, total_memory] = results[i].value();
      context.worker_memory.emplace_back(
          KVCacheMemorySnapshot{available_memory, total_memory});
    }
    if (config.enable_vmm_kvcache()) {
      const int64_t encoder_bytes =
          is_multimodal_ ? options_.max_encoder_cache_size() * 1024 * 1024 : 0;
      int64_t initial_bytes = std::numeric_limits<int64_t>::max();
      for (const KVCacheMemorySnapshot& memory : context.worker_memory) {
        const VmmArenaCapacity arena = calculate_vmm_arena_capacity(
            memory.available_memory,
            memory.total_memory,
            config.max_memory_utilization(),
            config.vmm_elastic(),
            config.max_cache_size(),
            reserved_memory_bytes_ + encoder_bytes);
        initial_bytes = std::min(initial_bytes, arena.initial_bytes);
      }
      // Persistent states must fit startup memory, rather than the larger
      // virtual arena that an elastic pool can grow into later.
      context.virtual_memory_cache_size = initial_bytes;
    }
  }
  return context;
}

void KVCapacityPlanner::prepare_virtual_memory(
    KVCacheCapacity& capacity,
    const VmmKVCacheLayout& layout,
    int64_t reserved_state_bytes) const {
  auto& coordinator = DistributedVmmCoordinator::get_instance();
  coordinator.prepare_kv_cache_pool(options_,
                                    world_size_,
                                    worker_clients_.size(),
                                    reserved_state_bytes,
                                    is_multimodal_);
  const size_t total_pages = coordinator.get_num_total_phy_pages();
  const size_t page_size =
      KVCacheConfig::get_instance().phy_page_granularity_size();
  capacity
      .n_blocks(
          static_cast<int64_t>(layout.block_capacity(total_pages, page_size)))
      .cache_size_in_bytes(static_cast<int64_t>(total_pages) * page_size +
                           reserved_state_bytes);
}

KVCacheCapacity KVCapacityPlanner::plan(bool initialize_virtual_memory) const {
  const KVCacheEstimateContext context = observe_memory();
  KVCacheCapacity capacity =
      KVCacheEstimator(model_args_).estimate(options_, context);
  if (KVCacheConfig::get_instance().enable_vmm_kvcache() &&
      initialize_virtual_memory) {
    const VmmKVCacheLayout layout(
        capacity, model_args_, world_size_, dtype_, options_.is_draft_engine());
    prepare_virtual_memory(
        capacity, layout, capacity.linear_cache_size_in_bytes());
  }
  GAUGE_SET(total_kv_cache_size_in_kilobytes,
            capacity.cache_size_in_bytes() / 1024);
  return capacity;
}

KVCacheAllocationPlan KVCapacityPlanner::estimate_kv_cache_capacity(
    const std::optional<KVCapacityPlanner>& draft) const {
  const KVCacheConfig& config = KVCacheConfig::get_instance();
  const bool shared_vmm = config.enable_vmm_kvcache() && draft.has_value();
  KVCacheCapacity target_capacity = plan(!shared_vmm);
  if (!draft) {
    return {{std::move(target_capacity), {}}, std::nullopt};
  }
  KVCacheCapacity draft_capacity = draft->plan(!shared_vmm);
  std::optional<VmmKVCacheLayout> layout;
  std::optional<int64_t> shared_vmm_blocks;
  if (shared_vmm && target_capacity.shared_slot_bytes() == 0) {
    layout.emplace(target_capacity, model_args_, world_size_, dtype_);
    const int64_t draft_world_size =
        options_.enable_mtp_draft_body_tp1() ? 1 : draft->world_size_;
    layout->append(VmmKVCacheLayout(draft_capacity,
                                    draft->model_args_,
                                    draft_world_size,
                                    draft->dtype_,
                                    /*is_draft_engine=*/true));
    const int64_t reserved_state_bytes =
        target_capacity.linear_cache_size_in_bytes() +
        draft_capacity.linear_cache_size_in_bytes();
    prepare_virtual_memory(target_capacity, *layout, reserved_state_bytes);
    shared_vmm_blocks = target_capacity.n_blocks();
    draft_capacity.cache_size_in_bytes(target_capacity.cache_size_in_bytes());
  }
  const int32_t layerwise_split_size =
      options_.is_draft_engine()
          ? 1
          : ParallelConfig::get_instance().layerwise_split_size();
  return {plan_shared_kv_cache(std::move(target_capacity),
                               std::move(draft_capacity),
                               model_args_,
                               draft->model_args_,
                               world_size_,
                               dtype_,
                               layerwise_split_size,
                               shared_vmm_blocks),
          std::move(layout)};
}

template <typename TargetEngine>
bool allocate_kv_cache(
    TargetEngine& target,
    std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator,
    LLMEngine* draft) {
  const auto plan = KVCapacityPlanner(target).estimate_kv_cache_capacity(
      draft ? std::make_optional<KVCapacityPlanner>(*draft) : std::nullopt);
  const auto* layout = plan.vmm_layout ? &*plan.vmm_layout : nullptr;
  return target.allocate_kv_cache(
             plan.capacities.target, transfer_coordinator, layout) &&
         (draft == nullptr ||
          draft->allocate_kv_cache(plan.capacities.draft,
                                   std::move(transfer_coordinator)));
}

template bool allocate_kv_cache<LLMEngine>(
    LLMEngine&,
    std::shared_ptr<KVCacheTransferCoordinatorBase>,
    LLMEngine*);
template bool allocate_kv_cache<VLMEngine>(
    VLMEngine&,
    std::shared_ptr<KVCacheTransferCoordinatorBase>,
    LLMEngine*);

}  // namespace xllm
