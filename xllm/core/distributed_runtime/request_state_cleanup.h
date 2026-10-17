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

#include <cstdint>
#include <memory>
#include <vector>

namespace xllm {
class KVCacheManager;
class WorkerClient;

// A just-released embedding id can be reused by a new request before the
// release notification reaches its consumer (the pool's free list is LIFO).
// BlockManagerPool::take_released_embedding_ids is the single filtering seam:
// it consults the embedding leaf's is_block_used state, so every consumer --
// the per-step mounting in LLMEngine::step, the worker-path readers, and the
// idle-RPC flush below -- only ever sees genuinely departed slots.
//
// Two delivery channels coexist deliberately. The piggyback path rides the
// next forward input because WorkerImpl::release_idle_request_state_async
// CHECKs !is_recorded_ (overlap output must be consumed first), which the
// engine cannot guarantee mid-step; the idle RPC below covers engine-idle
// flushes where that constraint is satisfiable. Collapsing them is a
// behavior change, not a refactor.
//
// Domain boundary: the per-step piggyback channel is LLM-only
// (WorkerRequestState exists on LlmForwardInput and its SHM serializer).
// VLM speculative deployments therefore reclaim per-slot state only through
// the idle flush below; extending the channel to VLM is a feature, not a
// refactor.
//
// Idle-only control dispatch. Each DP replica receives its own released slots
// on every TP/CP worker. No RPC is sent when there is nothing to reclaim.
void flush_released_request_state(
    KVCacheManager& manager,
    const std::vector<std::shared_ptr<WorkerClient>>& workers,
    int32_t workers_per_dp);
}  // namespace xllm
