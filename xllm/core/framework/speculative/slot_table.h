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

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xllm::detail {

// Bounds-checked access into an embedding-id-indexed slot table.
//
// Both speculative stores are an std::vector indexed by embedding_id, and both
// must reject the same two things before touching a slot: a negative id, and an
// id past the table. Sharing the check keeps that one invariant in one place.
//
// Ownership is deliberately NOT checked here: the stores disagree on what makes
// a slot belong to a caller (TargetDecodeStateStore requires its `valid` flag,
// DraftInputStateStore uses an empty-request-id mismatch), so each store keeps
// its own ownership predicate visible rather than hiding it behind this helper.
template <typename State>
inline const State& checked_slot(const std::vector<State>& states, int32_t id) {
  CHECK_GE(id, 0) << "embedding id must be non-negative";
  CHECK_LT(static_cast<size_t>(id), states.size())
      << "embedding id " << id << " exceeds the slot table of " << states.size()
      << " rows";
  return states[static_cast<size_t>(id)];
}

// Mutable peer of checked_slot: the write paths route through it so the
// bounds CHECK and the mutation cannot be reordered apart.
template <typename State>
inline State& checked_slot(std::vector<State>& states, int32_t id) {
  return const_cast<State&>(
      checked_slot(static_cast<const std::vector<State>&>(states), id));
}

}  // namespace xllm::detail
