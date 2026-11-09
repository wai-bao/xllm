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
#include <span>
#include <string>
#include <vector>

namespace xllm {

// Non-owning row-identity view (embedding id + owning request id per row)
// shared by both speculative state stores; built at the batch seam.
struct BatchRows {
  std::span<const int32_t> embedding_ids;
  std::span<const std::string> request_ids;

  size_t size() const { return embedding_ids.size(); }
};

inline void check_batch_rows(const BatchRows& rows) {
  CHECK_EQ(rows.embedding_ids.size(), rows.request_ids.size())
      << "embedding ids and request ids must be paired per row";
}

namespace detail {

// Bounds-checked access into an embedding-id-indexed slot table.
//
// Both speculative stores are an std::vector indexed by embedding_id, and both
// must reject the same two things before touching a slot: a negative id, and
// an id past the table. Sharing the check keeps that one invariant in one
// place. Ownership is deliberately NOT checked here: the stores disagree on
// what makes a slot belong to a caller, so each store keeps its own ownership
// predicate visible rather than hiding it behind this helper.
template <typename State>
inline State& checked_slot(std::vector<State>& states, int32_t id) {
  CHECK_GE(id, 0) << "embedding id must be non-negative";
  CHECK_LT(static_cast<size_t>(id), states.size())
      << "embedding id " << id << " exceeds the slot table of " << states.size()
      << " rows";
  return states[static_cast<size_t>(id)];
}

template <typename State>
inline const State& checked_slot(const std::vector<State>& states, int32_t id) {
  return checked_slot(*const_cast<std::vector<State>*>(&states), id);
}

}  // namespace detail

}  // namespace xllm
