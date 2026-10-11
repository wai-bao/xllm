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

#include <memory>
#include <utility>

#include "core/distributed_runtime/engine.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/profile/profile_manager.h"

namespace xllm {

// Test-only access to the scheduler-owned ProfileManager constructor
// (friend ProfileManagerTestPeer, the codebase's test-access convention).
// The callbacks come from the same shared factory the production scheduler
// constructors use, so the wiring cannot drift between them.
class ProfileManagerTestPeer {
 public:
  template <typename EngineT>
  static std::unique_ptr<ProfileManager> make(
      EngineT* engine,
      const ProfileManager::Options& options) {
    auto callbacks = make_engine_callbacks(engine);
    return std::unique_ptr<ProfileManager>(
        new ProfileManager(engine,
                           options,
                           std::move(callbacks.step),
                           std::move(callbacks.update_result)));
  }
};

}  // namespace xllm
