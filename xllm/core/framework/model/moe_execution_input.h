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

#include <type_traits>

#include "core/framework/model/model_input_types.h"

namespace xllm {

class ModelParamsView;

// Borrows only the batch and parallel state consumed by expert execution.
class MoeExecutionInput final {
 public:
  template <typename InputParams>
  explicit(!std::is_same_v<InputParams, ModelParamsView>)
      MoeExecutionInput(const InputParams& params)
      : meta(params.meta),
        parallel(params.parallel),
        expert(params.expert),
        enable_graph(params.enable_graph) {}

  const BatchInputMeta& meta;
  const ParallelInput& parallel;
  const ExpertInput& expert;
  bool enable_graph;
};

}  // namespace xllm
