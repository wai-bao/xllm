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

#include "runtime/options.h"

namespace xllm::spec_detail {

// Draft leaves are always plain-text LLMs even under a VLM target.
inline void mark_spec_draft_options(runtime::Options& opts) {
  opts.enable_schedule_overlap(false)
      .is_draft_model(true)
      .backend("llm")
      .enable_graph_aux_hidden_states(false);
}

}  // namespace xllm::spec_detail
