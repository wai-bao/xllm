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

#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <string_view>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

// The algorithm is the switch: NONE (empty user string) means speculative
// decoding is off; the token width never gates anything.
enum class SpeculativeAlgorithm : uint8_t {
  NONE,
  MTP,
  EAGLE3,
  SUFFIX,
  DFLASH,
  DFLASH2,
  DSPARK,
};

constexpr bool is_block_diffusion(SpeculativeAlgorithm algorithm) {
  return algorithm == SpeculativeAlgorithm::DFLASH ||
         algorithm == SpeculativeAlgorithm::DFLASH2 ||
         algorithm == SpeculativeAlgorithm::DSPARK;
}

// Whether the PD handoff must carry the bootstrap hidden state.
constexpr bool requires_bootstrap_hidden(SpeculativeAlgorithm algorithm) {
  return algorithm == SpeculativeAlgorithm::MTP ||
         algorithm == SpeculativeAlgorithm::EAGLE3;
}

// Width upper bound from the NPU fused-attention TND layout; mirrors the
// Python-side MAX_SPECULATIVE_TOKENS.
inline constexpr int32_t kMaxSpeculativeTokens = 16;

class SpeculativeConfig final {
 public:
  SpeculativeConfig() = default;
  ~SpeculativeConfig() = default;

  static SpeculativeConfig& get_instance();

  // Case-insensitive string -> enum parse.
  static SpeculativeAlgorithm from_string(std::string_view algorithm);

  // Lazy so PROPERTY-built snapshots (spawn path, tests) see the same truth
  // as the initialized singleton.
  SpeculativeAlgorithm algorithm() const {
    return from_string(speculative_algorithm());
  }

  // Lazy so partially-configured instances see the normalized truth.
  bool enable_speculative_decode() const {
    return !speculative_algorithm().empty();
  }

  // True when a fixed width is set (0 = unset, derived by normalize()).
  bool has_speculative_width() const { return num_speculative_tokens() > 0; }

  // Verify rows per decode step: width + bonus token.
  int32_t decode_query_len() const {
    return decode_query_len(num_speculative_tokens());
  }

  // Overload for call sites holding the raw token count.
  static constexpr int32_t decode_query_len(int32_t num_speculative_tokens) {
    return num_speculative_tokens + 1;
  }

  // Per-sequence verify-row padding, doubled under schedule overlap.
  size_t num_extra_kv_tokens(bool enable_schedule_overlap) const {
    const size_t verify_rows = decode_query_len();
    return enable_schedule_overlap ? verify_rows * 2 : verify_rows;
  }

  // Algorithm-level admission only; callers validate runtime geometry.
  static bool supports_task_pipeline(SpeculativeAlgorithm kind) {
    return kind == SpeculativeAlgorithm::MTP || is_block_diffusion(kind);
  }

  static bool supports_adaptive_target_verify(SpeculativeAlgorithm kind) {
    return kind == SpeculativeAlgorithm::DSPARK ||
           kind == SpeculativeAlgorithm::MTP;
  }

  // User flag ANDed with algorithm admission and the task-pipeline switch.
  bool effective_adaptive_target_verify(bool enable_task_pipeline) const {
    return enable_task_pipeline && enable_adaptive_target_verify() &&
           supports_adaptive_target_verify(algorithm());
  }

  static bool requires_aux_hidden_capture(SpeculativeAlgorithm kind) {
    return kind == SpeculativeAlgorithm::EAGLE3 || is_block_diffusion(kind);
  }

  // Aux-capture drafts share the target's KV layout, so they are included.
  static bool supports_host_kv_cache(SpeculativeAlgorithm kind) {
    return kind == SpeculativeAlgorithm::MTP ||
           kind == SpeculativeAlgorithm::EAGLE3 || is_block_diffusion(kind);
  }

  // DFlash2 carries the dense proposal only when probabilistic rejection
  // requires it; greedy-only DFlash and Suffix stay out.
  static bool is_probabilistic_draft_sampling_supported(
      SpeculativeAlgorithm kind) {
    return kind == SpeculativeAlgorithm::MTP ||
           kind == SpeculativeAlgorithm::DSPARK ||
           kind == SpeculativeAlgorithm::DFLASH2 ||
           kind == SpeculativeAlgorithm::EAGLE3;
  }

  std::string to_string() const;

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();
  void normalize();
  void validate() const;

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "SPECULATIVE OPTIONS",
        {"draft_model",
         "num_speculative_tokens",
         "speculative_algorithm",
         "speculative_suffix_cache_max_depth",
         "speculative_suffix_max_spec_factor",
         "speculative_suffix_max_spec_offset",
         "speculative_suffix_min_token_prob",
         "speculative_suffix_max_cached_requests",
         "speculative_suffix_use_tree_spec",
         "enable_probabilistic_draft",
         "enable_mtp_draft_body_tp1",
         "enable_atb_spec_kernel",
         "enable_adaptive_target_verify",
         "enable_adaptive_draft_step",
         "adaptive_draft_step_config",
         "adaptive_target_verify_min_gain",
         "adaptive_target_verify_cost_table",
         "adaptive_target_verify_calibration",
         "adaptive_target_verify_cost_table_output"}};
    return kOptionCategory;
  }

  // User-facing flag/JSON key stays "draft_model".
  PROPERTY(std::string, draft_model_path);

  PROPERTY(int32_t, num_speculative_tokens) = 0;

  // Raw user string; empty = not configured. Read algorithm() instead.
  PROPERTY(std::string, speculative_algorithm);

  PROPERTY(int32_t, speculative_suffix_cache_max_depth) = 64;

  PROPERTY(double, speculative_suffix_max_spec_factor) = 1.0;

  PROPERTY(double, speculative_suffix_max_spec_offset) = 0.0;

  PROPERTY(double, speculative_suffix_min_token_prob) = 0.1;

  PROPERTY(int32_t, speculative_suffix_max_cached_requests) = -1;

  PROPERTY(bool, speculative_suffix_use_tree_spec) = false;

  PROPERTY(bool, enable_probabilistic_draft) = false;

  PROPERTY(bool, enable_mtp_draft_body_tp1) = false;

  PROPERTY(bool, enable_atb_spec_kernel) = false;

  PROPERTY(bool, enable_adaptive_target_verify) = false;

  PROPERTY(bool, enable_adaptive_draft_step) = false;

  PROPERTY(std::string, adaptive_draft_step_config);

  PROPERTY(double, adaptive_target_verify_min_gain) = 0.0;

  PROPERTY(std::string, adaptive_target_verify_cost_table);

  PROPERTY(std::string, adaptive_target_verify_calibration);

  PROPERTY(std::string, adaptive_target_verify_cost_table_output);

 private:
  template <typename T>
  std::optional<T> read_draft_config(std::string_view key) const;
};

}  // namespace xllm
