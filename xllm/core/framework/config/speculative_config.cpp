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

#include "core/framework/config/speculative_config.h"

#include <glog/logging.h>

#include <boost/algorithm/string/predicate.hpp>
#include <filesystem>
#include <optional>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"
#include "core/framework/model/model_args.h"
#include "core/util/json_reader.h"

DEFINE_string(draft_model, "", "draft hf model path to the model file.");

DEFINE_int32(num_speculative_tokens, 0, "Number of speculative tokens.");

DEFINE_string(speculative_algorithm,
              "",
              "Speculative decoding algorithm. Supported options: MTP, "
              "Eagle3, Suffix, DFlash, DFlash2 (NPU only), DSpark. Empty "
              "(default) disables speculative decoding.");

DEFINE_int32(speculative_suffix_cache_max_depth,
             64,
             "Maximum suffix tree depth for suffix speculative decoding.");

DEFINE_double(speculative_suffix_max_spec_factor,
              1.0,
              "Suffix speculation max tokens factor relative to match length.");

DEFINE_double(speculative_suffix_max_spec_offset,
              0.0,
              "Suffix speculation max tokens additive offset.");

DEFINE_double(speculative_suffix_min_token_prob,
              0.1,
              "Minimum token probability used in suffix speculation.");

DEFINE_int32(speculative_suffix_max_cached_requests,
             -1,
             "Maximum globally cached requests for suffix speculation (-1 "
             "unlimited, 0 disabled).");

DEFINE_bool(speculative_suffix_use_tree_spec,
            false,
            "Whether to use tree-based suffix speculation instead of path "
            "speculation.");

DEFINE_bool(enable_probabilistic_draft,
            false,
            "Drafts sample from their distribution (emitting q for rejection "
            "sampling); the default is greedy argmax.");

DEFINE_bool(enable_mtp_draft_body_tp1,
            false,
            "Whether to run the MTP draft body with tensor-parallel size 1 "
            "while keeping the draft LMHead on the target TP group.");

DEFINE_bool(enable_atb_spec_kernel,
            false,
            "Whether to use ATB speculative kernel.");

DEFINE_bool(
    enable_adaptive_target_verify,
    false,
    "Adapt the target verification budget by pruning already-generated "
    "draft proposals using historical confidence and full-cycle costs, "
    "without reducing draft generation steps. Currently implemented "
    "for DSpark and greedy MTP Draft5/7 with Python ACL graphs and task "
    "pipeline. "
    "Costs are calibrated at startup unless a saved table is supplied.");

DEFINE_bool(enable_adaptive_draft_step,
            false,
            "Adapt the number of sequential draft forward steps executed "
            "before target verification using historical prefix acceptance "
            "and calibrated full-cycle throughput. Currently implemented for "
            "MTP with Python GLM, ordinary ACL graphs, async task pipeline "
            "and DP=CP=1.");
DEFINE_string(
    adaptive_draft_step_config,
    "",
    "Path to an optional JSON file containing draft-step selection policy "
    "and startup calibration sampling settings, not precomputed costs. Empty "
    "uses logged defaults. The current MTP service measures full-cycle costs "
    "for each depth using shared speculative SPS/additive fitting.");

DEFINE_double(
    adaptive_target_verify_min_gain,
    0.0,
    "Minimum predicted relative throughput gain required to change the "
    "adaptive target verification budget.");

DEFINE_string(
    adaptive_target_verify_cost_table,
    "",
    "JSON full-step cost table for adaptive DSpark task pipelines: "
    "one-dimensional SPS or two-dimensional additive model. Loading a "
    "compatible saved table skips startup sampling and fitting.");

DEFINE_string(
    adaptive_target_verify_calibration,
    "",
    "JSON configuration for internal startup DSpark calibration: "
    "capture graphs, drive the real asynchronous pipeline, fit CPU "
    "SPS/additive costs, and serve without a client or restart. Empty uses "
    "built-in defaults. Mutually exclusive with a cost-table input. "
    "The effective configuration is logged for defaults and JSON input.");

DEFINE_string(
    adaptive_target_verify_cost_table_output,
    "",
    "Optional new file for internally calibrated DSpark costs. "
    "Requires adaptive DSpark startup calibration; existing files are not "
    "overwritten.");

namespace xllm {

SpeculativeAlgorithm SpeculativeConfig::from_string(
    std::string_view algorithm) {
  if (boost::iequals(algorithm, "MTP")) {
    return SpeculativeAlgorithm::MTP;
  }
  if (boost::iequals(algorithm, "Eagle3")) {
    return SpeculativeAlgorithm::EAGLE3;
  }
  if (boost::iequals(algorithm, "Suffix")) {
    return SpeculativeAlgorithm::SUFFIX;
  }
  if (boost::iequals(algorithm, "DFlash")) {
    return SpeculativeAlgorithm::DFLASH;
  }
  if (boost::iequals(algorithm, "DFlash2")) {
    return SpeculativeAlgorithm::DFLASH2;
  }
  if (boost::iequals(algorithm, "DSpark")) {
    return SpeculativeAlgorithm::DSPARK;
  }
  return SpeculativeAlgorithm::NONE;
}

void SpeculativeConfig::from_flags() {
  draft_model_path(FLAGS_draft_model);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(num_speculative_tokens);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_algorithm);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_cache_max_depth);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_max_spec_factor);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_max_spec_offset);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_min_token_prob);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_max_cached_requests);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_use_tree_spec);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_probabilistic_draft);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_mtp_draft_body_tp1);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_atb_spec_kernel);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_adaptive_target_verify);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_adaptive_draft_step);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(adaptive_draft_step_config);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(adaptive_target_verify_min_gain);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(adaptive_target_verify_cost_table);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(adaptive_target_verify_calibration);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(adaptive_target_verify_cost_table_output);
}

void SpeculativeConfig::from_json(const JsonReader& json) {
  CHECK(!json.contains("enable_adaptive_speculative_decode"))
      << "enable_adaptive_speculative_decode was renamed to "
         "enable_adaptive_target_verify; update the JSON configuration.";
  CHECK(!json.contains("enable_adaptive_spec_prune"))
      << "enable_adaptive_spec_prune was renamed to "
         "enable_adaptive_target_verify; update the JSON configuration.";
  CHECK(!json.contains("enable_adaptive_mtp"))
      << "enable_adaptive_mtp was renamed to "
         "enable_adaptive_draft_step; update the JSON configuration.";
  CHECK(!json.contains("adaptive_mtp_config"))
      << "adaptive_mtp_config was renamed to adaptive_draft_step_config; "
         "update the JSON configuration.";
  CHECK(!json.contains("adaptive_dspark_cost_table"))
      << "adaptive_dspark_cost_table was renamed to "
         "adaptive_target_verify_cost_table; update the JSON configuration.";
  CHECK(!json.contains("adaptive_dspark_calibration"))
      << "adaptive_dspark_calibration was renamed to "
         "adaptive_target_verify_calibration; update the JSON configuration.";
  CHECK(!json.contains("adaptive_dspark_cost_table_output"))
      << "adaptive_dspark_cost_table_output was renamed to "
         "adaptive_target_verify_cost_table_output; update the JSON "
         "configuration.";
  CHECK(!json.contains("adaptive_speculative_min_gain"))
      << "adaptive_speculative_min_gain was renamed to "
         "adaptive_target_verify_min_gain; update the JSON configuration.";
  if (!config::is_flag_specified("draft_model")) {
    draft_model_path(
        json.value_or<std::string>("draft_model", draft_model_path()));
    FLAGS_draft_model = draft_model_path();
  }
  XLLM_CONFIG_ASSIGN_FROM_JSON(num_speculative_tokens);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_algorithm);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_cache_max_depth);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_max_spec_factor);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_max_spec_offset);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_min_token_prob);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_max_cached_requests);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_use_tree_spec);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_probabilistic_draft);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_mtp_draft_body_tp1);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_atb_spec_kernel);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_adaptive_target_verify);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_adaptive_draft_step);
  XLLM_CONFIG_ASSIGN_FROM_JSON(adaptive_draft_step_config);
  XLLM_CONFIG_ASSIGN_FROM_JSON(adaptive_target_verify_min_gain);
  XLLM_CONFIG_ASSIGN_FROM_JSON(adaptive_target_verify_cost_table);
  XLLM_CONFIG_ASSIGN_FROM_JSON(adaptive_target_verify_calibration);
  XLLM_CONFIG_ASSIGN_FROM_JSON(adaptive_target_verify_cost_table_output);
}

void SpeculativeConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const SpeculativeConfig default_config;
  APPEND_JSON_VALUE_IF_NOT_DEFAULT(config_json,
                                   "draft_model",
                                   draft_model_path(),
                                   default_config.draft_model_path());
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, num_speculative_tokens);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_algorithm);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_cache_max_depth);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_max_spec_factor);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_max_spec_offset);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_min_token_prob);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_max_cached_requests);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_use_tree_spec);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_probabilistic_draft);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_mtp_draft_body_tp1);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_atb_spec_kernel);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_adaptive_target_verify);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_adaptive_draft_step);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, adaptive_draft_step_config);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, adaptive_target_verify_min_gain);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, adaptive_target_verify_cost_table);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, adaptive_target_verify_calibration);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, adaptive_target_verify_cost_table_output);
}

std::string SpeculativeConfig::to_string() const {
  // Same field list as append_config_json; defaults are skipped.
  nlohmann::ordered_json config_json;
  append_config_json(config_json);
  return config_json.dump();
}

SpeculativeConfig& SpeculativeConfig::get_instance() {
  static SpeculativeConfig config;
  return config;
}

void SpeculativeConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
  // Unknown non-empty strings are rejected, not parsed as NONE.
  const SpeculativeAlgorithm kind = from_string(speculative_algorithm());
  CHECK(speculative_algorithm().empty() || kind != SpeculativeAlgorithm::NONE)
      << "Unknown speculative_algorithm: " << speculative_algorithm()
      << ". Supported: MTP, Eagle3, Suffix, DFlash, DFlash2, DSpark.";
  normalize();
  validate();
}

// Construction-time normalization: derive what the user left unset.
void SpeculativeConfig::normalize() {
  if (!enable_speculative_decode() && !draft_model_path().empty()) {
    // Prefer the checkpoint's model_type over the path heuristic.
    const std::optional<std::string> draft_model_type =
        read_draft_config<std::string>("model_type");
    if (draft_model_type.has_value()) {
      const std::string_view type = *draft_model_type;
      if (is_dspark_draft_model_type(type)) {
        speculative_algorithm("DSpark");
      } else if (is_dflash2_draft_model_type(type)) {
        speculative_algorithm("DFlash2");
      } else if (type == "DFlashDraftModel") {
        speculative_algorithm("DFlash");
      } else if (type.find("eagle3") != std::string_view::npos ||
                 type.find("eagle") != std::string_view::npos) {
        speculative_algorithm("Eagle3");
      } else {
        speculative_algorithm("MTP");
      }
      LOG(INFO) << "speculative_algorithm derived from draft model_type '"
                << type << "' of " << draft_model_path() << ": "
                << speculative_algorithm();
    } else {
      const std::string_view name = draft_model_path();
      if (boost::algorithm::contains(name, "eagle3") ||
          boost::algorithm::contains(name, "eagle-")) {
        speculative_algorithm("Eagle3");
      } else if (boost::algorithm::contains(name, "dflash2")) {
        speculative_algorithm("DFlash2");
      } else if (boost::algorithm::contains(name, "dspark")) {
        speculative_algorithm("DSpark");
      } else if (boost::algorithm::contains(name, "dflash")) {
        speculative_algorithm("DFlash");
      } else {
        speculative_algorithm("MTP");
      }
      LOG(WARNING) << "speculative_algorithm derived from the draft model "
                      "path '"
                   << draft_model_path()
                   << "' (no readable config.json model_type): "
                   << speculative_algorithm()
                   << ". Rename the directory and this guess changes; pass "
                      "speculative_algorithm explicitly.";
    }
  }
  if (enable_speculative_decode() && num_speculative_tokens() == 0) {
    // Width unset: derive it from the draft checkpoint.
    std::optional<int32_t> derived;
    if (algorithm() == SpeculativeAlgorithm::MTP) {
      derived = read_draft_config<int32_t>("num_nextn_predict_layers");
    } else if (algorithm() == SpeculativeAlgorithm::DFLASH2) {
      derived = read_draft_config<int32_t>("dflash2_block_size");
    } else if (algorithm() == SpeculativeAlgorithm::SUFFIX) {
      derived = speculative_suffix_cache_max_depth();
    }
    CHECK(derived.has_value() && derived.value() > 0)
        << "num_speculative_tokens was not provided and could not be "
           "derived for "
        << speculative_algorithm()
        << "; pass it explicitly or use a draft checkpoint that declares it.";
    num_speculative_tokens(derived.value());
    LOG(INFO) << "num_speculative_tokens derived: " << derived.value();
  }
}

template <typename T>
std::optional<T> SpeculativeConfig::read_draft_config(
    std::string_view key) const {
  const std::string config_path =
      (std::filesystem::path(draft_model_path()) / "config.json").string();
  JsonReader reader = config::load_json_file(config_path);
  return reader.value<T>(std::string(key));
}

template std::optional<int32_t> SpeculativeConfig::read_draft_config(
    std::string_view key) const;
template std::optional<std::string> SpeculativeConfig::read_draft_config(
    std::string_view key) const;

void SpeculativeConfig::validate() const {
  CHECK_GE(num_speculative_tokens(), 0)
      << "num_speculative_tokens must be >= 0 (0 means unset).";
#if defined(USE_NPU)
  // Fused attention TND layout bound on the per-step verify width.
  CHECK_LE(num_speculative_tokens(), kMaxSpeculativeTokens)
      << "NPU fused attention supports at most " << kMaxSpeculativeTokens
      << " speculative tokens per decode step.";
#endif
  CHECK(!enable_adaptive_draft_step() ||
        (enable_speculative_decode() &&
         algorithm() == SpeculativeAlgorithm::MTP &&
         num_speculative_tokens() >= 1 && num_speculative_tokens() <= 64 &&
         !enable_adaptive_target_verify()))
      << "Adaptive draft-step selection requires speculative_algorithm=MTP, "
         "maximum depth 1..64, and disabled adaptive target verification.";
  CHECK(adaptive_draft_step_config().empty() || enable_adaptive_draft_step())
      << "adaptive_draft_step_config requires enable_adaptive_draft_step.";
  CHECK(adaptive_target_verify_calibration().empty() ||
        (adaptive_target_verify_cost_table().empty() &&
         enable_adaptive_target_verify() &&
         supports_adaptive_target_verify(algorithm()) &&
         (num_speculative_tokens() == 5 || num_speculative_tokens() == 7)))
      << "adaptive_target_verify_calibration requires adaptive MTP/DSpark "
         "Draft5/7 "
         "without "
         "a cost-table input.";
  CHECK(adaptive_target_verify_cost_table_output().empty() ||
        (adaptive_target_verify_cost_table().empty() &&
         enable_adaptive_target_verify() &&
         supports_adaptive_target_verify(algorithm()) &&
         (num_speculative_tokens() == 5 || num_speculative_tokens() == 7)))
      << "adaptive_target_verify_cost_table_output requires startup "
         "calibration.";
  CHECK(adaptive_target_verify_cost_table().empty() ||
        (enable_adaptive_target_verify() &&
         supports_adaptive_target_verify(algorithm()) &&
         (num_speculative_tokens() == 5 || num_speculative_tokens() == 7)))
      << "adaptive_target_verify_cost_table requires adaptive MTP/DSpark "
         "Draft5/7.";
  CHECK(!enable_adaptive_target_verify() ||
        (supports_adaptive_target_verify(algorithm()) &&
         (num_speculative_tokens() == 5 || num_speculative_tokens() == 7)))
      << "Adaptive target verification supports MTP/DSpark Draft5/7; "
         "DFlash and DFlash2 do not support target verification pruning.";
  if (enable_probabilistic_draft()) {
    CHECK(is_probabilistic_draft_sampling_supported(algorithm()))
        << speculative_algorithm()
        << " does not support enable_probabilistic_draft. Supported: MTP, "
           "DSpark, DFlash2, Eagle3.";
  }
}

}  // namespace xllm
