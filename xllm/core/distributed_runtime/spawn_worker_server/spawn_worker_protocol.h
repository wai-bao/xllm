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
#include <cstdlib>
#include <optional>
#include <string>

#include "core/common/types.h"

namespace xllm::spawn_worker_protocol {

inline constexpr int32_t kArgumentCount = 41;
inline constexpr int32_t kMinimumArgumentCount = 34;
inline constexpr int32_t kSpecDecodeEnabledArgumentIndex = 14;
inline constexpr int32_t kNumSpeculativeTokensArgumentIndex = 15;
inline constexpr int32_t kSpeculativeAlgorithmArgumentIndex = 16;
inline constexpr int32_t kIndexerCacheDtypeArgumentIndex = 34;
inline constexpr int32_t kEnableMtpDraftBodyTp1ArgumentIndex = 35;
inline constexpr int32_t kTextEncoderTpSizeArgumentIndex = 36;
inline constexpr int32_t kEnableProbabilisticDraftArgumentIndex = 37;
inline constexpr int32_t kKvSplitSizeArgumentIndex = 38;
inline constexpr int32_t kModelImplArgumentIndex = 39;
inline constexpr int32_t kExtensionImplArgumentIndex = 40;
inline constexpr char kDefaultIndexerCacheDtype[] = "auto";

// Speculative-config fields carried via argv; a new SpeculativeConfig field
// must be added here, in the argv construction and to the singleton rebuild,
// or spawn deployments silently lose it.
struct SpawnSpeculativeConfig {
  int32_t num_speculative_tokens = 0;
  std::string speculative_algorithm;
  bool enable_mtp_draft_body_tp1 = false;
  bool enable_probabilistic_draft = false;
};

inline std::optional<std::string> parse_indexer_cache_dtype(
    int32_t argc,
    char* const argv[]) {
  if (argc < kMinimumArgumentCount || argv == nullptr) {
    return std::nullopt;
  }

  if (argc == kMinimumArgumentCount) {
    return std::string(kDefaultIndexerCacheDtype);
  }

  if (argv[kIndexerCacheDtypeArgumentIndex] == nullptr) {
    return std::nullopt;
  }
  return std::string(argv[kIndexerCacheDtypeArgumentIndex]);
}

inline std::string parse_model_impl(int32_t argc, char* const argv[]) {
  if (argv == nullptr || argc <= kModelImplArgumentIndex ||
      argv[kModelImplArgumentIndex] == nullptr) {
    return {};
  }
  return std::string(argv[kModelImplArgumentIndex]);
}

inline bool parse_bool_flag(int32_t argc, char* const argv[], int32_t index) {
  if (argv == nullptr || argc <= index || argv[index] == nullptr) {
    return false;
  }
  return atoi(argv[index]) > 0;
}

inline SpawnSpeculativeConfig parse_speculative_config(int32_t argc,
                                                       char* const argv[]) {
  SpawnSpeculativeConfig config;
  if (argv == nullptr || argc <= kSpeculativeAlgorithmArgumentIndex) {
    return config;
  }
  config.num_speculative_tokens =
      atoi(argv[kNumSpeculativeTokensArgumentIndex]);
  config.speculative_algorithm = argv[kSpeculativeAlgorithmArgumentIndex];
  config.enable_mtp_draft_body_tp1 =
      parse_bool_flag(argc, argv, kEnableMtpDraftBodyTp1ArgumentIndex);
  config.enable_probabilistic_draft =
      parse_bool_flag(argc, argv, kEnableProbabilisticDraftArgumentIndex);
  return config;
}

inline std::string parse_extension_impl(int32_t argc, char* const argv[]) {
  if (argv == nullptr || argc <= kExtensionImplArgumentIndex ||
      argv[kExtensionImplArgumentIndex] == nullptr) {
    return {};
  }
  return std::string(argv[kExtensionImplArgumentIndex]);
}

}  // namespace xllm::spawn_worker_protocol
