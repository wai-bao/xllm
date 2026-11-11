/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/framework/config/speculative_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/causal_vlm.h"
#include "core/framework/model/dit_model.h"
#include "core/framework/model_context/dit_model_context.h"
#include "core/framework/model_context/model_context.h"
#include "core/framework/tokenizer/tokenizer_args.h"
#include "core/util/json_reader.h"
#include "core/util/type_traits.h"  // IWYU pragma: keep
#include "processors/multimodal_processor.h"

namespace xllm {

// Target verification contract declared with each model registration; the
// runtime separately checks the current batch and device can execute it.
enum class TargetSpecVerifyMode : int8_t {
  GENERIC,
  CAUSAL_CHUNKED_PREFILL,
  // The target's verify kernel requires the same width for every sequence.
  UNIFORM_EXPANDED_VERIFY,
  // The Python executor's expanded verify path owns the verify graph.
  PYTHON_EXPANDED_VERIFY,
};

class Options;

namespace util {
struct DitDiscoveredComponent;

std::string resolve_dit_pipeline_type(
    const std::vector<DitDiscoveredComponent>& components);
}  // namespace util

using CausalLMFactory =
    std::function<std::unique_ptr<CausalLM>(const ModelContext& context)>;

using CausalVLMFactory =
    std::function<std::unique_ptr<CausalVLM>(const ModelContext& context)>;

using DiTModelFactory =
    std::function<std::unique_ptr<DiTModel>(const DiTModelContext& context)>;

using MultimodalProcessorFactory =
    std::function<std::unique_ptr<MultimodalProcessorBase>(
        const ModelArgs& model_args,
        std::shared_ptr<Tokenizer> tokenizer,
        const TokenizerArgs& tokenizer_args,
        int32_t max_sequence_length)>;

using ModelArgsLoader =
    std::function<bool(const JsonReader& json, ModelArgs* args)>;

class SpeechModelInterface;

using SpeechModelFactory =
    std::function<std::shared_ptr<const SpeechModelInterface>()>;

using TokenizerArgsLoader =
    std::function<bool(const JsonReader& json, TokenizerArgs* args)>;

struct CausalLMRegistration {
  CausalLMFactory factory;
  int32_t priority = 0;
};

struct CausalVLMRegistration {
  CausalVLMFactory factory;
  int32_t priority = 0;
};

using GraphKVCapacityAdapter = std::function<int64_t(const ModelArgs&)>;

enum class MtpModelFamily : int8_t {
  NONE,
  GLM5,
  QWEN35,
};

enum class GraphContextPolicy : int8_t {
  MAX_CONTEXT,
  QWEN_FULL_ATTENTION,
  KPOOL_FIXED,
};

struct MtpModelCapabilities {
  MtpModelFamily model_family = MtpModelFamily::NONE;
  bool supports_grouped_graph = false;
  GraphContextPolicy graph_context_policy = GraphContextPolicy::MAX_CONTEXT;
};

// TODO: add default args loader.
struct ModelMeta {
  std::unordered_map<std::string, CausalLMRegistration> causal_lm_factories;
  SpeechModelFactory speech_model_factory;
  std::unordered_map<std::string, CausalVLMRegistration> causal_vlm_factories;
  DiTModelFactory dit_model_factory;
  MultimodalProcessorFactory multimodal_processor_factory;
  ModelArgsLoader model_args_loader;
  TokenizerArgsLoader tokenizer_args_loader;
  bool supports_dcp = false;
  MtpModelCapabilities mtp_capabilities;
  GraphKVCapacityAdapter graph_kv_capacity_adapter;
  TargetSpecVerifyMode target_spec_verify_mode = TargetSpecVerifyMode::GENERIC;
  // Qwen3.5-family MegaGdn linear-state layout: the NPU linear-state cache
  // sizes its blocks for this family (targets and drafts share the layout).
  bool qwen3_5_linear_state_layout = false;
  std::string mtp_draft_model_type;
  std::string mtp_target_model_type;
  std::function<void(ModelArgs&)> mtp_draft_args_adapter;
  // Draft-input row hidden width for this target; null defaults to
  // args.hidden_size().
  std::function<int64_t(const ModelArgs&)> draft_hidden_width_fn;
};

// Model config names may be aliases, but registry lookups use one canonical
// key across backend, args-loader, and multimodal registrations.
inline std::string canonical_model_registration_name(
    std::string_view model_type) {
  if (model_type == "glm5_3_flash") {
    return "glm5_next";
  }
  // The DeepSeek-OCR release ships HF config model_type "deepseek_vl_v2"
  // (its architecture lineage); xllm registers the port under deepseek_ocr.
  if (model_type == "deepseek_vl_v2") {
    return "deepseek_ocr";
  }
  return std::string(model_type);
}

// Model registry is a singleton class that registers all models with the
// ModelFactory, ModelArgParser to facilitate model loading.
class ModelRegistry {
 public:
  static ModelRegistry* get_instance();

  static void register_causallm_factory(const std::string& name,
                                        CausalLMFactory factory);

  static bool register_causallm_factory(std::string name,
                                        CausalLMFactory factory,
                                        int32_t priority);

  static bool register_causallm_factory(std::string name,
                                        std::string implementation,
                                        CausalLMFactory factory,
                                        int32_t priority = 0);

  static void register_causalvlm_factory(const std::string& name,
                                         CausalVLMFactory factory);

  static bool register_causalvlm_factory(std::string name,
                                         CausalVLMFactory factory,
                                         int32_t priority);

  static bool register_causalvlm_factory(std::string name,
                                         std::string implementation,
                                         CausalVLMFactory factory,
                                         int32_t priority = 0);

  static void register_dit_model_factory(const std::string& name,
                                         DiTModelFactory factory);

  static void register_model_backend(const std::string& name,
                                     const std::string& backend);

  static void register_model_args_loader(const std::string& name,
                                         ModelArgsLoader loader);

  static void register_tokenizer_args_loader(const std::string& name,
                                             TokenizerArgsLoader loader);

  static void register_multimodal_processor_factory(
      const std::string& name,
      MultimodalProcessorFactory factory);

  static void register_speech_model_factory(const std::string& name,
                                            SpeechModelFactory factory);

  static void register_dcp_capability(const std::string& name);
  static bool supports_dcp(const std::string& name);

  static void register_mtp_capabilities(const std::string& name,
                                        MtpModelCapabilities capabilities);

  static MtpModelCapabilities get_mtp_capabilities(const std::string& name);

  static void register_graph_kv_capacity(const std::string& name,
                                         GraphKVCapacityAdapter adapter);

  static int64_t get_graph_kv_capacity(const std::string& name,
                                       const ModelArgs& args,
                                       int64_t default_capacity);

  static void register_mtp_model_pair(
      const std::string& target,
      const std::string& draft,
      std::function<void(ModelArgs&)> args_adapter = {});
  static std::string get_mtp_target_model_type(const std::string& draft);
  // Convert loaded target arguments using the paired model's own policy.
  static bool convert_to_mtp_draft(ModelArgs& args);
  static void register_draft_hidden_width(
      const std::string& name,
      std::function<int64_t(const ModelArgs&)> width_fn);
  // Key contract: `name` is the target model_type. A draft-converted
  // model_type resolves through its registered MTP pair first.
  static int64_t get_draft_hidden_width(const std::string& name,
                                        const ModelArgs& model_args);
  // Registers one model (target or draft) as using the Qwen3.5 MegaGdn
  // linear-state layout.
  static void register_qwen3_5_linear_state_layout(const std::string& name);
  static bool uses_qwen3_5_linear_state_layout(const std::string& name);
  static void register_target_spec_verify_mode(const std::string& name,
                                               TargetSpecVerifyMode mode);
  static TargetSpecVerifyMode get_target_spec_verify_mode(
      const std::string& name);

  static CausalLMFactory get_causallm_factory(const std::string& name);

  static CausalLMFactory get_causallm_factory(
      const std::string& name,
      const std::string& implementation);

  static CausalVLMFactory get_causalvlm_factory(const std::string& name);

  static CausalVLMFactory get_causalvlm_factory(
      const std::string& name,
      const std::string& implementation);

  static DiTModelFactory get_dit_model_factory(const std::string& name);

  static ModelArgsLoader get_model_args_loader(const std::string& name);

  static TokenizerArgsLoader get_tokenizer_args_loader(const std::string& name);

  static MultimodalProcessorFactory get_multimodal_processor_factory(
      const std::string& name);

  static SpeechModelFactory get_speech_model_factory(const std::string& name);

  static bool has_dit_model_factory(const std::string& name);

  static std::string get_model_backend(const std::string& name);

 private:
  // Metadata getters resolve aliases through the canonical key; the factory
  // getters intentionally use the raw registration key because the load path
  // canonicalizes the model type before calling them.
  static const ModelMeta* find_model_meta(const std::string& name);

  std::unordered_map<std::string, ModelMeta> model_registry_;
  std::unordered_map<std::string, std::string> model_backend_;
};

bool resolve_model_registration_name(const std::string& model_type,
                                     std::string* resolved_name,
                                     std::string* error_message = nullptr);

// Return whether a native NPU model owns context-parallel sharding.
bool is_npu_model_cp_capable(const std::string& resolved_name);

// Native DCP capability is distinct from prefill CP sharding capability.
bool is_npu_model_dcp_capable(const std::string& resolved_name);

int64_t get_npu_linear_state_cache_block_limit(const std::string& model_type);

bool is_mlu_model_cp_capable(const std::string& resolved_name);

bool resolve_model_registration(const std::string& model_type,
                                const std::string& requested_npu_kernel_backend,
                                std::string* effective_npu_kernel_backend,
                                std::string* resolved_name,
                                std::string* error_message = nullptr);

// Resolve a selected model extension before applying built-in backend rules.
// model_component is "causal_lm" or "causal_vlm". Registered model extensions
// use the same Torch backend as built-in NPU implementations.
bool resolve_model_registration(const std::string& model_type,
                                const std::string& requested_npu_kernel_backend,
                                const std::string& extension_impl,
                                std::string_view model_component,
                                std::string* effective_npu_kernel_backend,
                                std::string* resolved_name,
                                std::string* error_message = nullptr);

#if defined(USE_NPU)
// Resolve the model-specific NPU kernel backend before constructing a runtime
// engine. This applies only to the LLM/VLM model registry; DiT has its own
// model registry and backend selection. LLM/VLM pass their model component to
// apply Options::extension_impl before built-in backend validation.
void resolve_npu_kernel_backend(Options* options,
                                std::string_view model_component = "");
#endif

std::unique_ptr<CausalLM> create_llm_model(const ModelContext& context);

std::unique_ptr<CausalVLM> create_vlm_model(const ModelContext& context);

std::unique_ptr<DiTModel> create_dit_model(const DiTModelContext& context);

// Macro to register a model with the ModelRegistry
#define REGISTER_CAUSAL_MODEL_WITH_VARNAME(                        \
    VarName, ModelType, ModelClass, ...)                           \
  const bool VarName##_registered = []() {                         \
    ::xllm::ModelRegistry::register_causallm_factory(              \
        #ModelType, [](const ::xllm::ModelContext& context) {      \
          ModelClass model(context);                               \
          model->eval();                                           \
          return std::make_unique<xllm::CausalLMImpl<ModelClass>>( \
              std::move(model), context.get_tensor_options());     \
        } __VA_OPT__(, ) __VA_ARGS__);                             \
    return true;                                                   \
  }()

#define REGISTER_CAUSAL_MODEL(ModelType, ModelClass, ...) \
  REGISTER_CAUSAL_MODEL_WITH_VARNAME(                     \
      ModelType, ModelType, ModelClass __VA_OPT__(, ) __VA_ARGS__)

// Register a CausalLM subclass directly. Higher priorities replace lower
// priorities for the same model and implementation; ties keep the first.
#define CONCAT_MODEL_REGISTRATION_IMPL(left, right) left##right
#define CONCAT_MODEL_REGISTRATION(left, right) \
  CONCAT_MODEL_REGISTRATION_IMPL(left, right)

#define REGISTER_CAUSAL_LM_WITH_VARNAME(                                 \
    VarName, ModelType, Implementation, ModelClass, ...)                 \
  namespace {                                                            \
  const bool CONCAT_MODEL_REGISTRATION(VarName, _causal_lm_registered) = \
      ::xllm::ModelRegistry::register_causallm_factory(                  \
          #ModelType,                                                    \
          Implementation,                                                \
          [](const ::xllm::ModelContext& context) {                      \
            auto model = std::make_unique<ModelClass>(context);          \
            model->eval();                                               \
            return model;                                                \
          } __VA_OPT__(, ) __VA_ARGS__);                                 \
  }

#define REGISTER_CAUSAL_LM(ModelType, Implementation, ModelClass, ...)      \
  REGISTER_CAUSAL_LM_WITH_VARNAME(                                          \
      CONCAT_MODEL_REGISTRATION(xllm_causal_lm_registration_, __COUNTER__), \
      ModelType,                                                            \
      Implementation,                                                       \
      ModelClass __VA_OPT__(, ) __VA_ARGS__)

#define REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(                     \
    VarName, ModelType, ModelClass, ...)                            \
  const bool VarName##_registered = []() {                          \
    ::xllm::ModelRegistry::register_causalvlm_factory(              \
        #ModelType, [](const ::xllm::ModelContext& context) {       \
          ModelClass model(context);                                \
          model->eval();                                            \
          return std::make_unique<xllm::CausalVLMImpl<ModelClass>>( \
              std::move(model), context.get_tensor_options());      \
        } __VA_OPT__(, ) __VA_ARGS__);                              \
    return true;                                                    \
  }()

#define REGISTER_CAUSAL_VLM_MODEL(ModelType, ModelClass, ...) \
  REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(                     \
      ModelType, ModelType, ModelClass __VA_OPT__(, ) __VA_ARGS__)

#define REGISTER_CAUSAL_VLM_WITH_VARNAME(                                 \
    VarName, ModelType, Implementation, ModelClass, ...)                  \
  namespace {                                                             \
  const bool CONCAT_MODEL_REGISTRATION(VarName, _causal_vlm_registered) = \
      ::xllm::ModelRegistry::register_causalvlm_factory(                  \
          #ModelType,                                                     \
          Implementation,                                                 \
          [](const ::xllm::ModelContext& context) {                       \
            auto model = std::make_unique<ModelClass>(context);           \
            model->eval();                                                \
            return model;                                                 \
          } __VA_OPT__(, ) __VA_ARGS__);                                  \
  }

#define REGISTER_CAUSAL_VLM(ModelType, Implementation, ModelClass, ...)      \
  REGISTER_CAUSAL_VLM_WITH_VARNAME(                                          \
      CONCAT_MODEL_REGISTRATION(xllm_causal_vlm_registration_, __COUNTER__), \
      ModelType,                                                             \
      Implementation,                                                        \
      ModelClass __VA_OPT__(, ) __VA_ARGS__)

#define REGISTER_TARGET_SPEC_VERIFY_MODE(ModelType, Mode) \
  const bool ModelType##_spec_verify_registered = []() {  \
    ModelRegistry::register_target_spec_verify_mode(      \
        #ModelType, TargetSpecVerifyMode::Mode);          \
    return true;                                          \
  }()

#define REGISTER_MTP_MODEL_PAIR(TargetType, DraftType, ...) \
  const bool DraftType##_mtp_pair_registered = []() {       \
    ModelRegistry::register_mtp_model_pair(                 \
        #TargetType, #DraftType, ##__VA_ARGS__);            \
    return true;                                            \
  }()

#define REGISTER_QWEN3_5_LINEAR_STATE_LAYOUT(ModelType)              \
  const bool ModelType##_qwen35_layout_registered = []() {           \
    ModelRegistry::register_qwen3_5_linear_state_layout(#ModelType); \
    return true;                                                     \
  }()

#define REGISTER_DIT_MODEL_WITH_VARNAME(VarName, ModelType, ModelClass) \
  const bool VarName##_registered = []() {                              \
    ModelRegistry::register_dit_model_factory(                          \
        #ModelType, [](const DiTModelContext& context) {                \
          ModelClass model(context);                                    \
          model->eval();                                                \
          return std::make_unique<xllm::DiTModelImpl<ModelClass>>(      \
              std::move(model), context.get_tensor_options());          \
        });                                                             \
    return true;                                                        \
  }()

#define REGISTER_DIT_MODEL(ModelType, ModelClass) \
  REGISTER_DIT_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

#define REGISTER_MODEL_BACKEND_WITH_VARNAME(VarName, ModelType, Backend) \
  const bool VarName##_backend_registered = []() {                       \
    ModelRegistry::register_model_backend(#ModelType, Backend);          \
    return true;                                                         \
  }()

#define REGISTER_MODEL_BACKEND(ModelType, Backend) \
  REGISTER_MODEL_BACKEND_WITH_VARNAME(ModelType, ModelType, Backend)

#define REGISTER_MULTIMODAL_PROCESSOR_WITH_VARNAME(                     \
    VarName, ModelType, ProcessorClass)                                 \
  const bool VarName##_multimodal_processor_registered = []() {         \
    ModelRegistry::register_multimodal_processor_factory(               \
        #ModelType,                                                     \
        [](const ModelArgs& model_args,                                 \
           std::shared_ptr<Tokenizer> tokenizer,                        \
           const TokenizerArgs& tokenizer_args,                         \
           int32_t max_sequence_length) {                               \
          return std::make_unique<ProcessorClass>(model_args,           \
                                                  std::move(tokenizer), \
                                                  tokenizer_args,       \
                                                  max_sequence_length); \
        });                                                             \
    return true;                                                        \
  }()

#define REGISTER_MULTIMODAL_PROCESSOR(ModelType, ProcessorClass) \
  REGISTER_MULTIMODAL_PROCESSOR_WITH_VARNAME(                    \
      ModelType, ModelType, ProcessorClass)

#define REGISTER_SPEECH_MODEL_WITH_VARNAME(                                 \
    VarName, ModelType, SpeechModelClass)                                   \
  const bool VarName##_speech_model_registered = []() {                     \
    ModelRegistry::register_speech_model_factory(                           \
        #ModelType, []() { return std::make_shared<SpeechModelClass>(); }); \
    return true;                                                            \
  }()

#define REGISTER_SPEECH_MODEL(ModelType, SpeechModelClass) \
  REGISTER_SPEECH_MODEL_WITH_VARNAME(ModelType, ModelType, SpeechModelClass)

// Macro to register a model args loader with the ModelRegistry
#define REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(VarName, ModelType, Loader) \
  const bool VarName##_args_loader_registered = []() {                      \
    ModelRegistry::register_model_args_loader(#ModelType, Loader);          \
    return true;                                                            \
  }()

#define REGISTER_MODEL_ARGS_LOADER(ModelType, Loader) \
  REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(ModelType, ModelType, Loader)

#define REGISTER_MODEL_ARGS_WITH_VARNAME(VarName, ModelType, ...)       \
  REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(                              \
      VarName, ModelType, [](const JsonReader& json, ModelArgs* args) { \
        UNUSED_PARAMETER(json);                                         \
        UNUSED_PARAMETER(args);                                         \
        __VA_ARGS__();                                                  \
        return true;                                                    \
      })

#define REGISTER_MODEL_ARGS(ModelType, ...) \
  REGISTER_MODEL_ARGS_WITH_VARNAME(ModelType, ModelType, __VA_ARGS__)

// Macro to register a tokenizer args loader with the ModelRegistry
#define REGISTER_TOKENIZER_ARGS_LOADER_WITH_VARNAME(                   \
    VarName, ModelType, Loader)                                        \
  const bool VarName##_tokenizer_args_loader_registered = []() {       \
    ModelRegistry::register_tokenizer_args_loader(#ModelType, Loader); \
    return true;                                                       \
  }()

#define REGISTER_TOKENIZER_ARGS_LOADER(ModelType, Loader) \
  REGISTER_TOKENIZER_ARGS_LOADER_WITH_VARNAME(ModelType, ModelType, Loader)

#define REGISTER_TOKENIZER_ARGS_WITH_VARNAME(VarName, ModelType, ...)       \
  REGISTER_TOKENIZER_ARGS_LOADER_WITH_VARNAME(                              \
      VarName, ModelType, [](const JsonReader& json, TokenizerArgs* args) { \
        UNUSED_PARAMETER(json);                                             \
        UNUSED_PARAMETER(args);                                             \
        __VA_ARGS__();                                                      \
        return true;                                                        \
      })

#define REGISTER_TOKENIZER_ARGS(ModelType, ...) \
  REGISTER_TOKENIZER_ARGS_WITH_VARNAME(ModelType, ModelType, __VA_ARGS__)

#define LOAD_ARG(arg_name, json_name)                          \
  [&] {                                                        \
    auto value = args->arg_name();                             \
    using value_type = remove_optional_t<decltype(value)>;     \
    if (auto data_value = json.value<value_type>(json_name)) { \
      args->arg_name() = data_value.value();                   \
    }                                                          \
  }()

#define LOAD_ARG_OR(arg_name, json_name, default_value)                     \
  [&] {                                                                     \
    auto value = args->arg_name();                                          \
    using value_type = remove_optional_t<decltype(value)>;                  \
    args->arg_name() = json.value_or<value_type>(json_name, default_value); \
  }()

#define LOAD_ARG_OR_FUNC(arg_name, json_name, ...)             \
  [&] {                                                        \
    auto value = args->arg_name();                             \
    using value_type = remove_optional_t<decltype(value)>;     \
    if (auto data_value = json.value<value_type>(json_name)) { \
      args->arg_name() = data_value.value();                   \
    } else {                                                   \
      args->arg_name() = __VA_ARGS__();                        \
    }                                                          \
  }()

#define SET_ARG(arg_name, value) [&] { args->arg_name() = value; }()

// Rewrites args to the registered MTP draft model type when the checkpoint
// carries one; shared by the worker and engine speculative-decode paths.
inline void maybe_override_mtp_draft_model_type(ModelArgs& args) {
  const std::string current_type = args.model_type();
  if (ModelRegistry::convert_to_mtp_draft(args)) {
    LOG(INFO) << "Overriding draft model_type from " << current_type << " to "
              << args.model_type() << " for speculative decoding";
  }
}

// Resolves the MTP draft conversion and, on NPU, rewrites args to the
// registered pair-draft model type for a DRAFT_ENGINE_PAIR load. Returns the
// conversion so callers can act on EMBEDDED_LAYER themselves.
inline SpeculativeConfig::MtpDraftConversion maybe_load_mtp_pair_draft(
    bool is_draft_engine,
    std::string_view algorithm,
    int32_t num_speculative_tokens,
    ModelArgs& args) {
  const SpeculativeConfig::MtpDraftConversion conversion =
      SpeculativeConfig::resolve_mtp_draft_conversion(
          is_draft_engine,
          algorithm,
          num_speculative_tokens,
          args.num_nextn_predict_layers(),
          SpeculativeConfig::is_block_diffusion_algorithm(algorithm));
#if defined(USE_NPU)
  if (conversion == SpeculativeConfig::MtpDraftConversion::DRAFT_ENGINE_PAIR) {
    maybe_override_mtp_draft_model_type(args);
  }
#else
  (void)conversion;
#endif
  return conversion;
}

// Hidden width a DraftInput row must carry for this target model; defaults
// to hidden_size(), overridden per target when the draft embedding width
// differs (e.g. DeepSeek-V4 hyper-connections).
inline int64_t draft_input_hidden_width(const ModelArgs& model_args) {
  return ModelRegistry::get_draft_hidden_width(model_args.model_type(),
                                               model_args);
}

}  // namespace xllm
