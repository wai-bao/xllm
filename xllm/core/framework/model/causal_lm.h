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
// clang-format off
#if defined(USE_NPU)
#include "graph/types.h"
#endif
#include "core/framework/model/model.h"
// clang-format on
#include <c10/core/Device.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model/model_traits.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/framework/multimodal/mm_data_item.h"
#include "core/framework/quantization/quant_args.h"
#include "core/kv_cache/storage/kv_cache.h"

namespace xllm {

namespace layer {
struct AttentionMetadata;
class LmHead;
}  // namespace layer

struct ModelGraphMetadataState {
  virtual ~ModelGraphMetadataState() = default;
};

struct DFlash2CandidateOutput {
  // Candidate vocabulary ids [batch, draft_steps, top_k]. Edge logits use
  // [batch, draft_steps, predecessor_top_k, successor_top_k]; at step zero
  // every predecessor entry represents the same anchor token.
  torch::Tensor candidate_ids;
  torch::Tensor edge_logits;
};

class CausalLM : public torch::nn::Module, public Model {
 public:
  ~CausalLM() override = default;

  // An empty key keeps the runtime's default executor selection.
  virtual std::string executor_type() const { return {}; }

  // tokens: [num_tokens]
  // positions: [num_tokens]
  // returns: [num_tokens, hidden_size]
  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelParamsView& parameters) = 0;

  virtual bool requires_graph_forward_metadata() { return false; }

  virtual bool is_hybrid_linear_attention() { return false; }

  virtual bool supports_mla_graph_kv_bucketing() const { return false; }

  virtual std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return nullptr;
  }

  virtual void prepare_graph_forward_metadata(ModelGraphMetadataState*,
                                              const torch::Tensor&,
                                              ModelParamsView&) {}

  // Multimodal entry points, present only on VLM families.
  virtual MMDict encode(const ModelParamsView& /*parameters*/) {
    LOG(FATAL) << "encode is only available on multimodal models.";
    return {};
  }

  virtual torch::Tensor get_input_embeddings(
      const torch::Tensor& /*input_ids*/,
      const ModelParamsView& /*input_params*/) {
    LOG(FATAL) << "get_input_embeddings is only available on multimodal "
               << "models.";
    return {};
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_seqs, hidden_size]
  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    if (seleted_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, seleted_idxes);
    }
    return hidden_states;
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) = 0;

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // out_hidden: [num_selected_tokens, hidden_size]
  // returns: [num_selected_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes,
                               torch::Tensor& out_hidden) {
    if (seleted_idxes.defined()) {
      const auto idxes = seleted_idxes.to(
          torch::dtype(torch::kLong).device(hidden_states.device()));
      out_hidden = hidden_states.index_select(/*dim=*/0, idxes);
    } else {
      out_hidden = hidden_states;
    }
    return logits(hidden_states, seleted_idxes);
  }

  virtual bool share_weights_from(CausalLM& /*source*/) { return false; }

  virtual void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) {}

  virtual void start_expert_weight_transfer(int32_t /*layer_id*/) {}

  virtual void update_expert_weight(int32_t /*layer_id*/) {}

  // Returns whether the last prepare_expert_weight() call for the given layer
  // succeeded. Default is true so models that do not fail-report keep the old
  // behavior; DSV4 (npu_torch) overrides this to expose real prepare status
  // so EplbExecutor does not report a failed preparation as complete.
  virtual bool last_prepare_expert_weight_ok(int32_t /*layer_id*/) const {
    return true;
  }

  // DFlash-specific interface. Attention-free scatter-only pass that projects
  // target hidden into the draft's per-layer KV cache. Not part of forward()
  // because it has no attention and its shape doesn't match forward's decode
  // graph; sits outside the executor to avoid a per-flag branch in the eager
  // gate. Default: NOT_IMPLEMENTED. Overridden by DFlashDraftModel.
  virtual ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                                       const torch::Tensor& positions,
                                       const torch::Tensor& device_cache_slots,
                                       std::vector<KVCache>& kv_caches,
                                       const ModelParamsView& input_params) {
    NOT_IMPLEMENTED();
    return {};
  }

  virtual DFlash2CandidateOutput dflash2_candidates(
      const torch::Tensor& hidden_states,
      const torch::Tensor& unary_logits,
      const torch::Tensor& anchor_token_ids) {
    NOT_IMPLEMENTED();
    return {};
  }

  // DSpark-specific low-rank Markov projection. The draft worker owns the
  // sequential sampling lifecycle; the model owns only the trained weights and
  // bias computation.
  virtual torch::Tensor dspark_markov_bias(
      const torch::Tensor& previous_token_ids) {
    NOT_IMPLEMENTED();
    return {};
  }
};

template <typename ModelType>
class CausalLMImpl : public CausalLM {
 public:
  CausalLMImpl(ModelType model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_requires_graph_forward_metadata,
                       requires_graph_forward_metadata,
                       bool,
                       (),
                       ())
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_is_hybrid_linear_attention,
                       is_hybrid_linear_attention,
                       bool,
                       (),
                       ())
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_supports_mla_graph_kv_bucketing,
                       supports_mla_graph_kv_bucketing,
                       bool,
                       () const,
                       ())
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_create_graph_forward_metadata_state,
                       create_graph_forward_metadata_state,
                       std::unique_ptr<ModelGraphMetadataState>,
                       (),
                       ())
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_prepare_graph_forward_metadata,
                       prepare_graph_forward_metadata,
                       void,
                       (ModelGraphMetadataState * state,
                        const torch::Tensor& positions,
                        ModelParamsView& params),
                       (state, positions, params))
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_pooler,
                       pooler,
                       torch::Tensor,
                       (const torch::Tensor& hidden_states,
                        const torch::Tensor& seleted_idxes),
                       (hidden_states, seleted_idxes))

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->logits(hidden_states, seleted_idxes);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes,
                       torch::Tensor& out_hidden) override {
    if constexpr (detail::has_logits_with_hidden<ModelType>::value) {
      return model_->logits(hidden_states, seleted_idxes, out_hidden);
    } else {
      return CausalLM::logits(hidden_states, seleted_idxes, out_hidden);
    }
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  FORWARD_IF_SUPPORTED(
      ModelType,
      CausalLM,
      detail::has_write_context_kv,
      write_context_kv,
      ModelOutput,
      (const torch::Tensor& target_hidden,
       const torch::Tensor& positions,
       const torch::Tensor& device_cache_slots,
       std::vector<KVCache>& kv_caches,
       const ModelParamsView& input_params),
      (target_hidden, positions, device_cache_slots, kv_caches, input_params))
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_dflash2_candidates,
                       dflash2_candidates,
                       DFlash2CandidateOutput,
                       (const torch::Tensor& hidden_states,
                        const torch::Tensor& unary_logits,
                        const torch::Tensor& anchor_token_ids),
                       (hidden_states, unary_logits, anchor_token_ids))
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_dspark_markov_bias,
                       dspark_markov_bias,
                       torch::Tensor,
                       (const torch::Tensor& previous_token_ids),
                       (previous_token_ids))

  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_lazy_load_model,
                       lazy_load_model,
                       void,
                       (std::unique_ptr<ModelLoader> loader),
                       (std::move(loader)))
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_free_model_weights,
                       free_model_weights,
                       void,
                       (),
                       ())
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_reload_model_weights,
                       reload_model_weights,
                       void,
                       (),
                       ())
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_reload_model_weights_from_device,
                       reload_model_weights_from_device,
                       void,
                       (),
                       ())

  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_prepare_expert_weight,
                       prepare_expert_weight,
                       void,
                       (int32_t layer_id,
                        const std::vector<int32_t>& expert_ids),
                       (layer_id, expert_ids))
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_start_expert_weight_transfer,
                       start_expert_weight_transfer,
                       void,
                       (int32_t layer_id),
                       (layer_id))
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_update_expert_weight,
                       update_expert_weight,
                       void,
                       (int32_t layer_id),
                       (layer_id))
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_last_prepare_expert_weight_ok,
                       last_prepare_expert_weight_ok,
                       bool,
                       (int32_t layer_id) const,
                       (layer_id))

  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_get_lm_head,
                       get_lm_head,
                       layer::LmHead,
                       (),
                       ())
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_set_lm_head,
                       set_lm_head,
                       void,
                       (layer::LmHead & head),
                       (head))

  // The interface and backbone hook names differ, so the macro can't be used.
  MMDict encode(const ModelParamsView& parameters) override {
    if constexpr (detail::has_get_multimodal_embeddings<ModelType>::value) {
      return model_->get_multimodal_embeddings(parameters);
    }
    return CausalLM::encode(parameters);
  }

  torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelParamsView& input_params) override {
    if constexpr (detail::has_vlm_get_input_embeddings<ModelType>::value) {
      return model_->get_input_embeddings(input_ids, input_params);
    }
    return CausalLM::get_input_embeddings(input_ids, input_params);
  }

  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_get_word_embedding,
                       get_word_embedding,
                       layer::WordEmbedding,
                       (),
                       ())
  FORWARD_IF_SUPPORTED(ModelType,
                       CausalLM,
                       detail::has_set_word_embedding,
                       set_word_embedding,
                       void,
                       (layer::WordEmbedding & embedding),
                       (embedding))

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  ModelType model_;

  torch::TensorOptions options_;
};

}  // namespace xllm
