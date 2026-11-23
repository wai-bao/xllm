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

#include "core/framework/model/model_output.h"
#include "core/framework/multimodal/mposition.h"
#include "core/framework/multimodal/multimodal_utils.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/qwen2_decoder_layer.h"
#include "core/layers/qwen2_vision_layer.h"
#include "core/util/tensor_helper.h"
#include "models/llm/qwen2.h"
#include "models/model_registry.h"
#include "processors/multimodal_processor_pipeline.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/qwen2_vl_prompt_processor.h"
#include "processors/qwen2_vl_video_processor.h"

namespace xllm {

class Qwen2_VisionPatchMergerImpl : public torch::nn::Module {
 public:
  Qwen2_VisionPatchMergerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    int64_t d_model = model_args.mm_projection_dim();  // out_hidden_size
    int64_t context_dim = model_args.mm_hidden_size();
    int64_t spatial_merge_size = model_args.mm_spatial_merge_size();

    hidden_size_ =
        context_dim * static_cast<int>(std::pow(spatial_merge_size, 2));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({context_dim})
                                 .elementwise_affine(true)
                                 .eps(1e-6)));
    norm_->weight.set_data(norm_->weight.to(options));
    norm_->bias.set_data(norm_->bias.to(options));
    mlp_ = register_module(
        /*name=*/"mlp",
        layer::build_vision_merger_mlp(hidden_size_, d_model, options));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = norm_(x).view({-1, hidden_size_});
    return mlp_->forward(x);
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto& norm_dict = state_dict.get_dict_with_prefix("ln_q.");
    const auto& norm_weight = norm_dict.get_tensor("weight");
    if (norm_weight.defined()) {
      CHECK_EQ(norm_->weight.sizes(), norm_weight.sizes())
          << "weight size mismatch for " << name();
      norm_->weight.data().copy_(norm_weight);
    }
    const auto norm_bias = norm_dict.get_tensor("bias");
    if (norm_bias.defined()) {
      CHECK_EQ(norm_->bias.sizes(), norm_bias.sizes())
          << "bias size mismatch for " << name();
      norm_->bias.data().copy_(norm_bias);
    }
    layer::load_vision_merger_mlp(state_dict, mlp_, name());
  }

 private:
  int hidden_size_;
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Sequential mlp_{nullptr};
};
TORCH_MODULE(Qwen2_VisionPatchMerger);

class Qwen2_VisionTransformerImpl : public torch::nn::Module {
 public:
  Qwen2_VisionTransformerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();

    patch_embed_ = register_module(/*name=*/"patch_embed",
                                   layer::Qwen2_VisionPatchEmbed(context));
    rotary_pos_emb_ = register_module(
        /*name=*/"rotary_pos_emb", layer::Qwen2_VisionRotaryEmbedding(context));
    blocks_ = register_module("blocks", torch::nn::ModuleList());

    for (int32_t idx = 0; idx < model_args.mm_num_hidden_layers(); idx++) {
      auto block = layer::Qwen2_VisionLayer(context);
      blocks_->push_back(block);
      layers_.push_back(block);
    }
    merger_ = register_module("merger", Qwen2_VisionPatchMerger(context));
  }

  torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor grid_thw) {
    hidden_states = patch_embed_(hidden_states);
    auto rotary_pos_emb = std::get<0>(rotary_pos_emb_(grid_thw));

    auto cu_seqlens = layer::build_vision_cu_seqlens(grid_thw);

    auto m_cos = rotary_pos_emb.cos().type_as(hidden_states);
    m_cos = m_cos.repeat({1, 2});
    auto m_sin = rotary_pos_emb.sin().type_as(hidden_states);
    m_sin = m_sin.repeat({1, 2});
    auto cu_seqlens_vec = tensor_to_vector<int32_t>(cu_seqlens);
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      hidden_states = layers_[idx](
          hidden_states, m_cos, m_sin, cu_seqlens, cu_seqlens_vec, idx);
    }
    return merger_(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(
        state_dict.get_dict_with_prefix("patch_embed."));
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      layers_[idx]->load_state_dict(state_dict.get_dict_with_prefix(
          "blocks." + std::to_string(idx) + "."));
    }

    merger_->load_state_dict(state_dict.get_dict_with_prefix("merger."));
  }

 private:
  layer::Qwen2_VisionPatchEmbed patch_embed_{nullptr};
  layer::Qwen2_VisionRotaryEmbedding rotary_pos_emb_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<layer::Qwen2_VisionLayer> layers_;
  Qwen2_VisionPatchMerger merger_{nullptr};
};
TORCH_MODULE(Qwen2_VisionTransformer);

class Qwen2_VLForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Qwen2_VLForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", Qwen2_VisionTransformer(context));

    language_model_ =
        register_module("language_model", QWen2ForCausalLM(context));
  }

  MMDict get_multimodal_embeddings(const ModelParamsView& input_params) {
    const auto& mm_data = input_params.multimodal().mm_data;
    auto pixel_values = mm_data.get<torch::Tensor>(/*key=*/"pixel_values")
                            .value_or(torch::Tensor());
    auto image_grid_thw = mm_data.get<torch::Tensor>(/*key=*/"image_grid_thw")
                              .value_or(torch::Tensor());
    if (!pixel_values.defined() || !image_grid_thw.defined()) {
      return {};
    }

    auto image_embeds = visual_(pixel_values.to(options_), image_grid_thw);
    const int32_t merge_size = model_args_.mm_image_merge_size();
    MMDict multimodal_embeds;
    multimodal_embeds["image|embedding"] =
        split_vision_embeddings(image_embeds, image_grid_thw, merge_size);
    return multimodal_embeds;
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelParamsView& input_params) {
    const auto& mm_data = input_params.multimodal().mm_data;
    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    merge_vision_embeddings(mm_data, inputs_embeds);
    return inputs_embeds;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& input_params) {
    return language_model_->forward(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    namespace F = torch::nn::functional;
    return F::normalize(h, F::NormalizeFuncOptions().p(2).dim(1));
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(state_dict->get_dict_with_prefix("visual."));
    }
    if (!model_args_.encoder_embedding_mode()) {
      language_model_->load_model(std::move(loader));
    }
  }

  layer::LmHead get_lm_head() { return language_model_->get_lm_head(); }
  layer::WordEmbedding get_word_embedding() {
    return language_model_->get_word_embedding();
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;

  Qwen2_VisionTransformer visual_{nullptr};
  QWen2ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen2_VLForConditionalGeneration);

using Qwen2VLMultimodalProcessor = MultimodalProcessor<Qwen2VLPromptProcessor,
                                                       Qwen2VLImageProcessor,
                                                       Qwen2VLVideoProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(qwen2_vl, Qwen2VLMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen2_vl, Qwen2_VLForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(qwen2_vl, QwenVLMPositionGenerator);

REGISTER_MODEL_ARGS(qwen2_vl, [&] {
  // text config
  // LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_token_id, "vision_token_id", 151654);
  LOAD_ARG_OR(image_token_id, "image_token_id", 151655);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  // LOAD_ARG_OR(initializer_range, "initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);
  LOAD_ARG_OR(model_type, "model_type", "qwen2_vl");
  LOAD_ARG_OR(attention_bias,
              (std::vector<std::string>{"text_config.attention_bias",
                                        "attention_bias"}),
              true);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(sliding_window, "sliding_window", 32768);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  // LOAD_ARG_OR(transformers_version, "transformers_version", "4.41.2");
  // LOAD_ARG_OR(use_cache, "use_cache", true);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  // vision_config
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 32);
  // LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.embed_dim", 1280);
  // LOAD_ARG_OR(mm_mlp_ratio, "vision_config.mlp_ratio", 4);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.hidden_size", 3584);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_spatial_patch_size, "vision_config.spatial_patch_size", 14);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  LOAD_ARG_OR(rope_scaling_rope_type, "rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section, "rope_scaling.mrope_section");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
});
}  // namespace xllm
