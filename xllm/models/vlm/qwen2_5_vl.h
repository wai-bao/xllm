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

#include <algorithm>

#include "core/framework/config/model_config.h"
#include "core/framework/model/model_output.h"
#include "core/framework/multimodal/mposition.h"
#include "core/framework/multimodal/multimodal_utils.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/qwen2_5_vision_layer.h"
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

class Qwen2_5_VisionPatchMergerImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionPatchMergerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    int64_t d_model = model_args.mm_projection_dim();  // out_hidden_size
    int context_dim = model_args.mm_hidden_size();
    int spatial_merge_size = model_args.mm_spatial_merge_size();

    hidden_size_ =
        context_dim * static_cast<int>(std::pow(spatial_merge_size, 2));

    ln_q_ = register_module(
        "ln_q",
        layer::RMSNorm(context_dim, model_args.rms_norm_eps(), options));

    mlp_ = register_module(
        /*name=*/"mlp",
        layer::build_vision_merger_mlp(hidden_size_, d_model, options));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = std::get<0>(ln_q_(x));
    x = x.view({-1, hidden_size_});
    return mlp_->forward(x);
  }

  void load_state_dict(const StateDict& state_dict) {
    ln_q_->load_state_dict(state_dict.get_dict_with_prefix("ln_q."));

    layer::load_vision_merger_mlp(state_dict, mlp_, name());
  }

 private:
  int64_t hidden_size_;

  layer::RMSNorm ln_q_{nullptr};
  torch::nn::Sequential mlp_{nullptr};
};
TORCH_MODULE(Qwen2_5_VisionPatchMerger);

class Qwen2_5_VisionTransformerImpl : public torch::nn::Module {
 public:
  Qwen2_5_VisionTransformerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();

    window_size_ = model_args.mm_window_size();
    patch_size_ = model_args.mm_patch_size();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    const auto& block_indexes = model_args.mm_fullatt_block_indexes();
    fullatt_block_indexes_.insert(block_indexes.begin(), block_indexes.end());
    spatial_merge_unit_ = static_cast<int>(std::pow(spatial_merge_size_, 2));

    patch_embed_ = register_module(/*name=*/"patch_embed",
                                   layer::Qwen2_VisionPatchEmbed(context));
    rotary_pos_emb_ = register_module(
        /*name=*/"rotary_pos_emb", layer::Qwen2_VisionRotaryEmbedding(context));
    blocks_ = register_module("blocks", torch::nn::ModuleList());

    for (int32_t idx = 0; idx < model_args.mm_num_hidden_layers(); idx++) {
      auto block = layer::Qwen2_5_VisionLayer(context);
      blocks_->push_back(block);
      layers_.push_back(block);
    }
    merger_ = register_module("merger", Qwen2_5_VisionPatchMerger(context));
  }

  torch::Tensor get_window_index(torch::Tensor grid_thw,
                                 std::vector<int>& cu_window_seqlens) {
    auto count = grid_thw.sizes()[0];
    std::vector<torch::Tensor> window_index;
    window_index.reserve(count);
    cu_window_seqlens.reserve(count * 128);
    cu_window_seqlens.emplace_back(0);

    int window_index_id = 0;
    int vit_merger_window_size =
        window_size_ / spatial_merge_size_ / patch_size_;

    auto grid_thw_cpu = grid_thw.cpu();
    auto options =
        torch::TensorOptions().dtype(torch::kLong).device(grid_thw.device());

    for (int idx = 0; idx < count; ++idx) {
      auto grid_t = grid_thw_cpu[idx][0].item<int64_t>();
      auto grid_h = grid_thw_cpu[idx][1].item<int64_t>();
      auto grid_w = grid_thw_cpu[idx][2].item<int64_t>();

      auto llm_grid_h = grid_h / spatial_merge_size_;
      auto llm_grid_w = grid_w / spatial_merge_size_;

      auto index = torch::arange(grid_t * llm_grid_h * llm_grid_w, options)
                       .reshape({grid_t, llm_grid_h, llm_grid_w});
      auto pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
      auto pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;

      auto num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
      auto num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

      namespace F = torch::nn::functional;
      auto index_padded = F::pad(index,
                                 F::PadFuncOptions({0, pad_w, 0, pad_h})
                                     .mode(torch::kConstant)
                                     .value(-100));
      index_padded = index_padded.reshape({grid_t,
                                           num_windows_h,
                                           vit_merger_window_size,
                                           num_windows_w,
                                           vit_merger_window_size});

      index_padded = index_padded.permute({0, 1, 3, 2, 4})
                         .reshape({grid_t,
                                   num_windows_h * num_windows_w,
                                   vit_merger_window_size,
                                   vit_merger_window_size});

      auto index_padded_ne = torch::ne(index_padded, -100);
      auto seqlens = index_padded_ne.sum({2, 3}).reshape({-1});
      index_padded = index_padded.reshape({-1});
      auto index_new =
          index_padded.masked_select(index_padded_ne.reshape({-1}));

      window_index.push_back(index_new + window_index_id);
      auto cu_seqlens_tmp =
          (seqlens.cumsum(0, torch::kInt32) * spatial_merge_unit_ +
           cu_window_seqlens.back())
              .cpu();
      cu_window_seqlens.insert(
          cu_window_seqlens.end(),
          cu_seqlens_tmp.data_ptr<int>(),
          cu_seqlens_tmp.data_ptr<int>() + cu_seqlens_tmp.numel());
      window_index_id += grid_t * llm_grid_h * llm_grid_w;
    }

    return torch::cat(window_index, 0);
  }

  torch::Tensor forward(torch::Tensor hidden_states,
                        torch::Tensor grid_thw) {  // [batch,thw]
    hidden_states = patch_embed_(hidden_states);
    auto rotary_pos_emb = std::get<0>(rotary_pos_emb_(grid_thw));

    // windows attention
    std::vector<int> cu_window_seqlens_vec;
    auto window_index = get_window_index(grid_thw, cu_window_seqlens_vec);
    torch::TensorOptions options = torch::TensorOptions()
                                       .dtype(torch::kInt32)
                                       .device(hidden_states.device());
    cu_window_seqlens_vec.erase(
        std::unique(cu_window_seqlens_vec.begin(), cu_window_seqlens_vec.end()),
        cu_window_seqlens_vec.end());
    auto cu_window_seqlens = torch::tensor(cu_window_seqlens_vec, options);
    auto seq_len = hidden_states.sizes()[0];
    hidden_states = hidden_states.reshape(
        {seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    hidden_states = hidden_states.index(
        {window_index, torch::indexing::Slice(), torch::indexing::Slice()});
    hidden_states = hidden_states.reshape({seq_len, -1});

    rotary_pos_emb = rotary_pos_emb.reshape(
        {seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    rotary_pos_emb = rotary_pos_emb.index(
        {window_index, torch::indexing::Slice(), torch::indexing::Slice()});
    rotary_pos_emb = rotary_pos_emb.reshape({seq_len, -1});

    auto cu_seqlens = layer::build_vision_cu_seqlens(grid_thw);

    auto m_cos = rotary_pos_emb.cos().type_as(hidden_states).repeat({1, 2});
    auto m_sin = rotary_pos_emb.sin().type_as(hidden_states).repeat({1, 2});
    auto cu_seqlens_vec = tensor_to_vector<int32_t>(cu_seqlens);
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      const bool full_attention = fullatt_block_indexes_.contains(idx);
      hidden_states =
          layers_[idx](hidden_states,
                       m_cos,
                       m_sin,
                       full_attention ? cu_seqlens : cu_window_seqlens,
                       full_attention ? cu_seqlens_vec : cu_window_seqlens_vec,
                       idx);
    }
    // adapter
    hidden_states = merger_(hidden_states);

    auto reverse_indices = torch::argsort(window_index);
    hidden_states =
        hidden_states.index({reverse_indices, torch::indexing::Slice()});
    return hidden_states;
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
  int window_size_ = 0;
  int patch_size_ = 0;
  int spatial_merge_size_ = 0;
  std::unordered_set<int> fullatt_block_indexes_;
  int spatial_merge_unit_ = 0;

  layer::Qwen2_VisionPatchEmbed patch_embed_{nullptr};
  layer::Qwen2_VisionRotaryEmbedding rotary_pos_emb_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<layer::Qwen2_5_VisionLayer> layers_;
  Qwen2_5_VisionPatchMerger merger_{nullptr};
};
TORCH_MODULE(Qwen2_5_VisionTransformer);

class Qwen2_5_VLForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Qwen2_5_VLForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", Qwen2_5_VisionTransformer(context));

    language_model_ =
        register_module("language_model", QWen2ForCausalLM(context));
  }
  MMDict get_multimodal_embeddings(const ModelParamsView& input_params) {
    const auto& mm_data = input_params.multimodal().mm_data;
    const auto get_tensor = [&](const MMKey& key) {
      return mm_data.get<torch::Tensor>(key).value_or(torch::Tensor());
    };
    const auto pixel_values = get_tensor(/*key=*/"pixel_values");
    const auto image_grid_thw = get_tensor(/*key=*/"image_grid_thw");
    const auto pixel_values_videos = get_tensor(/*key=*/"pixel_values_videos");
    const auto video_grid_thw = get_tensor(/*key=*/"video_grid_thw");
    const auto second_per_grid_ts = get_tensor(/*key=*/"second_per_grid_ts");

    const int32_t merge_size = model_args_.mm_image_merge_size();
    MMDict multimodal_embeds;
    const auto add_embeddings = [&](const torch::Tensor& pixels,
                                    const torch::Tensor& grid_thw,
                                    const MMKey& embed_key) {
      if (!pixels.defined() || !grid_thw.defined()) {
        return;
      }
      auto embeds = visual_(pixels.to(options_), grid_thw);
      multimodal_embeds[embed_key] =
          split_vision_embeddings(embeds, grid_thw, merge_size);
    };
    add_embeddings(
        pixel_values, image_grid_thw, /*embed_key=*/"image|embedding");
    if (second_per_grid_ts.defined()) {
      add_embeddings(pixel_values_videos,
                     video_grid_thw,
                     /*embed_key=*/"video|embedding");
    }
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
    // return full embeddings if set flag
    if (::xllm::ModelConfig::get_instance()
            .enable_return_mm_full_embeddings()) {
      return h;
    }

    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    auto pooler_output = torch::nn::functional::normalize(
        h, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    return pooler_output;
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(state_dict->get_dict_with_prefix(
          std::vector<std::string>{"visual.", "model.visual."}));
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

  Qwen2_5_VisionTransformer visual_{nullptr};
  QWen2ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen2_5_VLForConditionalGeneration);

using Qwen25VLMultimodalProcessor = MultimodalProcessor<Qwen2VLPromptProcessor,
                                                        Qwen2VLImageProcessor,
                                                        Qwen2VLVideoProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(qwen2_5_vl, Qwen25VLMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen2_5_vl, Qwen2_5_VLForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(qwen2_5_vl, QwenVLMPositionGenerator);

// Macro to load Qwen2.5-VL model arguments (shared between qwen2_5_vl and
// Qwen2_5_VLForConditionalGeneration)
#define LOAD_QWEN2_5_VL_MODEL_ARGS()                                           \
  do {                                                                         \
    /* text config */                                                          \
    /* LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0); */            \
    LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);                         \
    LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);                         \
    LOAD_ARG_OR(pad_token_id, "pad_token_id", 151643);                         \
    LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);       \
    LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);           \
    LOAD_ARG_OR(vision_token_id, "vision_token_id", 151654);                   \
    LOAD_ARG_OR(image_token_id, "image_token_id", 151655);                     \
    LOAD_ARG_OR(video_token_id, "video_token_id", 151656);                     \
    LOAD_ARG_OR(hidden_act, "hidden_act", "silu");                             \
    LOAD_ARG_OR(hidden_size, "hidden_size", 3584);                             \
    /* LOAD_ARG_OR(initializer_range, "initializer_range", 0.02); */           \
    LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);                \
    LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 128000);   \
    LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);                   \
    LOAD_ARG_OR(model_type, "model_type", "qwen2_5_vl");                       \
    LOAD_ARG_OR(attention_bias,                                                \
                (std::vector<std::string>{"text_config.attention_bias",        \
                                          "attention_bias"}),                  \
                true);                                                         \
    LOAD_ARG_OR(n_heads, "num_attention_heads", 28);                           \
    LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);                            \
    LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);                         \
    LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-06);                          \
    LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);                         \
    LOAD_ARG_OR(sliding_window, "sliding_window", 32768);                      \
    LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);            \
    LOAD_ARG_OR(dtype, "torch_dtype", "");                                     \
    /* LOAD_ARG_OR(transformers_version, "transformers_version", "4.41.2"); */ \
    /* LOAD_ARG_OR(use_cache, "use_cache", true); */                           \
    LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);              \
    LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {                               \
      return args->hidden_size() / args->n_heads();                            \
    });                                                                        \
                                                                               \
    /* vision_config */                                                        \
    LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 32);              \
    LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");            \
    LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1280);            \
    LOAD_ARG_OR(                                                               \
        mm_intermediate_size, "vision_config.intermediate_size", 3420);        \
    LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);        \
    LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);                 \
    LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 3584);     \
    LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);                \
    LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2); \
    LOAD_ARG_OR_FUNC(mm_image_merge_size,                                      \
                     "vision_config.image_merge_size",                         \
                     [&] { return args->mm_spatial_merge_size(); });           \
    LOAD_ARG_OR(                                                               \
        mm_spatial_patch_size, "vision_config.spatial_patch_size", 14);        \
    LOAD_ARG_OR(mm_window_size, "vision_config.window_size", 112);             \
    LOAD_ARG_OR(mm_fullatt_block_indexes,                                      \
                "vision_config.fullatt_block_indexes",                         \
                std::vector<int64_t>({7, 15, 23, 31}));                        \
    LOAD_ARG_OR(mm_tokens_per_second, "vision_config.tokens_per_second", 2);   \
    LOAD_ARG_OR(                                                               \
        mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);       \
    LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {                            \
      return args->mm_hidden_size() / args->mm_num_attention_heads();          \
    });                                                                        \
                                                                               \
    LOAD_ARG_OR(rope_scaling_rope_type, "rope_scaling.type", "mrope");         \
    LOAD_ARG_OR(rope_scaling_mrope_section,                                    \
                "rope_scaling.mrope_section",                                  \
                std::vector<int64_t>({16, 24, 24}));                           \
    LOAD_ARG_OR(vocab_size, "vocab_size", 152064);                             \
    if (args->rope_scaling_rope_type() == "default") {                         \
      args->rope_scaling_rope_type() = "mrope";                                \
    }                                                                          \
  } while (0)

REGISTER_MODEL_ARGS(qwen2_5_vl, [&] { LOAD_QWEN2_5_VL_MODEL_ARGS(); });

REGISTER_MODEL_ARGS(Qwen2_5_VLForConditionalGeneration,
                    [&] { LOAD_QWEN2_5_VL_MODEL_ARGS(); });

#undef LOAD_QWEN2_5_VL_MODEL_ARGS

}  // namespace xllm
