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

#include "core/framework/config/kernel_config.h"
#include "core/framework/model/model_output.h"
#include "core/framework/multimodal/mm_data_item.h"
#include "core/layers/common/lm_head.h"
#include "models/model_registry.h"

namespace xllm {

struct Qwen3_VLImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

struct Qwen3_VLVideoInputs {
  torch::Tensor pixel_values_videos;
  torch::Tensor video_grid_thw;
};

template <typename VisionTransformer, typename LanguageModel>
class Qwen3VLForConditionalGenerationBase : public torch::nn::Module {
 public:
  Qwen3VLForConditionalGenerationBase(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", VisionTransformer(context));
    language_model_ = register_module("language_model", LanguageModel(context));
  }

  virtual ~Qwen3VLForConditionalGenerationBase() = default;

  void prepare_encoder_input(const MMBatchData& mm_data,
                             std::optional<Qwen3_VLImageInputs>& image_inputs,
                             std::optional<Qwen3_VLVideoInputs>& video_inputs) {
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values"))
      pixel_values = res.value();

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor pixel_values_videos;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values_videos"))
      pixel_values_videos = res.value();

    torch::Tensor video_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = Qwen3_VLImageInputs{pixel_values, image_grid_thw};

    if (pixel_values_videos.defined() && video_grid_thw.defined())
      video_inputs = Qwen3_VLVideoInputs{pixel_values_videos, video_grid_thw};
  }

  MMDict get_multimodal_embeddings(const MMBatchData& mm_data) {
    std::optional<Qwen3_VLImageInputs> image_input;
    std::optional<Qwen3_VLVideoInputs> video_input;
    prepare_encoder_input(mm_data, image_input, video_input);

    MMDict multimodal_embeds;
    auto merge_size = model_args_.mm_image_merge_size();
    if (image_input) {
      auto image_embeds =
          visual_(image_input->pixel_values.to(options_),
                  image_input->image_grid_thw.to(options_.device()));

      auto image_tokens =
          (image_input->image_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);

      std::vector<int64_t> image_tokens_vec(
          image_tokens.data_ptr<int64_t>(),
          image_tokens.data_ptr<int64_t>() + image_tokens.numel());
      multimodal_embeds["image|embedding"] =
          image_embeds.split(image_tokens_vec, /*dim=*/0);
    }
    if (video_input) {
      auto video_embeds =
          visual_(video_input->pixel_values_videos.to(options_),
                  video_input->video_grid_thw.to(options_.device()));

      auto video_tokens =
          (video_input->video_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);

      std::vector<int64_t> video_tokens_vec(
          video_tokens.data_ptr<int64_t>(),
          video_tokens.data_ptr<int64_t>() + video_tokens.numel());
      multimodal_embeds["video|embedding"] =
          video_embeds.split(video_tokens_vec, /*dim=*/0);
    }
    return multimodal_embeds;
  }

  MultimodalEmbeddings get_input_embeddings(const torch::Tensor& input_ids,
                                            const MMBatchData& mm_data) {
    return mm_data.merge_visual_embeddings(
        language_model_->get_input_embeddings(input_ids),
        model_args_.mm_deepstack_visual_indexes().size());
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& input_params) {
    return language_model_->forward(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->pooler(hidden_states, seleted_idxes);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(
          state_dict->get_dict_with_prefix("model.visual."));
    }

#if defined(USE_NPU)
    if constexpr (requires {
                    visual_->verify_loaded_weights("model.visual.");
                    visual_->merge_loaded_weights();
                  }) {
      visual_->verify_loaded_weights("model.visual.");
      visual_->merge_loaded_weights();
    }
#endif

    if (!model_args_.encoder_embedding_mode()) {
      language_model_->load_model(std::move(loader), "model.language_model.");
    }
  }

  bool is_hybrid_linear_attention() {
    if constexpr (detail::has_is_hybrid_linear_attention<
                      LanguageModel>::value) {
      return language_model_->is_hybrid_linear_attention();
    }
    return false;
  }

  layer::LmHead get_lm_head() { return language_model_->get_lm_head(); }
  layer::WordEmbedding get_word_embedding() {
    return language_model_->get_word_embedding();
  }

 protected:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  VisionTransformer visual_{nullptr};
  LanguageModel language_model_{nullptr};
};

}  // namespace xllm
