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
#include <cmath>
#include <memory>
#include <vector>

#include "models/dit/autoencoders/autoencoder_kl.h"
#include "torch/torch.h"

namespace xllm {

class Flux2ImageProcessorImpl final : public VAEImageProcessorImpl {
 public:
  explicit Flux2ImageProcessorImpl(const ModelContext& context,
                                   int64_t vae_scale_factor = 16)
      : VAEImageProcessorImpl(context, true, true, false, true, false, 32),
        vae_scale_factor_(vae_scale_factor),
        vae_latent_channels_(32) {}

  torch::Tensor check_image_input(const torch::Tensor& image,
                                  int64_t max_aspect_ratio = 8,
                                  int64_t min_side_length = 64,
                                  int64_t max_area = 1024 * 1024) {
    if (image.dim() != 3 && image.dim() != 4) {
      LOG(FATAL) << "Image must be 3D (C, H, W) or 4D (B, C, H, W), got dim: "
                 << image.dim();
    }

    int64_t height = image.size(-2);
    int64_t width = image.size(-1);

    if (width < min_side_length || height < min_side_length) {
      LOG(FATAL) << "Image too small: " << width << "x" << height
                 << ". Both dimensions must be at least " << min_side_length
                 << "px";
    }

    float aspect_ratio = std::max(static_cast<float>(width) / height,
                                  static_cast<float>(height) / width);
    if (aspect_ratio > max_aspect_ratio) {
      LOG(FATAL) << "Aspect ratio too extreme: " << width << "x" << height
                 << " (ratio: " << aspect_ratio
                 << ":1). Maximum allowed ratio is " << max_aspect_ratio
                 << ":1";
    }

    return image;
  }

  torch::Tensor resize_to_target_area(const torch::Tensor& image,
                                      int64_t target_area = 1024 * 1024) {
    int64_t image_width = image.size(-1);
    int64_t image_height = image.size(-2);

    float scale = std::sqrt(static_cast<float>(target_area) /
                            static_cast<float>(image_width * image_height));
    int64_t width = static_cast<int64_t>(image_width * scale);
    int64_t height = static_cast<int64_t>(image_height * scale);
    return VAEImageProcessorImpl::resize(image, height, width);
  }

 private:
  int64_t vae_scale_factor_;
  int64_t vae_latent_channels_;
};
TORCH_MODULE(Flux2ImageProcessor);
}  // namespace xllm
