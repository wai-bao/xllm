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

#include "core/runtime/vlm_executor_impl.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "core/framework/multimodal/mm_visitor.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace {

bool is_single_token_decode(const BatchInputMeta& meta) {
  return meta.batch_forward_type.is_decode() && meta.q_max_seq_len <= 1;
}

torch::Tensor single_axis_positions(const torch::Tensor& positions) {
  return positions.dim() == 2
             ? positions.select(/*dim=*/0, /*index=*/0).contiguous()
             : positions;
}

}  // namespace

VlmExecutorImpl::VlmExecutorImpl(CausalLM* model,
                                 const ModelArgs& args,
                                 const torch::Device& device,
                                 const runtime::Options& options)
    : model_(dynamic_cast<CausalVLM*>(model)),
      device_(device),
      enable_graph_(options.enable_graph()) {
  if (const auto cache_size_mb = options.max_encoder_cache_size();
      cache_size_mb > 0) {
    encoder_cache_ =
        std::make_unique<EncoderCache>(cache_size_mb * 1024 * 1024);
  }

  if (enable_graph_ || options.enable_prefill_piecewise_graph()) {
    llm_executor_ = ExecutorImplFactory::get_instance().create_executor_impl(
        model, args, device, options, Platform::type_str());
  }
}

MMDict VlmExecutorImpl::encode(const MMBatchData& mm_data) {
  return model_->encode(mm_data);
}

void VlmExecutorImpl::clear_multimodal_cache() {
  if (encoder_cache_ != nullptr) {
    encoder_cache_->clear();
  }
  if (llm_executor_ != nullptr) {
    llm_executor_->clear_multimodal_cache();
  }
}

bool VlmExecutorImpl::supports_multimodal_input() const { return true; }

ModelOutput VlmExecutorImpl::run(const torch::Tensor& tokens,
                                 const torch::Tensor& positions,
                                 std::vector<KVCache>& kv_caches,
                                 const ModelParamsView& params,
                                 MMBatchData* mm_data) {
  torch::NoGradGuard no_grad;
  const bool single_token_decode = is_single_token_decode(params.meta);
  const bool use_graph_executor =
      llm_executor_ &&
      (enable_graph_ || params.meta.batch_forward_type.is_prefill());
  if (mm_data != nullptr) {
    auto [input_embedding, deep_stacks] =
        prepare_inputs_embeds(tokens, params, *mm_data);
    params.embedding.input_embedding = std::move(input_embedding);
    params.embedding.deep_stacks_ = std::move(deep_stacks);
  } else if (use_graph_executor && !single_token_decode &&
             !params.embedding.input_embedding.defined()) {
    // Text and media prefills share the graph's explicit embedding buffer.
    params.embedding.input_embedding =
        model_->get_input_embeddings(tokens, MMBatchData{}).first;
  }

  const auto execution_positions = mm_data == nullptr && single_token_decode
                                       ? single_axis_positions(positions)
                                       : positions;
  if (use_graph_executor) {
    return llm_executor_->run(tokens, execution_positions, kv_caches, params);
  }
  return model_->forward(tokens, execution_positions, kv_caches, params);
}

MultimodalEmbeddings VlmExecutorImpl::prepare_inputs_embeds(
    const torch::Tensor& tokens,
    const ModelParamsView& params,
    MMBatchData& mm_data) {
  execute_mm_encoder(
      mm_data, encoder_cache_.get(), [this](const MMBatchData& encoder_inputs) {
        return encode(encoder_inputs);
      });

  EncoderEmbeddingGatherVisitor gather(device_,
                                       mm_data.type(),
                                       params.attention.host.kv_seq_lens,
                                       params.attention.host.q_seq_lens);
  CHECK(mm_data.foreach (gather));
  CHECK(gather.finish(mm_data));

  return model_->get_input_embeddings(tokens, mm_data);
}

void VlmExecutorImpl::prepare_graph_input(const torch::Tensor& tokens,
                                          const torch::Tensor& positions,
                                          std::vector<KVCache>& kv_caches,
                                          const ModelParamsView& params) {
  if (llm_executor_ && enable_graph_) {
    auto execution_positions = is_single_token_decode(params.meta)
                                   ? single_axis_positions(positions)
                                   : positions;
    llm_executor_->prepare_graph_input(
        tokens, execution_positions, kv_caches, params);
  }
}

void VlmExecutorImpl::set_dp_process_group(ProcessGroup* group) {
  if (llm_executor_) {
    llm_executor_->set_dp_process_group(group);
  }
}

}  // namespace xllm
