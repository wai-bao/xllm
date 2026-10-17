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
#include <memory>
#include <type_traits>
#include <utility>

#include "core/framework/model/llm_model_params.h"
#include "core/framework/model/rec_model_params.h"
#include "core/framework/model/vlm_model_params.h"
namespace xllm {
struct SpecEmbeddingExecutionInput {
  std::vector<int32_t> extra_token_ids;
  torch::Tensor mtp_shifted_token_ids;
};
struct SpecGraphExecutionInput {
  bool use_expanded_decode_for_spec_verify_attention = false;
  torch::Tensor expanded_kv_seq_lens;
  torch::Tensor expanded_block_tables;
  torch::Tensor expanded_paged_kv_indptr;
  torch::Tensor expanded_paged_kv_indices;
  torch::Tensor expanded_paged_kv_last_page_len;
  torch::Tensor expanded_tiling_data;
  std::vector<int32_t> expanded_kv_seq_lens_vec;
  torch::Tensor input_tokens_override;
  std::vector<torch::Tensor> spec_verify_draft_token_sources;
  bool spec_verify_source_addresses_stable = false;
  bool spec_verify_static_graph_tasks_prepared = false;
};
// Rec shares execution infrastructure without owning speculative input fields.
struct SpecExecutionState {
  SpecEmbeddingExecutionInput embedding;
  SpecGraphExecutionInput graph;
  std::vector<torch::Tensor> multi_block_tables;
  torch::Tensor mtp_shifted_token_ids;
  bool is_spec_verify = false;
  torch::Tensor num_accepted_tokens;
  MtpTopkStatePtr mtp_topk_state;
  std::vector<int64_t> num_accepted_tokens_host;
};
namespace detail {
inline SpecExecutionState spec_execution_state_to(
    const SpecExecutionState& source,
    const torch::Device& device) {
  SpecExecutionState out = source;
  out.embedding.mtp_shifted_token_ids =
      safe_to(source.embedding.mtp_shifted_token_ids, device, true);
  out.graph.expanded_kv_seq_lens =
      safe_to(source.graph.expanded_kv_seq_lens, device, true);
  out.graph.expanded_block_tables =
      safe_to(source.graph.expanded_block_tables, device, true);
  out.graph.expanded_paged_kv_indptr =
      safe_to(source.graph.expanded_paged_kv_indptr, device, true);
  out.graph.expanded_paged_kv_indices =
      safe_to(source.graph.expanded_paged_kv_indices, device, true);
  out.graph.expanded_paged_kv_last_page_len =
      safe_to(source.graph.expanded_paged_kv_last_page_len, device, true);
  out.graph.expanded_tiling_data =
      safe_to(source.graph.expanded_tiling_data, device, true);
  out.graph.input_tokens_override =
      safe_to(source.graph.input_tokens_override, device, true);
  out.graph.spec_verify_draft_token_sources.clear();
  out.graph.spec_verify_draft_token_sources.reserve(
      source.graph.spec_verify_draft_token_sources.size());
  for (const auto& token : source.graph.spec_verify_draft_token_sources) {
    out.graph.spec_verify_draft_token_sources.emplace_back(
        safe_to(token, device, true));
  }
  out.multi_block_tables.clear();
  out.multi_block_tables.reserve(source.multi_block_tables.size());
  for (const auto& table : source.multi_block_tables) {
    out.multi_block_tables.emplace_back(
        safe_to(table, table.options().device(torch::kCPU), true));
  }
  out.mtp_shifted_token_ids =
      safe_to(source.mtp_shifted_token_ids, device, true);
  out.num_accepted_tokens = safe_to(source.num_accepted_tokens, device, true);
  out.mtp_topk_state = source.mtp_topk_state == nullptr
                           ? nullptr
                           : source.mtp_topk_state->to(device);
  return out;
}
}  // namespace detail

class EmbeddingInputView final {
 public:
  template <typename Common, typename Spec>
  EmbeddingInputView(Common& common, Spec& spec)
      : input_embedding(common.input_embedding),
        embedding_ids(common.embedding_ids),
        linear_state_ids(common.linear_state_ids),
        linear_state_indices(common.linear_state_indices),
        request_ids(common.request_ids),
        extra_token_ids(spec.extra_token_ids),
        mtp_shifted_token_ids(spec.mtp_shifted_token_ids) {}
  torch::Tensor& input_embedding;
  std::vector<int32_t>& embedding_ids;
  std::vector<int32_t>& linear_state_ids;
  torch::Tensor& linear_state_indices;
  std::vector<std::string>& request_ids;
  std::vector<int32_t>& extra_token_ids;
  torch::Tensor& mtp_shifted_token_ids;
};

class GraphInputView final {
 public:
  template <typename Common, typename Spec>
  GraphInputView(Common& common, Spec& spec)
      : attn_mask(common.attn_mask),
        tiling_data(common.tiling_data),
        use_expanded_decode_for_spec_verify_attention(
            spec.use_expanded_decode_for_spec_verify_attention),
        expanded_kv_seq_lens(spec.expanded_kv_seq_lens),
        expanded_block_tables(spec.expanded_block_tables),
        expanded_paged_kv_indptr(spec.expanded_paged_kv_indptr),
        expanded_paged_kv_indices(spec.expanded_paged_kv_indices),
        expanded_paged_kv_last_page_len(spec.expanded_paged_kv_last_page_len),
        expanded_tiling_data(spec.expanded_tiling_data),
        expanded_kv_seq_lens_vec(spec.expanded_kv_seq_lens_vec),
        input_tokens_override(spec.input_tokens_override),
        spec_verify_draft_token_sources(spec.spec_verify_draft_token_sources),
        spec_verify_source_addresses_stable(
            spec.spec_verify_source_addresses_stable),
        spec_verify_static_graph_tasks_prepared(
            spec.spec_verify_static_graph_tasks_prepared)
#if defined(USE_DCU)
        ,
        use_dense_flash_attention(common.use_dense_flash_attention)
#endif
#if defined(USE_NPU)
        ,
        acl_graph_task_update_context(common.acl_graph_task_update_context)
#endif
  {
  }
  torch::Tensor& attn_mask;
  torch::Tensor& tiling_data;
  bool& use_expanded_decode_for_spec_verify_attention;
  torch::Tensor& expanded_kv_seq_lens;
  torch::Tensor& expanded_block_tables;
  torch::Tensor& expanded_paged_kv_indptr;
  torch::Tensor& expanded_paged_kv_indices;
  torch::Tensor& expanded_paged_kv_last_page_len;
  torch::Tensor& expanded_tiling_data;
  std::vector<int32_t>& expanded_kv_seq_lens_vec;
  torch::Tensor& input_tokens_override;
  std::vector<torch::Tensor>& spec_verify_draft_token_sources;
  bool& spec_verify_source_addresses_stable;
  bool& spec_verify_static_graph_tasks_prepared;

#if defined(USE_DCU)
  bool& use_dense_flash_attention;
#endif
#if defined(USE_NPU)
  std::shared_ptr<npu::AclGraphTaskUpdateContext>&
      acl_graph_task_update_context;
#endif
};

class ModelInputParams;
class ModelInputSnapshotOwner {
 public:
  virtual ~ModelInputSnapshotOwner() = default;
  virtual ModelInputParams view() = 0;
  virtual std::shared_ptr<ModelInputSnapshotOwner> to(
      const torch::Device& device) const = 0;
};
class ModelInputSnapshot final {
 public:
  ModelInputSnapshot() = default;
  template <typename NativeParams>
  explicit ModelInputSnapshot(NativeParams params);
  ModelInputSnapshot(const ModelInputSnapshot&) = delete;
  ModelInputSnapshot& operator=(const ModelInputSnapshot&) = delete;
  ModelInputSnapshot(ModelInputSnapshot&&) = default;
  ModelInputSnapshot& operator=(ModelInputSnapshot&&) = default;
  // A returned view retains this explicit native snapshot and can outlive
  // the snapshot handle, including when returned from a graph helper.
  ModelInputParams view() const;
  ModelInputSnapshot to(const torch::Device& device) const;

 private:
  friend class ModelInputParams;
  explicit ModelInputSnapshot(std::shared_ptr<ModelInputSnapshotOwner> owner)
      : owner_(std::move(owner)) {}
  std::shared_ptr<ModelInputSnapshotOwner> owner_;
};
// Shallow borrowed execution view: construction and copying never copy native
// host vectors or multimodal payloads.
class ModelInputParams final {
 private:
  enum class Domain { LLM, VLM, REC };
  std::shared_ptr<SpecExecutionState> rec_execution_state_;
  std::shared_ptr<ModelInputSnapshotOwner> snapshot_owner_;
  void* native_owner_ = nullptr;
  Domain domain_;
  VlmVisionInput* multimodal_ = nullptr;
  RecFeatureInput* features_ = nullptr;
  RecModelInputParams* rec_params_ = nullptr;

 public:
  explicit ModelInputParams(LlmModelParams& owner)
      : ModelInputParams(owner, owner, Domain::LLM) {}
  explicit ModelInputParams(VlmModelParams& owner)
      : ModelInputParams(owner, owner, Domain::VLM) {
    multimodal_ = &owner.multimodal;
  }
  explicit ModelInputParams(RecModelParams& owner)
      : ModelInputParams(owner, std::make_shared<SpecExecutionState>()) {}
  ModelInputParams(RecModelParams& owner,
                   std::shared_ptr<SpecExecutionState> state)
      : rec_execution_state_(std::move(state)),
        native_owner_(&owner),
        domain_(Domain::REC),
        features_(&owner.features),
        rec_params_(&owner.rec_params),
        meta(owner.meta),
        attention(owner.attention),
        embedding(owner.embedding, rec_execution_state_->embedding),
        parallel(owner.parallel),
        block_copy(owner.block_copy),
        expert(owner.expert),
        graph(owner.graph, rec_execution_state_->graph),
        linear_state_cache_ops(owner.linear_state_cache_ops),
        linear_state_validity_mask(owner.linear_state_validity_mask),
        multi_block_tables(rec_execution_state_->multi_block_tables),
        mtp_shifted_token_ids(rec_execution_state_->mtp_shifted_token_ids),
        is_spec_verify(rec_execution_state_->is_spec_verify),
        prefill_without_cache(owner.prefill_without_cache),
        num_accepted_tokens(rec_execution_state_->num_accepted_tokens),
        mtp_topk_state(rec_execution_state_->mtp_topk_state),
        num_accepted_tokens_host(
            rec_execution_state_->num_accepted_tokens_host),
        attn_metadata(owner.attn_metadata),
        python_attention_metadata(owner.python_attention_metadata),
        enable_graph(owner.enable_graph) {}
  ModelInputParams(const ModelInputParams&) = default;
  ModelInputParams(ModelInputParams&&) = default;
  ModelInputParams& operator=(const ModelInputParams&) = delete;
  ModelInputParams& operator=(ModelInputParams&&) = delete;
  ModelInputSnapshot clone() const;
  ModelInputSnapshot to(const torch::Device& device) const;
  bool has_multimodal() const { return multimodal_ != nullptr; }
  VlmVisionInput& multimodal() const {
    CHECK(multimodal_ != nullptr) << "vision input requires VLM parameters";
    return *multimodal_;
  }
  bool has_features() const { return features_ != nullptr; }
  RecFeatureInput& features() const {
    CHECK(features_ != nullptr) << "feature input requires Rec parameters";
    return *features_;
  }
  bool has_rec_params() const { return rec_params_ != nullptr; }
  RecModelInputParams& rec_params() const {
    CHECK(rec_params_ != nullptr) << "strategy input requires Rec parameters";
    return *rec_params_;
  }
  void clear_linear_attention_state() {
    embedding.linear_state_ids.clear();
    embedding.linear_state_indices = torch::Tensor();
    linear_state_cache_ops.clear();
    linear_state_validity_mask.clear();
  }

  void print() const {
    LOG(INFO) << "ModelInputParams: batch_forward_type is "
              << meta.batch_forward_type.to_string() << " , num_sequences is "
              << meta.num_sequences << " , kv_max_seq_len is "
              << meta.kv_max_seq_len << " , q_max_seq_len is "
              << meta.q_max_seq_len;
    LOG(INFO) << "ModelInputParams: attention.host.kv_seq_lens is "
              << attention.host.kv_seq_lens;
    LOG(INFO) << "ModelInputParams: attention.host.q_seq_lens is "
              << attention.host.q_seq_lens;
    LOG(INFO) << "ModelInputParams: batch_forward_type is "
              << meta.batch_forward_type.to_string();
    print_tensor(
        attention.device.kv_seq_lens, "ModelInputParams: kv_seq_lens", 4);
    print_tensor(
        attention.device.q_seq_lens, "ModelInputParams: q_seq_lens", 4);
    print_tensor(
        attention.device.q_cu_seq_lens, "ModelInputParams: q_cu_seq_lens", 4);
    print_tensor(attention.device.new_cache_slots,
                 "ModelInputParams: new_cache_slots",
                 4);
    print_tensor(
        attention.device.block_tables, "ModelInputParams: block_tables", 4);
    LOG(INFO) << "ModelInputParams: dp_global_token_nums is "
              << parallel.dp_global_token_nums
              << ", dp_is_decode: " << parallel.dp_is_decode;
    LOG(INFO) << "ModelInputParams: is_spec_verify is " << is_spec_verify;
    print_tensor(num_accepted_tokens,
                 "ModelInputParams: num_accepted_tokens",
                 /*max_elements=*/4);

    if (const auto* onerec_xattn = onerec_xattention_params()) {
      LOG(INFO) << "ModelInputParams: has onerec_xattention_params";
      onerec_xattn->print();
    } else if (const auto* onerec = onerec_params()) {
      LOG(INFO) << "ModelInputParams: has onerec_params";
      onerec->print();
    } else if (const auto* llmrec = llmrec_params()) {
      LOG(INFO) << "ModelInputParams: has llm_rec_multi_round_params"
                << ", beam_width=" << llmrec->beam_width
                << ", total_round=" << llmrec->total_round;
    }
  }

  int32_t get_q_seq_len(int32_t seq_idx) const {
    CHECK_GE(seq_idx, 0) << "seq_idx out of range";
#if defined(USE_NPU)
    CHECK_LT(seq_idx, static_cast<int32_t>(attention.host.q_seq_lens.size()))
        << "seq_idx out of range";
    return attention.host.q_seq_lens[seq_idx];
#else
    CHECK_LT(seq_idx + 1,
             static_cast<int32_t>(attention.host.q_seq_lens.size()))
        << "seq_idx out of range";
    return attention.host.q_seq_lens[seq_idx + 1] -
           attention.host.q_seq_lens[seq_idx];
#endif
  }

  bool synchronize_layer(int64_t layer_idx) const {
    if (parallel.layer_wise_load_synchronizer == nullptr) {
      return true;
    }
    CHECK_GE(layer_idx, 0) << "Layer index must be non-negative.";
    if (static_cast<uint64_t>(layer_idx) % parallel.layers_per_event == 0) {
      return parallel.layer_wise_load_synchronizer->synchronize_layer(
          layer_idx / parallel.layers_per_event);
    }
    return true;
  }

  bool synchronize_draft_layer() const {
    if (parallel.layer_wise_load_synchronizer == nullptr) {
      return true;
    }
    if (!parallel.draft_load_event_index.has_value()) {
      return true;
    }
    return parallel.layer_wise_load_synchronizer->synchronize_layer(
        static_cast<int64_t>(*parallel.draft_load_event_index));
  }

  bool record_layer(uint32_t layer_idx, const torch::Device& device) const {
#if defined(USE_MLU) || defined(USE_DCU)
    if (parallel.layer_synchronizer != nullptr) {
      return parallel.layer_synchronizer->record_current(layer_idx,
                                                         device.index());
    }
#else
    (void)layer_idx;
    (void)device;
#endif
    return true;
  }

  BatchInputMeta& meta;
  AttentionInputView attention;
  EmbeddingInputView embedding;
  ParallelInput& parallel;
  BlockCopyInput& block_copy;
  ExpertInput& expert;
  GraphInputView graph;
  std::vector<LinearStateCacheOp>& linear_state_cache_ops;
  LinearStateValidityMask& linear_state_validity_mask;
  std::vector<torch::Tensor>& multi_block_tables;
  torch::Tensor& mtp_shifted_token_ids;
  bool& is_spec_verify;
  bool& prefill_without_cache;
  torch::Tensor& num_accepted_tokens;
  MtpTopkStatePtr& mtp_topk_state;
  std::vector<int64_t>& num_accepted_tokens_host;
  std::shared_ptr<layer::AttentionMetadata>& attn_metadata;
  std::shared_ptr<PythonAttentionMetadata>& python_attention_metadata;
  bool& enable_graph;
  const OneRecModelInputParams* onerec_params() const {
    if (rec_params_ == nullptr) {
      return nullptr;
    }
    if (const auto* params = std::get_if<OneRecModelInputParams>(rec_params_)) {
      return params;
    }
    if (const auto* params = std::get_if<OneRecXAttentionParams>(rec_params_)) {
      return static_cast<const OneRecModelInputParams*>(params);
    }
    return nullptr;
  }

  bool has_onerec_params() const { return onerec_params() != nullptr; }

  OneRecModelInputParams& mutable_onerec_params() {
    CHECK(rec_params_ != nullptr);
    if (auto* params = std::get_if<OneRecModelInputParams>(rec_params_)) {
      return *params;
    }
    if (auto* params = std::get_if<OneRecXAttentionParams>(rec_params_)) {
      return static_cast<OneRecModelInputParams&>(*params);
    }
    if (!has_onerec_params()) {
      rec_params().emplace<OneRecModelInputParams>();
    }
    return std::get<OneRecModelInputParams>(rec_params());
  }

  const OneRecXAttentionParams* onerec_xattention_params() const {
    if (rec_params_ == nullptr) {
      return nullptr;
    }
    return std::get_if<OneRecXAttentionParams>(rec_params_);
  }

  bool has_onerec_xattention_params() const {
    return onerec_xattention_params() != nullptr;
  }

  OneRecXAttentionParams& mutable_onerec_xattention_params() {
    CHECK(rec_params_ != nullptr);
    if (!has_onerec_xattention_params()) {
      rec_params().emplace<OneRecXAttentionParams>();
    }
    return std::get<OneRecXAttentionParams>(rec_params());
  }

  // Accessors for LLM Rec multi-round params inside rec_params variant
  const LlmRecMultiRoundParams* llmrec_params() const {
    if (rec_params_ == nullptr) {
      return nullptr;
    }
    return std::get_if<LlmRecMultiRoundParams>(rec_params_);
  }

  bool has_llmrec_params() const { return llmrec_params() != nullptr; }

  LlmRecMultiRoundParams& mutable_llmrec_params() {
    CHECK(rec_params_ != nullptr);
    if (!has_llmrec_params()) {
      rec_params().emplace<LlmRecMultiRoundParams>();
    }
    return std::get<LlmRecMultiRoundParams>(rec_params());
  }

 private:
  friend class ModelInputSnapshot;
  template <typename Owner, typename Spec>
  ModelInputParams(Owner& owner, Spec& spec, Domain domain)
      : native_owner_(&owner),
        domain_(domain),
        meta(owner.meta),
        attention(owner.attention),
        embedding(owner.embedding, spec.embedding),
        parallel(owner.parallel),
        block_copy(owner.block_copy),
        expert(owner.expert),
        graph(owner.graph, spec.graph),
        linear_state_cache_ops(owner.linear_state_cache_ops),
        linear_state_validity_mask(owner.linear_state_validity_mask),
        multi_block_tables(spec.multi_block_tables),
        mtp_shifted_token_ids(spec.mtp_shifted_token_ids),
        is_spec_verify(spec.is_spec_verify),
        prefill_without_cache(spec.prefill_without_cache),
        num_accepted_tokens(spec.num_accepted_tokens),
        mtp_topk_state(spec.mtp_topk_state),
        num_accepted_tokens_host(spec.num_accepted_tokens_host),
        attn_metadata(owner.attn_metadata),
        python_attention_metadata(owner.python_attention_metadata),
        enable_graph(owner.enable_graph) {}
};
template <typename NativeParams>
class NativeModelInputSnapshotOwner final : public ModelInputSnapshotOwner {
 public:
  explicit NativeModelInputSnapshotOwner(NativeParams params)
      : params_(std::move(params)) {}
  NativeModelInputSnapshotOwner(NativeParams params, SpecExecutionState state)
      : params_(std::move(params)),
        rec_execution_state_(
            std::make_shared<SpecExecutionState>(std::move(state))) {}
  ModelInputParams view() override {
    if constexpr (std::is_same_v<NativeParams, RecModelParams>) {
      if (rec_execution_state_ == nullptr) {
        rec_execution_state_ = std::make_shared<SpecExecutionState>();
      }
      return ModelInputParams(params_, rec_execution_state_);
    } else {
      return ModelInputParams(params_);
    }
  }
  std::shared_ptr<ModelInputSnapshotOwner> to(
      const torch::Device& device) const override {
    if constexpr (std::is_same_v<NativeParams, RecModelParams>) {
      return std::make_shared<NativeModelInputSnapshotOwner>(
          params_.to(device),
          rec_execution_state_ == nullptr
              ? SpecExecutionState()
              : detail::spec_execution_state_to(*rec_execution_state_, device));
    } else {
      return std::make_shared<NativeModelInputSnapshotOwner>(
          params_.to(device));
    }
  }

 private:
  NativeParams params_;
  std::shared_ptr<SpecExecutionState> rec_execution_state_;
};
template <typename NativeParams>
ModelInputSnapshot::ModelInputSnapshot(NativeParams params)
    : owner_(std::make_shared<NativeModelInputSnapshotOwner<NativeParams>>(
          std::move(params))) {}
inline ModelInputParams ModelInputSnapshot::view() const {
  CHECK(owner_ != nullptr) << "cannot view an empty model input snapshot";
  ModelInputParams params = owner_->view();
  params.snapshot_owner_ = owner_;
  return params;
}
inline ModelInputSnapshot ModelInputSnapshot::to(
    const torch::Device& device) const {
  CHECK(owner_ != nullptr);
  return ModelInputSnapshot(owner_->to(device));
}
inline ModelInputSnapshot ModelInputParams::to(
    const torch::Device& device) const {
  CHECK(native_owner_ != nullptr);
  if (snapshot_owner_ != nullptr) {
    return ModelInputSnapshot(snapshot_owner_->to(device));
  }
  switch (domain_) {
    case Domain::LLM:
      return ModelInputSnapshot(
          static_cast<LlmModelParams*>(native_owner_)->to(device));
    case Domain::VLM:
      return ModelInputSnapshot(
          static_cast<VlmModelParams*>(native_owner_)->to(device));
    case Domain::REC:
      CHECK(rec_execution_state_ != nullptr);
      return ModelInputSnapshot(std::shared_ptr<ModelInputSnapshotOwner>(
          std::make_shared<NativeModelInputSnapshotOwner<RecModelParams>>(
              static_cast<RecModelParams*>(native_owner_)->to(device),
              detail::spec_execution_state_to(*rec_execution_state_, device))));
  }
  LOG(FATAL) << "unknown model input domain";
  return ModelInputSnapshot();
}
inline ModelInputSnapshot ModelInputParams::clone() const {
  switch (domain_) {
    case Domain::LLM:
      return ModelInputSnapshot(
          static_cast<LlmModelParams*>(native_owner_)->clone());
    case Domain::VLM:
      return ModelInputSnapshot(
          static_cast<VlmModelParams*>(native_owner_)->clone());
    case Domain::REC:
      return ModelInputSnapshot(std::shared_ptr<ModelInputSnapshotOwner>(
          std::make_shared<NativeModelInputSnapshotOwner<RecModelParams>>(
              static_cast<RecModelParams*>(native_owner_)->clone(),
              *rec_execution_state_)));
  }
  LOG(FATAL) << "unknown model input domain";
  return ModelInputSnapshot();
}
}  // namespace xllm
