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

#include "runtime/draft_model_spec_worker_impl.h"

#include <absl/strings/ascii.h>
#include <absl/strings/escaping.h>
#include <glog/logging.h>

#include <algorithm>
#include <filesystem>
#include <span>
#include <system_error>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "common/metrics.h"
#include "core/common/types.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model/model_type.h"
#include "core/framework/multimodal/content_hash.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_input_model_spec.h"
#include "core/framework/speculative/draft_input_sampling.h"
#include "core/kv_cache/layout/kv_cache_capacity.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/storage/static_kv_cache_tensor_allocator.h"
#include "core/kv_cache/transfer/hierarchy_kv_cache_transfer.h"
#include "core/kv_cache/transfer/mooncake_kv_cache_transfer.h"
#include "core/util/tensor_helper.h"
#include "framework/model_loader/model_loader.h"
#include "models/model_registry.h"
#include "runtime/forward_runtime_state.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/vlm_forward_params.h"
#include "runtime/worker_memory_resources.h"
#include "util/hash_util.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {

int64_t get_dp_local_tp_size(const ParallelArgs& parallel_args) {
  const int64_t dp_size = std::max<int64_t>(parallel_args.dp_size(), 1);
  const int64_t cp_size = std::max<int64_t>(parallel_args.cp_size(), 1);
  return std::max<int64_t>(parallel_args.world_size() / dp_size / cp_size, 1);
}

template <typename MakeShape, typename Allocate>
bool allocate_pool_if_loaded(WorkerImpl& worker,
                             MakeShape&& make_shape,
                             Allocate&& allocate) {
  const WorkerImpl::Status status = worker.get_status();
  if (status == WorkerImpl::Status::LOADED) {
    return allocate(worker, make_shape());
  }
  CHECK_EQ(status, WorkerImpl::Status::READY);
  return true;
}

std::string stable_path_digest(const std::string& path_string) {
  const std::filesystem::path path(path_string);
  std::error_code error;
  const std::filesystem::path canonical_path =
      std::filesystem::weakly_canonical(path, error);
  const std::string normalized_path =
      (error ? path.lexically_normal() : canonical_path).generic_string();
  const XXH3Key path_hash = hash_string(normalized_path);

  return absl::BytesToHexString(absl::string_view(
      reinterpret_cast<const char*>(path_hash.data), sizeof(path_hash.data)));
}

std::string draft_store_key_component(const runtime::Options& options) {
  const std::string algorithm =
      absl::AsciiStrToLower(options.speculative_algorithm());

  const std::string draft_model_path = options.draft_model_path().value_or("");
  if (draft_model_path.empty()) {
    return "spec_draft::" + algorithm + "::embedded";
  }

  std::string draft_model_name = std::filesystem::path(draft_model_path)
                                     .lexically_normal()
                                     .filename()
                                     .generic_string();
  if (draft_model_name.empty()) {
    draft_model_name = "checkpoint";
  }
  return "spec_draft::" + algorithm + "::" + draft_model_name +
         "::" + stable_path_digest(draft_model_path);
}

KVCacheEstimateOptions make_kv_cache_estimate_options(
    const ModelArgs& model_args,
    const runtime::Options& options,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    int64_t cache_size_in_bytes) {
  const int64_t dp_local_tp_size = get_dp_local_tp_size(parallel_args);
  const int64_t n_heads = model_args.n_heads();
  const int64_t n_kv_heads = model_args.n_kv_heads().value_or(n_heads);

  KVCacheEstimateOptions estimate_options;
  estimate_options.dtype = dtype;
  estimate_options.kv_cache_dtype = options.kv_cache_dtype();
  estimate_options.indexer_cache_dtype =
      KVCacheConfig::get_instance().indexer_cache_dtype();
  estimate_options.cache_size_in_bytes = cache_size_in_bytes;
  estimate_options.block_size = options.block_size();
  estimate_options.world_size = dp_local_tp_size;
  estimate_options.n_local_kv_heads =
      std::max<int64_t>(n_kv_heads / dp_local_tp_size, 1);
  if (has_linear_attention_layers(model_args)) {
    estimate_options.n_local_linear_k_heads = std::max<int64_t>(
        model_args.linear_num_key_heads() / dp_local_tp_size, 1);
    estimate_options.n_local_linear_v_heads = std::max<int64_t>(
        model_args.linear_num_value_heads() / dp_local_tp_size, 1);
  }
  estimate_options.max_seqs_per_batch =
      static_cast<int64_t>(options.max_seqs_per_batch());
  estimate_options.num_speculative_tokens =
      static_cast<int64_t>(options.num_speculative_tokens());
  estimate_options.max_tokens_per_batch =
      static_cast<int64_t>(options.max_tokens_per_batch());
  estimate_options.max_tokens_per_chunk_for_prefill =
      static_cast<int64_t>(options.max_tokens_per_chunk_for_prefill());
  estimate_options.max_linear_state_cache_slots =
      options.max_linear_state_cache_slots();
  estimate_options.linear_state_cache_block_limit =
      get_npu_linear_state_cache_block_limit(model_args.model_type());
  estimate_options.is_draft_engine = options.is_draft_engine();
  estimate_options.enable_chunked_prefill = options.enable_chunked_prefill();
  estimate_options.enable_schedule_overlap = options.enable_schedule_overlap();
  const KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  estimate_options.enable_prefix_cache =
      kv_cache_config.enable_prefix_cache() &&
      !kv_cache_config.enable_vmm_kvcache();
  estimate_options.enable_disagg_pd = options.enable_disagg_pd();
  estimate_options.instance_role = options.instance_role();
  estimate_options.dp_size = options.dp_size();
  estimate_options.enable_dp_fair_token_budget =
      SchedulerConfig::get_instance().enable_dp_fair_token_budget();
  return estimate_options;
}

}  // namespace

template <typename TargetInput>
DraftModelSpecWorkerImpl<TargetInput>::DraftModelSpecWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    const runtime::Options& target_options,
    WorkerType worker_type,
    const std::function<std::unique_ptr<LLMWorkerImpl>()>& draft_factory)
    : SpeculativeWorkerImpl<TargetInput>(parallel_args,
                                         device,
                                         options,
                                         target_options,
                                         worker_type),
      is_probabilistic_draft_(options.enable_probabilistic_draft()) {
  CHECK(draft_factory) << "Draft worker factory must not be empty.";
  draft_impl_ = draft_factory();
  CHECK(draft_impl_ != nullptr) << "Draft worker factory must return a worker.";
}

template <typename TargetInput>
DraftModelSpecWorkerImpl<TargetInput>::~DraftModelSpecWorkerImpl() {
  drain_pending_execution();
  // Stop asynchronous transfers before draft_impl_ releases its KV cache.
  if (hierarchy_kv_cache_transfer_ != nullptr) {
    hierarchy_kv_cache_transfer_->shutdown();
  }
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::drain_pending_execution() {
  SpeculativeWorkerImpl<TargetInput>::drain_pending_execution();
  if (draft_impl_ != nullptr) {
    draft_impl_->drain_pending_execution();
  }
}

template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::init_model(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  bool result;
  torch::Tensor draft_token_id_offsets;
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = SpeculativeWorkerImpl<TargetInput>::init_model(
        model_weights_path, random_seed, master_status);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    auto loader = ModelLoader::create(model_weights_path);
    if (!draft_impl_->options().model_path().empty() &&
        draft_impl_->options().model_path() != loader->model_weights_path()) {
      loader->set_reference_model_weights_path(
          draft_impl_->options().model_path());
    }
    draft_token_id_offsets = load_draft_vocab_mapping(*loader);
    auto context =
        draft_impl_->WorkerImpl::load_model_config(loader->model_args(),
                                                   loader->quant_args(),
                                                   /*num_speculative_tokens=*/0,
                                                   /*tokenizer=*/nullptr);
    result = draft_impl_->WorkerImpl::init_model(std::move(loader),
                                                 std::move(context),
                                                 nullptr,
                                                 random_seed,
                                                 master_status);
  }
  if (impl_->get_status() == WorkerImpl::Status::LOADED) {
    context_ = impl_->context_;
  }
  if (result && draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
    const int64_t draft_vocab_size = draft_args.draft_vocab_size() > 0
                                         ? draft_args.draft_vocab_size()
                                         : draft_args.vocab_size();
    draft_vocab_ =
        DraftVocabMapping(std::move(draft_token_id_offsets),
                          draft_vocab_size,
                          impl_->context_.get_model_args().vocab_size(),
                          device_);
  }
  return result;
}

template <typename TargetInput>
KVCacheShape DraftModelSpecWorkerImpl<TargetInput>::draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape) const {
  return build_draft_kv_cache_shape(target_kv_cache_shape);
}

template <typename TargetInput>
torch::Tensor DraftModelSpecWorkerImpl<TargetInput>::load_draft_vocab_mapping(
    ModelLoader& loader) {
  torch::Tensor draft_token_id_offsets;
  if (SpeculativeConfig::is_eagle3_algorithm(
          draft_impl_->options().speculative_algorithm()) ||
      SpeculativeConfig::is_dspark_algorithm(
          draft_impl_->options().speculative_algorithm())) {
    for (const auto& state_dict : loader.get_state_dicts()) {
      auto offsets = state_dict->get_tensor("d2t");
      if (offsets.defined()) {
        draft_token_id_offsets = std::move(offsets);
        break;
      }
    }
  }
  return draft_token_id_offsets;
}

template <typename TargetInput>
KVCacheShape DraftModelSpecWorkerImpl<TargetInput>::build_draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape,
    int64_t draft_world_size) const {
  if (draft_world_size <= 0 &&
      !target_kv_cache_shape.has_grouped_cache_layout()) {
    draft_world_size =
        get_dp_local_tp_size(draft_impl_->context_.get_parallel_args());
  }
  KVCacheCapacity draft_capacity;
  draft_capacity.enable_indexer_cache_quant(
      KVCacheConfig::get_instance().indexer_cache_dtype() == "int8");
  return build_speculative_draft_kv_cache_shape(
      target_kv_cache_shape,
      draft_capacity,
      draft_impl_->context_.get_model_args(),
      options_.block_size(),
      draft_world_size,
      options_.kv_cache_dtype());
}

template <typename TargetInput>
KVCacheCreateOptions
DraftModelSpecWorkerImpl<TargetInput>::build_draft_kv_cache_layout(
    const KVCacheShape& shape) const {
  auto layout = get_kv_cache_create_options(
      draft_impl_->context_.get_model_args(), shape, true);
  auto indexer_layers = resolve_indexer_cache_enabled_layers(
      draft_impl_->context_.get_model_args(), draft_num_layers());
  if (shape.has_kpool_tail_shape()) {
    indexer_layers.assign(draft_num_layers(), true);
  }
  layout.indexer_cache_enabled_layers(std::move(indexer_layers));
  layout.num_layers(draft_num_layers());
  if (shape.shared_draft_layers() > 0) {
    const auto shared = impl_->shared_kv_cache_layout();
    CHECK_EQ(shared.static_layout()->draft_layer_count(), layout.num_layers());
    CHECK(kv_cache_transfer_ == nullptr && !shape.has_grouped_cache_layout() &&
          !layout.enable_linear_attention() &&
          draft_impl_->context_.get_model_args().index_n_heads() == 0 &&
          draft_impl_->options().kv_cache_dtype() != "int8" &&
          KVCacheConfig::get_instance().indexer_cache_dtype() != "int8")
        << "Static shared draft storage requires ordinary Full-only layers";
    layout.static_layout(shared.static_layout())
        .shared_draft_cache(true)
        .full_block_slot_table(shared.full_block_slot_table())
        .tensor_allocator(create_static_draft_kv_tensor_allocator(
            std::static_pointer_cast<StaticKVCacheTensorAllocator>(
                shared.tensor_allocator())));
  }
  return layout;
}

template <typename TargetInput>
int64_t DraftModelSpecWorkerImpl<TargetInput>::draft_num_layers() const {
#if !defined(USE_NPU)
  const int32_t nextn_layers =
      draft_impl_->context_.get_model_args().num_nextn_predict_layers();
  if (nextn_layers > 0) {
    return nextn_layers;
  }
#endif
  return draft_impl_->context_.get_model_args().n_layers();
}

template <typename TargetInput>
std::tuple<int64_t, int64_t>
DraftModelSpecWorkerImpl<TargetInput>::estimate_kv_cache_capacity_with_draft(
    const runtime::Options& target_options,
    const runtime::Options& draft_options) {
  const auto [target_cache_bytes, target_total_bytes] =
      impl_->estimate_kv_cache_capacity();
  const auto [draft_cache_bytes, draft_total_bytes] =
      draft_impl_->estimate_kv_cache_capacity();
  const int64_t cache_size_in_bytes =
      std::min(target_cache_bytes, draft_cache_bytes);
  const int64_t total_memory = std::min(target_total_bytes, draft_total_bytes);

  const ModelArgs& target_model_args = impl_->context_.get_model_args();
  if (!util::is_deepseek_v4_model_type(target_model_args.model_type())) {
    return {cache_size_in_bytes, total_memory};
  }

  const ModelArgs& draft_model_args = draft_impl_->context_.get_model_args();
  KVCacheEstimateOptions target_estimate_options =
      make_kv_cache_estimate_options(target_model_args,
                                     target_options,
                                     parallel_args_,
                                     dtype_,
                                     cache_size_in_bytes);
  const KVCacheEstimateOptions draft_estimate_options =
      make_kv_cache_estimate_options(draft_model_args,
                                     draft_options,
                                     parallel_args_,
                                     dtype_,
                                     cache_size_in_bytes);
  target_estimate_options.draft_model_args = &draft_model_args;
  target_estimate_options.draft_options = &draft_estimate_options;

  const KVCacheCapacity capacity = ::xllm::estimate_kv_cache_capacity(
      target_model_args, target_estimate_options);
  return {capacity.cache_size_in_bytes(), total_memory};
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<
    TargetInput>::prepare_hierarchy_kv_cache_transfers() {
  if (options_.host_blocks_factor() <= 1.0) {
    return;
  }

  CHECK(impl_ != nullptr);
  const auto target_transfer = impl_->get_hierarchy_kv_cache_transfer();
  const auto draft_transfer = draft_impl_->get_hierarchy_kv_cache_transfer();
  auto unified_transfer = hierarchy_kv_cache_transfer_
                              ? hierarchy_kv_cache_transfer_
                              : target_transfer;
  if (unified_transfer == nullptr) {
    unified_transfer = draft_transfer
                           ? draft_transfer
                           : impl_->create_hierarchy_kv_cache_transfer();
  }

  if (target_transfer == nullptr) {
    impl_->bind_hierarchy_kv_cache_transfer(
        unified_transfer,
        HierarchyKVCacheTransfer::CacheRole::TARGET,
        compute_stream_.get(),
        /*store_key_component=*/"main");
  } else {
    CHECK_EQ(target_transfer.get(), unified_transfer.get())
        << "Speculative target worker hierarchy KV cache transfer changed "
           "unexpectedly.";
  }

  if (draft_transfer == nullptr) {
    draft_impl_->bind_hierarchy_kv_cache_transfer(
        unified_transfer,
        HierarchyKVCacheTransfer::CacheRole::DRAFT,
        compute_stream_.get(),
        draft_store_key_component(options_));
  } else {
    CHECK_EQ(draft_transfer.get(), unified_transfer.get())
        << "Speculative draft worker hierarchy KV cache transfer changed "
           "unexpectedly.";
  }

  if (hierarchy_kv_cache_transfer_ == nullptr) {
    set_hierarchy_kv_cache_transfer(std::move(unified_transfer));
  }
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<
    TargetInput>::finalize_hierarchy_kv_cache_transfers() {
  if (options_.host_blocks_factor() <= 1.0) {
    return;
  }

  CHECK(hierarchy_kv_cache_transfer_ != nullptr)
      << "Speculative hierarchy KV cache transfer is not prepared.";
  if (!hierarchy_kv_cache_transfer_->registration_finalized()) {
    CHECK(hierarchy_kv_cache_transfer_->finalize_registration());
  }
}

template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::allocate_pools(
    const KVCacheShape& kv_cache_shape,
    const AllocateFn& allocate) {
  const bool bind_vmm_group =
      KVCacheConfig::get_instance().enable_vmm_kvcache() &&
      impl_->get_status() == WorkerImpl::Status::LOADED &&
      draft_impl_->get_status() == WorkerImpl::Status::LOADED &&
      impl_->context_.get_model_id() != draft_impl_->context_.get_model_id();
  const bool target_allocated = allocate_pool_if_loaded(
      *impl_,
      [&]() -> const KVCacheShape& { return kv_cache_shape; },
      allocate);
  bool draft_allocated = true;
  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    const KVCacheShape draft_shape = draft_kv_cache_shape(kv_cache_shape);
    draft_impl_->allocate_kv_cache(draft_shape,
                                   build_draft_kv_cache_layout(draft_shape),
                                   kv_cache_transfer_);
    if (kv_cache_transfer_) {
      kv_cache_transfer_->configure_cache_layout(
          draft_impl_->context_.get_parallel_args(),
          draft_impl_->context_.get_model_args(),
          draft_impl_->options().block_size(),
          true);
      kv_cache_transfer_->register_kv_cache_spec(
          draft_impl_->kv_caches(), draft_shape, draft_impl_->dtype());
    }
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::READY);
  }

  const bool allocated = target_allocated && draft_allocated;
  if (allocated) {
    if (bind_vmm_group) {
      // The scheduler's block IDs address both caches. Bind their mappings
      // before the master reserves its first (padding) page group.
      CHECK(WorkerMemoryResources::get_instance()
                .kv_cache_memory()
                .bind_mapping_group(impl_->context_.get_model_id(),
                                    {impl_->context_.get_model_id(),
                                     draft_impl_->context_.get_model_id()}));
    }
    on_kv_pools_allocated(kv_cache_shape.key_cache_shape()[0]);
    finalize_hierarchy_kv_cache_transfers();
  }
  return allocated;
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::flush_pending_target_context() {
  if (!target_context_handoff_.has_pending()) {
    return;
  }
  CHECK(target_state_store_ != nullptr)
      << "target_state_store_ must be initialized before target state write";
  target_context_handoff_.flush(*target_state_store_);
}

template <typename TargetInput>
std::optional<ForwardOutput> DraftModelSpecWorkerImpl<TargetInput>::step(
    const TargetInput& input) {
  if (draft_impl_->uses_shared_static_kv_cache()) {
    const auto& embedding = input.input_params.embedding;
    // Decode can execute draft before target. Initialize their common leases
    // here so a newly admitted draft page cannot expose previous Linear state.
    // The compute stream orders this after preceding target/draft consumers.
    impl_->initialize_shared_full_blocks(embedding.full_block_slot_rows,
                                         embedding.full_block_lease_epochs,
                                         *compute_stream_);
  }
  return SpeculativeWorkerImpl<TargetInput>::step(input);
}

template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::allocate_kv_cache(
    const KVCacheShape& kv_cache_shape) {
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);
  prepare_hierarchy_kv_cache_transfers();
  return allocate_pools(kv_cache_shape,
                        [](WorkerImpl& worker, const KVCacheShape& shape) {
                          return worker.allocate_kv_cache(shape);
                        });
}

#if defined(USE_NPU) || defined(USE_MLU)
template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);
  prepare_hierarchy_kv_cache_transfers();
  if (kv_cache_transfer_ == nullptr) {
    kv_cache_transfer_ = std::make_shared<MooncakeKVCacheTransferDefault>(
        device_.index(),
        options_.transfer_listen_port(),
        device_,
        context_.get_model_args().model_type());
    kv_cache_transfer_->initialize(device_.index());
  }
  return allocate_pools(kv_cache_shape,
                        [this](WorkerImpl& worker, const KVCacheShape& shape) {
                          return worker.allocate_kv_cache_with_transfer(
                              kv_cache_transfer_, shape);
                        });
}
#endif

template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::prepare_model_decode_continuation(
    TargetInput& /*inputs*/,
    const ForwardOutput& /*prior_output*/) {
  return true;
}

namespace {

// Chunked-prefill verify normalization: clamps the decode position to the
// block-table capacity (leaving room for one validate row) and resolves the
// scheduler's KV-length lag or lead against that position.
void normalize_chunked_prefill_row(int32_t logical_block_size,
                                   int32_t validate_width,
                                   const torch::Tensor& block_tables,
                                   int32_t seq_id,
                                   int32_t& current_position,
                                   int32_t& current_kv_len,
                                   int32_t& expected_kv_len) {
  if (block_tables.defined() && block_tables.dim() == 2 &&
      seq_id < block_tables.size(0)) {
    const int32_t allocated_kv_len =
        static_cast<int32_t>(block_tables.size(1)) * logical_block_size;
    const int32_t max_valid_position = allocated_kv_len - validate_width;
    if (current_position > max_valid_position) {
      CHECK_GT(allocated_kv_len, 0)
          << "decode context has empty block table, seq_id=" << seq_id;
      CHECK_GE(max_valid_position, 0)
          << "decode context block table is too small for validation, "
          << "seq_id=" << seq_id << ", allocated_kv_len=" << allocated_kv_len
          << ", validate_width=" << validate_width;
      CHECK_LE(current_position - max_valid_position, validate_width)
          << "decode context position exceeds allocated blocks, seq_id="
          << seq_id << ", current_position=" << current_position
          << ", current_kv_len=" << current_kv_len
          << ", allocated_kv_len=" << allocated_kv_len;
      current_position = max_valid_position;
      expected_kv_len = current_position + 1;
      current_kv_len = std::min(current_kv_len, expected_kv_len);
    }
  }
  if (current_kv_len < expected_kv_len) {
    // Qwen3.5/MiMo MTP can receive a scheduler KV length that has not yet
    // caught up with the speculative placeholder resolved into
    // current_position. Normalize only the lag explainable by the current
    // speculative step.
    CHECK_LE(expected_kv_len - current_kv_len, validate_width)
        << "decode context kv_len lag is too large, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;
    current_kv_len = expected_kv_len;
  } else if (current_kv_len > expected_kv_len) {
    // The first decode step can carry the prompt KV length while the decode
    // position is still initialized to zero. Align the position to the KV
    // context before building the MTP draft input.
    current_position = current_kv_len - 1;
    expected_kv_len = current_kv_len;
  }
}

// Replays the JSON grammar check on the host-accepted tokens of the verify
// tail and maps failed sample-sequence ids to the producing batch's rows,
// on the launch-time snapshot the settle must see.
std::vector<size_t> json_failed_sequence_rows(
    const std::vector<JsonObjectGrammarState>& json_states,
    const std::vector<std::string>& sample_sequence_ids,
    std::span<const int32_t> embedding_ids,
    const torch::Tensor& accepted_tokens_host,
    const StreamEventPtr& accepted_tokens_ready,
    const std::vector<JsonObjectOutputError>& prior_errors,
    std::vector<JsonObjectOutputError>& errors) {
  CHECK_EQ(accepted_tokens_host.dim(), 2)
      << "JSON accepted output must be [sequence, token]";
  CHECK_EQ(sample_sequence_ids.size(), json_states.size())
      << "JSON sequence ids must match grammar state rows";
  CHECK_EQ(embedding_ids.size(), json_states.size())
      << "JSON embedding ids must match grammar state rows";
  CHECK(accepted_tokens_ready == nullptr ||
        accepted_tokens_ready->synchronize())
      << "failed to wait for accepted tokens before JSON replay";

  const torch::Tensor accepted_tokens =
      to_cpu_contiguous(accepted_tokens_host, torch::kLong);
  const std::vector<JsonTokenMismatch> mismatches =
      detail::find_json_accepted_token_mismatches(
          json_states,
          accepted_tokens.const_data_ptr<int64_t>(),
          static_cast<size_t>(accepted_tokens.size(0)),
          static_cast<size_t>(accepted_tokens.size(1)));
  std::unordered_set<std::string> failed_sequence_ids =
      detail::failed_sequence_ids_from_errors(prior_errors,
                                              errors,
                                              /*require_non_empty_id=*/true);

  for (const JsonTokenMismatch& mismatch : mismatches) {
    const std::string& sample_sequence_id =
        sample_sequence_ids[mismatch.sequence_index];
    CHECK(!sample_sequence_id.empty())
        << "JSON mismatch requires a sampled sequence id";
    LOG(ERROR) << "JSON accepted output replay mismatch: sequence_id="
               << sample_sequence_id
               << ", token_offset=" << mismatch.token_offset
               << ", token_id=" << mismatch.token_id
               << ", committed_tokens=" << mismatch.committed_tokens
               << ", state_fingerprint=" << mismatch.state_fingerprint;
    if (failed_sequence_ids.emplace(sample_sequence_id).second) {
      errors.push_back(
          {sample_sequence_id,
           "accepted output violates json_object grammar, token_id=" +
               std::to_string(mismatch.token_id)});
    }
  }
  std::vector<size_t> failed_sequence_rows;
  failed_sequence_rows.reserve(failed_sequence_ids.size());
  for (size_t row = 0; row < sample_sequence_ids.size(); ++row) {
    if (failed_sequence_ids.contains(sample_sequence_ids[row])) {
      failed_sequence_rows.emplace_back(row);
    }
  }
  CHECK_EQ(failed_sequence_rows.size(), failed_sequence_ids.size())
      << "JSON errors must reference sampled rows in the current batch";
  return failed_sequence_rows;
}

}  // namespace

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::
    rewrite_decode_rows_from_target_state(
        TargetInput& input,
        const std::vector<TargetDecodeStateStore::DecodeRow>& last_states,
        const DecodeRowPolicy& policy) const {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "decode context state count mismatch";
  const bool enable_cache_correction = this->enable_schedule_overlap();

  std::vector<int32_t> token_ids_vec;
  std::vector<int32_t> positions_vec;
  std::vector<int32_t> kv_seq_lens_vec;
  token_ids_vec.reserve(num_sequences);
  positions_vec.reserve(num_sequences);
  kv_seq_lens_vec.reserve(num_sequences);

  const torch::Tensor& token_ids_cpu = input.token_ids_host;
  const torch::Tensor& positions_cpu = input.positions_host;
  const Slice<int32_t> input_token_ids = tensor_slice(token_ids_cpu);
  const Slice<int32_t> input_positions = tensor_slice(positions_cpu);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size())
        << "decode context token seq_id out of range, seq_id=" << seq_id;
    CHECK_LT(static_cast<size_t>(seq_id), input_positions.size())
        << "decode context position seq_id out of range, seq_id=" << seq_id;
    const TargetDecodeStateStore::DecodeRow& state = last_states[seq_id];
    const int32_t input_token_id = input_token_ids[seq_id];
    const bool input_is_fake_token = input_token_id < 0;
    // Rewrite fake input tokens to the last committed real token so KV cache
    // scatter has a valid id; only apply the recorded position offset when the
    // cached state is still valid.
    const bool rewrite_fake_token =
        enable_cache_correction && input_is_fake_token;
    const bool use_cache_correction = rewrite_fake_token && state.valid;
    const bool use_fake_context = rewrite_fake_token && !state.valid &&
                                  policy.fake_context_falls_back_to_state_token;
    const int32_t position_offset =
        use_cache_correction ? state.position_offset : 0;
    int32_t current_position = input_positions[seq_id] + position_offset;
    int32_t current_kv_len =
        input.input_params.attention.host.kv_seq_lens[seq_id] + position_offset;
    int32_t expected_kv_len = policy.positions_decoupled_from_kv_len
                                  ? current_kv_len
                                  : current_position + 1;
    if (policy.chunked_prefill_verify) {
      normalize_chunked_prefill_row(
          policy.logical_block_size,
          policy.validate_width,
          input.input_params.attention.host.block_tables,
          seq_id,
          current_position,
          current_kv_len,
          expected_kv_len);
    }

    CHECK_EQ(expected_kv_len, current_kv_len)
        << "decode context position/kv_len mismatch, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;

    token_ids_vec.emplace_back((use_cache_correction || use_fake_context)
                                   ? state.token_id
                                   : input_token_id);
    positions_vec.emplace_back(current_position);
    kv_seq_lens_vec.emplace_back(current_kv_len);
  }

  input.token_ids_host = make_pinned_cpu_tensor(token_ids_vec);
  input.positions_host = make_pinned_cpu_tensor(positions_vec);
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  input.runtime.device_tensors_ready = false;
}

template <typename TargetInput>
DraftInputModelSpec DraftModelSpecWorkerImpl<TargetInput>::draft_input_spec()
    const {
  if (!draft_input_spec_valid_) {
    draft_input_spec_ = make_draft_input_spec(
        draft_impl_->context_.get_model_args(), draft_impl_->dtype());
    draft_input_spec_valid_ = true;
  }
  return draft_input_spec_;
}

template <typename TargetInput>
std::vector<uint8_t>
DraftModelSpecWorkerImpl<TargetInput>::compute_vocabulary_invalid_rows(
    SamplingParameters& sampling_params,
    const DraftTokenHistories& histories,
    const std::vector<JsonObjectGrammarState>& json_states,
    const std::vector<uint8_t>& sampling_invalid_rows) {
  return merge_invalid_rows(
      sampling_invalid_rows,
      filter_vocabulary_failures(
          prepare_draft_vocabulary_sampling(sampling_params, histories),
          json_states));
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::mark_invalid_draft_input_rows(
    const TargetInput& input,
    const std::vector<uint8_t>& invalid_rows,
    std::vector<size_t>* failed_rows,
    std::vector<JsonObjectOutputError>& errors) const {
  if (invalid_rows.empty()) {
    return;
  }
  const size_t batch_size = input.input_params.embedding.embedding_ids.size();
  CHECK_EQ(invalid_rows.size(), batch_size);
  std::unordered_set<size_t> already_failed;
  if (failed_rows != nullptr) {
    already_failed.insert(failed_rows->begin(), failed_rows->end());
  }
  for (size_t row = 0; row < batch_size; ++row) {
    if (invalid_rows[row] == 0 || !already_failed.insert(row).second) {
      continue;
    }
    CHECK_LT(row, input.sample_sequence_ids.size());
    if (failed_rows != nullptr) {
      failed_rows->emplace_back(row);
    }
    errors.push_back({input.sample_sequence_ids[row],
                      "draft input cannot advance committed sampling state"});
  }
}

template <typename TargetInput>
std::optional<ForwardOutput>
DraftModelSpecWorkerImpl<TargetInput>::dispatch_draft_worker(
    const TargetInput& input,
    TargetInput& processed_input) {
  // The colocated drafter is always an LLM worker; convert VLM target inputs
  // at this single boundary, and mirror the prepared sampling params back so
  // callers observe what the draft actually consumed.
  if constexpr (std::is_same_v<TargetInput, VlmForwardInput>) {
    LlmForwardInput draft_input =
        make_llm_draft_input(input, draft_impl_->uses_shared_static_kv_cache());
    LlmForwardInput processed;
    std::optional<ForwardOutput> output =
        detail::run_worker_no_sync(*draft_impl_,
                                   draft_input,
                                   *compute_stream_,
                                   *compute_stream_,
                                   &processed);
    processed_input.sampling_params = std::move(processed.sampling_params);
    return output;
  } else {
    return detail::run_worker_no_sync(*draft_impl_,
                                      input,
                                      *compute_stream_,
                                      *compute_stream_,
                                      &processed_input);
  }
}

template <typename TargetInput>
typename DraftModelSpecWorkerImpl<TargetInput>::PendingExtendRound
DraftModelSpecWorkerImpl<TargetInput>::launch_draft_extend(
    const TargetInput& input,
    const TargetInput& validate_input,
    const SampleOutput& accepted_output,
    torch::Tensor accepted_tokens_host,
    const StreamEventPtr& accepted_ready,
    std::vector<size_t> failed_rows,
    std::vector<JsonObjectOutputError>& errors) {
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  // The draft proposal never applies penalties or the grammar bitmask: the
  // rejection sampler enforces both on the target's verify logits, so the
  // draft forward launches from the device-side accepted output alone; the
  // settle-side consumers read the host mirror behind the event.
  SamplingParameters draft_sampling(input.sampling_params);
  prepare_draft_proposal_sampling(is_probabilistic_draft_, draft_sampling);
  TargetInput draft_extend_input = build_draft_extend_input(
      validate_input, accepted_output, std::move(draft_sampling));
  record_current_metadata_ready_event(draft_extend_input, *compute_stream_);

  // The source input's states still gate the empty-vocabulary rejection;
  // rejected rows surface on the producing step's output here (no host
  // dependency), while the failed rows ride the pending round into the
  // settle.
  std::vector<uint8_t> vocabulary_invalid_rows =
      compute_vocabulary_invalid_rows(draft_extend_input.sampling_params,
                                      /*histories=*/{},
                                      input.json_object_states,
                                      /*sampling_invalid_rows=*/{});

  PendingExtendRound pending;
  TargetInput& prepared = pending.prepared_input;
  Timer timer;
  pending.output = dispatch_draft_worker(draft_extend_input, prepared).value();
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  const BatchRows producing_rows = input.input_params.embedding.batch_rows();
  pending.embedding_ids.assign(producing_rows.embedding_ids.begin(),
                               producing_rows.embedding_ids.end());
  pending.request_ids.assign(producing_rows.request_ids.begin(),
                             producing_rows.request_ids.end());
  pending.sampling_params = input.sampling_params;
  pending.json_states = input.json_object_states;
  pending.sample_sequence_ids = input.sample_sequence_ids;
  pending.prior_errors = input.json_object_errors;
  pending.accepted_tokens_host = std::move(accepted_tokens_host);
  pending.accepted_ready = accepted_ready;
  pending.failed_rows = std::move(failed_rows);
  mark_invalid_draft_input_rows(
      input, vocabulary_invalid_rows, &pending.failed_rows, errors);
  return pending;
}

template <typename TargetInput>
std::vector<JsonObjectOutputError>
DraftModelSpecWorkerImpl<TargetInput>::settle_round(PendingExtendRound round) {
  // The D2H-event wait lives here and not in the launch.
  // json_failed_sequence_rows performs its own event wait; the explicit
  // synchronize below is only for the other host-token consumers.
  const bool scan_json = !round.json_states.empty();
  const bool needs_host_tokens =
      needs_committed_tokens(round.sampling_params, round.json_states);
  std::vector<JsonObjectOutputError> errors;
  if (!scan_json && needs_host_tokens) {
    CHECK(round.accepted_ready == nullptr ||
          round.accepted_ready->synchronize())
        << "failed to wait for accepted tokens before draft input sampling";
  }
  if (scan_json) {
    const std::vector<size_t> json_rows =
        json_failed_sequence_rows(round.json_states,
                                  round.sample_sequence_ids,
                                  round.embedding_ids,
                                  round.accepted_tokens_host,
                                  round.accepted_ready,
                                  round.prior_errors,
                                  errors);
    round.failed_rows.insert(
        round.failed_rows.end(), json_rows.begin(), json_rows.end());
  }
  publish_round_result(round);
  return errors;
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::on_kv_pools_allocated(
    int64_t num_blocks) {
  target_state_store_ = std::make_unique<TargetDecodeStateStore>(num_blocks);
}

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::release_request_state(
    const std::vector<int32_t>& released) {
  if (released.empty()) {
    return;
  }
  // Algorithm hooks run first: a pending-context flush may re-populate the
  // target store, so the shared clear must come after it.
  release_algorithm_state(released);
  if (target_state_store_ != nullptr) {
    target_state_store_->clear(released);
  }
}

template class DraftModelSpecWorkerImpl<LlmForwardInput>;
template class DraftModelSpecWorkerImpl<VlmForwardInput>;

}  // namespace xllm
