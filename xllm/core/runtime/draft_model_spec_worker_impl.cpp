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
#include <system_error>
#include <tuple>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/draft_sampling.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_input_model_spec.h"
#include "core/kv_cache/layout/kv_cache_capacity.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/transfer/hierarchy_kv_cache_transfer.h"
#include "core/kv_cache/transfer/mooncake_kv_cache_transfer.h"
#include "models/model_registry.h"
#include "runtime/llm_worker_impl.h"
#include "util/hash_util.h"
#include "util/tensor_helper.h"
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
      !kv_cache_config.enable_virtual_memory();
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
      draft_sampling_mode_(
          parse_draft_sampling_mode(options.draft_sampling_mode())) {
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
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = SpeculativeWorkerImpl<TargetInput>::init_model(
        model_weights_path, random_seed, master_status);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    result = draft_impl_->WorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  }
  if (impl_->get_status() == WorkerImpl::Status::LOADED) {
    context_ = impl_->context_;
  }
  return result;
}

template <typename TargetInput>
bool DraftModelSpecWorkerImpl<TargetInput>::task_models_loaded() const {
  return impl_ != nullptr && draft_impl_ != nullptr &&
         impl_->get_status() == WorkerImpl::Status::LOADED &&
         draft_impl_->get_status() == WorkerImpl::Status::LOADED;
}

template <typename TargetInput>
KVCacheShape DraftModelSpecWorkerImpl<TargetInput>::draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape) const {
  return build_draft_kv_cache_shape(target_kv_cache_shape);
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
                           : impl_->create_hierarchy_kv_cache_transfer(
                                 options_.enable_task_pipeline());
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
  const bool target_allocated = allocate_pool_if_loaded(
      *impl_,
      [&]() -> const KVCacheShape& { return kv_cache_shape; },
      allocate);
  const bool draft_allocated = allocate_pool_if_loaded(
      *draft_impl_,
      [&] { return draft_kv_cache_shape(kv_cache_shape); },
      allocate);

  const bool allocated = target_allocated && draft_allocated;
  if (allocated) {
    on_kv_pools_allocated(kv_cache_shape.key_cache_shape()[0]);
    finalize_hierarchy_kv_cache_transfers();
  }
  return allocated;
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
TargetInput
DraftModelSpecWorkerImpl<TargetInput>::update_input_by_last_step_output(
    TargetInput& inputs) {
  return inputs.clone();
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

}  // namespace

template <typename TargetInput>
void DraftModelSpecWorkerImpl<TargetInput>::
    rewrite_decode_rows_from_target_state(
        TargetInput& input,
        const std::vector<TargetDecodeStateStore::DecodeState>& last_states,
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
  kv_seq_lens_vec.reserve(num_sequences + 1);

  const torch::Tensor& token_ids_cpu = input.token_ids_host;
  const torch::Tensor& positions_cpu = input.positions_host;
  const Slice<int32_t> input_token_ids = tensor_slice(token_ids_cpu);
  const Slice<int32_t> input_positions = tensor_slice(positions_cpu);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size())
        << "decode context token seq_id out of range, seq_id=" << seq_id;
    CHECK_LT(static_cast<size_t>(seq_id), input_positions.size())
        << "decode context position seq_id out of range, seq_id=" << seq_id;
    const TargetDecodeStateStore::DecodeState& state = last_states[seq_id];
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
    int32_t current_kv_len = specBuilder::calc_kv_len(
        input.input_params.attention.host.kv_seq_lens, seq_id, position_offset);
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
    specBuilder::append_seq_len_by_layout(kv_seq_lens_vec, current_kv_len);
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
    draft_input_spec_ = {
        draft_input_hidden_width(draft_impl_->context_.get_model_args()),
        draft_impl_->dtype()};
    draft_input_spec_valid_ = true;
  }
  return draft_input_spec_;
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
