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

#include <aclnnop/aclnn_sparse_flash_attention.h>
#include <dlfcn.h>
#include <torch/library.h>

#if __has_include(<version/cann_version.h>)
#include <version/cann_version.h>
#endif

#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

at::Tensor construct_sparse_flash_attention_output_tensor(
    const at::Tensor& query) {
  return at::empty(query.sizes(), query.options().dtype(query.dtype()));
}

void check_sparse_flash_attention_shape_and_dtype(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    int64_t sparse_block_size,
    const c10::string_view& layout_query,
    const c10::string_view& layout_kv) {
  TORCH_CHECK(query.dim() >= 1,
              "query's dim num should be at least 1, actual ",
              query.dim(),
              ".");
  TORCH_CHECK(query.dtype() == at::kHalf || query.dtype() == at::kBFloat16,
              "query should be FLOAT16 or BFLOAT16.");
  TORCH_CHECK(key.dtype() == query.dtype(),
              "key's dtype should be equal to query's dtype.");
  TORCH_CHECK(value.dtype() == query.dtype(),
              "value's dtype should be equal to query's dtype.");
  TORCH_CHECK(sparse_indices.dtype() == at::kInt,
              "sparse_indices should be INT32.");
  TORCH_CHECK(sparse_block_size > 0,
              "sparse_block_size should be greater than 0, actual ",
              sparse_block_size,
              ".");
  TORCH_CHECK(!layout_query.empty(), "layout_query should not be empty.");
  TORCH_CHECK(!layout_kv.empty(), "layout_kv should not be empty.");
}

#if defined(CANN_MAJOR) && CANN_MAJOR >= 9
std::string_view resolved_op_api_provider(const char* function_name) {
  void* function = aclnn::detail::get_op_api_func_addr(function_name);
  CHECK(function != nullptr) << function_name << " is unavailable";
  Dl_info provider_info{};
  CHECK_NE(dladdr(function, &provider_info), 0);
  CHECK(provider_info.dli_fname != nullptr);
  return provider_info.dli_fname;
}

bool uses_legacy_sparse_flash_attention_abi() {
  static const bool legacy_abi = [] {
    const std::string_view provider =
        resolved_op_api_provider("aclnnSparseFlashAttentionGetWorkspaceSize");
    const std::string_view execute_provider =
        resolved_op_api_provider("aclnnSparseFlashAttention");
    CHECK(provider == execute_provider)
        << "SparseFlashAttention workspace and execute providers differ: "
        << provider << " vs. " << execute_provider;
    if (provider.find("/custom_xllm_math/") != std::string_view::npos) {
      CHECK(resolved_op_api_provider(
                "aclnnSparseFlashAttentionLseGetWorkspaceSize") == provider);
      CHECK(resolved_op_api_provider("aclnnSparseFlashAttentionLse") ==
            provider);
      return true;
    }
    CHECK(provider.find("/glm_next_transformer/") != std::string_view::npos ||
          provider.find("/libopapi.so") != std::string_view::npos ||
          provider.find("/libopapi_transformer.so") != std::string_view::npos)
        << "unknown SparseFlashAttention ABI provider: " << provider;
    return false;
  }();
  return legacy_abi;
}
#endif

void launch_sparse_flash_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& query_rope,
    const c10::optional<at::Tensor>& key_rope,
    double scale_value,
    int64_t sparse_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode,
    at::Tensor& output) {
  std::string query_layout_str(layout_query);
  std::string kv_layout_str(layout_kv);
  char* query_layout_ptr = query_layout_str.data();
  char* kv_layout_ptr = kv_layout_str.data();

#if defined(CANN_MAJOR) && CANN_MAJOR >= 9
  // CANN 9's built-in and glm_next_transformer providers add four arguments
  // and two required outputs. custom_xllm_math's old SparseFlashAttention
  // symbol cannot supply those outputs to CANN 9, but its Lse variant can.
  // Dispatch by the resolved provider, not the installed CANN version alone.
  static_assert(std::is_invocable_r_v<
                aclnnStatus,
                decltype(&aclnnSparseFlashAttentionGetWorkspaceSize),
                const aclTensor*,
                const aclTensor*,
                const aclTensor*,
                const aclTensor*,
                const aclTensor*,
                const aclTensor*,
                const aclTensor*,
                const aclTensor*,
                const aclTensor*,
                double,
                int64_t,
                char*,
                char*,
                int64_t,
                int64_t,
                int64_t,
                int64_t,
                bool,
                const aclTensor*,
                const aclTensor*,
                const aclTensor*,
                uint64_t*,
                aclOpExecutor**>);
  constexpr int64_t kUnboundedWindow = std::numeric_limits<int64_t>::max();
  constexpr int64_t kAttentionMode = 2;
  bool return_softmax_lse = false;
  at::Tensor softmax_max = at::empty({0}, query.options().dtype(at::kFloat));
  at::Tensor softmax_sum = at::empty({0}, query.options().dtype(at::kFloat));
  if (uses_legacy_sparse_flash_attention_abi()) {
    EXEC_NPU_CMD(aclnnSparseFlashAttentionLse,
                 query,
                 key,
                 value,
                 sparse_indices,
                 block_table,
                 actual_seq_lengths_query,
                 actual_seq_lengths_kv,
                 query_rope,
                 key_rope,
                 scale_value,
                 sparse_block_size,
                 query_layout_ptr,
                 kv_layout_ptr,
                 sparse_mode,
                 kUnboundedWindow,
                 kUnboundedWindow,
                 kAttentionMode,
                 return_softmax_lse,
                 output,
                 softmax_max,
                 softmax_sum);
  } else {
    EXEC_NPU_CMD(aclnnSparseFlashAttention,
                 query,
                 key,
                 value,
                 sparse_indices,
                 block_table,
                 actual_seq_lengths_query,
                 actual_seq_lengths_kv,
                 query_rope,
                 key_rope,
                 scale_value,
                 sparse_block_size,
                 query_layout_ptr,
                 kv_layout_ptr,
                 sparse_mode,
                 kUnboundedWindow,
                 kUnboundedWindow,
                 kAttentionMode,
                 return_softmax_lse,
                 output,
                 softmax_max,
                 softmax_sum);
  }
#else
  EXEC_NPU_CMD(aclnnSparseFlashAttention,
               query,
               key,
               value,
               sparse_indices,
               block_table,
               actual_seq_lengths_query,
               actual_seq_lengths_kv,
               query_rope,
               key_rope,
               scale_value,
               sparse_block_size,
               query_layout_ptr,
               kv_layout_ptr,
               sparse_mode,
               output);
#endif
}

}  // namespace

at::Tensor sparse_flash_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& query_rope,
    const c10::optional<at::Tensor>& key_rope,
    double scale_value,
    int64_t sparse_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode) {
  check_sparse_flash_attention_shape_and_dtype(query,
                                               key,
                                               value,
                                               sparse_indices,
                                               sparse_block_size,
                                               layout_query,
                                               layout_kv);
  at::Tensor out = construct_sparse_flash_attention_output_tensor(query);

  launch_sparse_flash_attention(query,
                                key,
                                value,
                                sparse_indices,
                                block_table,
                                actual_seq_lengths_query,
                                actual_seq_lengths_kv,
                                query_rope,
                                key_rope,
                                scale_value,
                                sparse_block_size,
                                layout_query,
                                layout_kv,
                                sparse_mode,
                                out);

  return out;
}

at::Tensor sparse_flash_attention_out(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& query_rope,
    const c10::optional<at::Tensor>& key_rope,
    double scale_value,
    int64_t sparse_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode,
    at::Tensor& output) {
  check_sparse_flash_attention_shape_and_dtype(query,
                                               key,
                                               value,
                                               sparse_indices,
                                               sparse_block_size,
                                               layout_query,
                                               layout_kv);
  CHECK(output.is_contiguous()) << "output must be contiguous";
  CHECK(output.sizes() == query.sizes())
      << "output shape must match query shape";
  CHECK(output.scalar_type() == query.scalar_type())
      << "output dtype must match query dtype";

  launch_sparse_flash_attention(query,
                                key,
                                value,
                                sparse_indices,
                                block_table,
                                actual_seq_lengths_query,
                                actual_seq_lengths_kv,
                                query_rope,
                                key_rope,
                                scale_value,
                                sparse_block_size,
                                layout_query,
                                layout_kv,
                                sparse_mode,
                                output);
  return output;
}

}  // namespace xllm::kernel::npu
