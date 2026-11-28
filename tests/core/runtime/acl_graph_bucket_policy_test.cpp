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

#include "core/runtime/acl_graph_bucket_policy.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(AclGraphBucketPolicyUnitTest, MapsGlobalWarmupBucketsToLocalDpBatches) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/2,
      /*max_batch_size=*/16,
      /*dp_size=*/4,
      /*is_graph_warmup=*/false,
      /*decode_query_len=*/1));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/3,
      /*max_batch_size=*/16,
      /*dp_size=*/4,
      /*is_graph_warmup=*/false,
      /*decode_query_len=*/1));
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/10,
      /*max_batch_size=*/40,
      /*dp_size=*/4,
      /*is_graph_warmup=*/false,
      /*decode_query_len=*/1));
}

TEST(AclGraphBucketPolicyUnitTest, RejectsNonCanonicalSingleDpBatches) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/20,
      /*max_batch_size=*/20,
      /*dp_size=*/1,
      /*is_graph_warmup=*/false,
      /*decode_query_len=*/1));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/12,
      /*max_batch_size=*/20,
      /*dp_size=*/1,
      /*is_graph_warmup=*/false,
      /*decode_query_len=*/1));
}

TEST(AclGraphBucketPolicyUnitTest, AdmitsNoPaddingMtpTokenBuckets) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/12,
      /*max_batch_size=*/16,
      /*dp_size=*/1,
      /*is_graph_warmup=*/false,
      /*decode_query_len=*/4));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/11,
      /*max_batch_size=*/16,
      /*dp_size=*/1,
      /*is_graph_warmup=*/false,
      /*decode_query_len=*/4));
}

TEST(AclGraphBucketPolicyUnitTest, AdmitsNonDivisibleDpMtpTailBucket) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/3,
      /*max_batch_size=*/7,
      /*dp_size=*/2,
      /*is_graph_warmup=*/false,
      /*decode_query_len=*/4));
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/4,
      /*max_batch_size=*/7,
      /*dp_size=*/2,
      /*is_graph_warmup=*/false,
      /*decode_query_len=*/4));
}

TEST(AclGraphBucketPolicyUnitTest, AppliesWarmupCapacityInLocalDpUnits) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/4,
      /*max_batch_size=*/16,
      /*dp_size=*/4,
      /*is_graph_warmup=*/true,
      /*decode_query_len=*/1));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/5,
      /*max_batch_size=*/16,
      /*dp_size=*/4,
      /*is_graph_warmup=*/true,
      /*decode_query_len=*/1));
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/15,
      /*max_batch_size=*/16,
      /*dp_size=*/1,
      /*is_graph_warmup=*/true,
      /*decode_query_len=*/1));
}

}  // namespace
}  // namespace xllm
