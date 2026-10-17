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

#include "core/distributed_runtime/collective_service.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace xllm {
namespace {

TEST(CollectiveServiceTest, RegistersWorkersWithoutUnusedCommunicationRoots) {
  for (int32_t world_size : {1, 2}) {
    CollectiveService service(world_size);
    for (int32_t rank = 0; rank < world_size; ++rank) {
      proto::AddressInfo request;
      request.set_global_rank(rank);
      request.set_address("worker-" + std::to_string(rank));
      proto::CommUniqueIdList response;
      service.Sync(nullptr, &request, &response, nullptr);
      EXPECT_EQ(response.comm_unique_ids_size(), 0);
    }
    const auto addresses = service.wait();
    ASSERT_EQ(addresses.size(), world_size);
    for (int32_t rank = 0; rank < world_size; ++rank) {
      EXPECT_EQ(addresses.at(rank), "worker-" + std::to_string(rank));
    }
  }
}

}  // namespace
}  // namespace xllm
