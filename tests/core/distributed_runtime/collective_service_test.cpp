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

#include <chrono>
#include <cstdint>
#include <future>
#include <string>

namespace xllm {
namespace {

void register_worker(CollectiveService& service,
                     int32_t rank,
                     const std::string& address) {
  proto::AddressInfo worker;
  proto::CommUniqueIdList response;
  worker.set_global_rank(rank);
  worker.set_address(address);
  service.Sync(nullptr, &worker, &response, nullptr);
}

TEST(CollectiveServiceTest, RendezvousNeedsNoDeviceOrCommunicator) {
  // Address discovery must work before device/communicator initialization.
  CollectiveService service(/*total_num=*/2);
  proto::AddressInfo worker;
  proto::CommUniqueIdList response;
  worker.set_global_rank(1);
  worker.set_address("127.0.0.1:9001");
  service.Sync(nullptr, &worker, &response, nullptr);
  EXPECT_EQ(response.comm_unique_ids_size(), 0);

  worker.set_global_rank(0);
  worker.set_address("127.0.0.1:9000");
  service.Sync(nullptr, &worker, &response, nullptr);
  EXPECT_EQ(response.comm_unique_ids_size(), 0);

  const auto addresses = service.wait();
  ASSERT_EQ(addresses.size(), 2);
  EXPECT_EQ(addresses.at(0), "127.0.0.1:9000");
  EXPECT_EQ(addresses.at(1), "127.0.0.1:9001");
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

  TEST(CollectiveServiceTest, MissingWorkersTimeOutWithoutPartialAddresses) {
    CollectiveService service(/*total_num=*/2);
    register_worker(service, /*rank=*/0, "127.0.0.1:9000");

    const auto addresses = service.wait_until(std::chrono::steady_clock::now());

    EXPECT_FALSE(addresses.has_value());
  }

  TEST(CollectiveServiceTest, TimedWaitAcceptsWorkersRegisteredBeforeWaiting) {
    CollectiveService service(/*total_num=*/2);
    register_worker(service, /*rank=*/0, "127.0.0.1:9000");
    register_worker(service, /*rank=*/1, "127.0.0.1:9001");

    const auto addresses = service.wait_until(std::chrono::steady_clock::now() +
                                              std::chrono::seconds(1));

    ASSERT_TRUE(addresses.has_value());
    ASSERT_EQ(addresses->size(), 2);
    EXPECT_EQ(addresses->at(0), "127.0.0.1:9000");
    EXPECT_EQ(addresses->at(1), "127.0.0.1:9001");
  }

  TEST(CollectiveServiceTest, FinalRegistrationWakesTimedWaitBeforeDeadline) {
    CollectiveService service(/*total_num=*/2);
    register_worker(service, /*rank=*/0, "127.0.0.1:9000");
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    auto result = std::async(std::launch::async, [&service, deadline]() {
      return service.wait_until(deadline);
    });
    EXPECT_EQ(result.wait_for(std::chrono::milliseconds(20)),
              std::future_status::timeout);

    register_worker(service, /*rank=*/1, "127.0.0.1:9001");
    EXPECT_EQ(result.wait_for(std::chrono::seconds(1)),
              std::future_status::ready);
    const auto addresses = result.get();
    ASSERT_TRUE(addresses.has_value());
    ASSERT_EQ(addresses->size(), 2);
    EXPECT_EQ(addresses->at(1), "127.0.0.1:9001");
  }

  TEST(CollectiveServiceTest, TimedOutRendezvousCanObserveLateWorkersOnRetry) {
    CollectiveService service(/*total_num=*/2);
    register_worker(service, /*rank=*/0, "127.0.0.1:9000");
    const auto expired_addresses = service.wait_until(
        std::chrono::steady_clock::now() + std::chrono::milliseconds(20));
    ASSERT_FALSE(expired_addresses.has_value());

    register_worker(service, /*rank=*/1, "127.0.0.1:9001");

    // A new rendezvous may observe registrations made after the previous
    // timeout.
    const auto addresses = service.wait_until(std::chrono::steady_clock::now() +
                                              std::chrono::seconds(1));
    ASSERT_TRUE(addresses.has_value());
    EXPECT_EQ(addresses->size(), 2);
  }

}  // namespace
}  // namespace xllm
