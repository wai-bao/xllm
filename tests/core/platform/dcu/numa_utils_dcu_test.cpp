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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <cstdint>

#include "core/platform/numa_utils.h"
#include "core/util/scope_guard.h"

namespace {

// =============================================================================
// Test environment
// =============================================================================

class NumaUtilsTestEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    google::InitGoogleLogging("numa_utils_test");
    google::SetStderrLogging(google::INFO);
  }

  void TearDown() override { google::ShutdownGoogleLogging(); }
};

::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new NumaUtilsTestEnvironment);

// =============================================================================
// File-local helpers
// =============================================================================

int32_t current_numa_node() {
  if (numa_available() < 0) {
    return -1;
  }
  const int32_t cpu = sched_getcpu();
  return cpu < 0 ? -1 : numa_node_of_cpu(cpu);
}

bool SaveCurrentThreadAffinity(cpu_set_t& cpu_set) {
  CPU_ZERO(&cpu_set);
  return pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set) ==
         0;
}

bool RestoreCurrentThreadAffinity(const cpu_set_t& cpu_set) {
  return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set) ==
         0;
}

bool HasHipDevice() {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

// =============================================================================
// Tests
// =============================================================================

// xllm::numa APIs under test: is_numa_available, get_num_numa_nodes
TEST(NumaUtilsTest, AvailabilityAndNodeCountAreConsistent) {
  const bool available = xllm::numa::is_numa_available();
  const int32_t num_nodes = xllm::numa::get_num_numa_nodes();

  if (!available) {
    EXPECT_EQ(num_nodes, -1);
  } else {
    EXPECT_GT(num_nodes, 0);
  }
}

TEST(NumaUtilsTest, InvalidNodeQueriesReturnEmptyOrError) {
  const bool available = xllm::numa::is_numa_available();

  if (!available) {
    GTEST_SKIP() << "NUMA is not available on this system";
  }

  EXPECT_NE(xllm::numa::bind_thread_to_numa_node(-1), 0);
  EXPECT_NE(xllm::numa::bind_process_to_numa_node(-1), 0);
  EXPECT_NE(
      xllm::numa::bind_thread_to_numa_node(xllm::numa::get_num_numa_nodes()),
      0);
  EXPECT_NE(
      xllm::numa::bind_process_to_numa_node(xllm::numa::get_num_numa_nodes()),
      0);
}

TEST(NumaUtilsTest, ValidNodeCpuListLooksReasonable) {
  const int32_t current_node = current_numa_node();
  if (current_node < 0) {
    GTEST_SKIP() << "Unable to determine current NUMA node";
  }

  cpu_set_t saved_affinity;
  ASSERT_TRUE(SaveCurrentThreadAffinity(saved_affinity));
  xllm::ScopeGuard restore_affinity(
      [&] { EXPECT_TRUE(RestoreCurrentThreadAffinity(saved_affinity)); });
  ASSERT_EQ(xllm::numa::bind_thread_to_numa_node(current_node), 0);
  cpu_set_t thread_affinity;
  ASSERT_TRUE(SaveCurrentThreadAffinity(thread_affinity));
  for (int32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    const bool allowed = CPU_ISSET(cpu, &saved_affinity) &&
                         numa_node_of_cpu(cpu) == current_node;
    EXPECT_EQ(CPU_ISSET(cpu, &thread_affinity), allowed) << "CPU " << cpu;
  }
}

TEST(NumaUtilsTest, BuildCpuSetRespectsProcessAffinityConstraints) {
  const bool available = xllm::numa::is_numa_available();

  if (!available) {
    GTEST_SKIP() << "NUMA is not available on this system";
  }

  const int32_t num_nodes = xllm::numa::get_num_numa_nodes();
  if (num_nodes < 2) {
    GTEST_SKIP() << "Need >=2 NUMA nodes for intersection test";
  }

  cpu_set_t saved_affinity;
  ASSERT_EQ(sched_getaffinity(getpid(), sizeof(cpu_set_t), &saved_affinity), 0);
  int32_t node0_cpu = -1;
  for (int32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &saved_affinity) && numa_node_of_cpu(cpu) == 0) {
      node0_cpu = cpu;
      break;
    }
  }
  if (node0_cpu < 0) {
    GTEST_SKIP() << "Node 0 has no allowed CPU";
  }

  cpu_set_t restricted;
  CPU_ZERO(&restricted);
  CPU_SET(node0_cpu, &restricted);
  ASSERT_EQ(sched_setaffinity(getpid(), sizeof(cpu_set_t), &restricted), 0);
  EXPECT_NE(xllm::numa::bind_thread_to_numa_node(1), 0);

  ASSERT_EQ(sched_setaffinity(getpid(), sizeof(cpu_set_t), &saved_affinity), 0);
}

// NOTE: bind_process_to_numa_node also calls numa_set_membind / numa_set_strict
// which are process-global memory policy settings with no standard undo API.
TEST(NumaUtilsTest, BindProcessToValidNodeSucceeds) {
  const bool available = xllm::numa::is_numa_available();

  if (!available) {
    GTEST_SKIP() << "NUMA is not available on this system";
  }

  const int32_t num_nodes = xllm::numa::get_num_numa_nodes();
  ASSERT_GT(num_nodes, 0);

  // Capture initial state before any binding.
  const int32_t initial_node = current_numa_node();

  // Pick a target node different from the initial one.
  int32_t target_node = (initial_node + 1) % num_nodes;

  cpu_set_t saved_affinity;
  CPU_ZERO(&saved_affinity);
  ASSERT_EQ(sched_getaffinity(getpid(), sizeof(cpu_set_t), &saved_affinity), 0);

  // Bind to target node.
  const int32_t bind_ret = xllm::numa::bind_process_to_numa_node(target_node);
  EXPECT_EQ(bind_ret, 0);

  // Yield to ensure the thread is migrated to a CPU within the new affinity
  // mask before querying the current NUMA node.
  sched_yield();

  const int32_t reported_node = current_numa_node();
  EXPECT_EQ(reported_node, target_node);

  // Rebind to initial node and verify the transition works both ways.
  // Note: bind_process_to_numa_node modifies process CPU affinity, so restore
  // the saved affinity first so that build_cpu_set_for_numa_node can find
  // the CPUs of the initial node (they would otherwise be filtered out).
  if (initial_node != target_node) {
    ASSERT_EQ(sched_setaffinity(getpid(), sizeof(cpu_set_t), &saved_affinity),
              0);
    const int32_t rebind_ret =
        xllm::numa::bind_process_to_numa_node(initial_node);
    EXPECT_EQ(rebind_ret, 0);

    sched_yield();

    const int32_t restored_node = current_numa_node();
    EXPECT_EQ(restored_node, initial_node);
  }

  ASSERT_EQ(sched_setaffinity(getpid(), sizeof(cpu_set_t), &saved_affinity), 0);
}

TEST(NumaUtilsTest, BindThreadToNumaNodeSucceeds) {
  const bool available = xllm::numa::is_numa_available();

  if (!available) {
    GTEST_SKIP() << "NUMA is not available on this system";
  }

  const int32_t num_nodes = xllm::numa::get_num_numa_nodes();
  ASSERT_GT(num_nodes, 0);

  const int32_t initial_node = current_numa_node();

  const int32_t target_node = (initial_node + 2) % num_nodes;

  // Save process affinity so we can restore it before rebinding (thread
  // affinity changes affect sched_getaffinity(0, ...) in
  // build_cpu_set_for_numa_node).
  cpu_set_t saved_affinity;
  CPU_ZERO(&saved_affinity);
  ASSERT_EQ(sched_getaffinity(getpid(), sizeof(cpu_set_t), &saved_affinity), 0);

  // Bind the thread to target node.
  const int32_t bind_ret = xllm::numa::bind_thread_to_numa_node(target_node);
  EXPECT_EQ(bind_ret, 0);

  sched_yield();

  const int32_t reported_node = current_numa_node();
  EXPECT_EQ(reported_node, target_node);

  // Bind back to initial node.
  if (initial_node != target_node) {
    ASSERT_EQ(sched_setaffinity(getpid(), sizeof(cpu_set_t), &saved_affinity),
              0);
    const int32_t rebind_ret =
        xllm::numa::bind_thread_to_numa_node(initial_node);
    EXPECT_EQ(rebind_ret, 0);

    sched_yield();

    const int32_t restored_node = current_numa_node();
    EXPECT_EQ(restored_node, initial_node);
  }

  ASSERT_EQ(sched_setaffinity(getpid(), sizeof(cpu_set_t), &saved_affinity), 0);
}

// xllm::numa APIs under test: get_device_numa_node, is_numa_available,
// get_num_numa_nodes, device_count
TEST(NumaUtilsTest, GetDeviceNumaNodeReturnsValidNodeOrMinusOneForHipDevice) {
  if (!HasHipDevice()) {
    GTEST_SKIP() << "No HIP/DCU device available";
  }

  const bool available = xllm::numa::is_numa_available();
  const int32_t num_numa_nodes = xllm::numa::get_num_numa_nodes();
  int device_count = 0;
  ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
  ASSERT_GT(device_count, 0);

  for (int device_id = 0; device_id < device_count; ++device_id) {
    const int32_t numa_node = xllm::numa::get_device_numa_node(device_id);

    if (!available) {
      EXPECT_EQ(numa_node, -1);
    } else {
      // Some environments may not expose sysfs; -1 is acceptable.
      if (numa_node != -1) {
        EXPECT_GE(numa_node, 0);
        EXPECT_LT(numa_node, num_numa_nodes);
      }
    }
  }
}

}  // namespace
