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

#pragma once
#include <folly/Function.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "concurrent_queue.h"
#include "concurrentqueue.h"
#include "lightweightsemaphore.h"
#include "util/blocking_counter.h"
namespace xllm {

// Fan-out coordinator that replaces the `std::vector<std::promise<void>>` +
// `std::vector<std::future<void>>::get()` pattern. Uses a single
// `BlockingCounter` for completion plus first-exception capture for
// propagation. Compared with the per-task promise/future model this avoids
//   * N heap allocations for `promise<void>`'s shared state,
//   * N atomic refcount ops on `shared_ptr`,
//   * N condvar wakeups in `future.get()` (one per future),
// and replaces them with one condvar wakeup at `wait()` time.
//
// Designed for the schedule-then-wait pattern where the caller knows the
// exact task count up front.
//
// Usage:
//   TaskGroup group(num_tasks);
//   for (...) thread_pool->schedule(group.wrap([&] { work(); }));
//   group.wait();  // rethrows the first exception, if any
class TaskGroup final {
 public:
  explicit TaskGroup(int32_t count) : counter_(count) {}

  TaskGroup(const TaskGroup&) = delete;
  TaskGroup& operator=(const TaskGroup&) = delete;
  TaskGroup(TaskGroup&&) = delete;
  TaskGroup& operator=(TaskGroup&&) = delete;

  // Wrap a callable so that the first thrown exception is captured and the
  // counter is decremented exactly once, regardless of how the body exits.
  // Must be called exactly `count` times in total across all wrapped tasks.
  template <typename F>
  auto wrap(F body) {
    return [this, body = std::move(body)]() mutable {
      try {
        body();
      } catch (...) {
        capture_exception();
      }
      counter_.decrement_count();
    };
  }

  // Block until all wrapped tasks have completed. Rethrows the first
  // exception captured by any task, if any.
  void wait() {
    counter_.wait();
    if (first_exception_) {
      std::rethrow_exception(first_exception_);
    }
  }

 private:
  void capture_exception() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!first_exception_) {
      first_exception_ = std::current_exception();
    }
  }

  BlockingCounter counter_;
  std::mutex mu_;
  std::exception_ptr first_exception_;
};

// A FIFO queue-based thread pool.
//
// Internally the pool maintains:
// * one private FIFO queue per worker (`queues_`), used by
//   `schedule_with_tid`. Tasks pushed via `schedule_with_tid(task, tid)`
//   are guaranteed to run on worker `tid` in FIFO order. This preserves
//   the strict per-tid ordering and worker affinity required by callers
class ThreadPool final {
 public:
  // a runnable is an object intended to be executed by the threadpool
  // it must be invocable with no arguments and return void.
  using Runnable = folly::Function<void()>;

  // constructors
  ThreadPool() : ThreadPool(1) {}
  // destructor
  ~ThreadPool();

  // disable copy/move constructor and assignment
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  explicit ThreadPool(size_t num_threads,
                      bool cpu_binding = false,
                      const std::string& pool_name = "");
  explicit ThreadPool(size_t num_threads,
                      Runnable init_func,
                      bool cpu_binding = false,
                      const std::string& pool_name = "");
  // Bind each worker thread to the corresponding CPU core in cpu_cores.
  // cpu_cores[i] is the CPU core ID for thread i. If cpu_cores is empty,
  // no binding is performed. If cpu_cores.size() does not equal num_threads,
  // a warning will be logged and CPU core binding will be skipped.
  explicit ThreadPool(size_t num_threads,
                      Runnable init_func,
                      std::vector<int32_t> cpu_cores,
                      const std::string& pool_name = "");

  // schedule a runnable to be executed
  int32_t schedule(Runnable runnable);

  void schedule_with_tid(Runnable runnable, size_t tid);

  bool empty() {
    return std::all_of(queues_.begin(), queues_.end(), [](auto& queue) {
      return queue.empty();
    });
  }

  size_t size() { return threads_.size(); }

 private:
  void internal_loop(size_t tid,
                     std::shared_ptr<Runnable> init_func,
                     std::shared_ptr<BlockingCounter> block_counter,
                     int32_t cpu_core);

  std::vector<std::thread> threads_;
  std::vector<ConcurrentQueue<Runnable>> queues_;

  std::atomic<size_t> index_{0};
  std::string pool_name_;
};

// A lock-free MPMC queue-based thread pool.
//
// Internally the pool maintains:
// * one lock-free MPMC queue shared by all workers (`global_queue_`),
//   used by `schedule`. Any idle worker can steal from this queue, which
//   eliminates the head-of-line blocking that the previous round-robin /
//   per-worker queue design exhibited when one worker was preempted by
//   the OS or ran a long task.
class MPMCThreadPool final {
 public:
  // a runnable is an object intended to be executed by the threadpool
  // it must be invocable with no arguments and return void.
  using Runnable = folly::Function<void()>;

  // constructors
  MPMCThreadPool() : MPMCThreadPool(1) {}
  // destructor
  ~MPMCThreadPool();

  // disable copy/move constructor and assignment
  MPMCThreadPool(const MPMCThreadPool&) = delete;
  MPMCThreadPool& operator=(const MPMCThreadPool&) = delete;
  MPMCThreadPool(MPMCThreadPool&&) = delete;
  MPMCThreadPool& operator=(MPMCThreadPool&&) = delete;

  explicit MPMCThreadPool(size_t num_threads,
                          bool cpu_binding = false,
                          const char* fn = nullptr,
                          int32_t ln = 0,
                          const std::string& pool_name = "");
  explicit MPMCThreadPool(size_t num_threads,
                          Runnable init_func,
                          bool cpu_binding = false,
                          const char* fn = nullptr,
                          int32_t ln = 0,
                          const std::string& pool_name = "");
  // Bind each worker thread to the corresponding CPU core in cpu_cores.
  // cpu_cores[i] is the CPU core ID for thread i. If cpu_cores is empty,
  // no binding is performed. If cpu_cores.size() does not equal num_threads,
  // a warning will be logged and CPU core binding will be skipped.
  explicit MPMCThreadPool(size_t num_threads,
                          Runnable init_func,
                          std::vector<int32_t> cpu_cores,
                          const char* fn = nullptr,
                          int32_t ln = 0,
                          const std::string& pool_name = "");

  // Schedule a runnable on the global queue. Any idle worker may pick it up.
  void schedule(Runnable runnable);

  bool empty() {
    if (global_queue_.size_approx() != 0) {
      return false;
    }
    return true;
  }

  size_t size() { return threads_.size(); }

 private:
  void internal_loop(size_t tid,
                     Runnable* init_func,
                     BlockingCounter* block_counter,
                     int32_t cpu_core);

  // Drain the global queue from within a worker. Returns true if at least
  // one task was executed.
  bool drain_global_queue();

  std::vector<std::thread> threads_;

  // Lock-free MPMC queue used by `schedule`. Any worker can steal.
  moodycamel::ConcurrentQueue<Runnable> global_queue_;

  // Wakeup carrier: one semaphore per worker. `schedule` broadcasts to all.
  std::vector<std::unique_ptr<moodycamel::LightweightSemaphore>> sems_;

  // Flips to true in the destructor to make idle workers exit.
  std::atomic<bool> stopped_{false};

  std::string pool_name_;
};

}  // namespace xllm
