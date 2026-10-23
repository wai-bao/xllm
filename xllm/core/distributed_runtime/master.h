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

#include <torch/types.h>

#include <atomic>
#include <thread>
#include <vector>

#include "common/options.h"
#include "common/rate_limiter.h"
#include "core/distributed_runtime/xservice_client.h"
#include "core/framework/request/request_output.h"
namespace xllm {

class Engine;
class ModelArgs;
class Scheduler;

class Master {
 public:
  explicit Master(const Options& options);
  virtual ~Master();
  // Rank-0 masters override this to start the scheduler loop. Non-zero
  // ranks use the default implementation, which starts a background idle
  // thread. The destructor stops that thread.
  virtual void run();
  // Block until the idle thread exits (SIGINT/SIGTERM). Used by the
  // binary when no HTTP server is started on a non-leader rank.
  void wait();
  const Options& options() const { return options_; }

  virtual const ModelArgs* model_args() const = 0;

  RateLimiter* get_rate_limiter() { return &rate_limiter_; }

  // The response transport can outlive this master's serving registration.
  OutputCallback make_rpc_response_callback() const;
  // A last-resort sink for error paths that run before any master (and its
  // xservice client) is resolved: reports the failure status and drops the
  // response, so early validation failures never call an empty callback.
  static OutputCallback make_error_response_callback();
  // Stamps the request's streaming mode onto every error output forwarded
  // through the wrapped callback. Masters install this once at their request
  // entry, so every downstream error path (verify_params, request factory,
  // add_request) reports the envelope the transport layer needs: it picks
  // the SSE frame vs plain-JSON error body from RequestOutput::stream.
  static OutputCallback stamp_stream_on_error(OutputCallback callback,
                                              bool stream);

 protected:
  // node_rank == 0. Only the base class and derived masters branch on this;
  // no external caller needs it.
  bool is_leader() const { return options_.node_rank() == 0; }

  // Acquire the shared discovery client when service routing is enabled, and
  // register this instance once its scheduler exists. The acquire/bind
  // lifecycle lives in one place here, but the registration handle is returned
  // rather than stored: it borrows the derived class's scheduler and engine, so
  // the derived class must own it and declare it after them. Member
  // destruction order then releases the registration first, which the language
  // guarantees -- a base-class member could not.
  void acquire_service_client();
  [[nodiscard]] std::unique_ptr<XServiceRegistration> bind_service_registration(
      Scheduler* scheduler,
      Engine* engine);

  Options options_;
  std::vector<torch::Device> devices_;
  RateLimiter rate_limiter_;
  std::shared_ptr<XServiceClient> xservice_client_;

 private:
  static void handle_shutdown_signal(int signum);
  static std::atomic<bool> idle_running_;
  std::thread idle_thread_;
};

}  // namespace xllm
