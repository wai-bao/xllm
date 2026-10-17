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

#include "core/distributed_runtime/comm_channel.h"

#include <brpc/closure_guard.h>
#include <brpc/details/controller_private_accessor.h>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include <gtest/gtest.h>
#include <sys/socket.h>

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <thread>

#include "core/framework/speculative/draft_input_state_store.h"
#include "runtime/rec_forward_params.h"
#include "runtime/vlm_forward_params.h"

namespace xllm {
namespace {

class ModelRoutingService final : public proto::DistributeWorker {
 public:
  void ExecuteLLMModel(google::protobuf::RpcController* controller,
                       const proto::LLMForwardRequest* request,
                       proto::LLMForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (!request->has_packed_input() ||
        request->packed_input().payload().size() < 11 ||
        static_cast<uint8_t>(request->packed_input().payload()[10]) != 1) {
      controller->SetFailed("Unexpected packed input domain");
      return;
    }
    response->add_outputs()->add_tokens()->set_id(1);
  }

  void ExecuteVLMModel(google::protobuf::RpcController* controller,
                       const proto::VLMForwardRequest* request,
                       proto::VLMForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (!request->has_packed_input() ||
        request->packed_input().payload().size() < 11 ||
        static_cast<uint8_t>(request->packed_input().payload()[10]) != 2) {
      controller->SetFailed("Unexpected packed input domain");
      return;
    }
    response->add_outputs()->add_tokens()->set_id(2);
  }

  void ExecuteRECModel(google::protobuf::RpcController* controller,
                       const proto::RECForwardRequest* request,
                       proto::RECForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (!request->has_packed_input() ||
        request->packed_input().payload().size() < 11 ||
        static_cast<uint8_t>(request->packed_input().payload()[10]) != 3) {
      controller->SetFailed("Unexpected packed input domain");
      return;
    }
    response->add_outputs()->add_tokens()->set_id(3);
  }

  void ExecuteDITModel(google::protobuf::RpcController* controller,
                       const proto::DITForwardRequest* request,
                       proto::DITForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (!request->has_packed_input() ||
        request->packed_input().payload().size() < 11 ||
        static_cast<uint8_t>(request->packed_input().payload()[10]) != 4) {
      controller->SetFailed("Unexpected packed input domain");
      return;
    }
    response->mutable_dit_forward_output()->add_text_output("diffusion");
  }

  void GetLastLLMStepResult(google::protobuf::RpcController* controller,
                            const proto::Empty* /*request*/,
                            proto::LLMForwardResponse* /*response*/,
                            google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    controller->SetFailed("result unavailable");
  }

  void GetLastVLMStepResult(google::protobuf::RpcController* controller,
                            const proto::Empty* /*request*/,
                            proto::VLMForwardResponse* /*response*/,
                            google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    controller->SetFailed("result unavailable");
  }
};

class CommChannelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
    ASSERT_TRUE(channel_.init_brpc(
        butil::endpoint2str(server_.listen_address()).c_str()));
  }

  void TearDown() override {
    server_.Stop(/*timeout_ms=*/0);
    server_.Join();
  }

  ModelRoutingService service_;
  brpc::Server server_;
  CommChannel channel_;
};

TEST_F(CommChannelTest, RoutesLLMInputToItsOwnRpc) {
  LlmForwardInput input;
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  channel_.execute_llm_model_async(input, promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->outputs.size(), 1);
  ASSERT_EQ(output->outputs[0].tokens.size(), 1);
  EXPECT_EQ(output->outputs[0].tokens[0].id, 1);
}

TEST_F(CommChannelTest, RoutesVLMInputToItsOwnRpc) {
  VlmForwardInput input;
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  channel_.execute_vlm_model_async(input, promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->outputs.size(), 1);
  ASSERT_EQ(output->outputs[0].tokens.size(), 1);
  EXPECT_EQ(output->outputs[0].tokens[0].id, 2);
}

TEST_F(CommChannelTest, RoutesRECInputToItsOwnRpc) {
  RecForwardInput input;
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  channel_.execute_rec_model_async(input, promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->outputs.size(), 1);
  ASSERT_EQ(output->outputs[0].tokens.size(), 1);
  EXPECT_EQ(output->outputs[0].tokens[0].id, 3);
}

TEST_F(CommChannelTest, RoutesDITInputToItsOwnRpc) {
  DiTForwardInput input;
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  channel_.execute_dit_model_async(input, promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->dit_forward_output.text_output.size(), 1);
  EXPECT_EQ(output->dit_forward_output.text_output[0], "diffusion");
}

TEST_F(CommChannelTest, CompletesPromiseWhenLastLLMResultRpcFails) {
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  EXPECT_FALSE(channel_.get_last_llm_step_result_async(promise));
  ASSERT_TRUE(future.isReady());
  EXPECT_FALSE(std::move(future).get().has_value());
}

TEST_F(CommChannelTest, CompletesPromiseWhenLastVLMResultRpcFails) {
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  EXPECT_FALSE(channel_.get_last_vlm_step_result_async(promise));
  ASSERT_TRUE(future.isReady());
  EXPECT_FALSE(std::move(future).get().has_value());
}

class ReleaseService final : public proto::DistributeWorker {
 public:
  explicit ReleaseService(bool disconnect_first)
      : disconnect_first_(disconnect_first),
        resume_(resume_promise_.get_future().share()) {}
  ~ReleaseService() override {
    resume_original();
    if (original_.joinable()) {
      original_.join();
    }
  }

  void store(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.write_draft_input(
        {1},
        {request_id},
        {torch::tensor({3}, torch::kLong), torch::ones({1, 2}), {}, {}});
  }

  bool contains(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_.read_draft_input({1}, {request_id}).has_value();
  }

  void resume_original() {
    if (!resumed_.exchange(true)) {
      resume_promise_.set_value();
    }
  }

  std::future<void> original_done() { return done_promise_.get_future(); }
  int32_t calls() const { return calls_.load(); }

  void ReleaseIdleRequestState(google::protobuf::RpcController* controller,
                               const proto::RequestStateRelease* request,
                               proto::Status* response,
                               google::protobuf::Closure* done) override {
    std::vector<int32_t> ids(request->embedding_ids().begin(),
                             request->embedding_ids().end());
    const int32_t attempt = calls_.fetch_add(1);
    if (disconnect_first_ && attempt == 0) {
      auto* socket = brpc::ControllerPrivateAccessor(
                         static_cast<brpc::Controller*>(controller))
                         .get_sending_socket();
      CHECK(socket != nullptr);
      // Drop the connection while the server operation remains pending.
      CHECK_EQ(::shutdown(socket->fd(), SHUT_RDWR), 0);
      original_ = std::thread([this, ids = std::move(ids), response, done] {
        resume_.wait();
        {
          std::lock_guard<std::mutex> lock(cache_mutex_);
          cache_.clear(ids);
        }
        response->set_ok(true);
        done->Run();
        done_promise_.set_value();
      });
      return;
    }
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      cache_.clear(ids);
    }
    response->set_ok(true);
    done->Run();
  }

 private:
  const bool disconnect_first_;
  DraftInputStateStore cache_{2};
  std::mutex cache_mutex_;
  std::atomic<int32_t> calls_{0};
  std::atomic<bool> resumed_{false};
  std::promise<void> resume_promise_;
  std::shared_future<void> resume_;
  std::promise<void> done_promise_;
  std::thread original_;
};

class ReleaseServerGuard final {
 public:
  ReleaseServerGuard(ReleaseService& service, brpc::Server& server)
      : service_(service), server_(server) {}
  ~ReleaseServerGuard() {
    service_.resume_original();
    server_.Stop(/*closewait_ms=*/0);
    server_.Join();
  }

 private:
  ReleaseService& service_;
  brpc::Server& server_;
};

TEST(CommChannelReleaseTest, IdleReleaseAcknowledgesCompletedCleanup) {
  ReleaseService service(/*disconnect_first=*/false);
  service.store("request");
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  ReleaseServerGuard guard(service, server);
  CommChannel channel;
  ASSERT_TRUE(channel.init_brpc("127.0.0.1:" +
                                std::to_string(server.listen_address().port)));
  ASSERT_TRUE(channel.release_idle_request_state({1}));
  EXPECT_EQ(service.calls(), 1);
  EXPECT_FALSE(service.contains("request"));
}

TEST(CommChannelReleaseTest, DisconnectedIdleReleaseFailsWithoutRetry) {
  ReleaseService service(/*disconnect_first=*/true);
  service.store("request");
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  ReleaseServerGuard guard(service, server);
  CommChannel channel;
  ASSERT_TRUE(channel.init_brpc("127.0.0.1:" +
                                std::to_string(server.listen_address().port)));
  EXPECT_FALSE(channel.release_idle_request_state({1}));
  EXPECT_EQ(service.calls(), 1);
  EXPECT_TRUE(service.contains("request"));
  auto completed = service.original_done();
  service.resume_original();
  ASSERT_EQ(completed.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  completed.get();
  EXPECT_FALSE(service.contains("request"));
  EXPECT_EQ(service.calls(), 1);
}

}  // namespace
}  // namespace xllm
