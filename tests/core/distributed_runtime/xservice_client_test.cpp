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

#include "core/distributed_runtime/xservice_client.h"

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm {

class XServiceClientTestPeer final {
 public:
  static std::unique_ptr<XServiceClient> registration_client(
      const std::string& address) {
    auto client = std::unique_ptr<XServiceClient>(new XServiceClient());
    client->master_xservice_addr_ = address;
    client->http_service_protocol_ = true;
    client->chan_options_.protocol = "http";
    client->chan_options_.timeout_ms = 1000;
    client->chan_options_.max_retry = 0;
    return client;
  }

  static bool register_with_retry(XServiceClient& client) {
    return client.register_direct_instance_with_retry();
  }

  static void stop(XServiceClient& client) {
    {
      std::lock_guard<std::mutex> lock(client.registration_mutex_);
      client.exited_.store(true);
    }
    client.registration_cv_.notify_all();
    class XServiceGenerationTestPeer {
     public:
      static std::shared_ptr<XServiceClient> make_client() {
        return std::shared_ptr<XServiceClient>(new XServiceClient());
      }
    };

    namespace {

    enum class ResponseMode {
      NORMAL,
      TRANSPORT_FAILURE,
      MISMATCHED_STATUS_COUNT
    };

    class GenerationTestService final
        : public xllm_service::proto::XllmRpcService {
     public:
      explicit GenerationTestService(
          ResponseMode mode = ResponseMode::NORMAL,
          std::unordered_map<std::string, bool> results = {})
          : mode_(mode), results_(std::move(results)) {}

      void RegisterInstance(
          google::protobuf::RpcController* controller,
          const xllm_service::proto::InstanceMetaInfo* /*request*/,
          xllm_service::proto::StatusCode* response,
          google::protobuf::Closure* done) override {
        brpc::ClosureGuard done_guard(done);
        const int32_t attempt = ++registration_calls;
        if (attempt == 1) {
          registration_started.set_value();
        }
        if (attempt <= registration_failures) {
          controller->SetFailed("injected transient registration failure");
          return;
        }
        response->set_status_code(registration_status);
      }

      int32_t registration_failures = 0;
      int32_t registration_status = 0;
      std::atomic<int32_t> registration_calls{0};
      std::promise<void> registration_started;

      void Generations(google::protobuf::RpcController* controller,
                       const proto::DisaggStreamGenerations* request,
                       proto::StatusSet* response,
                       google::protobuf::Closure* done) override {
        brpc::ClosureGuard done_guard(done);
        {
          std::lock_guard<std::mutex> lock(mutex_);
          request_ = *request;
          ++calls_;
        }
        if (mode_ == ResponseMode::TRANSPORT_FAILURE) {
          controller->SetFailed("injected generations transport failure");
          return;
        }
        if (mode_ == ResponseMode::MISMATCHED_STATUS_COUNT) {
          return;
        }
        for (const auto& generation : request->gens()) {
          auto result = results_.find(generation.req_id());
          response->add_all_status()->set_ok(result == results_.end() ||
                                             result->second);
        }
      }

      proto::DisaggStreamGenerations request() {
        std::lock_guard<std::mutex> lock(mutex_);
        return request_;
      }

      int32_t calls() {
        std::lock_guard<std::mutex> lock(mutex_);
        return calls_;
      }

     private:
      const ResponseMode mode_;
      const std::unordered_map<std::string, bool> results_;
      std::mutex mutex_;
      proto::DisaggStreamGenerations request_;
      int32_t calls_ = 0;
    };

    class GenerationTestServer final {
     public:
      explicit GenerationTestServer(
          ResponseMode mode = ResponseMode::NORMAL,
          std::unordered_map<std::string, bool> results = {})
          : service_(mode, std::move(results)) {}

      ~GenerationTestServer() {
        server_.Stop(/*timeout_ms=*/0);
        server_.Join();
      }

      int32_t start() {
        const int32_t result =
            server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE);
        if (result != 0) {
          return result;
        }
        return server_.Start("127.0.0.1:0", nullptr);
      }

      std::string address() const {
        return "127.0.0.1:" + std::to_string(server_.listen_address().port);
      }

      GenerationTestService& service() { return service_; }

     private:
      GenerationTestService service_;
      brpc::Server server_;
    };

    RequestOutput make_output(std::string request_id, std::string target) {
      RequestOutput output;
      output.request_id = std::move(request_id);
      output.service_request_id = "service-" + output.request_id;
      output.target_xservice_addr = std::move(target);
      return output;
    }

    TEST(XServiceClientTest, DirectRegistrationRecoversFromTransientFailure) {
      GenerationTestServer server;
      server.service().registration_failures = 1;
      ASSERT_EQ(server.start(), 0);
      auto client =
          XServiceClientTestPeer::registration_client(server.address());
      EXPECT_TRUE(XServiceClientTestPeer::register_with_retry(*client));
      EXPECT_EQ(server.service().registration_calls.load(), 2);
    }

    TEST(XServiceClientTest, DirectRegistrationDoesNotRetryExplicitRejection) {
      GenerationTestServer server;
      server.service().registration_status = 1;
      ASSERT_EQ(server.start(), 0);
      auto client =
          XServiceClientTestPeer::registration_client(server.address());
      EXPECT_FALSE(XServiceClientTestPeer::register_with_retry(*client));
      EXPECT_EQ(server.service().registration_calls.load(), 1);
    }

    TEST(XServiceClientTest, ShutdownInterruptsRegistrationBackoff) {
      GenerationTestServer server;
      server.service().registration_failures = 100;
      auto started = server.service().registration_started.get_future();
      ASSERT_EQ(server.start(), 0);
      auto client =
          XServiceClientTestPeer::registration_client(server.address());
      auto registration = std::async(std::launch::async, [&client] {
        return XServiceClientTestPeer::register_with_retry(*client);
      });
      const auto first_attempt = started.wait_for(std::chrono::seconds(5));
      XServiceClientTestPeer::stop(*client);
      EXPECT_EQ(first_attempt, std::future_status::ready);
      // Shutdown interrupts the backoff, but not an in-flight HTTP RPC, whose
      // channel timeout can be the configured value or the 10-second default.
      EXPECT_EQ(registration.wait_for(std::chrono::seconds(15)),
                std::future_status::ready);
      EXPECT_FALSE(registration.get());
      EXPECT_EQ(server.service().registration_calls.load(), 1);
    }

    TEST(XServiceClientDeathTest, MismatchedResponseCountIsFatal) {
      GTEST_FLAG_SET(death_test_style, "threadsafe");
      EXPECT_DEATH(
          {
            GenerationTestServer server(ResponseMode::MISMATCHED_STATUS_COUNT);
            CHECK_EQ(server.start(), 0);
            const RequestOutput output =
                make_output("mismatch", server.address());
            const auto client = XServiceGenerationTestPeer::make_client();
            client->generation(output);
          },
          "The size of status set is not equal to the size of outputs");
    }

    TEST(XServiceClientTest, SingleGenerationPreservesPayload) {
      GenerationTestServer server;
      ASSERT_EQ(server.start(), 0);
      RequestOutput output = make_output("payload", server.address());
      output.status = Status(StatusCode::CANCELLED, "cancelled by caller");
      output.finished = true;
      output.finished_on_prefill_instance = true;
      output.usage = Usage{/*num_prompt_tokens=*/11,
                           /*num_generated_tokens=*/2,
                           /*num_total_tokens=*/13,
                           /*num_cached_tokens=*/7};

      LogProb logprob;
      logprob.token = "answer";
      logprob.token_id = 42;
      logprob.logprob = -0.25f;
      logprob.finished_token = false;
      logprob.top_logprobs =
          std::vector<LogProbData>{{"other", 43, -1.5f, true}};
      SequenceOutput first;
      first.index = 3;
      first.text = "answer";
      first.token_ids = {41, 42};
      first.finish_reason = "stop";
      first.stop_reason = int32_t{42};
      first.logprobs = std::vector<LogProb>{std::move(logprob)};
      output.outputs.reserve(2);
      output.outputs.emplace_back(std::move(first));
      SequenceOutput second;
      second.index = 4;
      second.text = "second";
      second.stop_reason = std::string("done");
      output.outputs.emplace_back(std::move(second));

      EXPECT_TRUE(
          XServiceGenerationTestPeer::make_client()->generation(output));
      EXPECT_EQ(server.service().calls(), 1);
      const auto request = server.service().request();
      ASSERT_EQ(request.gens_size(), 1);
      const auto& generation = request.gens(0);
      EXPECT_EQ(generation.req_id(), "payload");
      EXPECT_EQ(generation.service_req_id(), "service-payload");
      ASSERT_TRUE(generation.has_gen_status());
      EXPECT_EQ(generation.gen_status().status_code(),
                static_cast<int32_t>(StatusCode::CANCELLED));
      EXPECT_EQ(generation.gen_status().status_msg(), "cancelled by caller");
      EXPECT_TRUE(generation.finished());
      EXPECT_TRUE(generation.finished_on_prefill_instance());
      ASSERT_TRUE(generation.has_usage());
      EXPECT_EQ(generation.usage().num_prompt_tokens(), 11);
      EXPECT_EQ(generation.usage().num_generated_tokens(), 2);
      EXPECT_EQ(generation.usage().num_total_tokens(), 13);
      EXPECT_EQ(generation.usage().num_cached_tokens(), 7);
      ASSERT_EQ(generation.outputs_size(), 2);
      const auto& sequence = generation.outputs(0);
      EXPECT_EQ(sequence.index(), 3);
      EXPECT_EQ(sequence.text(), "answer");
      ASSERT_EQ(sequence.token_ids_size(), 2);
      EXPECT_EQ(sequence.token_ids(0), 41);
      EXPECT_EQ(sequence.token_ids(1), 42);
      EXPECT_EQ(sequence.finish_reason(), "stop");
      EXPECT_EQ(sequence.stop_reason().token_id(), 42);
      ASSERT_EQ(sequence.logprobs_size(), 1);
      const auto& received_logprob = sequence.logprobs(0);
      EXPECT_EQ(received_logprob.log_prob_data().token(), "answer");
      EXPECT_EQ(received_logprob.log_prob_data().token_id(), 42);
      EXPECT_FLOAT_EQ(received_logprob.log_prob_data().logprob(), -0.25f);
      EXPECT_FALSE(received_logprob.log_prob_data().finished_token());
      ASSERT_EQ(received_logprob.top_logprobs_size(), 1);
      EXPECT_EQ(received_logprob.top_logprobs(0).token(), "other");
      EXPECT_EQ(received_logprob.top_logprobs(0).token_id(), 43);
      EXPECT_FLOAT_EQ(received_logprob.top_logprobs(0).logprob(), -1.5f);
      EXPECT_TRUE(received_logprob.top_logprobs(0).finished_token());
      EXPECT_EQ(generation.outputs(1).index(), 4);
      EXPECT_EQ(generation.outputs(1).stop_reason().stop_string(), "done");
      EXPECT_TRUE(generation.outputs(1).finish_reason().empty());
      EXPECT_EQ(generation.outputs(1).logprobs_size(), 0);

      // The borrowed output remains usable after the asynchronous RPC joins.
      EXPECT_EQ(output.outputs[0].text, "answer");
      ASSERT_TRUE(output.outputs[0].logprobs.has_value());
      EXPECT_EQ(output.outputs[0].logprobs->front().token, "answer");
    }

    TEST(XServiceClientTest, BatchRoutesResultsToOriginalInputOrder) {
      GenerationTestServer first(ResponseMode::NORMAL, {{"rejected", false}});
      GenerationTestServer second;
      ASSERT_EQ(first.start(), 0);
      ASSERT_EQ(second.start(), 0);
      std::vector<RequestOutput> outputs;
      outputs.reserve(3);
      outputs.emplace_back(make_output("rejected", first.address()));
      outputs.emplace_back(make_output("other-service", second.address()));
      outputs.emplace_back(make_output("accepted", first.address()));

      EXPECT_EQ(XServiceGenerationTestPeer::make_client()->generations(outputs),
                (std::vector<bool>{false, true, true}));
      const auto first_request = first.service().request();
      ASSERT_EQ(first_request.gens_size(), 2);
      EXPECT_EQ(first_request.gens(0).req_id(), "rejected");
      EXPECT_EQ(first_request.gens(1).req_id(), "accepted");
      EXPECT_FALSE(first_request.gens(0).has_gen_status());
      EXPECT_FALSE(first_request.gens(0).has_usage());
      EXPECT_EQ(first_request.gens(0).outputs_size(), 0);
      const auto second_request = second.service().request();
      ASSERT_EQ(second_request.gens_size(), 1);
      EXPECT_EQ(second_request.gens(0).req_id(), "other-service");
      EXPECT_EQ(first.service().calls(), 1);
      EXPECT_EQ(second.service().calls(), 1);
    }

    TEST(XServiceClientTest, SingleGenerationReturnsRejectedStatus) {
      GenerationTestServer server(ResponseMode::NORMAL, {{"rejected", false}});
      ASSERT_EQ(server.start(), 0);
      const RequestOutput output = make_output("rejected", server.address());
      EXPECT_FALSE(
          XServiceGenerationTestPeer::make_client()->generation(output));
    }

    TEST(XServiceClientTest, TransportFailureOnlyFailsAffectedService) {
      GenerationTestServer failing(ResponseMode::TRANSPORT_FAILURE);
      GenerationTestServer healthy;
      ASSERT_EQ(failing.start(), 0);
      ASSERT_EQ(healthy.start(), 0);
      std::vector<RequestOutput> outputs;
      outputs.reserve(3);
      outputs.emplace_back(make_output("failed-first", failing.address()));
      outputs.emplace_back(make_output("healthy", healthy.address()));
      outputs.emplace_back(make_output("failed-last", failing.address()));

      EXPECT_EQ(XServiceGenerationTestPeer::make_client()->generations(outputs),
                (std::vector<bool>{false, true, false}));
      EXPECT_FALSE(XServiceGenerationTestPeer::make_client()->generation(
          outputs.front()));
    }

    TEST(XServiceClientTest,
         EmptyBatchAndMissingTargetDoNotRequireInitialization) {
      EXPECT_TRUE(
          XServiceGenerationTestPeer::make_client()->generations({}).empty());
      const RequestOutput output = make_output("missing-target", "");
      EXPECT_FALSE(
          XServiceGenerationTestPeer::make_client()->generation(output));
    }

    }  // namespace
  }  // namespace xllm
