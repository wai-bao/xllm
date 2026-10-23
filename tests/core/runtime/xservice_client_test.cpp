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

#include <brpc/server.h>
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <thread>

#include "core/framework/block/block_manager_pool.h"
#include "core/framework/config/distributed_config.h"
#include "core/scheduler/scheduler.h"
#include "proto/rpc.grpc.pb.h"

namespace xllm {

class XServiceClientTestPeer final {
 public:
  static std::unique_ptr<XServiceClient> create() {
    return std::unique_ptr<XServiceClient>(new XServiceClient);
  }

  static void start_observers(XServiceClient& client) {
    client.heartbeat_thread_ =
        std::make_unique<std::thread>([&client] { client.heartbeat(); });
    client.reconcile_thread_ = std::make_unique<std::thread>(
        [&client] { client.reconcile_registration_loop(); });
  }

  static bool register_after_stop(XServiceClient& client) {
    // No EtcdClient exists: a stopped client must reject before accessing it.
    return client.register_instance_with_retry(
        "key", "value", /*generation=*/0);
  }

  static std::function<bool()> snapshot_registration(XServiceClient& client) {
    std::lock_guard<std::mutex> lock(client.registration_mutex_);
    return [&client,
            key = client.registration_key_,
            value = client.registration_value_,
            generation = client.registration_generation_] {
      return client.register_instance_with_retry(key, value, generation);
    };
  }

  static void configure_heartbeat(XServiceClient& client,
                                  const std::string& address,
                                  Scheduler& scheduler,
                                  const BlockManagerPool& pool) {
    client.master_xservice_addr_ = address;
    client.chan_options_.timeout_ms = -1;
    CHECK(client.connect_to_xservice(address));
    auto observer = std::make_shared<XServiceClient::Observer>();
    observer->metrics_provider = [&scheduler, &pool] {
      XServiceMetricsSnapshot snapshot;
      snapshot.gpu_cache_usage_perc = pool.get_gpu_cache_usage_perc();
      snapshot.waiting_requests_num = scheduler.get_waiting_requests_num();
      scheduler.get_latency_metrics(snapshot.ttft, snapshot.tbt);
      return snapshot;
    };
    client.observers_.emplace_back(std::move(observer));
    client.register_done_.store(true);
  }
};

namespace {

class EmptyScheduler final : public Scheduler {
 public:
  explicit EmptyScheduler(InstanceInfo info = {}, uint32_t waiting = 0)
      : info_(std::move(info)), waiting_(waiting) {}
  void step(const absl::Duration&) override {}
  void generate() override {}
  bool add_request(std::shared_ptr<Request>&) override { return true; }
  uint32_t get_waiting_requests_num() const override {
    if (before_metrics) {
      before_metrics();
    }
    return waiting_;
  }
  void get_latency_metrics(std::vector<int64_t>&,
                           std::vector<int64_t>&) override {}
  const InstanceInfo& get_instance_info() override { return info_; }
  std::function<void()> before_metrics;

 private:
  InstanceInfo info_;
  uint32_t waiting_;
};

std::unique_ptr<XServiceRegistration> bind_scheduler(
    const std::shared_ptr<XServiceClient>& client,
    Scheduler& scheduler,
    InstanceInfo info) {
  return client->bind(std::move(info), [&scheduler] {
    XServiceMetricsSnapshot snapshot;
    snapshot.waiting_requests_num = scheduler.get_waiting_requests_num();
    scheduler.get_latency_metrics(snapshot.ttft, snapshot.tbt);
    return snapshot;
  });
}

class RecordingDiscoveryService final
    : public xllm_service::proto::XllmRpcService {
 public:
  void Heartbeat(google::protobuf::RpcController*,
                 const xllm_service::proto::HeartbeatRequest* request,
                 xllm_service::proto::Status* response,
                 google::protobuf::Closure* done) override {
    waiting.store(request->load_metrics().waiting_requests_num());
    response->set_ok(true);
    done->Run();
  }

  void Generations(google::protobuf::RpcController*,
                   const proto::DisaggStreamGenerations* request,
                   proto::StatusSet* response,
                   google::protobuf::Closure* done) override {
    for (int32_t row = 0; row < request->gens_size(); ++row) {
      response->add_all_status()->set_ok(true);
    }
    done->Run();
  }

  std::atomic<uint32_t> waiting{0};
};

class DiscoverySessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_interval_ = DistributedConfig::get_instance().heart_beat_interval();
    const char* endpoint = std::getenv("XLLM_TEST_ETCD_ENDPOINT");
    if (endpoint == nullptr) {
      GTEST_SKIP() << "Set XLLM_TEST_ETCD_ENDPOINT to an isolated test etcd.";
    }
    endpoint_ = endpoint;
    namespace_ =
        "discovery-test-" + std::to_string(getpid()) + "-" +
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    store_ = std::make_unique<etcd::SyncClient>(endpoint_);
    store_->set_grpc_timeout(std::chrono::seconds(2));
    ASSERT_EQ(server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
    ASSERT_TRUE(
        store_
            ->put("/" + namespace_ + "/XLLM:SERVICE:MASTER",
                  "127.0.0.1:" + std::to_string(server_.listen_address().port))
            .is_ok());
    DistributedConfig::get_instance().heart_beat_interval(0.01);
  }

  void TearDown() override {
    server_.Stop(/*closewait_ms=*/0);
    server_.Join();
    DistributedConfig::get_instance().heart_beat_interval(saved_interval_);
  }

  std::shared_ptr<XServiceClient> acquire(const std::string& name = "node") {
    return XServiceClient::acquire(endpoint_, name, namespace_);
  }

  static InstanceInfo info(const std::string& rpc_address,
                           const std::string& name = "node") {
    InstanceInfo result;
    result.name = name;
    result.type = "DEFAULT";
    result.rpc_address = rpc_address;
    return result;
  }

  template <typename Predicate>
  static bool wait_for(Predicate predicate) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!predicate()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
  }

  std::string endpoint_;
  std::string namespace_;
  std::unique_ptr<etcd::SyncClient> store_;
  RecordingDiscoveryService service_;
  brpc::Server server_;
  double saved_interval_ = 0.5;
};

TEST_F(DiscoverySessionTest,
       SharedSessionSurvivesObserverRemovalAndRecreation) {
  auto first = acquire();
  auto second = acquire();
  ASSERT_NE(first, nullptr);
  ASSERT_EQ(first, second);
  EmptyScheduler scheduler_a(info("rpc-a"), /*waiting=*/11);
  EmptyScheduler scheduler_b(info("rpc-b"), /*waiting=*/22);
  auto a = bind_scheduler(first, scheduler_a, scheduler_a.get_instance_info());
  auto b = bind_scheduler(second, scheduler_b, scheduler_b.get_instance_info());
  ASSERT_TRUE(wait_for([&] { return service_.waiting.load() == 11; }));
  a.reset();
  first.reset();
  EXPECT_TRUE(second->initialize_done());
  EXPECT_TRUE(wait_for([&] { return service_.waiting.load() == 22; }));
  EXPECT_TRUE(wait_for([&] {
    auto value = store_->get("/" + namespace_ + "/XLLM:DEFAULT:node");
    return value.is_ok() &&
           nlohmann::json::parse(value.value().as_string())["rpc_address"] ==
               "rpc-b";
  }));
  std::weak_ptr<XServiceClient> old = second;
  b.reset();
  second.reset();
  EXPECT_TRUE(old.expired());
  auto replacement = acquire();
  ASSERT_NE(replacement, nullptr);
  auto c =
      bind_scheduler(replacement, scheduler_a, scheduler_a.get_instance_info());
  EXPECT_TRUE(wait_for([&] { return service_.waiting.load() == 11; }));
}

TEST_F(DiscoverySessionTest, LastUnbindRetiresLeaseWhileSessionRemainsUsable) {
  auto client = acquire();
  ASSERT_NE(client, nullptr);
  EmptyScheduler scheduler(info("rpc-old"));
  auto registration =
      bind_scheduler(client, scheduler, scheduler.get_instance_info());
  ASSERT_NE(registration, nullptr);
  auto stale_registration =
      XServiceClientTestPeer::snapshot_registration(*client);
  const std::string key = "/" + namespace_ + "/XLLM:DEFAULT:node";
  const auto registered = store_->get(key);
  ASSERT_TRUE(registered.is_ok());
  ASSERT_NE(registered.value().lease(), 0);

  registration.reset();
  EXPECT_TRUE(client->initialize_done());
  EXPECT_EQ(store_->get(key).error_code(), 100);  // etcd key not found
  EXPECT_FALSE(stale_registration());
  EXPECT_EQ(store_->get(key).error_code(), 100);

  RequestOutput output;
  output.request_id = "finishing-request";
  output.service_request_id = "finishing-service-request";
  output.finished = true;
  EXPECT_EQ(client->generations({output}), std::vector<bool>{true});

  EmptyScheduler replacement(info("rpc-new"), /*waiting=*/88);
  auto rebound =
      bind_scheduler(client, replacement, replacement.get_instance_info());
  ASSERT_NE(rebound, nullptr);
  EXPECT_FALSE(stale_registration());
  const auto current = store_->get(key);
  ASSERT_TRUE(current.is_ok());
  EXPECT_EQ(nlohmann::json::parse(current.value().as_string())["rpc_address"],
            "rpc-new");
  EXPECT_NE(current.value().lease(), registered.value().lease());
  EXPECT_TRUE(wait_for([&] { return service_.waiting.load() == 88; }));
}

TEST_F(DiscoverySessionTest, UnbindDoesNotDeleteAnotherOwnersRegistration) {
  // Keep reconciliation out of this test's replacement/retirement interval.
  DistributedConfig::get_instance().heart_beat_interval(60);
  auto client = acquire();
  ASSERT_NE(client, nullptr);
  EmptyScheduler scheduler(info("rpc-old"));
  auto registration =
      bind_scheduler(client, scheduler, scheduler.get_instance_info());
  ASSERT_NE(registration, nullptr);
  const std::string key = "/" + namespace_ + "/XLLM:DEFAULT:node";
  // Another process may publish the same name under a new lease.
  const auto lease = store_->leasegrant(/*ttl=*/30);
  ASSERT_TRUE(lease.is_ok());
  ASSERT_NE(lease.value().lease(), 0);
  ASSERT_TRUE(store_->put(key, "replacement", lease.value().lease()).is_ok());
  registration.reset();
  const auto current = store_->get(key);
  ASSERT_TRUE(current.is_ok());
  EXPECT_EQ(current.value().as_string(), "replacement");
  EXPECT_EQ(current.value().lease(), lease.value().lease());
  EXPECT_TRUE(store_->leaserevoke(lease.value().lease()).is_ok());
}

TEST_F(DiscoverySessionTest, StoppedSessionIsReplacedWithoutFalseReadyState) {
  auto first = acquire();
  ASSERT_NE(first, nullptr);
  first->stop();
  EXPECT_FALSE(first->initialize_done());
  EXPECT_FALSE(first->init(endpoint_, "node", namespace_));
  EmptyScheduler stopped_scheduler(info("rpc-stopped"));
  EXPECT_EQ(
      bind_scheduler(
          first, stopped_scheduler, stopped_scheduler.get_instance_info()),
      nullptr);
  auto replacement = acquire();
  ASSERT_NE(replacement, nullptr);
  EXPECT_NE(first, replacement);
  EmptyScheduler scheduler(info("rpc-new"), /*waiting=*/33);
  auto registration =
      bind_scheduler(replacement, scheduler, scheduler.get_instance_info());
  EXPECT_TRUE(wait_for([&] { return service_.waiting.load() == 33; }));
}

TEST_F(DiscoverySessionTest, DifferentServingInstancesHaveIndependentSessions) {
  auto first = acquire("first");
  auto second = acquire("second");
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first, second);
  first->stop();
  EXPECT_TRUE(second->initialize_done());
  EmptyScheduler scheduler(info("rpc-second", "second"), /*waiting=*/44);
  auto registration =
      bind_scheduler(second, scheduler, scheduler.get_instance_info());
  EXPECT_TRUE(wait_for([&] { return service_.waiting.load() == 44; }));
}

TEST_F(DiscoverySessionTest, InvalidRegistrationDoesNotPreventLaterBinding) {
  auto client = acquire();
  ASSERT_NE(client, nullptr);
  EmptyScheduler scheduler(info("rpc-good"), /*waiting=*/55);
  InstanceInfo invalid = scheduler.get_instance_info();
  invalid.type = "INVALID";
  EXPECT_EQ(bind_scheduler(client, scheduler, invalid), nullptr);
  auto registration =
      bind_scheduler(client, scheduler, scheduler.get_instance_info());
  ASSERT_NE(registration, nullptr);
  EXPECT_TRUE(wait_for([&] { return service_.waiting.load() == 55; }));
}

TEST_F(DiscoverySessionTest, SecondaryBindingsPreserveThePrimaryRegistration) {
  auto client = acquire();
  ASSERT_NE(client, nullptr);
  EXPECT_EQ(client,
            XServiceClient::acquire(endpoint_, "node", "/" + namespace_ + "/"));
  EmptyScheduler primary(info("rpc-primary"), /*waiting=*/66);
  EmptyScheduler secondary(info("rpc-secondary"), /*waiting=*/77);
  auto registration =
      bind_scheduler(client, primary, primary.get_instance_info());
  ASSERT_NE(registration, nullptr);
  EXPECT_EQ(bind_scheduler(client, secondary, info("rpc", "other")), nullptr);
  InstanceInfo wrong_role = secondary.get_instance_info();
  wrong_role.type = "DECODE";
  EXPECT_EQ(bind_scheduler(client, secondary, wrong_role), nullptr);
  auto second =
      bind_scheduler(client, secondary, secondary.get_instance_info());
  ASSERT_NE(second, nullptr);
  second.reset();
  EXPECT_TRUE(client->initialize_done());
  EXPECT_TRUE(wait_for([&] { return service_.waiting.load() == 66; }));
  auto value = store_->get("/" + namespace_ + "/XLLM:DEFAULT:node");
  ASSERT_TRUE(value.is_ok());
  EXPECT_EQ(nlohmann::json::parse(value.value().as_string())["rpc_address"],
            "rpc-primary");
}

TEST_F(DiscoverySessionTest, UnbindWaitsUntilBorrowedMetricStateIsReleased) {
  auto client = acquire();
  ASSERT_NE(client, nullptr);
  std::promise<void> entered;
  auto entry = entered.get_future();
  std::promise<void> release;
  auto gate = release.get_future().share();
  std::atomic<bool> first{true};
  EmptyScheduler scheduler(info("rpc-blocked"));
  scheduler.before_metrics = [&] {
    if (first.exchange(false)) {
      entered.set_value();
    }
    gate.wait();
  };
  auto registration =
      bind_scheduler(client, scheduler, scheduler.get_instance_info());
  EXPECT_EQ(entry.wait_for(std::chrono::seconds(3)), std::future_status::ready);
  EmptyScheduler secondary(info("rpc-secondary"));
  auto secondary_binding = std::async(std::launch::async, [&] {
    return bind_scheduler(client, secondary, secondary.get_instance_info());
  });
  const std::future_status binding_status =
      secondary_binding.wait_for(std::chrono::seconds(1));
  auto unbound = std::async(std::launch::async, [&] { registration.reset(); });
  EXPECT_EQ(unbound.wait_for(std::chrono::milliseconds(30)),
            std::future_status::timeout);
  release.set_value();
  unbound.get();
  EXPECT_EQ(binding_status, std::future_status::ready);
  EXPECT_NE(secondary_binding.get(), nullptr);
}

class RegistrationKeyService final : public etcdserverpb::KV::Service {
 public:
  grpc::Status Put(grpc::ServerContext*,
                   const etcdserverpb::PutRequest* request,
                   etcdserverpb::PutResponse*) override {
    if (request->key() == "/lease-test/instance") {
      lease_id_.store(request->lease());
      writes_.fetch_add(1);
    }
    return grpc::Status::OK;
  }

  uint32_t writes() const { return writes_.load(); }
  int64_t lease_id() const { return lease_id_.load(); }

 private:
  std::atomic<uint32_t> writes_{0};
  std::atomic<int64_t> lease_id_{0};
};

class RegistrationLeaseService final : public etcdserverpb::Lease::Service {
 public:
  enum class GrantMode { ERROR, EMPTY, VALID };

  void set_mode(GrantMode mode) { mode_.store(mode); }
  uint32_t grants() const { return grants_.load(); }

  grpc::Status LeaseGrant(grpc::ServerContext*,
                          const etcdserverpb::LeaseGrantRequest* request,
                          etcdserverpb::LeaseGrantResponse* response) override {
    grants_.fetch_add(1);
    const GrantMode mode = mode_.load();
    if (mode == GrantMode::ERROR) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                          "lease temporarily unavailable");
    }
    response->set_id(mode == GrantMode::VALID ? 42 : 0);
    response->set_ttl(request->ttl());
    return grpc::Status::OK;
  }

  grpc::Status LeaseKeepAlive(
      grpc::ServerContext*,
      grpc::ServerReaderWriter<etcdserverpb::LeaseKeepAliveResponse,
                               etcdserverpb::LeaseKeepAliveRequest>* stream)
      override {
    etcdserverpb::LeaseKeepAliveRequest request;
    while (stream->Read(&request)) {
      etcdserverpb::LeaseKeepAliveResponse response;
      response.set_id(request.id());
      response.set_ttl(3);
      if (!stream->Write(response)) {
        break;
      }
    }
    return grpc::Status::OK;
  }

 private:
  std::atomic<GrantMode> mode_{GrantMode::VALID};
  std::atomic<uint32_t> grants_{0};
};

class EtcdRegistrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    grpc::ServerBuilder builder;
    int port = 0;  // gRPC's selected-port API requires int.
    builder.AddListeningPort(
        "127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&keys_);
    builder.RegisterService(&leases_);
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
    client_ =
        std::make_unique<EtcdClient>("http://127.0.0.1:" + std::to_string(port),
                                     "lease-test",
                                     std::chrono::milliseconds(200));
  }

  void TearDown() override {
    client_.reset();
    if (server_ != nullptr) {
      server_->Shutdown();
      server_->Wait();
    }
  }

  RegistrationKeyService keys_;
  RegistrationLeaseService leases_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<EtcdClient> client_;
};

TEST_F(EtcdRegistrationTest, FailedLeaseGrantDoesNotPublishAnUnleasedKey) {
  leases_.set_mode(RegistrationLeaseService::GrantMode::ERROR);
  EXPECT_FALSE(client_->register_instance("instance", "value", /*ttl=*/3));
  EXPECT_EQ(keys_.writes(), 0);
}

TEST_F(EtcdRegistrationTest, EmptyLeaseGrantDoesNotPublishAnUnleasedKey) {
  leases_.set_mode(RegistrationLeaseService::GrantMode::EMPTY);
  EXPECT_FALSE(client_->register_instance("instance", "value", /*ttl=*/3));
  EXPECT_EQ(keys_.writes(), 0);
}

TEST_F(EtcdRegistrationTest, RegistrationRecoversWithTheGrantedLease) {
  leases_.set_mode(RegistrationLeaseService::GrantMode::ERROR);
  EXPECT_FALSE(client_->register_instance("instance", "value", /*ttl=*/3));
  leases_.set_mode(RegistrationLeaseService::GrantMode::VALID);
  EXPECT_TRUE(client_->register_instance("instance", "value", /*ttl=*/3));
  EXPECT_EQ(leases_.grants(), 2);
  EXPECT_EQ(keys_.writes(), 1);
  EXPECT_EQ(keys_.lease_id(), 42);
}

TEST_F(EtcdRegistrationTest, NonpositiveTtlCannotPublishAnUnleasedKey) {
  EXPECT_FALSE(client_->register_instance("instance", "value", /*ttl=*/0));
  EXPECT_FALSE(client_->register_instance("instance", "value", /*ttl=*/-1));
  EXPECT_EQ(leases_.grants(), 0);
  EXPECT_EQ(keys_.writes(), 0);
}

class StallingLeaseService final : public etcdserverpb::Lease::Service {
 public:
  explicit StallingLeaseService(bool stall_refresh)
      : stall_refresh_(stall_refresh), gate_(release_.get_future().share()) {}

  grpc::Status LeaseGrant(grpc::ServerContext*,
                          const etcdserverpb::LeaseGrantRequest* request,
                          etcdserverpb::LeaseGrantResponse* response) override {
    response->set_id(1);
    response->set_ttl(request->ttl());
    return grpc::Status::OK;
  }

  grpc::Status LeaseKeepAlive(
      grpc::ServerContext*,
      grpc::ServerReaderWriter<etcdserverpb::LeaseKeepAliveResponse,
                               etcdserverpb::LeaseKeepAliveRequest>* stream)
      override {
    started_.set_value();
    etcdserverpb::LeaseKeepAliveRequest request;
    while (stream->Read(&request)) {
      if (stall_refresh_) {
        refreshing_.set_value();
        gate_.wait();
      }
      etcdserverpb::LeaseKeepAliveResponse response;
      response.set_id(request.id());
      response.set_ttl(60);
      if (!stream->Write(response)) {
        break;
      }
    }
    // Withhold the server's final status even after the client half-closes.
    gate_.wait();
    return grpc::Status::OK;
  }

  std::future<void> started() { return started_.get_future(); }
  std::future<void> refreshing() { return refreshing_.get_future(); }
  void release() { release_.set_value(); }

 private:
  bool stall_refresh_;
  std::promise<void> started_;
  std::promise<void> refreshing_;
  std::promise<void> release_;
  std::shared_future<void> gate_;
};

void check_keep_alive_cancellation(bool stall_refresh,
                                   bool expire_renewal = false) {
  StallingLeaseService service(stall_refresh);
  auto started = service.started();
  auto refreshing = service.refreshing();
  grpc::ServerBuilder builder;
  int port = 0;  // gRPC's selected-port API requires int.
  builder.AddListeningPort(
      "127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  etcd::SyncClient client("http://127.0.0.1:" + std::to_string(port));
  client.set_grpc_timeout(expire_renewal ? std::chrono::milliseconds(100)
                                         : std::chrono::milliseconds(60000));
  auto keep_alive =
      std::make_unique<etcd::KeepAlive>(client, stall_refresh ? 1 : 60);
  EXPECT_EQ(started.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  if (stall_refresh) {
    EXPECT_EQ(refreshing.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
  }
  if (expire_renewal) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  auto cancelled = std::async(std::launch::async, [&] {
    keep_alive->Cancel();
    keep_alive.reset();
  });
  const auto status = cancelled.wait_for(std::chrono::seconds(1));
  // Release the handler even when the cancellation assertion fails.
  service.release();
  cancelled.get();
  server->Shutdown();
  server->Wait();
  EXPECT_EQ(status, std::future_status::ready);
}

TEST(EtcdCancellationTest, CancelsWithoutWaitingForTheRemoteFinalStatus) {
  check_keep_alive_cancellation(/*stall_refresh=*/false);
}

TEST(EtcdCancellationTest, CancelsAnOutstandingRenewalBeforeJoiningItsThread) {
  check_keep_alive_cancellation(/*stall_refresh=*/true);
}

TEST(EtcdCancellationTest, DrainsTimedOutRenewalBeforeFinishingTheStream) {
  check_keep_alive_cancellation(/*stall_refresh=*/true,
                                /*expire_renewal=*/true);
}

class StallingAuthService final : public etcdserverpb::Auth::Service {
 public:
  StallingAuthService() : gate_(release_.get_future().share()) {}

  grpc::Status Authenticate(
      grpc::ServerContext* context,
      const etcdserverpb::AuthenticateRequest*,
      etcdserverpb::AuthenticateResponse* response) override {
    has_deadline.store(context->deadline() !=
                       std::chrono::system_clock::time_point::max());
    if (stall.load()) {
      gate_.wait();
    }
    response->set_token("test-token");
    return grpc::Status::OK;
  }

  void release() { release_.set_value(); }
  std::atomic<bool> stall{true};
  std::atomic<bool> has_deadline{false};

 private:
  std::promise<void> release_;
  std::shared_future<void> gate_;
};

TEST(EtcdCancellationTest, AuthenticationUsesTheInitialRequestDeadline) {
  StallingAuthService service;
  grpc::ServerBuilder builder;
  int port = 0;  // gRPC's selected-port API requires int.
  builder.AddListeningPort(
      "127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  auto initialized = std::async(std::launch::async, [&] {
    etcd::SyncClient client("http://127.0.0.1:" + std::to_string(port),
                            "test-user",
                            "test-password",
                            /*auth_token_ttl=*/300,
                            /*load_balancer=*/"round_robin",
                            std::chrono::milliseconds(100));
  });
  const auto status = initialized.wait_for(std::chrono::seconds(1));
  service.release();
  initialized.get();
  server->Shutdown();
  server->Wait();
  EXPECT_EQ(status, std::future_status::ready);
  EXPECT_TRUE(service.has_deadline.load());
}

TEST(EtcdCancellationTest, AuthenticationRenewalUsesTheUpdatedDeadline) {
  StallingAuthService service;
  service.stall.store(false);
  grpc::ServerBuilder builder;
  int port = 0;  // gRPC's selected-port API requires int.
  builder.AddListeningPort(
      "127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  etcd::SyncClient client("http://127.0.0.1:" + std::to_string(port),
                          "test-user",
                          "test-password",
                          /*auth_token_ttl=*/1);
  client.set_grpc_timeout(std::chrono::milliseconds(100));
  service.stall.store(true);
  // The dependency renews only after the one-second minimum token TTL.
  std::this_thread::sleep_for(std::chrono::milliseconds(2100));
  auto renewed = std::async(std::launch::async,
                            [&] { (void)client.current_auth_token(); });
  const auto status = renewed.wait_for(std::chrono::seconds(1));
  service.release();
  renewed.get();
  server->Shutdown();
  server->Wait();
  EXPECT_EQ(status, std::future_status::ready);
  EXPECT_TRUE(service.has_deadline.load());
}

class PendingHeartbeatService final
    : public xllm_service::proto::XllmRpcService {
 public:
  void Heartbeat(google::protobuf::RpcController*,
                 const xllm_service::proto::HeartbeatRequest*,
                 xllm_service::proto::Status* response,
                 google::protobuf::Closure* done) override {
    response_ = response;
    done_ = done;
    entered_.set_value();
  }

  std::future<void> entered() { return entered_.get_future(); }
  void release() {
    response_->set_ok(true);
    done_->Run();
  }

 private:
  std::promise<void> entered_;
  xllm_service::proto::Status* response_ = nullptr;
  google::protobuf::Closure* done_ = nullptr;
};

TEST(XServiceClientTest, StopCancelsHeartbeatBeforeTheServerReplies) {
  PendingHeartbeatService service;
  auto entered = service.entered();
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  const double saved_interval =
      DistributedConfig::get_instance().heart_beat_interval();
  DistributedConfig::get_instance().heart_beat_interval(0.01);
  BlockManagerPool::Options options;
  options.num_blocks(4).block_size(2).enable_prefix_cache(false);
  BlockManagerPool pool(options);
  EmptyScheduler scheduler;
  auto client = XServiceClientTestPeer::create();
  XServiceClientTestPeer::configure_heartbeat(
      *client,
      "127.0.0.1:" + std::to_string(server.listen_address().port),
      scheduler,
      pool);
  XServiceClientTestPeer::start_observers(*client);
  ASSERT_EQ(entered.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  auto stopped = std::async(std::launch::async, [&] { client->stop(); });
  EXPECT_EQ(stopped.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  // Always unblock the remote handler, including when the assertion fails.
  service.release();
  stopped.get();
  EXPECT_TRUE(client->get_static_decode_list().empty());
  EXPECT_TRUE(client->get_instance_info("unused").name.empty());
  server.Stop(/*closewait_ms=*/0);
  server.Join();
  DistributedConfig::get_instance().heart_beat_interval(saved_interval);
}

TEST(XServiceClientDeathTest, StopsIdleObserversBeforeTheNextHeartbeat) {
  EXPECT_EXIT(([] {
                alarm(10);
                DistributedConfig::get_instance().heart_beat_interval(60);
                auto client = XServiceClientTestPeer::create();
                XServiceClientTestPeer::start_observers(*client);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                const auto start = std::chrono::steady_clock::now();
                client->stop();
                client->stop();
                CHECK(std::chrono::steady_clock::now() - start <
                      std::chrono::seconds(1));
                CHECK(!XServiceClientTestPeer::register_after_stop(*client));
                client.reset();
                _exit(0);
              }()),
              ::testing::ExitedWithCode(0),
              "");
}

}  // namespace
}  // namespace xllm
