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

#include <brpc/channel.h>
#include <brpc/controller.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/etcd_client.h"
#include "common/types.h"
#include "framework/request/request_output.h"
#include "xservice.pb.h"
namespace xllm {

class XServiceRegistration;

struct XServiceMetricsSnapshot {
  float gpu_cache_usage_perc = 0.0F;
  uint32_t waiting_requests_num = 0;
  std::vector<int64_t> ttft;
  std::vector<int64_t> tbt;
  bool has_virtual_memory_info = false;
  std::vector<size_t> worker_free_phy_pages;
  std::unordered_map<std::string, std::vector<WeightSegment>>
      model_weight_segments;
};

class XServiceClient final
    : public std::enable_shared_from_this<XServiceClient> {
 public:
  static std::shared_ptr<XServiceClient> acquire(
      const std::string& etcd_addr,
      const std::string& instance_name,
      const std::string& etcd_namespace = "");

  ~XServiceClient();
  // Stop background metric collection before its owners destroy the provider.
  // Terminal and idempotent; in-flight response RPCs keep their connections.
  void stop();
  bool init(const std::string& etcd_addr,
            const std::string& instance_name = "",
            const std::string& etcd_namespace = "");
  bool initialize_done() const {
    return initialize_done_.load() && !exited_.load();
  }
  using MetricsProvider = std::function<XServiceMetricsSnapshot()>;
  std::unique_ptr<XServiceRegistration> bind(InstanceInfo info,
                                             MetricsProvider metrics_provider);

  std::string get_instance_name();
  void heartbeat();
  InstanceInfo get_instance_info(const std::string& instance_name);
  std::vector<std::string> get_static_decode_list();

  // response generation tokens to xllm service
  bool generation(const RequestOutput& output);
  std::vector<bool> generations(const std::vector<RequestOutput>& outputs);
  std::vector<bool> send_generations(std::span<const RequestOutput> outputs);
  std::vector<std::string> get_all_xservice_addrs();

 private:
  friend class XServiceRegistration;
  void unbind(uint64_t registration_id);
  bool register_instance(const InstanceInfo& instance_info);
  bool set_registration_info(const InstanceInfo& instance_info);
  bool wait_for_stop(std::chrono::milliseconds timeout);
  bool track_control_rpc(brpc::CallId id);
  void forget_control_rpc(brpc::CallId id);

  // RAII scope for a control-plane RPC: applies the discovery timeout to the
  // controller, registers its call id so a concurrent stop() can cancel it, and
  // forgets the id on scope exit. tracked() is false when the client is already
  // exiting, in which case the caller must abandon the RPC.
  class ControlRpcScope final {
   public:
    ControlRpcScope(XServiceClient& client, brpc::Controller& cntl);
    ~ControlRpcScope();
    ControlRpcScope(const ControlRpcScope&) = delete;
    ControlRpcScope& operator=(const ControlRpcScope&) = delete;
    bool tracked() const { return tracked_; }

   private:
    XServiceClient& client_;
    brpc::CallId call_id_;
    bool tracked_ = false;
  };

  bool register_instance_with_retry(const std::string& key,
                                    const std::string& value,
                                    uint64_t generation);
  bool reconcile_registration();
  void reconcile_registration_loop();

  void handle_master_service_watch(const etcd::Response& response,
                                   const uint64_t& prefix_len);
  void handle_xservices_watch(const etcd::Response& response,
                              const uint64_t& prefix_len);

  // connect to specific xllm_service
  bool connect_to_xservice(const std::string& xservice_addr);

  // remove the connection to specific xllm_service
  void disconnect_xservice(const std::string& xservice_addr);

  // call rpc with current master stub atomically.
  bool with_master_stub(
      const std::function<void(xllm_service::proto::XllmRpcService_Stub*)>& fn,
      std::string* master_addr);

  // find stub by address, caller should hold mutex_
  xllm_service::proto::XllmRpcService_Stub* find_stub_locked(
      const std::string& xservice_addr);

 private:
  friend class XServiceClientTestPeer;
  friend class XServiceGenerationTestPeer;
  XServiceClient() = default;

  std::atomic_bool exited_{false};
  std::once_flag stop_once_;
  std::mutex stop_mutex_;
  std::condition_variable stop_cv_;
  std::unordered_map<uint64_t, brpc::CallId> control_rpcs_;
  std::atomic_bool register_done_{false};
  std::atomic_bool initialize_done_{false};
  std::mutex init_mutex_;
  std::string etcd_address_;
  std::string etcd_namespace_;
  std::string instance_name_;
  std::string incarnation_id_;
  std::string registration_key_;
  std::string registration_value_;
  // Protected by registration_mutex_. Each retry checks this generation while
  // holding the mutex through its write, so unbind cannot be republished.
  uint64_t registration_generation_ = 0;

  std::string master_xservice_addr_;
  std::unordered_map<std::string, std::unique_ptr<brpc::Channel>>
      xservice_channels_;
  std::unordered_map<std::string,
                     std::unique_ptr<xllm_service::proto::XllmRpcService_Stub>>
      xservice_stubs_;
  std::unique_ptr<std::thread> heartbeat_thread_;
  std::unique_ptr<std::thread> reconcile_thread_;

  std::shared_mutex mutex_;
  std::mutex registration_mutex_;
  brpc::ChannelOptions chan_options_;
  std::unique_ptr<EtcdClient> etcd_client_;
  struct Observer {
    uint64_t id = 0;
    MetricsProvider metrics_provider;
    InstanceInfo info;
    size_t in_flight = 0;
  };
  // A registration outlives the borrowed objects' use by heartbeat. Unbinding
  // waits for current metric collection before the owning Master destroys them.
  std::mutex observer_mutex_;
  std::condition_variable observer_cv_;
  uint64_t next_observer_id_ = 1;
  std::vector<std::shared_ptr<Observer>> observers_;
};

class XServiceRegistration final {
 public:
  ~XServiceRegistration();
  XServiceRegistration(const XServiceRegistration&) = delete;
  XServiceRegistration& operator=(const XServiceRegistration&) = delete;

 private:
  friend class XServiceClient;
  XServiceRegistration(std::shared_ptr<XServiceClient> client,
                       uint64_t registration_id);
  std::shared_ptr<XServiceClient> client_;
  uint64_t registration_id_;
};

}  // namespace xllm
