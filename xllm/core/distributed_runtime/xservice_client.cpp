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

#include "core/distributed_runtime/xservice_client.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <nlohmann/json.hpp>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "api_service/openai_json.h"
#include "core/common/etcd_utils.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/service_config.h"
#include "util/env_var.h"
#include "util/hash_util.h"
#include "util/net.h"
#include "util/uuid.h"

namespace xllm {
namespace {
static std::string ETCD_MASTER_SERVICE_KEY = "XLLM:SERVICE:MASTER";
static std::string ETCD_XSERVICES_KEY_PREFIX =
    "XLLM:SERVICE:";  // all xllm_service registration prefix
constexpr const char* kEtcdUsernameEnvVar = "ETCD_USERNAME";
constexpr const char* kEtcdPasswordEnvVar = "ETCD_PASSWORD";
constexpr int32_t kDefaultHttpTimeoutMs = 10000;
constexpr size_t kMaxHttpBatchBytes = 1024 * 1024;
constexpr size_t kMaxHttpBatchResults = 256;
constexpr std::string_view kHttpBatchPrefix = "{\"results\":[";
constexpr std::chrono::seconds kDiscoveryRequestTimeout{5};

std::chrono::milliseconds heartbeat_interval() {
  return std::chrono::milliseconds(static_cast<int64_t>(
      ::xllm::DistributedConfig::get_instance().heart_beat_interval() * 1000));
}
static std::unordered_map<xllm_service::proto::InstanceType, std::string>
    ETCD_KEYS_PREFIX_MAP = {
        {xllm_service::proto::InstanceType::DEFAULT, "XLLM:DEFAULT:"},
        {xllm_service::proto::InstanceType::PREFILL, "XLLM:PREFILL:"},
        {xllm_service::proto::InstanceType::DECODE, "XLLM:DECODE:"},
        {xllm_service::proto::InstanceType::MIX, "XLLM:MIX:"},
};

std::string parse_instance_name(const std::string& name) {
  if (name.empty()) return "";
  // Validate the format of instance name
  // The format is `ip:port` currently.
  auto pos = name.find(':');
  if (pos == std::string::npos) {
    // only offer the port, we need to fill the ip address
    return xllm::net::get_local_ip_addr() + ":" + name;
  }
  return name;
}

bool check_instance_name(const std::string& name) {
  std::vector<std::string> addr = absl::StrSplit(name, ':');
  // Now only support `ip:port` format
  if (addr.size() != 2 || addr[0].empty() || addr[1].empty()) {
    LOG(ERROR)
        << "Invalid instance name format, now only support `ip:port` style.";
    return false;
  }

  return true;
}

void fill_generation_request(const RequestOutput& output,
                             proto::DisaggStreamGeneration* request) {
  request->set_req_id(output.request_id);
  request->set_service_req_id(output.service_request_id);
  if (output.status.has_value()) {
    auto gen_status = request->mutable_gen_status();
    gen_status->set_status_code(
        static_cast<int32_t>(output.status.value().code()));
    gen_status->set_status_msg(output.status.value().message());
  }
  request->set_finished(output.finished || output.cancelled);
  request->set_finished_on_prefill_instance(
      output.finished_on_prefill_instance);
  if (output.usage.has_value()) {
    proto::OutputUsage* proto_usage = request->mutable_usage();
    proto_usage->set_num_prompt_tokens(output.usage.value().num_prompt_tokens);
    proto_usage->set_num_generated_tokens(
        output.usage.value().num_generated_tokens);
    proto_usage->set_num_total_tokens(output.usage.value().num_total_tokens);
    proto_usage->set_num_cached_tokens(output.usage.value().num_cached_tokens);
  }
  request->mutable_outputs()->Reserve(
      static_cast<int32_t>(output.outputs.size()));
  for (const auto& seq_output : output.outputs) {
    auto* proto_seq_out = request->mutable_outputs()->Add();
    proto_seq_out->set_index(seq_output.index);
    proto_seq_out->set_text(seq_output.text);
    if (seq_output.finish_reason.has_value()) {
      proto_seq_out->set_finish_reason(seq_output.finish_reason.value());
    } else {
      proto_seq_out->set_finish_reason("");
    }
    if (const auto* token = std::get_if<int32_t>(&seq_output.stop_reason)) {
      proto_seq_out->mutable_stop_reason()->set_token_id(*token);
    } else if (const auto* text =
                   std::get_if<std::string>(&seq_output.stop_reason)) {
      proto_seq_out->mutable_stop_reason()->set_stop_string(*text);
    }
    proto_seq_out->mutable_token_ids()->Reserve(
        static_cast<int32_t>(seq_output.token_ids.size()));
    for (const auto& value : seq_output.token_ids) {
      *proto_seq_out->mutable_token_ids()->Add() = value;
    }
    if (!seq_output.logprobs.has_value()) {
      continue;
    }
    const auto& logprobs = seq_output.logprobs.value();
    proto_seq_out->mutable_logprobs()->Reserve(
        static_cast<int32_t>(logprobs.size()));
    for (const auto& logprob : logprobs) {
      auto* proto_logprob = proto_seq_out->mutable_logprobs()->Add();
      proto::LogProbData* log_prob_data =
          proto_logprob->mutable_log_prob_data();
      log_prob_data->set_token(logprob.token);
      log_prob_data->set_token_id(logprob.token_id);
      log_prob_data->set_logprob(logprob.logprob);
      log_prob_data->set_finished_token(logprob.finished_token);
      if (!logprob.top_logprobs.has_value()) {
        continue;
      }
      const auto& top_logprobs = logprob.top_logprobs.value();
      proto_logprob->mutable_top_logprobs()->Reserve(
          static_cast<int32_t>(top_logprobs.size()));
      for (const auto& top_logprob : top_logprobs) {
        auto* top_log_prob_data = proto_logprob->mutable_top_logprobs()->Add();
        top_log_prob_data->set_token(top_logprob.token);
        top_log_prob_data->set_token_id(top_logprob.token_id);
        top_log_prob_data->set_logprob(top_logprob.logprob);
        top_log_prob_data->set_finished_token(top_logprob.finished_token);
      }
    }
  }
}

std::string service_channel_key(const std::string& address, bool http) {
  return (http ? "http://" : "baidu_std://") + address;
}

}  // namespace

std::shared_ptr<XServiceClient> XServiceClient::acquire(
    const std::string& etcd_addr,
    const std::string& instance_name,
    const std::string& etcd_namespace) {
  static std::mutex registry_mutex;
  static std::unordered_map<std::string, std::weak_ptr<XServiceClient>>
      sessions;
  const std::string normalized_namespace =
      normalize_etcd_namespace(etcd_namespace);
  const std::string key =
      absl::StrCat(etcd_addr, normalized_namespace, instance_name);
  std::shared_ptr<XServiceClient> client;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    std::erase_if(sessions,
                  [](const auto& entry) { return entry.second.expired(); });
    auto& entry = sessions[key];
    client = entry.lock();
    if (client == nullptr || client->exited_.load()) {
      client = std::shared_ptr<XServiceClient>(new XServiceClient);
      entry = client;
    }
  }
  if (!client->init(etcd_addr, instance_name, etcd_namespace)) {
    return nullptr;
  }
  return client;
}

bool XServiceClient::init(const std::string& etcd_addr,
                          const std::string& instance_name,
                          const std::string& etcd_namespace) {
  std::lock_guard<std::mutex> init_lock(init_mutex_);
  if (exited_.load()) {
    return false;
  }
  // Normalize once: the idempotence compare and the assignment below must see
  // the same value, and init is also reachable without acquire's normalization.
  const std::string normalized_namespace =
      normalize_etcd_namespace(etcd_namespace);
  if (initialize_done_.load()) {
    return etcd_address_ == etcd_addr && instance_name_ == instance_name &&
           etcd_namespace_ == normalized_namespace;
  }

  const std::string& direct_address =
      DistributedConfig::get_instance().xservice_address();
  if (etcd_addr.empty() && direct_address.empty()) {
    LOG(ERROR) << "Either etcd_addr or xservice_address must be configured.";
    return false;
  }
  if (!etcd_addr.empty() && !direct_address.empty()) {
    LOG(ERROR) << "etcd_addr and xservice_address are mutually exclusive.";
    return false;
  }
  direct_service_ = !direct_address.empty();

  instance_name_ = instance_name;
  etcd_address_ = etcd_addr;
  etcd_namespace_ = normalized_namespace;
  if (incarnation_id_.empty()) {
    ShortUUID uuid;
    incarnation_id_ = uuid.random();
  }
  const std::string& service_protocol =
      DistributedConfig::get_instance().xservice_rpc_protocol();
  if (service_protocol != "baidu_std" && service_protocol != "http") {
    LOG(ERROR) << "Unsupported xservice_rpc_protocol: " << service_protocol;
    return false;
  }
  chan_options_.protocol = service_protocol;
  http_service_protocol_ = service_protocol == "http";
  // Generations contains deltas without sequence numbers. An ambiguous HTTP
  // ACK must fail the request, rather than transparently replaying a delta.
  chan_options_.max_retry = service_protocol == "http" ? 0 : 3;
  chan_options_.timeout_ms =
      ::xllm::ServiceConfig::get_instance().rpc_channel_timeout_ms();
  if (service_protocol == "http" && chan_options_.timeout_ms <= 0) {
    LOG(ERROR) << "HTTP service RPC requires rpc_channel_timeout_ms > 0.";
    return false;
  }

  if (direct_service_) {
    if (service_protocol != "http" || !check_instance_name(direct_address)) {
      LOG(ERROR) << "Direct service mode requires HTTP and a host:port "
                    "xservice_address.";
      return false;
    }
    master_xservice_addr_ = direct_address;
    if (!connect_to_xservice(master_xservice_addr_)) {
      return false;
    }
    heartbeat_thread_ =
        std::make_unique<std::thread>(&XServiceClient::heartbeat, this);
    initialize_done_ = true;
    return true;
  }

  const std::string etcd_username =
      util::get_optional_string_env(kEtcdUsernameEnvVar).value_or("");
  const std::string etcd_password =
      util::get_optional_string_env(kEtcdPasswordEnvVar).value_or("");
  const bool has_etcd_auth_user = !etcd_username.empty();
  const bool has_etcd_auth_password = !etcd_password.empty();
  if (has_etcd_auth_user != has_etcd_auth_password) {
    LOG(ERROR) << "Both " << kEtcdUsernameEnvVar << " and "
               << kEtcdPasswordEnvVar << " must be set together.";
    return false;
  }
  if (has_etcd_auth_user) {
    etcd_client_ = std::make_unique<EtcdClient>(etcd_addr,
                                                etcd_username,
                                                etcd_password,
                                                etcd_namespace,
                                                kDiscoveryRequestTimeout);
  } else {
    etcd_client_ = std::make_unique<EtcdClient>(
        etcd_addr, etcd_namespace, kDiscoveryRequestTimeout);
  }

  // connect master xllm_service
  while (!etcd_client_->get_master_service(ETCD_MASTER_SERVICE_KEY,
                                           &master_xservice_addr_)) {
    LOG(ERROR) << "Master service not set, wait 2s!";
    if (wait_for_stop(std::chrono::seconds(2))) {
      return false;
    }
  }

  if (!check_instance_name(master_xservice_addr_)) {
    LOG(FATAL) << "Invalid master service name format, now only support "
                  "`ip:port` style.";
    return false;
  }

  if (!connect_to_xservice(master_xservice_addr_)) {
    LOG(FATAL) << "Fail to initialize connection to master xservice server "
               << master_xservice_addr_;
    return false;
  }

  // Get and connect to all existing xllm_service instances.
  std::vector<std::string> all_services;
  if (etcd_client_->get_all_xservices(ETCD_XSERVICES_KEY_PREFIX,
                                      &all_services)) {
    for (const auto& service_addr : all_services) {
      if (service_addr != master_xservice_addr_ &&
          check_instance_name(service_addr)) {
        connect_to_xservice(service_addr);
      }
    }
  }

  // heartbeat thread
  heartbeat_thread_ =
      std::make_unique<std::thread>(&XServiceClient::heartbeat, this);
  reconcile_thread_ = std::make_unique<std::thread>(
      &XServiceClient::reconcile_registration_loop, this);

  // watch master xllm_service change
  auto master_func = [this](const etcd::Response& response,
                            uint64_t prefix_len) {
    handle_master_service_watch(response, prefix_len);
  };
  etcd_client_->add_watch(ETCD_MASTER_SERVICE_KEY, master_func);

  // watch all xllm_service changes
  auto xservices_func = [this](const etcd::Response& response,
                               uint64_t prefix_len) {
    handle_xservices_watch(response, prefix_len);
  };
  etcd_client_->add_watch(ETCD_XSERVICES_KEY_PREFIX, xservices_func);

  initialize_done_.store(!exited_.load());
  return initialize_done();
}

XServiceRegistration::XServiceRegistration(
    std::shared_ptr<XServiceClient> client,
    uint64_t registration_id)
    : client_(std::move(client)), registration_id_(registration_id) {}

XServiceRegistration::~XServiceRegistration() {
  client_->unbind(registration_id_);
}

std::unique_ptr<XServiceRegistration> XServiceClient::bind(
    InstanceInfo info,
    MetricsProvider metrics_provider) {
  CHECK(metrics_provider != nullptr);
  if (!initialize_done()) {
    return nullptr;
  }
  std::unique_ptr<XServiceRegistration> registration;
  std::unique_lock<std::mutex> lock(observer_mutex_);
  if (exited_.load() || info.name != instance_name_ ||
      (!observers_.empty() && info.type != observers_.front()->info.type)) {
    return nullptr;
  }
  auto observer = std::make_shared<Observer>();
  observer->id = next_observer_id_++;
  observer->metrics_provider = std::move(metrics_provider);
  observer->info = std::move(info);
  registration.reset(
      new XServiceRegistration(shared_from_this(), observer->id));
  observers_.emplace_back(observer);
  const bool needs_registration = observers_.size() == 1;
  InstanceInfo registration_info = observer->info;
  lock.unlock();
  // The etcd RPC runs off observer_mutex_ so heartbeats keep flowing while
  // registration retries; register_instance re-checks the generation under
  // registration_mutex_.
  if (needs_registration && !register_instance(registration_info)) {
    lock.lock();
    auto it = std::find(observers_.begin(), observers_.end(), observer);
    if (it != observers_.end()) {
      observers_.erase(it);
    }
    return nullptr;
  }
  return registration;
}

void XServiceClient::unbind(uint64_t registration_id) {
  std::unique_lock<std::mutex> lock(observer_mutex_);
  const bool was_primary =
      !observers_.empty() && observers_.front()->id == registration_id;
  auto it = std::find_if(observers_.begin(),
                         observers_.end(),
                         [registration_id](const auto& observer) {
                           return observer->id == registration_id;
                         });
  if (it == observers_.end()) {
    return;
  }
  std::shared_ptr<Observer> removed = *it;
  observers_.erase(it);
  std::string unregistered_key;
  if (observers_.empty()) {
    std::lock_guard<std::mutex> registration_lock(registration_mutex_);
    register_done_.store(false);
    ++registration_generation_;
    unregistered_key = registration_key_;
    registration_key_.clear();
    registration_value_.clear();
  } else if (was_primary) {
    CHECK(set_registration_info(observers_.front()->info));
  }
  lock.unlock();
  // The etcd RPC runs off observer_mutex_; the generation bump already makes
  // any concurrent re-registration authoritative.
  if (!unregistered_key.empty()) {
    etcd_client_->unregister_instance(unregistered_key);
  }
  lock.lock();
  observer_cv_.wait(lock, [&] { return removed->in_flight == 0; });
}

XServiceClient::~XServiceClient() { stop(); }

void XServiceClient::stop() {
  std::call_once(stop_once_, [this] {
    std::vector<brpc::CallId> pending;
    {
      std::lock_guard<std::mutex> lock(stop_mutex_);
      exited_.store(true);
      pending.reserve(control_rpcs_.size());
      for (const auto& entry : control_rpcs_) {
        pending.emplace_back(entry.second);
      }
    }
    stop_cv_.notify_all();
    for (brpc::CallId id : pending) {
      brpc::StartCancel(id);
    }
    // Initialization observes exited_ and uses bounded etcd operations.
    // Wait for publication of its threads before joining them.
    std::lock_guard<std::mutex> init_lock(init_mutex_);
    if (etcd_client_ != nullptr) {
      etcd_client_->stop_watch();
    }
    if (heartbeat_thread_ && heartbeat_thread_->joinable()) {
      heartbeat_thread_->join();
    }
    if (reconcile_thread_ && reconcile_thread_->joinable()) {
      reconcile_thread_->join();
    }
    if (etcd_client_ != nullptr) {
      etcd_client_->stop();
    }
  });
}

bool XServiceClient::track_control_rpc(brpc::CallId id) {
  std::lock_guard<std::mutex> lock(stop_mutex_);
  if (exited_.load()) {
    return false;
  }
  control_rpcs_.emplace(id.value, id);
  return true;
}

void XServiceClient::forget_control_rpc(brpc::CallId id) {
  std::lock_guard<std::mutex> lock(stop_mutex_);
  control_rpcs_.erase(id.value);
}

XServiceClient::ControlRpcScope::ControlRpcScope(XServiceClient& client,
                                                 brpc::Controller& cntl)
    : client_(client) {
  cntl.set_timeout_ms(std::chrono::duration_cast<std::chrono::milliseconds>(
                          kDiscoveryRequestTimeout)
                          .count());
  call_id_ = cntl.call_id();
  tracked_ = client_.track_control_rpc(call_id_);
}

XServiceClient::ControlRpcScope::~ControlRpcScope() {
  if (tracked_) {
    client_.forget_control_rpc(call_id_);
  }
}

bool XServiceClient::wait_for_stop(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(stop_mutex_);
  return stop_cv_.wait_for(lock, timeout, [this] { return exited_.load(); });
}

std::string XServiceClient::get_instance_name() { return instance_name_; }

bool XServiceClient::register_instance_with_retry(const std::string& key,
                                                  const std::string& value,
                                                  uint64_t generation) {
  int32_t retry_cnt = 0;
  while (!exited_.load()) {
    {
      std::lock_guard<std::mutex> lock(registration_mutex_);
      if (generation != registration_generation_ || registration_key_.empty()) {
        return false;
      }
    }
    // Run the etcd RPC unlocked so a stuck call cannot stall unbind or
    // reconcile_registration waiting on registration_mutex_; a superseded
    // registration is caught by the generation check on the next pass.
    if (etcd_client_->register_instance(
            key, value, DistributedConfig::get_instance().etcd_ttl())) {
      std::lock_guard<std::mutex> lock(registration_mutex_);
      return generation == registration_generation_;
    }
    if (retry_cnt >= 30) {
      LOG(ERROR) << "Register instance failed! key: " << key;
      return false;
    }

    LOG(WARNING) << "Register instance failed, wait 2s! key: " << key;
    if (wait_for_stop(std::chrono::seconds(2))) {
      return false;
    }
    retry_cnt++;
  }
  return false;
}

bool XServiceClient::reconcile_registration() {
  std::string registration_key;
  std::string registration_value;
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(registration_mutex_);
    if (!register_done_.load() || registration_key_.empty() ||
        registration_value_.empty()) {
      return true;
    }
    registration_key = registration_key_;
    registration_value = registration_value_;
    generation = registration_generation_;
  }

  std::string current_value;
  if (etcd_client_->get(registration_key, &current_value) &&
      current_value == registration_value) {
    return true;
  }

  LOG(WARNING) << "Detected missing instance registration in etcd, "
                  "re-registering instance: "
               << registration_key;
  if (register_instance_with_retry(
          registration_key, registration_value, generation)) {
    return true;
  }
  std::lock_guard<std::mutex> lock(registration_mutex_);
  // A removed or replaced registration is an expected cancellation.
  return generation != registration_generation_;
}

void XServiceClient::reconcile_registration_loop() {
  const auto interval = heartbeat_interval();
  while (!wait_for_stop(interval)) {
    if (!register_done_.load()) continue;

    if (!reconcile_registration() && !exited_.load()) {
      LOG(ERROR) << "Failed to reconcile instance registration in etcd.";
    }
  }
}

bool XServiceClient::set_registration_info(const InstanceInfo& instance_info) {
  if (instance_info.name != instance_name_) {
    LOG(ERROR) << "Discovery registration does not match its serving instance";
    return false;
  }
  InstanceInfo registered_info = instance_info;
  registered_info.incarnation_id = incarnation_id_;
  if (registered_info.register_ts_ms == 0) {
    registered_info.register_ts_ms =
        static_cast<uint64_t>(absl::ToUnixMillis(absl::Now()));
  }

  if (direct_service_) {
    xllm_service::proto::InstanceMetaInfo registration;
    registration.set_name(registered_info.name);
    registration.set_rpc_address(registered_info.rpc_address);
    registration.set_incarnation_id(registered_info.incarnation_id);
    registration.set_register_ts_ms(registered_info.register_ts_ms);
    if (InstanceRole(registered_info.type) == InstanceRole::PREFILL) {
      registration.set_type(xllm_service::proto::PREFILL);
    } else if (InstanceRole(registered_info.type) == InstanceRole::DECODE) {
      registration.set_type(xllm_service::proto::DECODE);
    } else {
      LOG(FATAL) << "Direct service mode supports PREFILL or DECODE only.";
    }
    registration.set_dp_size(registered_info.dp_size);
    registration.set_kv_split_size(registered_info.kv_split_size);
    for (uint64_t cluster_id : registered_info.cluster_ids) {
      registration.add_cluster_ids(cluster_id);
    }
    for (const std::string& address : registered_info.addrs) {
      registration.add_addrs(address);
    }
    for (uint16_t port : registered_info.ports) {
      registration.add_ports(port);
    }
    {
      std::lock_guard<std::mutex> lock(registration_mutex_);
      instance_name_ = registered_info.name;
      direct_registration_ = std::move(registration);
    }
    const bool registered = register_direct_instance_with_retry();
    if (exited_.load()) {
      return false;
    }
    CHECK(registered) << "Direct service registration failed.";
    register_done_.store(true);
    return true;
  }

  std::string key_prefix = "";
  if (InstanceRole(registered_info.type) == InstanceRole::DEFAULT) {
    key_prefix =
        ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::DEFAULT];
  } else if (InstanceRole(registered_info.type) == InstanceRole::PREFILL) {
    key_prefix =
        ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::PREFILL];
  } else if (InstanceRole(registered_info.type) == InstanceRole::DECODE) {
    key_prefix =
        ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::DECODE];
  } else if (InstanceRole(registered_info.type) == InstanceRole::MIX) {
    key_prefix = ETCD_KEYS_PREFIX_MAP[xllm_service::proto::InstanceType::MIX];
  } else {
    LOG(ERROR) << "Unsupported instance type: " << registered_info.type;
    return false;
  }

  const std::string key = key_prefix + registered_info.name;
  const std::string value = registered_info.serialize_to_json().dump();
  {
    std::lock_guard<std::mutex> lock(registration_mutex_);
    registration_key_ = key;
    registration_value_ = value;
    ++registration_generation_;
  }

  return true;
}

bool XServiceClient::register_instance(const InstanceInfo& info) {
  if (!set_registration_info(info)) {
    return false;
  }
  std::string key;
  std::string value;
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(registration_mutex_);
    key = registration_key_;
    value = registration_value_;
    generation = registration_generation_;
  }
  if (!register_instance_with_retry(key, value, generation)) {
    return false;
  }
  std::lock_guard<std::mutex> lock(registration_mutex_);
  if (generation != registration_generation_ || exited_.load()) {
    return false;
  }
  register_done_.store(true);
  LOG(INFO) << "Success register instance to etcd.";
  return true;
}

XServiceClient::DirectRegistrationResult
XServiceClient::register_direct_instance() {
  xllm_service::proto::InstanceMetaInfo request;
  {
    std::lock_guard<std::mutex> lock(registration_mutex_);
    request = direct_registration_;
  }
  brpc::Controller cntl;
  xllm_service::proto::StatusCode response;
  std::string master_address;
  if (!with_master_stub(
          [&](xllm_service::proto::XllmRpcService_Stub* stub) {
            stub->RegisterInstance(&cntl, &request, &response, nullptr);
          },
          &master_address)) {
    return DirectRegistrationResult::REJECTED;
  }
  if (cntl.Failed() || response.status_code() != 0) {
    LOG(ERROR) << "Direct service registration failed: " << master_address
               << ", RPC error: " << cntl.ErrorText()
               << ", status: " << response.status_code();
    if (cntl.Failed()) {
      const int32_t http_status = cntl.http_response().status_code();
      if (!http_service_protocol_ || http_status < 400 || http_status >= 500 ||
          http_status == 408 || http_status == 429) {
        return DirectRegistrationResult::RETRYABLE_FAILURE;
      }
    }
    return DirectRegistrationResult::REJECTED;
  }
  return DirectRegistrationResult::SUCCESS;
}

bool XServiceClient::register_direct_instance_with_retry() {
  constexpr int32_t kMaxAttempts = 31;
  for (int32_t attempt = 1; attempt <= kMaxAttempts && !exited_.load();
       ++attempt) {
    const DirectRegistrationResult result = register_direct_instance();
    if (result == DirectRegistrationResult::SUCCESS) {
      return true;
    }
    if (result == DirectRegistrationResult::REJECTED ||
        attempt == kMaxAttempts) {
      return false;
    }
    LOG(WARNING) << "Direct service registration retry, attempt=" << attempt;
    if (wait_for_stop(std::chrono::seconds(2))) {
      return false;
    }
  }
  return false;
}

InstanceInfo XServiceClient::get_instance_info(
    const std::string& instance_name) {
  InstanceInfo result;
  brpc::Controller cntl;
  ControlRpcScope rpc(*this, cntl);
  if (!rpc.tracked()) {
    return result;
  }
  xllm_service::proto::InstanceID req;
  xllm_service::proto::InstanceMetaInfo resp;
  req.set_name(instance_name);

  std::string master_addr;
  if (!with_master_stub(
          [&](xllm_service::proto::XllmRpcService_Stub* master_stub) {
            master_stub->GetInstanceInfo(&cntl, &req, &resp, nullptr);
          },
          &master_addr)) {
    return result;
  }

  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to get instance info from xservice server "
               << master_addr << ", error text: " << cntl.ErrorText();
    return result;
  }
  result.name = resp.name();
  result.rpc_address = resp.rpc_address();
  result.incarnation_id = resp.incarnation_id();
  result.register_ts_ms = resp.register_ts_ms();
  if (resp.type() == xllm_service::proto::InstanceType::PREFILL) {
    result.type = "PREFILL";
  } else if (resp.type() == xllm_service::proto::InstanceType::DECODE) {
    result.type = "DECODE";
  } else if (resp.type() == xllm_service::proto::InstanceType::MIX) {
    result.type = "MIX";
  } else {
    result.type = "DEFAULT";
  }
  // parse kv cache info
  for (auto& cluster_id : resp.cluster_ids()) {
    result.cluster_ids.emplace_back(cluster_id);
  }
  for (auto& addr : resp.addrs()) {
    result.addrs.emplace_back(addr);
  }
  result.dp_size = resp.dp_size();
  if (resp.kv_split_size() > 0) {
    result.kv_split_size = resp.kv_split_size();
  }
  for (auto& port : resp.ports()) {
    result.ports.emplace_back(port);
  }

  return result;
}

void XServiceClient::heartbeat() {
  const auto interval = heartbeat_interval();
  while (!wait_for_stop(interval)) {
    if (!register_done_.load()) continue;

    brpc::Controller cntl;
    ControlRpcScope rpc(*this, cntl);
    if (!rpc.tracked()) {
      return;
    }
    xllm_service::proto::HeartbeatRequest req;
    req.set_name(instance_name_);
    req.set_incarnation_id(incarnation_id_);

    std::shared_ptr<Observer> observer;
    {
      std::lock_guard<std::mutex> lock(observer_mutex_);
      if (observers_.empty()) {
        continue;
      }
      observer = observers_.front();
      ++observer->in_flight;
    }
    XServiceMetricsSnapshot metrics = observer->metrics_provider();
    bool still_primary = false;
    {
      std::lock_guard<std::mutex> lock(observer_mutex_);
      --observer->in_flight;
      still_primary = !observers_.empty() && observers_.front() == observer;
      observer_cv_.notify_all();
    }
    if (!still_primary) {
      continue;
    }
    req.mutable_load_metrics()->set_gpu_cache_usage_perc(
        metrics.gpu_cache_usage_perc);
    req.mutable_load_metrics()->set_waiting_requests_num(
        metrics.waiting_requests_num);
    if (!metrics.ttft.empty()) {
      auto max_ttft =
          std::max_element(metrics.ttft.begin(), metrics.ttft.end());
      req.mutable_latency_metrics()->set_recent_max_ttft(*max_ttft);
    }

    if (!metrics.tbt.empty()) {
      auto max_tbt = std::max_element(metrics.tbt.begin(), metrics.tbt.end());
      req.mutable_latency_metrics()->set_recent_max_tbt(*max_tbt);
    }

    xllm_service::proto::Status resp;
    std::string master_addr;
    if (!with_master_stub(
            [&](xllm_service::proto::XllmRpcService_Stub* master_stub) {
              master_stub->Heartbeat(&cntl, &req, &resp, nullptr);
            },
            &master_addr)) {
      continue;
    }

    if (cntl.Failed()) {
      if (!exited_.load()) {
        LOG(ERROR) << "Failed to send heartbeat to master xservice "
                   << master_addr << ", error msg is: " << cntl.ErrorText();
      }
    } else if (!resp.ok()) {
      LOG(ERROR) << "Failed to send heartbeat to master xservice "
                 << master_addr;
      if (direct_service_ &&
          register_direct_instance() != DirectRegistrationResult::SUCCESS) {
        LOG(ERROR) << "Failed to renew direct service registration.";
      }
    }
  }
}

std::vector<std::string> XServiceClient::get_static_decode_list() {
  brpc::Controller cntl;
  ControlRpcScope rpc(*this, cntl);
  if (!rpc.tracked()) {
    return {};
  }
  xllm_service::proto::InstanceID req;
  xllm_service::proto::InstanceIDs resp;
  req.set_name(instance_name_);

  std::string master_addr;
  if (!with_master_stub(
          [&](xllm_service::proto::XllmRpcService_Stub* master_stub) {
            master_stub->GetStaticDecodeList(&cntl, &req, &resp, nullptr);
          },
          &master_addr)) {
    return {};
  }

  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to get static decode list from master xservice server "
               << master_addr << ", error text: " << cntl.ErrorText();
    return {};
  }
  return std::vector<std::string>(resp.names().begin(), resp.names().end());
}

std::vector<std::string> XServiceClient::get_all_xservice_addrs() {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  std::vector<std::string> addrs;
  for (const auto& pair : xservice_stubs_) {
    addrs.push_back(pair.first);
  }
  return addrs;
}

bool XServiceClient::generation(const RequestOutput& output) {
  const std::string& protocol =
      DistributedConfig::get_instance().xservice_rpc_protocol();
  if (protocol == "http" || output.response_formatter) {
    auto results = send_generations(std::span<const RequestOutput>(&output, 1));
    CHECK_EQ(results.size(), 1);
    return results.front();
  }
  CHECK_EQ(protocol, "baidu_std");
  std::string master_addr;
  if (output.target_xservice_addr.empty()) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    master_addr = master_xservice_addr_;
  }
  const std::string& target_service = output.target_xservice_addr.empty()
                                          ? master_addr
                                          : output.target_xservice_addr;
  if (target_service.empty()) {
    LOG(ERROR) << "No target xservice address available for request_id: "
               << output.request_id;
    return false;
  }
  if (!connect_to_xservice(target_service)) {
    LOG(ERROR) << "Failed to connect target xservice: " << target_service;
    return false;
  }

  proto::DisaggStreamGenerations request;
  fill_generation_request(output, request.mutable_gens()->Add());
  proto::StatusSet response;
  brpc::Controller cntl;
  std::shared_ptr<brpc::Channel> channel;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto* service_stub = find_stub_locked(target_service);
    if (service_stub == nullptr) {
      LOG(ERROR) << "No stub available for xservice: " << target_service;
      return false;
    }
    const auto channel_it = xservice_channels_.find(
        service_channel_key(target_service, http_service_protocol_));
    CHECK(channel_it != xservice_channels_.end());
    channel = channel_it->second;
    service_stub->Generations(&cntl, &request, &response, brpc::DoNothing());
  }
  brpc::Join(cntl.call_id());
  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to response tokens to xservice server "
               << target_service << ", error text: " << cntl.ErrorText();
    return false;
  }
  CHECK_EQ(response.all_status_size(), 1)
      << "The size of status set is not equal to the size of outputs";
  return response.all_status(0).ok();
}

std::vector<bool> XServiceClient::generations(
    const std::vector<RequestOutput>& outputs) {
  return send_generations(outputs);
}

std::vector<bool> XServiceClient::send_generations(
    std::span<const RequestOutput> outputs) {
  const std::string& protocol =
      DistributedConfig::get_instance().xservice_rpc_protocol();
  if (protocol == "http") {
    return generations_http(outputs);
  }
  CHECK_EQ(protocol, "baidu_std");
  return generations_rpc(outputs);
}

std::vector<bool> XServiceClient::generations_rpc(
    std::span<const RequestOutput> outputs) {
  std::vector<bool> results(outputs.size(), false);
  if (outputs.empty()) {
    return results;
  }
  std::string master_addr;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    master_addr = master_xservice_addr_;
  }

  struct ServiceBatch {
    std::vector<size_t> output_indices;
    proto::DisaggStreamGenerations gens;
  };

  // Group requests by target xllm_service. Keeping the indices and protobuf
  // request together avoids a second address-keyed map and a later lookup.
  std::unordered_map<std::string, ServiceBatch> service_batches;

  auto mark_batch_failed = [&](const ServiceBatch& batch) {
    for (size_t index : batch.output_indices) {
      if (outputs[index].response_formatter) {
        *outputs[index].response_formatter = {};
      }
    }
  };

  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    const std::string& target_service = output.target_xservice_addr.empty()
                                            ? master_addr
                                            : output.target_xservice_addr;

    if (target_service.empty()) {
      LOG(ERROR) << "No target xservice address available for request_id: "
                 << output.request_id;
      continue;
    }

    std::optional<ServiceGenerationOutput> formatted;
    if (output.response_formatter) {
      auto& formatter = *output.response_formatter;
      if (!formatter) {
        continue;
      }
      const bool keep_running = formatter(output, /*http=*/false, &formatted);
      if (!keep_running) {
        formatter = {};
      }
      if (!formatted.has_value()) {
        results[i] = keep_running;
        continue;
      }
    }

    auto& batch = service_batches[target_service];
    batch.output_indices.push_back(i);

    // Construct the request to corresponding service.
    auto* request = batch.gens.mutable_gens()->Add();
    fill_generation_request(output, request);
    if (formatted.has_value() && formatted->chat_output) {
      request->mutable_chat_output()->Swap(formatted->chat_output.get());
    }
  }

  // Use brpc semi-synchronous RPC pattern (DoNothing + Join) instead of
  // folly thread pool. brpc::Join is bthread-aware: it yields the brpc worker
  // thread via butex_wait, avoiding the deadlock that occurs when
  // folly::SemiFuture::wait() blocks all brpc workers with a pthread futex.
  struct AsyncCallContext {
    const std::string* service_addr = nullptr;
    const ServiceBatch* batch = nullptr;
    std::shared_ptr<brpc::Channel> channel;
    brpc::Controller cntl;
    proto::StatusSet resp;
    bool issued = false;
  };

  const size_t num_services = service_batches.size();
  std::vector<AsyncCallContext> contexts(num_services);

  // Fire all RPCs asynchronously
  size_t idx = 0;
  for (auto& pair : service_batches) {
    auto& ctx = contexts[idx++];
    ctx.service_addr = &pair.first;
    ctx.batch = &pair.second;

    if (!connect_to_xservice(*ctx.service_addr)) {
      LOG(ERROR) << "Failed to connect target xservice: " << *ctx.service_addr;
      continue;
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto* service_stub = find_stub_locked(*ctx.service_addr);
    if (service_stub == nullptr) {
      LOG(ERROR) << "No stub available for xservice: " << *ctx.service_addr;
      continue;
    }
    const auto channel_it = xservice_channels_.find(
        service_channel_key(*ctx.service_addr, http_service_protocol_));
    CHECK(channel_it != xservice_channels_.end());
    ctx.channel = channel_it->second;
    ctx.issued = true;
    service_stub->Generations(
        &ctx.cntl, &pair.second.gens, &ctx.resp, brpc::DoNothing());
  }

  // Wait for all RPCs — bthread-aware, yields brpc worker properly
  for (auto& ctx : contexts) {
    if (ctx.issued) {
      brpc::Join(ctx.cntl.call_id());
    }
  }

  // Process results
  for (auto& ctx : contexts) {
    CHECK(ctx.service_addr != nullptr);
    CHECK(ctx.batch != nullptr);
    const std::string& service_addr = *ctx.service_addr;
    const auto& indices = ctx.batch->output_indices;

    if (!ctx.issued || ctx.cntl.Failed()) {
      if (ctx.cntl.Failed()) {
        LOG(ERROR) << "Fail to response tokens to xservice server "
                   << service_addr << ", error text: " << ctx.cntl.ErrorText();
      }
      mark_batch_failed(*ctx.batch);
      continue;
    }

    if (ctx.resp.all_status_size() != static_cast<int32_t>(indices.size())) {
      LOG(ERROR) << "Invalid generation ACK count from service: "
                 << service_addr;
      mark_batch_failed(*ctx.batch);
      continue;
    }

    for (size_t j = 0; j < indices.size(); ++j) {
      size_t original_idx = indices[j];
      results[original_idx] = ctx.resp.all_status(j).ok();
      if (!results[original_idx] && outputs[original_idx].response_formatter) {
        *outputs[original_idx].response_formatter = {};
      }
    }
  }

  return results;
}

std::vector<bool> XServiceClient::generations_http(
    std::span<const RequestOutput> outputs) {
  std::vector<bool> results(outputs.size(), false);
  std::vector<ServiceGenerationOutput> formatted;
  std::vector<size_t> indices;
  formatted.reserve(outputs.size());
  indices.reserve(outputs.size());
  for (size_t index = 0; index < outputs.size(); ++index) {
    const auto& output = outputs[index];
    const bool failed = output.status.has_value() && !output.status->ok();
    if (output.outputs.empty() && !output.finished && !output.cancelled &&
        !failed) {
      results[index] = true;
      continue;
    }
    std::optional<ServiceGenerationOutput> result;
    if (output.response_formatter) {
      auto& formatter = *output.response_formatter;
      if (!formatter) {
        continue;
      }
      const bool keep_running = formatter(output, /*http=*/true, &result);
      if (!keep_running) {
        formatter = {};
      }
      if (!result.has_value()) {
        results[index] = keep_running;
        continue;
      }
    } else if (failed) {
      // Prefill only reports terminal errors and needs no token parser.
      result.emplace();
      result->stream = output.stream;
      result->finished = true;
      result->status_code =
          api_service::openai_http_status(output.status->code());
      result->payload = api_service::openai_error_json(
          output.status->code(), output.status->message());
      if (output.stream) {
        result->payload = "data: " + result->payload + "\n\ndata: [DONE]\n\n";
      }
    } else {
      LOG(ERROR) << "HTTP generation requires a response formatter, request_id="
                 << output.request_id;
      continue;
    }
    result->service_request_id = output.service_request_id;
    result->target_xservice_addr = output.target_xservice_addr;
    formatted.emplace_back(std::move(*result));
    indices.emplace_back(index);
  }
  const auto accepted = send_http_generations(formatted);
  CHECK_EQ(accepted.size(), indices.size());
  for (size_t index = 0; index < indices.size(); ++index) {
    const size_t original = indices[index];
    results[original] = accepted[index];
    if (!accepted[index] && outputs[original].response_formatter) {
      // Pending outputs share this context, so an ambiguous failure cannot
      // generate and send later deltas after the failed batch.
      *outputs[original].response_formatter = {};
    }
  }
  return results;
}

std::vector<bool> XServiceClient::send_http_generations(
    const std::vector<ServiceGenerationOutput>& outputs) {
  struct Batch {
    std::string body{kHttpBatchPrefix};
    std::vector<size_t> indices;
  };
  struct Target {
    std::shared_ptr<brpc::Channel> channel;
    std::vector<Batch> batches;
    size_t next = 0;
  };
  std::vector<bool> accepted(outputs.size(), false);
  std::vector<bool> oversized(outputs.size(), false);
  std::unordered_map<std::string, Target> targets;
  targets.reserve(outputs.size());
  for (size_t index = 0; index < outputs.size(); ++index) {
    const auto& output = outputs[index];
    if (output.target_xservice_addr.empty() ||
        output.service_request_id.empty() || output.payload.empty()) {
      LOG(ERROR) << "HTTP batch output requires an owner and a payload";
      continue;
    }
    nlohmann::json entry = {
        {"attempt_id", output.service_request_id},
        {"sequence", output.sequence},
        {"finished", output.finished},
        {"status_code", output.status_code},
        {"content_type",
         output.stream ? "text/event-stream" : "application/json"},
        {"payload", output.payload}};
    std::string encoded = entry.dump(-1,
                                     ' ',
                                     /*ensure_ascii=*/false,
                                     nlohmann::json::error_handler_t::replace);
    if (encoded.size() + kHttpBatchPrefix.size() + 2 > kMaxHttpBatchBytes) {
      LOG(ERROR) << "HTTP batch item exceeds the body limit, attempt_id="
                 << output.service_request_id;
      // This is a local rejection before delivery, so a compact terminal
      // error is safe. Ambiguous transport failures below are never replayed.
      oversized[index] = true;
      std::string error = api_service::openai_error_json(
          StatusCode::RESOURCE_EXHAUSTED,
          "The generated response exceeds the HTTP callback body limit");
      entry["finished"] = true;
      entry["status_code"] = 500;
      entry["payload"] =
          output.stream ? "data: " + error + "\n\ndata: [DONE]\n\n" : error;
      encoded = entry.dump(-1,
                           ' ',
                           /*ensure_ascii=*/false,
                           nlohmann::json::error_handler_t::replace);
      if (encoded.size() + kHttpBatchPrefix.size() + 2 > kMaxHttpBatchBytes) {
        LOG(ERROR) << "HTTP callback identity exceeds the body limit";
        continue;
      }
    }
    auto& target = targets[output.target_xservice_addr];
    if (target.batches.empty() ||
        target.batches.back().indices.size() >= kMaxHttpBatchResults ||
        target.batches.back().body.size() + encoded.size() + 3 >
            kMaxHttpBatchBytes) {
      target.batches.emplace_back();
      target.batches.back().indices.reserve(
          std::min(outputs.size(), kMaxHttpBatchResults));
    }
    auto& batch = target.batches.back();
    if (!batch.indices.empty()) {
      batch.body += ',';
    }
    batch.body += encoded;
    batch.indices.emplace_back(index);
  }

  for (auto& [address, target] : targets) {
    target.channel = get_http_callback_channel(address);
    if (target.channel == nullptr) {
      target.next = target.batches.size();
    }
  }
  // Issue one chunk per owner concurrently. Keep later chunks behind the
  // previous ACK so splitting a large batch never reorders an attempt.
  while (true) {
    struct Call {
      Target* target;
      brpc::Controller controller;
    };
    std::vector<std::unique_ptr<Call>> calls;
    calls.reserve(targets.size());
    for (auto& [address, target] : targets) {
      if (target.next == target.batches.size()) {
        continue;
      }
      auto call = std::make_unique<Call>();
      call->target = &target;
      auto& controller = call->controller;
      controller.http_request().set_method(brpc::HTTP_METHOD_POST);
      controller.http_request().uri() = "/xllm/v1/results";
      controller.http_request().set_content_type("application/json");
      controller.http_request().SetHeader("Connection", "keep-alive");
      auto& batch = target.batches[target.next];
      controller.request_attachment().append(batch.body);
      controller.request_attachment().append("]}");
      target.channel->CallMethod(/*method=*/nullptr,
                                 &controller,
                                 /*request=*/nullptr,
                                 /*response=*/nullptr,
                                 brpc::DoNothing());
      calls.emplace_back(std::move(call));
    }
    if (calls.empty()) {
      break;
    }
    for (auto& call : calls) {
      auto& controller = call->controller;
      brpc::Join(controller.call_id());
      auto& target = *call->target;
      const auto& indices = target.batches[target.next].indices;
      const nlohmann::json response =
          nlohmann::json::parse(controller.response_attachment().to_string(),
                                /*cb=*/nullptr,
                                /*allow_exceptions=*/false);
      const bool valid =
          !controller.Failed() &&
          controller.http_response().status_code() == 200 &&
          response.is_object() && response.contains("accepted") &&
          response["accepted"].is_array() &&
          response["accepted"].size() == indices.size() &&
          std::all_of(response["accepted"].begin(),
                      response["accepted"].end(),
                      [](const auto& value) { return value.is_boolean(); });
      if (!valid) {
        LOG(ERROR) << "HTTP batch output failed or returned an invalid ACK, "
                   << "HTTP status=" << controller.http_response().status_code()
                   << ", error=" << controller.ErrorText();
        // Delivery may already have happened. Do not replay this batch or
        // issue its remaining chunks after an ambiguous failure.
        target.next = target.batches.size();
        continue;
      }
      for (size_t index = 0; index < indices.size(); ++index) {
        // Stop engine output after a size rejection even if the peer accepted
        // its terminal error. generations_http also closes the shared parser.
        accepted[indices[index]] = !oversized[indices[index]] &&
                                   response["accepted"][index].get<bool>();
      }
      ++target.next;
    }
  }
  return accepted;
}

std::shared_ptr<brpc::Channel> XServiceClient::get_service_channel(
    const std::string& xservice_addr,
    bool http) {
  const std::string key = service_channel_key(xservice_addr, http);
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const auto it = xservice_channels_.find(key);
    if (it != xservice_channels_.end()) {
      return it->second;
    }
  }

  auto channel = create_service_channel(xservice_addr, http);
  if (channel == nullptr) {
    return nullptr;
  }
  std::unique_lock<std::shared_mutex> lock(mutex_);
  return xservice_channels_.try_emplace(key, std::move(channel)).first->second;
}

std::shared_ptr<brpc::Channel> XServiceClient::get_http_callback_channel(
    const std::string& address) {
  auto channel = http_callback_channels_.find(address);
  if (channel != nullptr) {
    return channel;
  }
  channel = create_service_channel(address, /*http=*/true);
  return channel == nullptr
             ? nullptr
             : http_callback_channels_.insert(address, std::move(channel));
}

std::shared_ptr<brpc::Channel> XServiceClient::create_service_channel(
    const std::string& address,
    bool http) {
  brpc::ChannelOptions options = chan_options_;
  if (http) {
    options.protocol = "http";
    options.connection_type = brpc::CONNECTION_TYPE_POOLED;
    // The peer may have applied a delta before its ACK is lost. Never replay
    // it.
    options.max_retry = 0;
    const int32_t timeout_ms =
        ServiceConfig::get_instance().rpc_channel_timeout_ms();
    options.timeout_ms = timeout_ms > 0 ? timeout_ms : kDefaultHttpTimeoutMs;
  }
  auto channel = std::make_shared<brpc::Channel>();
  if (channel->Init(address.c_str(), /*load_balancer_name=*/"", &options) !=
      0) {
    LOG(ERROR) << "Fail to initialize service channel to " << address;
    return nullptr;
  }

  return channel;
}

bool XServiceClient::connect_to_xservice(const std::string& xservice_addr) {
  if (!check_instance_name(xservice_addr)) {
    LOG(ERROR) << "Invalid xservice address format: " << xservice_addr;
    return false;
  }

  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (xservice_stubs_.find(xservice_addr) != xservice_stubs_.end()) {
      return true;
    }
  }

  auto channel = get_service_channel(xservice_addr, http_service_protocol_);
  if (channel == nullptr) {
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);
  // Reinsert if discovery removed the cache entry while the channel was built.
  const auto channel_it =
      xservice_channels_
          .try_emplace(
              service_channel_key(xservice_addr, http_service_protocol_),
              std::move(channel))
          .first;
  xservice_stubs_.try_emplace(
      xservice_addr,
      std::make_unique<xllm_service::proto::XllmRpcService_Stub>(
          channel_it->second.get()));

  LOG(INFO) << "Successfully connected to xservice: " << xservice_addr;
  return true;
}

bool XServiceClient::with_master_stub(
    const std::function<void(xllm_service::proto::XllmRpcService_Stub*)>& fn,
    std::string* master_addr) {
  if (master_addr == nullptr) {
    return false;
  }

  // wrapper in a whole lambda function
  auto run_with_current_master_stub = [&](bool* has_master_addr) -> bool {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    *master_addr = master_xservice_addr_;
    *has_master_addr = !master_addr->empty();
    if (!*has_master_addr) {
      LOG(ERROR) << "Master xservice address is empty";
      return false;
    }

    auto* master_stub = find_stub_locked(*master_addr);
    if (master_stub == nullptr) {
      return false;
    }

    fn(master_stub);
    return true;
  };

  bool has_master_addr = false;
  if (run_with_current_master_stub(&has_master_addr)) {
    return true;
  }
  if (!has_master_addr) {
    return false;
  }

  // try re-connecting once
  if (!connect_to_xservice(*master_addr)) {
    LOG(ERROR) << "Failed to connect to master xservice: " << *master_addr;
    return false;
  }

  if (run_with_current_master_stub(&has_master_addr)) {
    return true;
  }
  if (!has_master_addr) {
    return false;
  }

  LOG(ERROR) << "No master stub available for address: " << *master_addr;
  return false;
}

xllm_service::proto::XllmRpcService_Stub* XServiceClient::find_stub_locked(
    const std::string& xservice_addr) {
  auto it = xservice_stubs_.find(xservice_addr);
  if (it == xservice_stubs_.end() || it->second == nullptr) {
    return nullptr;
  }
  return it->second.get();
}

void XServiceClient::disconnect_xservice(const std::string& xservice_addr) {
  http_callback_channels_.erase(xservice_addr);
  std::unique_lock<std::shared_mutex> lock(mutex_);

  if (xservice_stubs_.erase(xservice_addr) > 0) {
    xservice_channels_.erase(
        service_channel_key(xservice_addr, /*http=*/false));
    xservice_channels_.erase(service_channel_key(xservice_addr, /*http=*/true));
    LOG(INFO) << "Disconnected from xservice: " << xservice_addr;

    // if master disconnected，need to update master address
    if (xservice_addr == master_xservice_addr_) {
      LOG(WARNING) << "Master xservice disconnected: " << master_xservice_addr_;
    }
  }
}

void XServiceClient::handle_master_service_watch(const etcd::Response& response,
                                                 const uint64_t& prefix_len) {
  if (response.events().empty() || exited_.load()) {
    return;
  }

  for (const auto& event : response.events()) {
    if (event.event_type() == etcd::Event::EventType::PUT) {
      auto new_master_addr = event.kv().as_string();

      {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (master_xservice_addr_.compare(new_master_addr) == 0) {
          continue;
        }

        LOG(INFO) << "Master service changed from " << master_xservice_addr_
                  << " to " << new_master_addr;

        master_xservice_addr_ = new_master_addr;
      }

      if (!connect_to_xservice(new_master_addr)) {
        LOG(ERROR) << "Failed to connect to new master: " << new_master_addr;
      }
    } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      if (!master_xservice_addr_.empty()) {
        LOG(WARNING) << "Master service key deleted, clear cached master addr: "
                     << master_xservice_addr_;
        master_xservice_addr_.clear();
      }
    }
  }
}

void XServiceClient::handle_xservices_watch(const etcd::Response& response,
                                            const uint64_t& prefix_len) {
  if (response.events().empty() || exited_.load()) {
    return;
  }

  for (const auto& event : response.events()) {
    std::string event_key;
    std::string service_addr;
    if (event.event_type() == etcd::Event::EventType::PUT) {
      if (event.has_kv()) {
        event_key = event.kv().key().substr(prefix_len);
        service_addr = event.kv().as_string();
      }
    } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
      if (event.has_prev_kv()) {
        event_key = event.prev_kv().key().substr(prefix_len);
        service_addr = event.prev_kv().as_string();
      }
      if (service_addr.empty() && event.has_kv()) {
        if (event_key.empty()) {
          event_key = event.kv().key().substr(prefix_len);
        }
        service_addr = event.kv().as_string();
      }
    }

    if (event_key == ETCD_MASTER_SERVICE_KEY) {
      continue;
    }

    if (service_addr.empty() && !event_key.empty() &&
        event_key.rfind(ETCD_XSERVICES_KEY_PREFIX, 0) == 0) {
      service_addr = event_key.substr(ETCD_XSERVICES_KEY_PREFIX.size());
    }

    if (service_addr.empty()) {
      continue;
    }

    if (!check_instance_name(service_addr)) {
      continue;
    }

    if (event.event_type() == etcd::Event::EventType::PUT) {
      std::string master_xservice_addr;
      {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        master_xservice_addr = master_xservice_addr_;
      }

      if (service_addr != master_xservice_addr) {
        connect_to_xservice(service_addr);
      }
    } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
      disconnect_xservice(service_addr);
    }
  }
}

}  // namespace xllm
