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

#include "etcd_client.h"

#include <glog/logging.h>

#include <etcd/Client.hpp>
#include <etcd/Response.hpp>
#include <nlohmann/json.hpp>

#include "xllm/core/common/etcd_utils.h"

namespace xllm {

EtcdClient::EtcdClient(const std::string& etcd_addr,
                       const std::string& etcd_namespace,
                       std::chrono::milliseconds timeout)
    : client_(etcd_addr),
      etcd_addr_(etcd_addr),
      etcd_namespace_prefix_(normalize_etcd_namespace(etcd_namespace)) {
  client_.set_grpc_timeout(timeout);
  auto response = client_.put(namespaced_key("XLLM_PING"), "PING");
  if (!response.is_ok()) {
    LOG(FATAL) << "etcd connect to etcd server failed: "
               << response.error_message();
  }
}

EtcdClient::EtcdClient(const std::string& etcd_addr,
                       const std::string& username,
                       const std::string& password,
                       const std::string& etcd_namespace,
                       std::chrono::milliseconds timeout)
    : client_(etcd_addr,
              username,
              password,
              /*auth_token_ttl=*/300,
              /*load_balancer=*/"round_robin",
              timeout),
      etcd_addr_(etcd_addr),
      etcd_namespace_prefix_(normalize_etcd_namespace(etcd_namespace)) {
  // The patched auth constructor already applies the request timeout to both
  // the authentication and the per-RPC deadline.
  auto response = client_.put(namespaced_key("XLLM_PING"), "PING");
  if (!response.is_ok()) {
    LOG(FATAL) << "etcd connect to etcd server failed: "
               << response.error_message();
  }
}

std::string EtcdClient::namespaced_key(const std::string& logical_key) const {
  if (etcd_namespace_prefix_.empty()) {
    return logical_key;
  }
  return etcd_namespace_prefix_ + logical_key;
}

EtcdClient::~EtcdClient() { stop(); }

void EtcdClient::stop() {
  if (stopped_.exchange(true)) {
    return;
  }
  decltype(keep_alives_) keep_alives;
  {
    std::lock_guard<std::mutex> lock(keep_alives_mutex_);
    keep_alives.swap(keep_alives_);
  }
  stop_watch();
  for (auto& [key, keep_alive] : keep_alives) {
    keep_alive->Cancel();
  }
}

void EtcdClient::add_watch(const std::string& key_prefix,
                           Callback callback,
                           bool recursive) {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  if (watchers_.find(key_prefix) != watchers_.end()) {
    watchers_[key_prefix].watcher->Cancel();
  }

  uint64_t prefix_len = etcd_namespace_prefix_.size();
  auto watcher = std::make_unique<etcd::Watcher>(
      client_,
      namespaced_key(key_prefix),
      [callback, prefix_len](const etcd::Response& response) {
        callback(response, prefix_len);
      },
      recursive);

  watchers_[key_prefix] = {std::move(watcher), callback};
}

void EtcdClient::remove_watch(const std::string& key_prefix) {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  auto it = watchers_.find(key_prefix);
  if (it != watchers_.end()) {
    it->second.watcher->Cancel();
    watchers_.erase(it);
  }
}

void EtcdClient::stop_watch() {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  for (auto& pair : watchers_) {
    pair.second.watcher->Cancel();
  }

  watchers_.clear();
}

bool EtcdClient::get_master_service(const std::string& key,
                                    std::string* values) {
  auto response = client_.get(namespaced_key(key));
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key << " failed: " << response.error_message();
    return false;
  }
  *values = response.value().as_string();
  return true;
}

bool EtcdClient::get(const std::string& key, std::string* value) {
  auto response = client_.get(namespaced_key(key));
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key << " failed: " << response.error_message();
    return false;
  }

  if (value != nullptr) {
    *value = response.value().as_string();
  }
  return true;
}

bool EtcdClient::get_all_xservices(const std::string& key_prefix,
                                   std::vector<std::string>* values) {
  auto response = client_.ls(namespaced_key(key_prefix));
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key_prefix
               << " failed: " << response.error_message();
    return false;
  }

  if (values == nullptr) {
    return true;
  }

  values->clear();
  values->reserve(response.keys().size());
  for (int i = 0; i < response.keys().size(); ++i) {
    values->emplace_back(response.value(i).as_string());
  }

  return true;
}

void EtcdClient::unregister_instance(const std::string& key) {
  std::shared_ptr<etcd::KeepAlive> keep_alive;
  {
    std::lock_guard<std::mutex> lock(keep_alives_mutex_);
    auto it = keep_alives_.find(key);
    if (it == keep_alives_.end()) {
      return;
    }
    keep_alive = std::move(it->second);
    keep_alives_.erase(it);
  }
  keep_alive->Cancel();
  const auto response = client_.leaserevoke(keep_alive->Lease());
  if (!response.is_ok()) {
    // Renewal has stopped, so an unavailable server cannot keep the old
    // registration alive beyond its TTL.
    LOG(WARNING) << "Failed to revoke instance lease, key: " << key
                 << ", error: " << response.error_message();
  }
}

bool EtcdClient::register_instance(const std::string& key,
                                   const std::string& value,
                                   const int ttl) {
  if (ttl <= 0) {
    LOG(ERROR) << "Instance lease TTL must be positive, key: " << key;
    return false;
  }
  if (stopped_.load()) {
    return false;
  }
  const auto lease_response = client_.leasegrant(ttl);
  if (!lease_response.is_ok() || lease_response.value().lease() == 0) {
    LOG(ERROR) << "Failed to grant instance lease, key: " << key
               << ", error: " << lease_response.error_message();
    return false;
  }
  const int64_t lease_id = lease_response.value().lease();
  // KeepAlive's implicit grant discards grant errors and can leave lease ID 0,
  // which Put interprets as an unleased, persistent key.
  auto keep_alive = std::make_shared<etcd::KeepAlive>(client_, ttl, lease_id);
  auto response = client_.put(namespaced_key(key), value, lease_id);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd set " << key << " failed: " << response.error_message();
    keep_alive->Cancel();
    return false;
  }

  std::shared_ptr<etcd::KeepAlive> old_keep_alive;
  {
    std::lock_guard<std::mutex> lock(keep_alives_mutex_);
    if (stopped_.load()) {
      return false;
    }
    auto it = keep_alives_.find(key);
    if (it != keep_alives_.end()) {
      old_keep_alive = it->second;
    }
    keep_alives_[key] = keep_alive;
  }
  if (old_keep_alive != nullptr) {
    old_keep_alive->Cancel();
  }

  return true;
}

}  // namespace xllm
