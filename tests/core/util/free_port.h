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

#pragma once

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <mutex>
#include <unordered_set>

namespace xllm::net {

inline int get_local_free_port() {
  static std::mutex mutex;
  static std::unordered_set<int> allocated_ports;
  std::lock_guard<std::mutex> lock(mutex);
  int port = 0;
  do {
    sockaddr_in addr;
    const int fd = socket(AF_INET, SOCK_STREAM, /*protocol=*/0);
    if (fd < 0) {
      return -1;
    }
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
      close(fd);
      return -1;
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == -1) {
      close(fd);
      return -1;
    }
    port = ntohs(addr.sin_port);
    close(fd);
  } while (allocated_ports.contains(port));
  allocated_ports.insert(port);
  return port;
}

}  // namespace xllm::net
