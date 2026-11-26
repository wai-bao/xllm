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

#include <sys/mman.h>
#include <unistd.h>

#include <string>

#include "core/util/json_reader.h"

namespace xllm {

// Feed in-memory JSON through JsonReader's production file parser. A memfd
// avoids leaving fixture files behind while keeping text parsing test-only.
inline bool parse_json_text(JsonReader& reader, const std::string& text) {
  const int fd = memfd_create(/*name=*/"xllm-json-test", MFD_CLOEXEC);
  if (fd < 0) {
    return false;
  }
  size_t offset = 0;
  while (offset < text.size()) {
    const ssize_t written =
        write(fd, text.data() + offset, text.size() - offset);
    if (written <= 0) {
      close(fd);
      return false;
    }
    offset += static_cast<size_t>(written);
  }
  const bool parsed = reader.parse("/proc/self/fd/" + std::to_string(fd));
  close(fd);
  return parsed;
}

}  // namespace xllm
