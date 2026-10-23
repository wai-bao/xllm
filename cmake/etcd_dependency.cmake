# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build a versioned source copy so dependency fixes do not modify the submodule
# or require a gitlink pointing at an unpublished commit.
set(_etcd_source "${CMAKE_SOURCE_DIR}/third_party/etcd_cpp_apiv3")
set(_etcd_patch "${CMAKE_CURRENT_LIST_DIR}/patches/etcd-cancellable-requests.patch")
file(GLOB_RECURSE _etcd_files CONFIGURE_DEPENDS
  LIST_DIRECTORIES false RELATIVE "${_etcd_source}" "${_etcd_source}/*")
list(FILTER _etcd_files EXCLUDE REGEX "(^|/)\\.git($|/)")
list(SORT _etcd_files)
file(SHA256 "${_etcd_patch}" _etcd_fingerprint)
foreach(_file IN LISTS _etcd_files)
  file(SHA256 "${_etcd_source}/${_file}" _digest)
  string(APPEND _etcd_fingerprint "${_file}:${_digest}\n")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_etcd_source}/${_file}")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_etcd_patch}")
string(SHA256 _etcd_fingerprint "${_etcd_fingerprint}")
set(XLLM_ETCD_SOURCE_DIR
  "${CMAKE_BINARY_DIR}/third_party/etcd_cpp_apiv3-source/${_etcd_fingerprint}")
if(NOT EXISTS "${XLLM_ETCD_SOURCE_DIR}/.xllm-patched")
  foreach(_file IN LISTS _etcd_files)
    configure_file("${_etcd_source}/${_file}"
      "${XLLM_ETCD_SOURCE_DIR}/${_file}" COPYONLY)
  endforeach()
  find_package(Git REQUIRED)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}"
      "${GIT_EXECUTABLE}" apply "${_etcd_patch}"
    WORKING_DIRECTORY "${XLLM_ETCD_SOURCE_DIR}"
    RESULT_VARIABLE _etcd_patch_result
    ERROR_VARIABLE _etcd_patch_error)
  if(NOT _etcd_patch_result EQUAL 0)
    message(FATAL_ERROR "Cannot apply etcd cancellation fix: ${_etcd_patch_error}")
  endif()
  file(WRITE "${XLLM_ETCD_SOURCE_DIR}/.xllm-patched" "${_etcd_fingerprint}\n")
  file(GLOB _etcd_stale_sources
       LIST_DIRECTORIES true
       "${CMAKE_BINARY_DIR}/third_party/etcd_cpp_apiv3-source/*")
  list(REMOVE_ITEM _etcd_stale_sources "${XLLM_ETCD_SOURCE_DIR}")
  foreach(_stale IN LISTS _etcd_stale_sources)
    file(REMOVE_RECURSE "${_stale}")
  endforeach()
endif()

# Stable public include paths avoid invalidating every xLLM compiler command
# when only the dependency implementation changes.
set(XLLM_ETCD_INCLUDE_DIR
  "${CMAKE_BINARY_DIR}/third_party/etcd_cpp_apiv3-include")
foreach(_file IN LISTS _etcd_files)
  if(_file MATCHES "^etcd/.*\\.hpp$")
    configure_file("${XLLM_ETCD_SOURCE_DIR}/${_file}"
      "${XLLM_ETCD_INCLUDE_DIR}/${_file}" COPYONLY)
  endif()
endforeach()
