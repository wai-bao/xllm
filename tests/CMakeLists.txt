if(NOT BUILD_TESTING)
  return()
endif()

if(USE_NPU)
  # Run against the package assembled by the build so its native extension is
  # available alongside the Python sources. Load tcmalloc at process startup
  # when present; loading it after torch_npu can exhaust static TLS slots.
  set(XLLM_PYTHON_IMPORT_ROOT
      "$<PATH:NORMAL_PATH,$<TARGET_FILE_DIR:xllm_export>/..>")
  find_library(XLLM_TEST_TCMALLOC_LIBRARY NAMES tcmalloc)
  set(XLLM_PYTHON_TEST_ENV
      "PYTHONPATH=${XLLM_PYTHON_IMPORT_ROOT}:${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}")
  if(XLLM_TEST_TCMALLOC_LIBRARY)
    list(APPEND XLLM_PYTHON_TEST_ENV
         "LD_PRELOAD=${XLLM_TEST_TCMALLOC_LIBRARY}:$ENV{LD_PRELOAD}")
  endif()
  add_test(
    NAME python_executor_test
    COMMAND
      "${CMAKE_COMMAND}" -E env
      ${XLLM_PYTHON_TEST_ENV}
      "${Python3_EXECUTABLE}" -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  set_tests_properties(
    python_executor_test
    PROPERTIES WORKING_DIRECTORY "${XLLM_PYTHON_IMPORT_ROOT}"
  )

  # Run separately with the source-tree PYTHONPATH so tests/python runtime
  # stubs and the build-tree environment cannot affect the probe.
  add_test(
    NAME python_hccl_decode_graph_test
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
      "${Python3_EXECUTABLE}" -m pytest -q
      "${CMAKE_CURRENT_SOURCE_DIR}/npu/test_hccl_decode_graph.py"
  )
  set_tests_properties(
    python_hccl_decode_graph_test
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      RUN_SERIAL TRUE
  )
endif()

add_subdirectory(api_service)
add_subdirectory(core)
add_subdirectory(function_call)
add_subdirectory(models)
