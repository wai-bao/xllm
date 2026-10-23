include(cc_test)

add_subdirectory(adaptive)

cc_test(
  NAME
    draft_config_test
  SRCS
    draft_config_test.cpp
  DEPS
    :speculative
    GTest::gtest_main
)

cc_test(
  NAME
    mtp_json_object_state_test
  NO_NPU_RUNTIME
  SRCS
    mtp_json_object_state_test.cpp
  DEPS
    :speculative
    :json_object_grammar
    GTest::gtest_main
)



cc_test(
  NAME
    vocab_map_test
  NO_NPU_RUNTIME

  SRCS
    vocab_map_test.cpp
  DEPS
    :speculative
    :sampler
    torch
    GTest::gtest_main
)
target_link_libraries(vocab_map_test
                      PUBLIC
                      $<$<BOOL:${USE_NPU}>:torch_npu>
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)

cc_test(
  NAME
    mtp_draft_input_sampling_test
  NO_NPU_RUNTIME
  SRCS
    mtp_draft_input_sampling_test.cpp
  DEPS
    :speculative
    :runtime
    :json_object_grammar
    :sampler
    :platform
    torch
    GTest::gtest_main
)
target_link_libraries(mtp_draft_input_sampling_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)

cc_test(
  NAME
    mtp_request_state_lifecycle_test
  NO_NPU_RUNTIME
  SRCS
    mtp_request_state_lifecycle_test.cpp
  DEPS
    :speculative
    :platform
    GTest::gtest_main
)
target_link_libraries(mtp_request_state_lifecycle_test
                      PUBLIC
                      $<$<BOOL:${USE_NPU}>:torch_npu>
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)

cc_test(
  NAME
    spec_metrics_test
  SRCS
    spec_metrics_test.cpp
  DEPS
    :speculative
    :sampler
    GTest::gtest_main
)
