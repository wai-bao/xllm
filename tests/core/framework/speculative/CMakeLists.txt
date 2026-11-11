include(cc_test)

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
  CPU_ONLY
  SRCS
    mtp_json_object_state_test.cpp
  DEPS
    :speculative
    :json_object_grammar
    GTest::gtest_main
)

cc_test(
  NAME
    draft_extend_input_test
  CPU_ONLY
  SRCS
    draft_extend_input_test.cpp
  DEPS
    :speculative
    :platform
    GTest::gtest_main
)
target_link_libraries(draft_extend_input_test
                      PUBLIC ${XLLM_SPEC_TEST_NPU_LIBS})

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
                      PUBLIC ${XLLM_SPEC_TEST_NPU_LIBS})

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
                      PUBLIC $<$<BOOL:${USE_NPU}>:torch_npu>
                              ${XLLM_SPEC_TEST_NPU_LIBS})

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

