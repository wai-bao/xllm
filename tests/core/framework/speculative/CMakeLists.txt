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
    spec_metrics_test
  SRCS
    spec_metrics_test.cpp
  DEPS
    :speculative
    :sampler
    GTest::gtest_main
)

