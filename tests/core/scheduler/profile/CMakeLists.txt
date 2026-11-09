include(cc_test)

cc_test(
  NAME
    profile_graph_warmup_test
  SRCS
    profile_graph_warmup_test.cpp
  DEPS
    :xllm_server
    :block
    :profile
    :platform
    :request
    models
    GTest::gtest_main
)
