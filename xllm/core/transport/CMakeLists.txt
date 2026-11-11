include(cc_library)

cc_library(
  NAME
    speculative_transport_adapter
  HDRS
    draft_input_proto.h
  DEPS
    :speculative
    :util
    torch
    proto::xllm_proto
)
