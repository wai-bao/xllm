include(cc_library)

cc_library(
  NAME
    speculative_transport_adapter
  HDRS
    mtp_draft_seed_proto.h
    pd_speculative_handoff.h
  SRCS
    pd_speculative_handoff.cpp
  DEPS
    :speculative
    :util
    torch
    proto::xllm_proto
)
