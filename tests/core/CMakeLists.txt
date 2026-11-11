# Shared NPU-side link set for tests that host torch in-process
# (Python::Python) and drive NPU ops directly.
set(XLLM_SPEC_TEST_NPU_LIBS
    Python::Python
    $<$<BOOL:${USE_NPU}>:ascendcl>
    $<$<BOOL:${USE_NPU}>:hccl>
    $<$<BOOL:${USE_NPU}>:c_sec>
    $<$<BOOL:${USE_NPU}>:nnopbase>)

add_subdirectory(common)
add_subdirectory(distributed_runtime)
add_subdirectory(framework)
add_subdirectory(kv_cache)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(platform)
add_subdirectory(runtime)
add_subdirectory(scheduler)
add_subdirectory(util)
