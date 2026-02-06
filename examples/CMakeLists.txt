# Par2_Core Examples
#
# Two examples showing how to embed TransportEngine in an HPC pipeline:
#   1. pipeline_example       — time-dependent velocity (streams + events)
#   2. steady_flow_example    — constant velocity (simplest hot loop)

add_executable(par2_pipeline_example pipeline_example.cu)
target_link_libraries(par2_pipeline_example PRIVATE par2_core)
set_target_properties(par2_pipeline_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(par2_steady_flow_example steady_flow_example.cu)
target_link_libraries(par2_steady_flow_example PRIVATE par2_core)
set_target_properties(par2_steady_flow_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
