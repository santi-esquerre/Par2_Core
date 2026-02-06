# Par2_Core — Pipeline Integration Example
#
# Single example showing how to embed TransportEngine in an HPC pipeline:
#   bind → prepare → step loop → explicit synchronize for host I/O.

add_executable(par2_pipeline_example pipeline_example.cu)
target_link_libraries(par2_pipeline_example PRIVATE par2_core)
set_target_properties(par2_pipeline_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
