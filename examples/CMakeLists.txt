# Par2_Core Examples

# Minimal example demonstrating the API
add_executable(minimal_bind_ptrs minimal_bind_ptrs.cu)
target_link_libraries(minimal_bind_ptrs PRIVATE par2_core)
set_target_properties(minimal_bind_ptrs PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# M1 verification: RWPT semantic parity tests
add_executable(verify_rwpt_semantics verify_rwpt_semantics.cu)
target_link_libraries(verify_rwpt_semantics PRIVATE par2_core)
set_target_properties(verify_rwpt_semantics PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# M3 verification: Boundary conditions + tracking tests
add_executable(verify_boundary_conditions verify_boundary_conditions.cu)
target_link_libraries(verify_boundary_conditions PRIVATE par2_core)
set_target_properties(verify_boundary_conditions PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# P2: Export CSV and Statistics example
add_executable(export_csv_and_stats export_csv_and_stats.cu)
target_link_libraries(export_csv_and_stats PRIVATE par2_core)
set_target_properties(export_csv_and_stats PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# HPC: Pipeline integration with multi-stream coordination
add_executable(pipeline_integration pipeline_integration.cu)
target_link_libraries(pipeline_integration PRIVATE par2_core)
set_target_properties(pipeline_integration PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
