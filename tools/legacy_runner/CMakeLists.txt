# =============================================================================
# par2core_legacy_runner - Legacy PAR² compatibility tool
# =============================================================================
#
# Reads legacy YAML configuration and MODFLOW FTL files, runs Par2_Core,
# and produces compatible output for regression testing against original PAR².
#
# Dependencies:
#   - yaml-cpp (bundled in external/yaml-cpp-0.8.0)
#
# Build:
#   cmake -DPAR2_BUILD_LEGACY_RUNNER=ON ..
#   make par2core_legacy_runner
#
# =============================================================================

# Use bundled yaml-cpp
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_FORMAT_SOURCE OFF CACHE BOOL "" FORCE)
add_subdirectory(external/yaml-cpp-0.8.0 yaml-cpp EXCLUDE_FROM_ALL)

# Legacy runner executable
add_executable(par2core_legacy_runner
    main.cu
    legacy_config.cpp
)

target_include_directories(par2core_legacy_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(par2core_legacy_runner PRIVATE
    par2_core
    yaml-cpp
)

# CUDA settings
set_target_properties(par2core_legacy_runner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(par2core_legacy_runner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
