cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(par2_core VERSION 0.1.0 LANGUAGES CXX CUDA)

# =============================================================================
# Build options
# =============================================================================
option(PAR2_BUILD_EXAMPLES "Build examples" ON)
option(PAR2_BUILD_TESTS "Build tests" OFF)
option(PAR2_BUILD_LEGACY_RUNNER "Build legacy runner tool (requires yaml-cpp)" OFF)
option(PAR2_BUILD_ORIGINAL "Build original PAR2 executable for comparison" OFF)

# =============================================================================
# C++ and CUDA standards
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default CUDA architectures if not specified (75+ for CUDA 12.x)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

# clangd compatibility: avoid CUDA response files
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)

# =============================================================================
# Library target: par2_core
# =============================================================================
add_library(par2_core STATIC
    src/transport_engine.cu
    src/injectors.cu
    src/stats.cu
    src/io.cu
    src/kernels/move_particles.cu
    src/kernels/cornerfield.cu
    src/kernels/drift_correction.cu
)

target_include_directories(par2_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# CUDA-specific settings
set_target_properties(par2_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Compiler warnings
target_compile_options(par2_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Debug/Release flags
target_compile_options(par2_core PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math>
)

# =============================================================================
# Examples
# =============================================================================
if(PAR2_BUILD_EXAMPLES)
    add_subdirectory(examples EXCLUDE_FROM_ALL)
endif()

# =============================================================================
# Tests
# =============================================================================
if(PAR2_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)

    # M4 Pipeline tests
    add_executable(test_m4_pipeline tests/test_m4_pipeline.cu)
    target_link_libraries(test_m4_pipeline PRIVATE par2_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_m4_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(test_m4_pipeline PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME M4_Pipeline COMMAND test_m4_pipeline)

    # M5 Data Access tests
    add_executable(test_m5_dataaccess tests/test_m5_dataaccess.cu)
    target_link_libraries(test_m5_dataaccess PRIVATE par2_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_m5_dataaccess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(test_m5_dataaccess PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME M5_DataAccess COMMAND test_m5_dataaccess)

    # P2 Stats tests
    add_executable(test_stats tests/test_stats.cu)
    target_link_libraries(test_stats PRIVATE par2_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_stats PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(test_stats PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME Stats COMMAND test_stats)
endif()

# =============================================================================
# Legacy Runner Tool (optional)
# =============================================================================
if(PAR2_BUILD_LEGACY_RUNNER)
    add_subdirectory(tools/legacy_runner)
endif()

# =============================================================================
# Original PAR2 Executable (for comparison testing)
# =============================================================================
if(PAR2_BUILD_ORIGINAL)
    # Find CUDA toolkit for curand
    find_package(CUDAToolkit REQUIRED)

    # Use bundled yaml-cpp from legacy_runner if available, otherwise add it
    if(NOT TARGET yaml-cpp)
        set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
        set(YAML_CPP_FORMAT_SOURCE OFF CACHE BOOL "" FORCE)
        add_subdirectory(tools/legacy_runner/external/yaml-cpp-0.8.0 yaml-cpp-original EXCLUDE_FROM_ALL)
    endif()

    # Original PAR2 executable
    # Note: PParticles.cu is included by PParticles.cuh (template implementation)
    add_executable(par2_original
        legacy/maingpu.cu
    )

    target_include_directories(par2_original PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/legacy
    )

    # Define PAR2_FLOAT (use double to match legacy runner for comparison)
    target_compile_definitions(par2_original PRIVATE
        PAR2_FLOAT=double
    )

    target_link_libraries(par2_original PRIVATE
        yaml-cpp
        CUDA::cudart
        CUDA::curand
    )

    set_target_properties(par2_original PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_options(par2_original PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math>
    )
endif()

# =============================================================================
# M6 Comparison Tests (original vs core)
# =============================================================================
if(PAR2_BUILD_ORIGINAL AND PAR2_BUILD_LEGACY_RUNNER)
    enable_testing()

    # Find Python for comparison script
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    # Test Example01-basic (self-contained, always works)
    add_test(
        NAME par2_compare_example01
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tools/compare_original_vs_core.py
            --yaml ${CMAKE_SOURCE_DIR}/legacy/Examples/Example01-basic/example.yaml
            --par2_original $<TARGET_FILE:par2_original>
            --par2_core $<TARGET_FILE:par2core_legacy_runner>
            --seed 12345
            --stat-rtol 0.05
    )

    # Note: Example02-flopy requires MODFLOW files generated by FloPy.
    # It cannot be run standalone without first generating tmp/model.ftl
    # via the FlopyPar2.ipynb notebook. Skipping for automated testing.
endif()

# =============================================================================
# Installation
# =============================================================================
include(GNUInstallDirs)

install(TARGETS par2_core
    EXPORT par2_core-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/par2_core
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT par2_core-targets
    FILE par2_core-targets.cmake
    NAMESPACE par2::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/par2_core
)
