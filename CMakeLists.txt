cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(par2_core VERSION 0.1.0 LANGUAGES CXX CUDA)

# =============================================================================
# Build options
# =============================================================================

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
# Install targets
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
