/**
 * @file cuda_check.cuh
 * @brief CUDA error checking macros for Par2_Core.
 *
 * These macros check CUDA errors WITHOUT calling cudaDeviceSynchronize().
 * They are suitable for use in the hot path.
 *
 * In release builds (PAR2_DEBUG_LEVEL=0), these macros expand to no-ops
 * for maximum performance.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_INTERNAL_CUDA_CHECK_CUH
#define PAR2_INTERNAL_CUDA_CHECK_CUH

#include <par2_core/debug_policy.hpp>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace par2 {
namespace detail {

/**
 * @brief Check a CUDA error and throw if not success.
 *
 * @note Does NOT synchronize - only checks the error code.
 */
inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::string msg = "CUDA error at ";
        msg += file;
        msg += ":";
        msg += std::to_string(line);
        msg += ": ";
        msg += cudaGetErrorString(err);
        throw std::runtime_error(msg);
    }
}

/**
 * @brief Check last CUDA error (e.g., after kernel launch).
 *
 * @note Does NOT synchronize - only checks cudaPeekAtLastError().
 */
inline void cuda_check_last(const char* file, int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::string msg = "CUDA kernel error at ";
        msg += file;
        msg += ":";
        msg += std::to_string(line);
        msg += ": ";
        msg += cudaGetErrorString(err);
        throw std::runtime_error(msg);
    }
}

} // namespace detail
} // namespace par2

/**
 * @brief Check CUDA API call result (always enabled, needed for correctness).
 *
 * Usage: PAR2_CUDA_CHECK(cudaMalloc(&ptr, size));
 *
 * @note This is always enabled because allocation failures must be caught.
 */
#define PAR2_CUDA_CHECK(call) \
    ::par2::detail::cuda_check((call), __FILE__, __LINE__)

/**
 * @brief Check for kernel launch errors (debug builds only).
 *
 * Usage: kernel<<<...>>>(...); PAR2_CUDA_CHECK_LAST();
 *
 * @note This only catches launch configuration errors, not runtime errors.
 *       Runtime errors are detected on the next synchronization.
 * @note In release builds (PAR2_ENABLE_CUDA_CHECKS=0), this is a no-op.
 */
#if PAR2_ENABLE_CUDA_CHECKS
#  define PAR2_CUDA_CHECK_LAST() \
      ::par2::detail::cuda_check_last(__FILE__, __LINE__)
#else
#  define PAR2_CUDA_CHECK_LAST() ((void)0)
#endif

#endif // PAR2_INTERNAL_CUDA_CHECK_CUH
