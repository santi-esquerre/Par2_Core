/**
 * @file unwrap_positions.cuh
 * @brief Kernel for computing unwrapped (continuous) positions.
 *
 * For periodic BC, particles wrap around the domain. This kernel computes
 * the continuous (unwrapped) positions using the wrap counters:
 *
 *   x_u = x + wrapX * Lx
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_KERNELS_UNWRAP_POSITIONS_CUH
#define PAR2_KERNELS_UNWRAP_POSITIONS_CUH

#include <par2_core/grid.hpp>
#include <par2_core/views.hpp>
#include <par2_core/types.hpp>
#include <cuda_runtime.h>

namespace par2 {
namespace kernels {

/**
 * @brief Kernel to compute unwrapped positions from wrapped + wrapCount.
 *
 * For each particle i (if active):
 *   x_u[i] = x[i] + wrapX[i] * Lx  (if wrapX != nullptr)
 *   x_u[i] = x[i]                  (if wrapX == nullptr)
 *
 * Inactive/exited particles are still processed (their position is copied).
 */
template <typename T>
__global__ void unwrap_positions_kernel(
    const T* __restrict__ x,
    const T* __restrict__ y,
    const T* __restrict__ z,
    const int32_t* __restrict__ wrapX,  // May be nullptr
    const int32_t* __restrict__ wrapY,  // May be nullptr
    const int32_t* __restrict__ wrapZ,  // May be nullptr
    T Lx, T Ly, T Lz,  // Domain sizes
    T* __restrict__ x_u,
    T* __restrict__ y_u,
    T* __restrict__ z_u,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // X component
    if (wrapX != nullptr) {
        x_u[i] = x[i] + static_cast<T>(wrapX[i]) * Lx;
    } else {
        x_u[i] = x[i];
    }

    // Y component
    if (wrapY != nullptr) {
        y_u[i] = y[i] + static_cast<T>(wrapY[i]) * Ly;
    } else {
        y_u[i] = y[i];
    }

    // Z component
    if (wrapZ != nullptr) {
        z_u[i] = z[i] + static_cast<T>(wrapZ[i]) * Lz;
    } else {
        z_u[i] = z[i];
    }
}

/**
 * @brief Launch wrapper for unwrap_positions_kernel.
 */
template <typename T>
inline void launch_unwrap_positions(
    const GridDesc<T>& grid,
    const ConstParticlesView<T>& particles,
    T* x_u, T* y_u, T* z_u,
    int num_blocks,
    int block_size,
    cudaStream_t stream
) {
    // Domain sizes
    const T Lx = grid.dx * grid.nx;
    const T Ly = grid.dy * grid.ny;
    const T Lz = grid.dz * grid.nz;

    unwrap_positions_kernel<<<num_blocks, block_size, 0, stream>>>(
        particles.x, particles.y, particles.z,
        particles.wrapX, particles.wrapY, particles.wrapZ,
        Lx, Ly, Lz,
        x_u, y_u, z_u,
        particles.n
    );
}

} // namespace kernels
} // namespace par2

#endif // PAR2_KERNELS_UNWRAP_POSITIONS_CUH
