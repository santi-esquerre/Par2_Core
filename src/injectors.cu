/**
 * @file injectors.cu
 * @brief GPU kernels for particle injection/initialization.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#include <par2_core/injectors.hpp>
#include "internal/cuda_check.cuh"

#include <curand_kernel.h>

namespace par2 {

// =============================================================================
// Kernels
// =============================================================================

namespace {

template <typename T>
__global__ void inject_box_kernel(
    T* __restrict__ x,
    T* __restrict__ y,
    T* __restrict__ z,
    int n,
    T x0, T y0, T z0,
    T x1, T y1, T z1,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Initialize local RNG with unique sequence per particle
    curandState_t state;
    curand_init(seed, tid, 0, &state);

    // Generate uniform random positions in box
    T u = curand_uniform_double(&state);
    T v = curand_uniform_double(&state);
    T w = curand_uniform_double(&state);

    x[tid] = x0 + u * (x1 - x0);
    y[tid] = y0 + v * (y1 - y0);
    z[tid] = z0 + w * (z1 - z0);
}

template <typename T>
__global__ void inject_grid_kernel(
    T* __restrict__ x,
    T* __restrict__ y,
    T* __restrict__ z,
    int n,
    T x0, T y0, T z0,
    T x1, T y1, T z1,
    int nx, int ny, int nz
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Convert linear index to 3D grid position
    int total_grid = nx * ny * nz;
    if (tid >= total_grid) return;

    int iz = tid / (nx * ny);
    int iy = (tid / nx) % ny;
    int ix = tid % nx;

    // Compute normalized positions [0,1]
    T tx = (nx > 1) ? T(ix) / T(nx - 1) : T(0.5);
    T ty = (ny > 1) ? T(iy) / T(ny - 1) : T(0.5);
    T tz = (nz > 1) ? T(iz) / T(nz - 1) : T(0.5);

    // Map to box
    x[tid] = x0 + tx * (x1 - x0);
    y[tid] = y0 + ty * (y1 - y0);
    z[tid] = z0 + tz * (z1 - z0);
}

} // anonymous namespace

// =============================================================================
// Public API Implementation
// =============================================================================

template <typename T>
void inject_box(
    const ParticlesView<T>& particles,
    T x0, T y0, T z0,
    T x1, T y1, T z1,
    unsigned long long seed,
    cudaStream_t stream
) {
    if (!particles.valid() || particles.n <= 0) return;

    const int block_size = 256;
    const int num_blocks = (particles.n + block_size - 1) / block_size;

    inject_box_kernel<<<num_blocks, block_size, 0, stream>>>(
        particles.x, particles.y, particles.z,
        particles.n,
        x0, y0, z0, x1, y1, z1,
        seed
    );

    PAR2_CUDA_CHECK_LAST();
}

template <typename T>
void inject_grid(
    const ParticlesView<T>& particles,
    T x0, T y0, T z0,
    T x1, T y1, T z1,
    int nx, int ny, int nz,
    cudaStream_t stream
) {
    if (!particles.valid() || particles.n <= 0) return;

    const int block_size = 256;
    const int num_blocks = (particles.n + block_size - 1) / block_size;

    inject_grid_kernel<<<num_blocks, block_size, 0, stream>>>(
        particles.x, particles.y, particles.z,
        particles.n,
        x0, y0, z0, x1, y1, z1,
        nx, ny, nz
    );

    PAR2_CUDA_CHECK_LAST();
}

// Explicit instantiations
template void inject_box<float>(
    const ParticlesView<float>&, float, float, float, float, float, float,
    unsigned long long, cudaStream_t);
template void inject_box<double>(
    const ParticlesView<double>&, double, double, double, double, double, double,
    unsigned long long, cudaStream_t);

template void inject_grid<float>(
    const ParticlesView<float>&, float, float, float, float, float, float,
    int, int, int, cudaStream_t);
template void inject_grid<double>(
    const ParticlesView<double>&, double, double, double, double, double, double,
    int, int, int, cudaStream_t);

} // namespace par2
