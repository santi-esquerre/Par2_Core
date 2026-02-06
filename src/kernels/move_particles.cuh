/**
 * @file move_particles.cuh
 * @brief Internal header for particle movement kernel.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_INTERNAL_KERNELS_MOVE_PARTICLES_CUH
#define PAR2_INTERNAL_KERNELS_MOVE_PARTICLES_CUH

#include <par2_core/grid.hpp>
#include <par2_core/types.hpp>
#include <par2_core/views.hpp>
#include <par2_core/boundary.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace par2 {
namespace kernels {

/**
 * @brief Initialize RNG states for particles.
 */
__global__ void init_rng_states(
    curandState_t* states,
    int n,
    unsigned long long seed
);

/**
 * @brief Launch the particle movement kernel.
 *
 * This is the host-side wrapper that configures and launches the kernel.
 */
template <typename T>
void launch_move_particles(
    const GridDesc<T>& grid,
    const TransportParams<T>& params,
    const BoundaryConfig<T>& boundary,
    const EngineConfig& config,
    T dt,
    const VelocityView<T>& velocity,
    const CornerVelocityView<T>& corner_velocity,
    const ParticlesView<T>& particles,
    const DriftCorrectionView<T>& drift_correction,
    curandState_t* rng_states,
    int num_blocks,
    int block_size,
    cudaStream_t stream
);

/**
 * @brief Launch the box injection kernel.
 */
template <typename T>
void launch_inject_box(
    T* x, T* y, T* z,
    int n,
    T x0, T y0, T z0,
    T x1, T y1, T z1,
    unsigned long long seed,
    int offset,
    int num_blocks,
    int block_size,
    cudaStream_t stream
);

} // namespace kernels
} // namespace par2

#endif // PAR2_INTERNAL_KERNELS_MOVE_PARTICLES_CUH
