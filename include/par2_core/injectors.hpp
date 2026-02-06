/**
 * @file injectors.hpp
 * @brief Particle injection functions public API.
 *
 * These functions initialize particle positions on the GPU without
 * requiring host-device data transfers. The particles must already
 * be allocated on device.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_INJECTORS_HPP
#define PAR2_CORE_INJECTORS_HPP

#include "views.hpp"
#include <cuda_runtime.h>

namespace par2 {

/**
 * @brief Initialize particles uniformly distributed in a box.
 *
 * @tparam T Floating point type (float or double)
 *
 * @param particles Particle view (device pointers)
 * @param x0, y0, z0 Minimum corner of box
 * @param x1, y1, z1 Maximum corner of box
 * @param seed Random seed for reproducibility
 * @param stream CUDA stream (nullptr for default)
 *
 * Particles are distributed uniformly within the box using
 * a deterministic algorithm based on particle index and seed.
 *
 * @note This is the standalone function version. TransportEngine::inject_box
 *       provides the same functionality through the engine API.
 */
template <typename T>
void inject_box(
    const ParticlesView<T>& particles,
    T x0, T y0, T z0,
    T x1, T y1, T z1,
    unsigned long long seed = 12345ULL,
    cudaStream_t stream = nullptr
);

/**
 * @brief Initialize particles at regular grid positions within a box.
 *
 * @tparam T Floating point type
 *
 * @param particles Particle view
 * @param x0, y0, z0 Minimum corner
 * @param x1, y1, z1 Maximum corner
 * @param nx, ny, nz Number of particles per axis
 * @param stream CUDA stream
 *
 * Creates a regular grid of particles. Total particles = nx * ny * nz.
 * If particles.n != nx*ny*nz, only min(particles.n, nx*ny*nz) are set.
 *
 * @note Useful for debugging and visualization.
 */
template <typename T>
void inject_grid(
    const ParticlesView<T>& particles,
    T x0, T y0, T z0,
    T x1, T y1, T z1,
    int nx, int ny, int nz,
    cudaStream_t stream = nullptr
);

// Explicit instantiation declarations
extern template void inject_box<float>(
    const ParticlesView<float>&, float, float, float, float, float, float,
    unsigned long long, cudaStream_t);
extern template void inject_box<double>(
    const ParticlesView<double>&, double, double, double, double, double, double,
    unsigned long long, cudaStream_t);

} // namespace par2

#endif // PAR2_CORE_INJECTORS_HPP
