/**
 * @file cornerfield.cuh
 * @brief Internal header for cornerfield computation kernel.
 *
 * Corner velocities are computed by averaging adjacent face-centered velocities.
 * This is needed for trilinear interpolation and displacement matrix computation.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_INTERNAL_KERNELS_CORNERFIELD_CUH
#define PAR2_INTERNAL_KERNELS_CORNERFIELD_CUH

#include <par2_core/grid.hpp>
#include <par2_core/views.hpp>
#include <cuda_runtime.h>

namespace par2 {
namespace kernels {

/**
 * @brief Compute corner velocities from face velocities.
 *
 * @tparam T Floating point type
 *
 * @param grid Grid descriptor
 * @param face_vel Face-centered velocity field (input)
 * @param corner_vel Corner-centered velocity field (output, must be pre-allocated)
 * @param stream CUDA stream
 *
 * The corner velocity at each grid vertex is the average of the
 * adjacent face velocities. This matches the legacy PAR² algorithm.
 *
 * @note Output arrays must have size grid.num_corners() each.
 */
template <typename T>
void launch_compute_corner_velocities(
    const GridDesc<T>& grid,
    const VelocityView<T>& face_vel,
    T* corner_Uc,
    T* corner_Vc,
    T* corner_Wc,
    int num_blocks,
    int block_size,
    cudaStream_t stream
);

} // namespace kernels
} // namespace par2

#endif // PAR2_INTERNAL_KERNELS_CORNERFIELD_CUH
