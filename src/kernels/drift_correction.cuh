/**
 * @file drift_correction.cuh
 * @brief Internal header for drift correction kernel.
 *
 * The drift correction term div(D) compensates for the spatial variation
 * of the dispersion tensor in the Fokker-Planck equation:
 *
 *   drift_x = ∂D_xx/∂x + ∂D_xy/∂y + ∂D_xz/∂z
 *   drift_y = ∂D_xy/∂x + ∂D_yy/∂y + ∂D_yz/∂z  
 *   drift_z = ∂D_xz/∂x + ∂D_yz/∂y + ∂D_zz/∂z
 *
 * SOURCE OF TRUTH: legacy/Geometry/CellField.cuh, computeDriftCorrection()
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_INTERNAL_KERNELS_DRIFT_CORRECTION_CUH
#define PAR2_INTERNAL_KERNELS_DRIFT_CORRECTION_CUH

#include <par2_core/grid.hpp>
#include <par2_core/types.hpp>
#include <par2_core/views.hpp>
#include <cuda_runtime.h>

namespace par2 {
namespace kernels {

/**
 * @brief Compute precomputed drift correction using finite differences.
 *
 * This is the PREFERRED mode for production runs:
 *   1. Compute D tensor at cell centers (from face-centered velocities)
 *   2. Compute div(D) using finite differences
 *
 * Requires temporary buffers for D tensor components (6 arrays of num_cells).
 *
 * @param grid Grid descriptor
 * @param velocity Face-centered velocity field
 * @param params Transport parameters (Dm, alphaL, alphaT)
 * @param nan_prevention If true, apply zero-velocity tolerance (safe). If false, match legacy exactly.
 * @param drift_x, drift_y, drift_z Output drift correction components
 * @param temp_D11..temp_D23 Temporary buffers for D tensor (6 arrays)
 * @param num_blocks, block_size Kernel launch configuration
 * @param stream CUDA stream
 */
template <typename T>
void launch_compute_drift_precomputed(
    const GridDesc<T>& grid,
    const VelocityView<T>& velocity,
    const TransportParams<T>& params,
    bool nan_prevention,
    T* drift_x, T* drift_y, T* drift_z,
    T* temp_D11, T* temp_D22, T* temp_D33,
    T* temp_D12, T* temp_D13, T* temp_D23,
    int num_blocks, int block_size, cudaStream_t stream
);

/**
 * @brief [DEPRECATED] Original drift correction interface.
 *
 * This is a stub that asserts false - use launch_compute_drift_precomputed instead.
 */
template <typename T>
void launch_compute_drift_correction(
    const GridDesc<T>& grid,
    const CornerVelocityView<T>& corner_vel,
    const TransportParams<T>& params,
    T* dc_x,
    T* dc_y,
    T* dc_z,
    int num_blocks,
    int block_size,
    cudaStream_t stream
);

} // namespace kernels
} // namespace par2

#endif // PAR2_INTERNAL_KERNELS_DRIFT_CORRECTION_CUH
