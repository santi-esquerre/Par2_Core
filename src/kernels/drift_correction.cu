/**
 * @file drift_correction.cu
 * @brief GPU kernel for computing precomputed drift correction (div(D)) field.
 *
 * SOURCE OF TRUTH: legacy/Geometry/CellField.cuh, computeDriftCorrection()
 *
 * The drift correction compensates for the spatial variation of the
 * dispersion tensor, ensuring correct Fokker-Planck behavior:
 *
 *   drift_x = ∂D_xx/∂x + ∂D_xy/∂y + ∂D_xz/∂z
 *   drift_y = ∂D_xy/∂x + ∂D_yy/∂y + ∂D_yz/∂z
 *   drift_z = ∂D_xz/∂x + ∂D_yz/∂y + ∂D_zz/∂z
 *
 * The precomputed mode computes drift at cell centers using finite differences.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#include "drift_correction.cuh"
#include "../internal/cuda_check.cuh"
#include "../internal/fields/facefield_accessor.cuh"
#include "../internal/math/dispersion.cuh"

#include <cmath>

namespace par2 {
namespace kernels {

// =============================================================================
// Compute D Tensor at Cell Centers (Step 1)
// =============================================================================

/**
 * @brief Compute dispersion tensor D at cell centers.
 *
 * SOURCE: legacy/Geometry/CellField.cuh, lines 91-116
 *
 * @param nan_prevention If true, apply tolerance check when |v| < toll (safe).
 *                       If false, match legacy behavior exactly (may produce NaN if |v|=0).
 */
template <typename T>
__global__ void compute_D_tensor_kernel(
    const GridDesc<T> grid,
    const T* __restrict__ U,
    const T* __restrict__ V,
    const T* __restrict__ W,
    T Dm, T alphaL, T alphaT,
    bool nan_prevention,
    T* __restrict__ D11,
    T* __restrict__ D22,
    T* __restrict__ D33,
    T* __restrict__ D12,
    T* __restrict__ D13,
    T* __restrict__ D23
) {
    int num_cells = grid.num_cells();

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < num_cells;
         tid += blockDim.x * gridDim.x)
    {
        int idz = tid / (grid.ny * grid.nx);
        int rem = tid % (grid.ny * grid.nx);
        int idy = rem / grid.nx;
        int idx = rem % grid.nx;

        T cx = grid.px + (T(idx) + T(0.5)) * grid.dx;
        T cy = grid.py + (T(idy) + T(0.5)) * grid.dy;
        T cz = grid.pz + (T(idz) + T(0.5)) * grid.dz;

        T vx, vy, vz;
        internal::sample_velocity_facefield(U, V, W, grid, idx, idy, idz, true,
                                            cx, cy, cz, vx, vy, vz);

        T vnorm = sqrt(vx*vx + vy*vy + vz*vz);

        // NaN-prevention guard (NOT in legacy - only when nan_prevention=true)
        if (nan_prevention) {
            const T toll = internal::compute_velocity_tolerance(Dm, alphaL);
            if (vnorm < toll) {
                D11[tid] = Dm; D22[tid] = Dm; D33[tid] = Dm;
                D12[tid] = T(0); D13[tid] = T(0); D23[tid] = T(0);
                continue;
            }
        }

        T isotropic = alphaT * vnorm + Dm;
        T aniso_factor = (alphaL - alphaT) / vnorm;

        D11[tid] = isotropic + aniso_factor * vx * vx;
        D22[tid] = isotropic + aniso_factor * vy * vy;
        D33[tid] = isotropic + aniso_factor * vz * vz;
        D12[tid] = aniso_factor * vx * vy;
        D13[tid] = aniso_factor * vx * vz;
        D23[tid] = aniso_factor * vy * vz;
    }
}

// =============================================================================
// Compute Drift from D Tensor (Step 2) - Finite Differences
// =============================================================================

/**
 * @brief Compute drift correction from D tensor using finite differences.
 *
 * SOURCE: legacy/Geometry/CellField.cuh, lines 121-229
 */
template <typename T>
__global__ void compute_drift_from_D_kernel(
    const GridDesc<T> grid,
    const T* __restrict__ D11,
    const T* __restrict__ D22,
    const T* __restrict__ D33,
    const T* __restrict__ D12,
    const T* __restrict__ D13,
    const T* __restrict__ D23,
    T* __restrict__ drift_x,
    T* __restrict__ drift_y,
    T* __restrict__ drift_z
) {
    int num_cells = grid.num_cells();

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < num_cells;
         tid += blockDim.x * gridDim.x)
    {
        int idz = tid / (grid.ny * grid.nx);
        int rem = tid % (grid.ny * grid.nx);
        int idy = rem / grid.nx;
        int idx = rem % grid.nx;

        auto cell_id = [&](int ix, int iy, int iz) {
            return iz * grid.ny * grid.nx + iy * grid.nx + ix;
        };

        // X derivatives
        T ddx;
        int idx1, idx2;
        if (idx == 0) { ddx = grid.dx; idx1 = idx; idx2 = idx + 1; }
        else if (idx == grid.nx - 1) { ddx = grid.dx; idx1 = idx - 1; idx2 = idx; }
        else { ddx = T(2) * grid.dx; idx1 = idx - 1; idx2 = idx + 1; }

        int id1_x = cell_id(idx1, idy, idz);
        int id2_x = cell_id(idx2, idy, idz);
        T dD11x = (D11[id2_x] - D11[id1_x]) / ddx;
        T dD12x = (D12[id2_x] - D12[id1_x]) / ddx;
        T dD13x = (D13[id2_x] - D13[id1_x]) / ddx;

        // Y derivatives
        T ddy;
        int idy1, idy2;
        if (idy == 0) { ddy = grid.dy; idy1 = idy; idy2 = idy + 1; }
        else if (idy == grid.ny - 1) { ddy = grid.dy; idy1 = idy - 1; idy2 = idy; }
        else { ddy = T(2) * grid.dy; idy1 = idy - 1; idy2 = idy + 1; }

        int id1_y = cell_id(idx, idy1, idz);
        int id2_y = cell_id(idx, idy2, idz);
        T dD12y = (D12[id2_y] - D12[id1_y]) / ddy;
        T dD22y = (D22[id2_y] - D22[id1_y]) / ddy;
        T dD23y = (D23[id2_y] - D23[id1_y]) / ddy;

        // Z derivatives (only if 3D)
        T dD13z = T(0), dD23z = T(0), dD33z = T(0);
        if (grid.nz > 1) {
            T ddz;
            int idz1, idz2;
            if (idz == 0) { ddz = grid.dz; idz1 = idz; idz2 = idz + 1; }
            else if (idz == grid.nz - 1) { ddz = grid.dz; idz1 = idz - 1; idz2 = idz; }
            else { ddz = T(2) * grid.dz; idz1 = idz - 1; idz2 = idz + 1; }

            int id1_z = cell_id(idx, idy, idz1);
            int id2_z = cell_id(idx, idy, idz2);
            dD13z = (D13[id2_z] - D13[id1_z]) / ddz;
            dD23z = (D23[id2_z] - D23[id1_z]) / ddz;
            dD33z = (D33[id2_z] - D33[id1_z]) / ddz;
        }

        drift_x[tid] = dD11x + dD12y + dD13z;
        drift_y[tid] = dD12x + dD22y + dD23z;
        drift_z[tid] = dD13x + dD23y + dD33z;

        if (grid.nz == 1) drift_z[tid] = T(0);
    }
}

// =============================================================================
// Launch Functions
// =============================================================================

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
) {
    compute_D_tensor_kernel<<<num_blocks, block_size, 0, stream>>>(
        grid, velocity.U, velocity.V, velocity.W,
        params.molecular_diffusion, params.alpha_l, params.alpha_t,
        nan_prevention,
        temp_D11, temp_D22, temp_D33, temp_D12, temp_D13, temp_D23
    );

    compute_drift_from_D_kernel<<<num_blocks, block_size, 0, stream>>>(
        grid, temp_D11, temp_D22, temp_D33, temp_D12, temp_D13, temp_D23,
        drift_x, drift_y, drift_z
    );
}

template <typename T>
void launch_compute_drift_correction(
    const GridDesc<T>&, const CornerVelocityView<T>&, const TransportParams<T>&,
    T*, T*, T*, int, int, cudaStream_t
) {
    assert(false && "Use launch_compute_drift_precomputed instead");
}

// Explicit instantiations
template void launch_compute_drift_precomputed<float>(
    const GridDesc<float>&, const VelocityView<float>&, const TransportParams<float>&,
    bool,
    float*, float*, float*, float*, float*, float*, float*, float*, float*,
    int, int, cudaStream_t);
template void launch_compute_drift_precomputed<double>(
    const GridDesc<double>&, const VelocityView<double>&, const TransportParams<double>&,
    bool,
    double*, double*, double*, double*, double*, double*, double*, double*, double*,
    int, int, cudaStream_t);

template void launch_compute_drift_correction<float>(
    const GridDesc<float>&, const CornerVelocityView<float>&, const TransportParams<float>&,
    float*, float*, float*, int, int, cudaStream_t);
template void launch_compute_drift_correction<double>(
    const GridDesc<double>&, const CornerVelocityView<double>&, const TransportParams<double>&,
    double*, double*, double*, int, int, cudaStream_t);

} // namespace kernels
} // namespace par2
