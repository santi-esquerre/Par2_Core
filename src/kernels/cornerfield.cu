/**
 * @file cornerfield.cu
 * @brief GPU kernel for computing corner-field velocities from face-field.
 *
 * =============================================================================
 * M2 CONTRACT: CORNER VELOCITY FIELD
 * =============================================================================
 *
 * SOURCE OF TRUTH: legacy/Geometry/CornerField.cuh, computeCornerVelocities()
 *                  lines 507-634
 *
 * ## Purpose
 *
 * Corner velocities (Uc, Vc, Wc) are required for:
 *   - InterpolationMode::Trilinear (smooth trilinear velocity sampling)
 *   - DriftCorrectionMode::TrilinearOnFly (drift via cornerfield derivatives)
 *   - Displacement matrix B computation (uses corner velocity for eigenvectors)
 *
 * ## Inputs
 *
 *   - U, V, W: Face-centered velocity arrays (staggered MAC grid)
 *   - Size: (nx+1) * (ny+1) * (nz+1) per component
 *   - Layout: U[mergeId(ix,iy,iz)] = velocity at face perpendicular to X at (ix,iy,iz)
 *
 * ## Outputs
 *
 *   - Uc, Vc, Wc: Corner-centered velocity arrays
 *   - Size: (nx+1) * (ny+1) * (nz+1) per component (same as face arrays)
 *   - Layout: Uc[cornerIdx] = X-velocity at corner (ix,iy,iz)
 *
 * ## Algorithm (EXACT legacy semantics)
 *
 * For each corner (idx, idy, idz) where idx ∈ [0, nx], idy ∈ [0, ny], idz ∈ [0, nz]:
 *
 * ### X-component (Uc):
 *   - If idx < nx: use XM faces (face at idx position)
 *   - If idx == nx: use XP faces (face at idx-1 position, +1 offset)
 *   - Average over adjacent valid cells in Y and Z directions (2x2 stencil)
 *   - Uc[corner] = sum(faces) / count
 *
 * ### Y-component (Vc):
 *   - If idy < ny: use YM faces
 *   - If idy == ny: use YP faces
 *   - Average over adjacent valid cells in X and Z directions
 *
 * ### Z-component (Wc):
 *   - If idz < nz: use ZM faces
 *   - If idz == nz: use ZP faces
 *   - Average over adjacent valid cells in X and Y directions
 *
 * ## 2D Support (nz == 1)
 *
 * When nz == 1:
 *   - Z-component Wc is computed but typically zero
 *   - Only corners at idz=0 and idz=1 exist
 *   - validId check with idz-1 will be false for idz=0
 *
 * ## Face Field Indexing (legacy facefield::get)
 *
 *   facefield::get<XM>(U,V,W, g, idx, idy, idz) = U[mergeId(idx, idy, idz)]
 *   facefield::get<XP>(U,V,W, g, idx, idy, idz) = U[mergeId(idx+1, idy, idz)]
 *   facefield::get<YM>(U,V,W, g, idx, idy, idz) = V[mergeId(idx, idy, idz)]
 *   facefield::get<YP>(U,V,W, g, idx, idy, idz) = V[mergeId(idx, idy+1, idz)]
 *   facefield::get<ZM>(U,V,W, g, idx, idy, idz) = W[mergeId(idx, idy, idz)]
 *   facefield::get<ZP>(U,V,W, g, idx, idy, idz) = W[mergeId(idx, idy, idz+1)]
 *
 * =============================================================================
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#include "cornerfield.cuh"
#include "../internal/cuda_check.cuh"

namespace par2 {
namespace kernels {

// =============================================================================
// Corner Velocity Computation Kernel (LEGACY-FAITHFUL)
// =============================================================================

namespace {

/**
 * @brief Compute face field index (same as legacy facefield::mergeId)
 */
template <typename T>
__device__ __forceinline__
int face_idx(const GridDesc<T>& g, int ix, int iy, int iz) {
    return iz * (g.ny + 1) * (g.nx + 1) + iy * (g.nx + 1) + ix;
}

/**
 * @brief Check if cell is valid (same as legacy grid::validId)
 *
 * A cell (idx, idy, idz) is valid if it's within [0, nx) x [0, ny) x [0, nz)
 */
template <typename T>
__device__ __forceinline__
bool valid_cell(const GridDesc<T>& g, int idx, int idy, int idz) {
    return idx >= 0 && idx < g.nx &&
           idy >= 0 && idy < g.ny &&
           idz >= 0 && idz < g.nz;
}

/**
 * @brief Get face velocity using legacy facefield::get semantics.
 *
 * Direction codes (matching legacy grid::Direction):
 *   XM=0: U at face (idx, idy, idz)     -> U[mergeId(idx, idy, idz)]
 *   XP=1: U at face (idx+1, idy, idz)   -> U[mergeId(idx+1, idy, idz)]
 *   YM=2: V at face (idx, idy, idz)     -> V[mergeId(idx, idy, idz)]
 *   YP=3: V at face (idx, idy+1, idz)   -> V[mergeId(idx, idy+1, idz)]
 *   ZM=4: W at face (idx, idy, idz)     -> W[mergeId(idx, idy, idz)]
 *   ZP=5: W at face (idx, idy, idz+1)   -> W[mergeId(idx, idy, idz+1)]
 */
enum FaceDir { XM=0, XP=1, YM=2, YP=3, ZM=4, ZP=5 };

template <typename T>
__device__ __forceinline__
T get_face(const T* U, const T* V, const T* W,
           const GridDesc<T>& g, int idx, int idy, int idz, int dir)
{
    int id;
    T val;
    switch (dir) {
        case XP: id = face_idx(g, idx+1, idy, idz); val = U[id]; break;
        case XM: id = face_idx(g, idx,   idy, idz); val = U[id]; break;
        case YP: id = face_idx(g, idx, idy+1, idz); val = V[id]; break;
        case YM: id = face_idx(g, idx, idy,   idz); val = V[id]; break;
        case ZP: id = face_idx(g, idx, idy, idz+1); val = W[id]; break;
        case ZM: id = face_idx(g, idx, idy, idz);   val = W[id]; break;
        default: return T(0);
    }
    
    return val;
}

/**
 * @brief Kernel to compute corner velocities from face velocities.
 *
 * EXACT replication of legacy cornerfield::computeCornerVelocities().
 * See legacy/Geometry/CornerField.cuh lines 507-634.
 *
 * For each corner (idx, idy, idz):
 *   Uc = average of X-velocity from up to 4 adjacent cells
 *   Vc = average of Y-velocity from up to 4 adjacent cells
 *   Wc = average of Z-velocity from up to 4 adjacent cells
 */
template <typename T>
__global__ void compute_corner_velocities_kernel(
    const GridDesc<T> grid,
    const T* __restrict__ U,  // Face velocity X
    const T* __restrict__ V,  // Face velocity Y
    const T* __restrict__ W,  // Face velocity Z
    T* __restrict__ Uc,       // Corner velocity X (output)
    T* __restrict__ Vc,       // Corner velocity Y (output)
    T* __restrict__ Wc,       // Corner velocity Z (output)
    int num_corners
) {
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < num_corners;
         tid += blockDim.x * gridDim.x)
    {
        // Convert linear index to 3D corner indices
        // Corner indices: idx ∈ [0, nx], idy ∈ [0, ny], idz ∈ [0, nz]
        const int idx = tid % (grid.nx + 1);
        const int idy = (tid / (grid.nx + 1)) % (grid.ny + 1);
        const int idz = tid / ((grid.nx + 1) * (grid.ny + 1));

        // =====================================================================
        // X-component (Uc) - LEGACY EXACT
        // =====================================================================
        // tx=0 if idx < nx (use XM face), tx=1 if idx==nx (use XP face from cell idx-1)
        T vx = T(0);
        int fx = 0;
        const int tx = (idx == grid.nx) ? 1 : 0;

        // Loop over 2x2 stencil in Y-Z directions
        for (int ty = 0; ty <= 1; ty++) {
            for (int tz = 0; tz <= 1; tz++) {
                // Check if cell (idx-tx, idy-ty, idz-tz) is valid
                if (valid_cell(grid, idx - tx, idy - ty, idz - tz)) {
                    // tx==0: use XM direction (face at idx position)
                    // tx==1: use XP direction (face at idx position from cell idx-1)
                    const int dir = (tx == 0) ? XM : XP;
                    vx += get_face(U, V, W, grid, idx - tx, idy - ty, idz - tz, dir);
                    fx++;
                }
            }
        }
        Uc[tid] = (fx > 0) ? vx / T(fx) : T(0);

        // =====================================================================
        // Y-component (Vc) - LEGACY EXACT
        // =====================================================================
        T vy = T(0);
        int fy = 0;
        const int tyb = (idy == grid.ny) ? 1 : 0;

        for (int tx2 = 0; tx2 <= 1; tx2++) {
            for (int tz = 0; tz <= 1; tz++) {
                if (valid_cell(grid, idx - tx2, idy - tyb, idz - tz)) {
                    const int dir = (tyb == 0) ? YM : YP;
                    vy += get_face(U, V, W, grid, idx - tx2, idy - tyb, idz - tz, dir);
                    fy++;
                }
            }
        }
        Vc[tid] = (fy > 0) ? vy / T(fy) : T(0);

        // =====================================================================
        // Z-component (Wc) - LEGACY EXACT
        // =====================================================================
        T vz = T(0);
        int fz = 0;
        const int tzb = (idz == grid.nz) ? 1 : 0;

        for (int tx3 = 0; tx3 <= 1; tx3++) {
            for (int ty3 = 0; ty3 <= 1; ty3++) {
                if (valid_cell(grid, idx - tx3, idy - ty3, idz - tzb)) {
                    const int dir = (tzb == 0) ? ZM : ZP;
                    vz += get_face(U, V, W, grid, idx - tx3, idy - ty3, idz - tzb, dir);
                    fz++;
                }
            }
        }
        Wc[tid] = (fz > 0) ? vz / T(fz) : T(0);
    }
}

} // anonymous namespace

// =============================================================================
// Launch Function
// =============================================================================

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
) {
    int num_corners = grid.num_corners();

    compute_corner_velocities_kernel<<<num_blocks, block_size, 0, stream>>>(
        grid,
        face_vel.U, face_vel.V, face_vel.W,
        corner_Uc, corner_Vc, corner_Wc,
        num_corners
    );

    PAR2_CUDA_CHECK_LAST();
}

// Explicit instantiations
template void launch_compute_corner_velocities<float>(
    const GridDesc<float>&, const VelocityView<float>&,
    float*, float*, float*, int, int, cudaStream_t);
template void launch_compute_corner_velocities<double>(
    const GridDesc<double>&, const VelocityView<double>&,
    double*, double*, double*, int, int, cudaStream_t);

} // namespace kernels
} // namespace par2
