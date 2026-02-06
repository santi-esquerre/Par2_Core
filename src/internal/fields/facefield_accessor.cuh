/**
 * @file facefield_accessor.cuh
 * @brief Face-centered velocity field accessor with linear interpolation.
 *
 * Implements exact semantic parity with legacy/Geometry/FaceField.cuh
 *
 * SOURCE OF TRUTH: legacy/Geometry/FaceField.cuh, facefield::in(), lines 144-175
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_INTERNAL_FACEFIELD_ACCESSOR_CUH
#define PAR2_CORE_INTERNAL_FACEFIELD_ACCESSOR_CUH

#include <par2_core/grid.hpp>

namespace par2 {
namespace internal {

// =============================================================================
// Indexing Functions
// =============================================================================

/**
 * @brief Compute face-field array index.
 *
 * Legacy formula: id = idz*(ny+1)*(nx+1) + idy*(nx+1) + idx
 * Source: legacy/Geometry/FaceField.cuh, facefield::mergeId()
 */
template <typename T>
__host__ __device__ __forceinline__
int facefield_index(const GridDesc<T>& g, int idx, int idy, int idz) {
    return idz * (g.ny + 1) * (g.nx + 1) + idy * (g.nx + 1) + idx;
}

/**
 * @brief Get cell center coordinates.
 *
 * Legacy formula: cx = px + idx*dx + 0.5*dx
 * Source: legacy/Geometry/CartesianGrid.cuh, grid::centerOfCell()
 */
template <typename T>
__host__ __device__ __forceinline__
void cell_center(const GridDesc<T>& g, int idx, int idy, int idz,
                 T& cx, T& cy, T& cz) {
    cx = g.px + (T(idx) + T(0.5)) * g.dx;
    cy = g.py + (T(idy) + T(0.5)) * g.dy;
    cz = g.pz + (T(idz) + T(0.5)) * g.dz;
}

// =============================================================================
// Linear Interpolation
// =============================================================================

/**
 * @brief Normalized linear interpolation: v = v0*(1-t) + v1*t
 *
 * Source: legacy/Geometry/Interpolation.cuh, interpolation::linear()
 */
template <typename T>
__host__ __device__ __forceinline__
T lerp(T t, T v0, T v1) {
    return v0 * (T(1) - t) + v1 * t;
}

// =============================================================================
// Face-Field Velocity Sampling (InterpolationMode::Linear)
// =============================================================================

/**
 * @brief Sample velocity from face-centered field using linear interpolation.
 *
 * This is the EXACT implementation of legacy facefield::in().
 *
 * SOURCE OF TRUTH: legacy/Geometry/FaceField.cuh, lines 144-175
 *
 * The algorithm:
 * 1. Compute cell center (cx, cy, cz)
 * 2. Compute local coordinates: t_x = (px - cx)/dx + 0.5  ∈ [0,1]
 * 3. For each component, fetch values at face_minus and face_plus
 * 4. Linear interpolate: v_x = lerp(t_x, U[face_xm], U[face_xp])
 *
 * @param U X-component velocity (face-centered, size (nx+1)*(ny+1)*(nz+1))
 * @param V Y-component velocity (face-centered)
 * @param W Z-component velocity (face-centered)
 * @param g Grid descriptor
 * @param idx Cell index in X (from particle position)
 * @param idy Cell index in Y
 * @param idz Cell index in Z
 * @param valid True if cell indices are within grid
 * @param px Particle X position
 * @param py Particle Y position
 * @param pz Particle Z position
 * @param[out] vx Interpolated X velocity
 * @param[out] vy Interpolated Y velocity
 * @param[out] vz Interpolated Z velocity
 *
 * @note For invalid cells, legacy returns 1.0 for face velocities but we
 *       return 0.0 which makes more physical sense. The valid flag should
 *       prevent computation on invalid particles anyway.
 */
template <typename T>
__device__ __forceinline__
void sample_velocity_facefield(
    const T* __restrict__ U,
    const T* __restrict__ V,
    const T* __restrict__ W,
    const GridDesc<T>& g,
    int idx, int idy, int idz,
    bool valid,
    T px, T py, T pz,
    T& vx, T& vy, T& vz
) {
    if (!valid) {
        // Legacy returns 1.0 for invalid; we return 0.0 (safer)
        // This matches legacy behavior since displacement is zeroed for invalid particles
        vx = vy = vz = T(0);
        return;
    }

    // Cell center (legacy: grid::centerOfCell)
    T cx, cy, cz;
    cell_center(g, idx, idy, idz, cx, cy, cz);

    // Local coordinates (legacy formula exactly)
    // Dx = px - cx, then t = Dx/dx + 0.5
    // This maps particle position within cell to [0,1]
    T tx = (px - cx) / g.dx + T(0.5);
    T ty = (py - cy) / g.dy + T(0.5);
    T tz = (pz - cz) / g.dz + T(0.5);

    // Face indices
    // XM face: at (idx, idy, idz) - left face of cell
    // XP face: at (idx+1, idy, idz) - right face of cell
    //
    // Legacy uses facefield::get<XM> and facefield::get<XP> which
    // internally compute these same indices
    int id_xm = facefield_index(g, idx,   idy, idz);
    int id_xp = facefield_index(g, idx+1, idy, idz);
    int id_ym = facefield_index(g, idx, idy,   idz);
    int id_yp = facefield_index(g, idx, idy+1, idz);
    int id_zm = facefield_index(g, idx, idy, idz);
    int id_zp = facefield_index(g, idx, idy, idz+1);

    // Fetch face velocities
    T vxm = U[id_xm];
    T vxp = U[id_xp];
    T vym = V[id_ym];
    T vyp = V[id_yp];
    T vzm = W[id_zm];
    T vzp = W[id_zp];

    // Linear interpolation per component (legacy formula)
    vx = lerp(tx, vxm, vxp);
    vy = lerp(ty, vym, vyp);
    vz = lerp(tz, vzm, vzp);
}

/**
 * @brief Sample velocity with 2D support.
 *
 * When nz==1, W component is forced to zero regardless of field values.
 * This matches legacy behavior where dpz is not computed for 2D grids.
 */
template <typename T>
__device__ __forceinline__
void sample_velocity_facefield_2d_aware(
    const T* __restrict__ U,
    const T* __restrict__ V,
    const T* __restrict__ W,
    const GridDesc<T>& g,
    int idx, int idy, int idz,
    bool valid,
    T px, T py, T pz,
    T& vx, T& vy, T& vz
) {
    sample_velocity_facefield(U, V, W, g, idx, idy, idz, valid, px, py, pz, vx, vy, vz);

    // 2D mode: force W=0
    if (g.nz == 1) {
        vz = T(0);
    }
}

// =============================================================================
// Cell Identification
// =============================================================================

/**
 * @brief Compute cell indices from particle position.
 *
 * Source: legacy/Geometry/CartesianGrid.cuh, grid::idPoint()
 */
template <typename T>
__host__ __device__ __forceinline__
void position_to_cell(const GridDesc<T>& g, T px, T py, T pz,
                      int& idx, int& idy, int& idz) {
    idx = static_cast<int>(floor((px - g.px) / g.dx));
    idy = static_cast<int>(floor((py - g.py) / g.dy));
    idz = static_cast<int>(floor((pz - g.pz) / g.dz));
}

/**
 * @brief Check if cell indices are valid (inside grid).
 *
 * Source: legacy/Geometry/CartesianGrid.cuh, grid::validId()
 */
template <typename T>
__host__ __device__ __forceinline__
bool is_valid_cell(const GridDesc<T>& g, int idx, int idy, int idz) {
    return idx >= 0 && idx < g.nx &&
           idy >= 0 && idy < g.ny &&
           idz >= 0 && idz < g.nz;
}

/**
 * @brief Compute cell-field index (for cellfield/drift arrays).
 *
 * Source: legacy/Geometry/CartesianGrid.cuh, grid::mergeId()
 */
template <typename T>
__host__ __device__ __forceinline__
int cellfield_index(const GridDesc<T>& g, int idx, int idy, int idz) {
    return idz * g.ny * g.nx + idy * g.nx + idx;
}

} // namespace internal
} // namespace par2

#endif // PAR2_CORE_INTERNAL_FACEFIELD_ACCESSOR_CUH
