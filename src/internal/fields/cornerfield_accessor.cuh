/**
 * @file cornerfield_accessor.cuh
 * @brief Corner-centered velocity field accessor with trilinear interpolation.
 *
 * Implements exact semantic parity with:
 * - legacy/Geometry/CornerField.cuh, cornerfield::in()
 * - legacy/Geometry/CornerField.cuh, cornerfield::velocityCorrection()
 *
 * SOURCE OF TRUTH:
 * - Trilinear interpolation: lines 146-195
 * - Drift correction (div(D)): lines 217-334
 * - Zero-velocity tolerance: lines 277-286
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_INTERNAL_CORNERFIELD_ACCESSOR_CUH
#define PAR2_CORE_INTERNAL_CORNERFIELD_ACCESSOR_CUH

#include <par2_core/grid.hpp>
#include "facefield_accessor.cuh"  // For cell_center, lerp
#include "../math/dispersion.cuh"  // For compute_velocity_tolerance

namespace par2 {
namespace internal {

// =============================================================================
// Corner Field Indexing
// =============================================================================

/**
 * @brief Compute corner-field array index.
 *
 * Legacy formula (same as facefield): id = idz*(ny+1)*(nx+1) + idy*(nx+1) + idx
 * SOURCE: legacy/Geometry/CornerField.cuh, cornerfield::mergeId()
 */
template <typename T>
__host__ __device__ __forceinline__
int cornerfield_index(const GridDesc<T>& g, int idx, int idy, int idz) {
    return idz * (g.ny + 1) * (g.nx + 1) + idy * (g.nx + 1) + idx;
}

// =============================================================================
// Trilinear Interpolation Functions
// =============================================================================

/**
 * @brief Trilinear interpolation of 8 corner values.
 *
 * SOURCE: legacy/Geometry/Interpolation.cuh, interpolation::trilinear()
 *
 * @param x,y,z Normalized coordinates ∈ [0,1]³
 * @param v000..v111 Values at 8 corners (binary encoding: vXYZ)
 */
template <typename T>
__host__ __device__ __forceinline__
T trilinear(T x, T y, T z,
            T v000, T v100, T v010, T v110,
            T v001, T v101, T v011, T v111) {
    // Interpolate along X first
    T v00 = lerp(x, v000, v100);
    T v01 = lerp(x, v001, v101);
    T v10 = lerp(x, v010, v110);
    T v11 = lerp(x, v011, v111);

    // Then Y
    T v0 = lerp(y, v00, v10);
    T v1 = lerp(y, v01, v11);

    // Finally Z
    return lerp(z, v0, v1);
}

/**
 * @brief Trilinear interpolation of X-derivative.
 *
 * SOURCE: legacy/Geometry/Interpolation.cuh, interpolation::trilinearDevX()
 */
template <typename T>
__host__ __device__ __forceinline__
T trilinear_dx(T x, T y, T z, T dx,
               T v000, T v100, T v010, T v110,
               T v001, T v101, T v011, T v111) {
    // Derivative along X: (v1 - v0) / dx
    T v00 = (v100 - v000) / dx;
    T v01 = (v101 - v001) / dx;
    T v10 = (v110 - v010) / dx;
    T v11 = (v111 - v011) / dx;

    // Interpolate in Y, then Z
    T v0 = lerp(y, v00, v10);
    T v1 = lerp(y, v01, v11);
    return lerp(z, v0, v1);
}

/**
 * @brief Trilinear interpolation of Y-derivative.
 *
 * SOURCE: legacy/Geometry/Interpolation.cuh, interpolation::trilinearDevY()
 */
template <typename T>
__host__ __device__ __forceinline__
T trilinear_dy(T x, T y, T z, T dy,
               T v000, T v100, T v010, T v110,
               T v001, T v101, T v011, T v111) {
    // Derivative along Y
    T v00 = (v010 - v000) / dy;
    T v01 = (v011 - v001) / dy;
    T v10 = (v110 - v100) / dy;
    T v11 = (v111 - v101) / dy;

    // Interpolate in X, then Z
    T v0 = lerp(x, v00, v10);
    T v1 = lerp(x, v01, v11);
    return lerp(z, v0, v1);
}

/**
 * @brief Trilinear interpolation of Z-derivative.
 *
 * SOURCE: legacy/Geometry/Interpolation.cuh, interpolation::trilinearDevZ()
 */
template <typename T>
__host__ __device__ __forceinline__
T trilinear_dz(T x, T y, T z, T dz,
               T v000, T v100, T v010, T v110,
               T v001, T v101, T v011, T v111) {
    // Derivative along Z
    T v00 = (v001 - v000) / dz;
    T v01 = (v011 - v010) / dz;
    T v10 = (v101 - v100) / dz;
    T v11 = (v111 - v110) / dz;

    // Interpolate in X, then Y
    T v0 = lerp(x, v00, v10);
    T v1 = lerp(x, v01, v11);
    return lerp(y, v0, v1);
}

// =============================================================================
// Corner Field Velocity Sampling
// =============================================================================

/**
 * @brief Sample velocity from corner-centered field using trilinear interpolation.
 *
 * SOURCE OF TRUTH: legacy/Geometry/CornerField.cuh, cornerfield::in(), lines 146-195
 *
 * @param Uc,Vc,Wc Corner velocity components (size (nx+1)*(ny+1)*(nz+1))
 * @param g Grid descriptor
 * @param idx,idy,idz Cell indices containing the particle
 * @param valid True if cell indices are within grid
 * @param px,py,pz Particle position
 * @param[out] vx,vy,vz Interpolated velocity
 */
template <typename T>
__device__ __forceinline__
void sample_velocity_cornerfield(
    const T* __restrict__ Uc,
    const T* __restrict__ Vc,
    const T* __restrict__ Wc,
    const GridDesc<T>& g,
    int idx, int idy, int idz,
    bool valid,
    T px, T py, T pz,
    T& vx, T& vy, T& vz
) {
    if (!valid) {
        vx = vy = vz = T(0);
        return;
    }

    // Cell center
    T cx, cy, cz;
    cell_center(g, idx, idy, idz, cx, cy, cz);

    // Normalized coordinates within cell [0,1]³
    // SOURCE: legacy lines 168-170
    T x = (px - cx) / g.dx + T(0.5);
    T y = (py - cy) / g.dy + T(0.5);
    T z = (pz - cz) / g.dz + T(0.5);

    // Corner indices
    int c000 = cornerfield_index(g, idx,   idy,   idz);
    int c100 = cornerfield_index(g, idx+1, idy,   idz);
    int c010 = cornerfield_index(g, idx,   idy+1, idz);
    int c110 = cornerfield_index(g, idx+1, idy+1, idz);
    int c001 = cornerfield_index(g, idx,   idy,   idz+1);
    int c101 = cornerfield_index(g, idx+1, idy,   idz+1);
    int c011 = cornerfield_index(g, idx,   idy+1, idz+1);
    int c111 = cornerfield_index(g, idx+1, idy+1, idz+1);

    // Trilinear interpolation for each component
    vx = trilinear(x, y, z,
                   Uc[c000], Uc[c100], Uc[c010], Uc[c110],
                   Uc[c001], Uc[c101], Uc[c011], Uc[c111]);
    vy = trilinear(x, y, z,
                   Vc[c000], Vc[c100], Vc[c010], Vc[c110],
                   Vc[c001], Vc[c101], Vc[c011], Vc[c111]);
    vz = trilinear(x, y, z,
                   Wc[c000], Wc[c100], Wc[c010], Wc[c110],
                   Wc[c001], Wc[c101], Wc[c011], Wc[c111]);
}

// =============================================================================
// Drift Correction (div(D)) - TrilinearOnFly mode
// =============================================================================

/**
 * @brief Compute drift correction velocity using trilinear derivatives.
 *
 * SOURCE OF TRUTH: legacy/Geometry/CornerField.cuh, velocityCorrection(), lines 217-334
 *
 * The drift correction is ∇·D where D is the dispersion tensor:
 *   v_drift_x = ∂D_xx/∂x + ∂D_xy/∂y + ∂D_xz/∂z
 *   v_drift_y = ∂D_xy/∂x + ∂D_yy/∂y + ∂D_yz/∂z
 *   v_drift_z = ∂D_xz/∂x + ∂D_yz/∂y + ∂D_zz/∂z
 *
 * The derivatives are computed using trilinearDevX/Y/Z at the 8 corners.
 *
 * @param Uc,Vc,Wc Corner velocity components
 * @param g Grid descriptor
 * @param idx,idy,idz Cell indices
 * @param valid True if cell is valid
 * @param px,py,pz Particle position
 * @param Dm Molecular diffusion
 * @param alphaL Longitudinal dispersivity
 * @param alphaT Transverse dispersivity
 * @param[out] vcx,vcy,vcz Drift correction velocity
 */
template <typename T>
__device__ void compute_drift_trilinear(
    const T* __restrict__ Uc,
    const T* __restrict__ Vc,
    const T* __restrict__ Wc,
    const GridDesc<T>& g,
    int idx, int idy, int idz,
    bool valid,
    T px, T py, T pz,
    T Dm, T alphaL, T alphaT,
    T& vcx, T& vcy, T& vcz
) {
    if (!valid) {
        vcx = vcy = vcz = T(0);
        return;
    }

    // Cell center and normalized coordinates
    T cx, cy, cz;
    cell_center(g, idx, idy, idz, cx, cy, cz);

    T x = (px - cx) / g.dx + T(0.5);
    T y = (py - cy) / g.dy + T(0.5);
    T z = (pz - cz) / g.dz + T(0.5);

    // Corner indices
    int c000 = cornerfield_index(g, idx,   idy,   idz);
    int c100 = cornerfield_index(g, idx+1, idy,   idz);
    int c010 = cornerfield_index(g, idx,   idy+1, idz);
    int c110 = cornerfield_index(g, idx+1, idy+1, idz);
    int c001 = cornerfield_index(g, idx,   idy,   idz+1);
    int c101 = cornerfield_index(g, idx+1, idy,   idz+1);
    int c011 = cornerfield_index(g, idx,   idy+1, idz+1);
    int c111 = cornerfield_index(g, idx+1, idy+1, idz+1);

    // Fetch corner velocities
    // SOURCE: legacy lines 238-261
    T vx000 = Uc[c000], vx100 = Uc[c100], vx010 = Uc[c010], vx110 = Uc[c110];
    T vx001 = Uc[c001], vx101 = Uc[c101], vx011 = Uc[c011], vx111 = Uc[c111];

    T vy000 = Vc[c000], vy100 = Vc[c100], vy010 = Vc[c010], vy110 = Vc[c110];
    T vy001 = Vc[c001], vy101 = Vc[c101], vy011 = Vc[c011], vy111 = Vc[c111];

    T vz000 = Wc[c000], vz100 = Wc[c100], vz010 = Wc[c010], vz110 = Wc[c110];
    T vz001 = Wc[c001], vz101 = Wc[c101], vz011 = Wc[c011], vz111 = Wc[c111];

    // Zero-velocity tolerance handling (CRITICAL for avoiding NaN)
    // SOURCE: legacy lines 277-286
    // "const T toll = 0.01*Dm/alphaL;"
    // "if (all components < toll) vx = toll"
    const T toll = compute_velocity_tolerance(Dm, alphaL);

    // Helper macro to apply tolerance at each corner
    #define APPLY_TOLL(vx, vy, vz) \
        if ((vx) < toll && (vy) < toll && (vz) < toll) (vx) = toll

    APPLY_TOLL(vx000, vy000, vz000);
    APPLY_TOLL(vx100, vy100, vz100);
    APPLY_TOLL(vx010, vy010, vz010);
    APPLY_TOLL(vx110, vy110, vz110);
    APPLY_TOLL(vx001, vy001, vz001);
    APPLY_TOLL(vx101, vy101, vz101);
    APPLY_TOLL(vx011, vy011, vz011);
    APPLY_TOLL(vx111, vy111, vz111);

    #undef APPLY_TOLL

    // Compute |v| at each corner
    // SOURCE: legacy lines 288-295
    T vnorm000 = sqrt(vx000*vx000 + vy000*vy000 + vz000*vz000);
    T vnorm100 = sqrt(vx100*vx100 + vy100*vy100 + vz100*vz100);
    T vnorm010 = sqrt(vx010*vx010 + vy010*vy010 + vz010*vz010);
    T vnorm110 = sqrt(vx110*vx110 + vy110*vy110 + vz110*vz110);
    T vnorm001 = sqrt(vx001*vx001 + vy001*vy001 + vz001*vz001);
    T vnorm101 = sqrt(vx101*vx101 + vy101*vy101 + vz101*vz101);
    T vnorm011 = sqrt(vx011*vx011 + vy011*vy011 + vz011*vz011);
    T vnorm111 = sqrt(vx111*vx111 + vy111*vy111 + vz111*vz111);

    // Compute D_ij at each corner and interpolate derivatives
    // D_ij = (alphaT*|v| + Dm)*delta_ij + (alphaL - alphaT)*vi*vj/|v|
    //
    // We need: dD_xx/dx, dD_yy/dy, dD_zz/dz (diagonal derivatives)
    //          dD_xy/dx, dD_xy/dy           (off-diagonal)
    //          dD_xz/dx, dD_xz/dz
    //          dD_yz/dy, dD_yz/dz

    // Helper: D_ii = (alphaT*|v| + Dm) + (alphaL - alphaT)*vi^2/|v|
    // Helper: D_ij = (alphaL - alphaT)*vi*vj/|v|

    #define D_XX(vx, vy, vz, vnorm) ((alphaT*(vnorm) + Dm) + (alphaL - alphaT)*(vx)*(vx)/(vnorm))
    #define D_YY(vx, vy, vz, vnorm) ((alphaT*(vnorm) + Dm) + (alphaL - alphaT)*(vy)*(vy)/(vnorm))
    #define D_ZZ(vx, vy, vz, vnorm) ((alphaT*(vnorm) + Dm) + (alphaL - alphaT)*(vz)*(vz)/(vnorm))
    #define D_XY(vx, vy, vz, vnorm) ((alphaL - alphaT)*(vx)*(vy)/(vnorm))
    #define D_XZ(vx, vy, vz, vnorm) ((alphaL - alphaT)*(vx)*(vz)/(vnorm))
    #define D_YZ(vx, vy, vz, vnorm) ((alphaL - alphaT)*(vy)*(vz)/(vnorm))

    // dD_xx/dx
    // SOURCE: legacy lines 297-305 (trilinearDevX)
    T dDxxx = trilinear_dx(x, y, z, g.dx,
        D_XX(vx000, vy000, vz000, vnorm000), D_XX(vx100, vy100, vz100, vnorm100),
        D_XX(vx010, vy010, vz010, vnorm010), D_XX(vx110, vy110, vz110, vnorm110),
        D_XX(vx001, vy001, vz001, vnorm001), D_XX(vx101, vy101, vz101, vnorm101),
        D_XX(vx011, vy011, vz011, vnorm011), D_XX(vx111, vy111, vz111, vnorm111));

    // dD_yy/dy
    T dDyyy = trilinear_dy(x, y, z, g.dy,
        D_YY(vx000, vy000, vz000, vnorm000), D_YY(vx100, vy100, vz100, vnorm100),
        D_YY(vx010, vy010, vz010, vnorm010), D_YY(vx110, vy110, vz110, vnorm110),
        D_YY(vx001, vy001, vz001, vnorm001), D_YY(vx101, vy101, vz101, vnorm101),
        D_YY(vx011, vy011, vz011, vnorm011), D_YY(vx111, vy111, vz111, vnorm111));

    // dD_zz/dz
    T dDzzz = trilinear_dz(x, y, z, g.dz,
        D_ZZ(vx000, vy000, vz000, vnorm000), D_ZZ(vx100, vy100, vz100, vnorm100),
        D_ZZ(vx010, vy010, vz010, vnorm010), D_ZZ(vx110, vy110, vz110, vnorm110),
        D_ZZ(vx001, vy001, vz001, vnorm001), D_ZZ(vx101, vy101, vz101, vnorm101),
        D_ZZ(vx011, vy011, vz011, vnorm011), D_ZZ(vx111, vy111, vz111, vnorm111));

    // dD_xy/dx and dD_xy/dy
    T dDxyx = trilinear_dx(x, y, z, g.dx,
        D_XY(vx000, vy000, vz000, vnorm000), D_XY(vx100, vy100, vz100, vnorm100),
        D_XY(vx010, vy010, vz010, vnorm010), D_XY(vx110, vy110, vz110, vnorm110),
        D_XY(vx001, vy001, vz001, vnorm001), D_XY(vx101, vy101, vz101, vnorm101),
        D_XY(vx011, vy011, vz011, vnorm011), D_XY(vx111, vy111, vz111, vnorm111));

    T dDxyy = trilinear_dy(x, y, z, g.dy,
        D_XY(vx000, vy000, vz000, vnorm000), D_XY(vx100, vy100, vz100, vnorm100),
        D_XY(vx010, vy010, vz010, vnorm010), D_XY(vx110, vy110, vz110, vnorm110),
        D_XY(vx001, vy001, vz001, vnorm001), D_XY(vx101, vy101, vz101, vnorm101),
        D_XY(vx011, vy011, vz011, vnorm011), D_XY(vx111, vy111, vz111, vnorm111));

    // dD_xz/dx and dD_xz/dz
    T dDxzx = trilinear_dx(x, y, z, g.dx,
        D_XZ(vx000, vy000, vz000, vnorm000), D_XZ(vx100, vy100, vz100, vnorm100),
        D_XZ(vx010, vy010, vz010, vnorm010), D_XZ(vx110, vy110, vz110, vnorm110),
        D_XZ(vx001, vy001, vz001, vnorm001), D_XZ(vx101, vy101, vz101, vnorm101),
        D_XZ(vx011, vy011, vz011, vnorm011), D_XZ(vx111, vy111, vz111, vnorm111));

    T dDxzz = trilinear_dz(x, y, z, g.dz,
        D_XZ(vx000, vy000, vz000, vnorm000), D_XZ(vx100, vy100, vz100, vnorm100),
        D_XZ(vx010, vy010, vz010, vnorm010), D_XZ(vx110, vy110, vz110, vnorm110),
        D_XZ(vx001, vy001, vz001, vnorm001), D_XZ(vx101, vy101, vz101, vnorm101),
        D_XZ(vx011, vy011, vz011, vnorm011), D_XZ(vx111, vy111, vz111, vnorm111));

    // dD_yz/dy and dD_yz/dz
    T dDyzy = trilinear_dy(x, y, z, g.dy,
        D_YZ(vx000, vy000, vz000, vnorm000), D_YZ(vx100, vy100, vz100, vnorm100),
        D_YZ(vx010, vy010, vz010, vnorm010), D_YZ(vx110, vy110, vz110, vnorm110),
        D_YZ(vx001, vy001, vz001, vnorm001), D_YZ(vx101, vy101, vz101, vnorm101),
        D_YZ(vx011, vy011, vz011, vnorm011), D_YZ(vx111, vy111, vz111, vnorm111));

    T dDyzz = trilinear_dz(x, y, z, g.dz,
        D_YZ(vx000, vy000, vz000, vnorm000), D_YZ(vx100, vy100, vz100, vnorm100),
        D_YZ(vx010, vy010, vz010, vnorm010), D_YZ(vx110, vy110, vz110, vnorm110),
        D_YZ(vx001, vy001, vz001, vnorm001), D_YZ(vx101, vy101, vz101, vnorm101),
        D_YZ(vx011, vy011, vz011, vnorm011), D_YZ(vx111, vy111, vz111, vnorm111));

    #undef D_XX
    #undef D_YY
    #undef D_ZZ
    #undef D_XY
    #undef D_XZ
    #undef D_YZ

    // Compute drift correction (div(D))
    // SOURCE: legacy lines 330-333
    // *vx = dDxxx + dDxyy + dDxzz
    // *vy = dDxyx + dDyyy + dDyzz
    // *vz = dDxzx + dDyzy + dDzzz
    vcx = dDxxx + dDxyy + dDxzz;
    vcy = dDxyx + dDyyy + dDyzz;
    vcz = dDxzx + dDyzy + dDzzz;

    // 2D mode: zero out Z drift
    if (g.nz == 1) {
        vcz = T(0);
    }
}

/**
 * @brief Sample velocity from corner field with 2D awareness.
 */
template <typename T>
__device__ __forceinline__
void sample_velocity_cornerfield_2d_aware(
    const T* __restrict__ Uc,
    const T* __restrict__ Vc,
    const T* __restrict__ Wc,
    const GridDesc<T>& g,
    int idx, int idy, int idz,
    bool valid,
    T px, T py, T pz,
    T& vx, T& vy, T& vz
) {
    sample_velocity_cornerfield(Uc, Vc, Wc, g, idx, idy, idz, valid, px, py, pz, vx, vy, vz);

    if (g.nz == 1) {
        vz = T(0);
    }
}

} // namespace internal
} // namespace par2

#endif // PAR2_CORE_INTERNAL_CORNERFIELD_ACCESSOR_CUH
