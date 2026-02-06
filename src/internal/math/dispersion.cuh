/**
 * @file dispersion.cuh
 * @brief Dispersion tensor D(v) and displacement matrix B computation.
 *
 * Implements exact semantic parity with:
 * - legacy/Geometry/CornerField.cuh, displacementMatrix() (lines 413-487)
 * - legacy/Geometry/CornerField.cuh, velocityCorrection() zero-handling
 *
 * =============================================================================
 * MATHEMATICAL BACKGROUND
 * =============================================================================
 *
 * The dispersion tensor D is:
 *
 *   D_ij = (α_T |v| + D_m) δ_ij + (α_L - α_T) v_i v_j / |v|
 *
 * Eigenvalues:
 *   - λ_L = α_L |v| + D_m  (longitudinal, along v)
 *   - λ_T = α_T |v| + D_m  (transverse, perpendicular to v, 2× degenerate)
 *
 * The displacement matrix B satisfies:
 *   B · B^T = 2 · D · dt
 *
 * B is computed via eigendecomposition:
 *   B = √(2dt) Σ_i √(λ_i) (e_i ⊗ e_i) / |e_i|²
 *
 * Orthonormal eigenvectors:
 *   e_0 = v                                (along flow, eigenvalue λ_L)
 *   e_1 = (-v_y, v_x, 0)                   (⊥ in XY plane, eigenvalue λ_T)
 *   e_2 = (-v_z v_x, -v_z v_y, v_x² + v_y²)  (⊥ to both, eigenvalue λ_T)
 *
 * =============================================================================
 * ZERO-VELOCITY HANDLING (SOURCE OF TRUTH: legacy)
 * =============================================================================
 *
 * The legacy code has TWO mechanisms to handle |v|→0:
 *
 * 1. In velocityCorrection() (drift), line 278-286:
 *      toll = 0.01 * Dm / alphaL
 *      if (all components < toll) vx = toll
 *
 * 2. In displacementMatrix(), line 432:
 *      toll = 0.01 * Dm / alphaL
 *      vx = max(vx, toll)  // Only clamp vx
 *
 * The clamping ensures |v| > 0 preventing NaN in division.
 * We replicate both behaviors exactly.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_INTERNAL_DISPERSION_CUH
#define PAR2_CORE_INTERNAL_DISPERSION_CUH

#include <cmath>

namespace par2 {
namespace internal {

// =============================================================================
// Zero-Velocity Tolerance
// =============================================================================

/**
 * @brief Compute tolerance for zero-velocity handling.
 *
 * SOURCE: legacy/Geometry/CornerField.cuh, line 277
 *   toll = 0.01 * Dm / alphaL
 *
 * @note If alphaL is zero, we use a small absolute tolerance instead.
 */
template <typename T>
__host__ __device__ __forceinline__
T compute_velocity_tolerance(T Dm, T alphaL) {
    if (alphaL > T(0)) {
        return T(0.01) * Dm / alphaL;
    }
    // Fallback for pure molecular diffusion or no-diffusion (alphaL=0).
    //
    // LEGACY BUG: Legacy computes toll = 0.01*Dm/alphaL which is NaN when
    // alphaL=0. This NaN propagates through the drift correction and
    // displacement matrix, making dpx=NaN. The BC check then rejects the
    // NaN displacement, causing particles to freeze near boundaries where
    // corner velocities are zero.
    //
    // We return a small positive value instead, which correctly produces
    // zero drift correction (since D=0 when Dm=alphaL=alphaT=0) without NaN.
    return T(1e-15);
}

// =============================================================================
// Displacement Matrix B (displacementMatrix semantic copy)
// =============================================================================

/**
 * @brief Compute displacement matrix B from velocity and dispersion parameters.
 *
 * SOURCE OF TRUTH: legacy/Geometry/CornerField.cuh, displacementMatrix(), lines 413-487
 *
 * The matrix B satisfies B·B^T = 2·D·dt where D is the dispersion tensor.
 *
 * @param vx X-component of velocity (typically from trilinear cornerfield)
 * @param vy Y-component of velocity
 * @param vz Z-component of velocity
 * @param Dm Molecular diffusion coefficient [L²/T]
 * @param alphaL Longitudinal dispersivity [L]
 * @param alphaT Transverse dispersivity [L]
 * @param dt Time step [T]
 * @param[out] B00,B11,B22,B01,B02,B12 Symmetric matrix components
 *
 * @note The output B is symmetric: B01=B10, B02=B20, B12=B21
 * @note For |v|→0, B degenerates to isotropic √(2·Dm·dt)·I
 * @note nan_prevention: If true, applies extra guards for |e1|²,|e2|²→0 (NOT in legacy).
 */
template <typename T>
__host__ __device__ __forceinline__
void compute_displacement_matrix_legacy(
    T vx, T vy, T vz,
    T Dm, T alphaL, T alphaT, T dt,
    bool nan_prevention,
    T& B00, T& B11, T& B22, T& B01, T& B02, T& B12
) {
    // Legacy tolerance for avoiding NaN
    // SOURCE: line 432: "const T toll = 0.01*Dm/alphaL;"
    const T toll = compute_velocity_tolerance(Dm, alphaL);

    // Legacy clamping (line 434):
    //   "vx = (vx < toll) ? toll : vx;"
    // NOTE: Only vx is clamped, not vy or vz. This is legacy behavior.
    vx = (vx < toll) ? toll : vx;

    // Velocity magnitude squared and magnitude
    // SOURCE: lines 436-437
    T vnorm2 = vx*vx + vy*vy + vz*vz;
    T vnorm = sqrt(vnorm2);

    // Dispersion coefficients
    // SOURCE: lines 440-441
    //   "T alpha = alphaT*vnorm + Dm;"
    //   "T beta  = (alphaL - alphaT)/vnorm;"
    T alpha = alphaT * vnorm + Dm;
    T beta = (alphaL - alphaT) / vnorm;

    // Eigenvectors (not normalized - legacy uses outer product / |e|²)
    // SOURCE: lines 443-451
    //
    // e0 = v                              (eigenvalue: alpha + beta*|v|² = alphaL*|v| + Dm)
    // e1 = (-vy, vx, 0)                   (eigenvalue: alpha = alphaT*|v| + Dm)
    // e2 = (-vz*vx, -vz*vy, vx²+vy²)      (eigenvalue: alpha = alphaT*|v| + Dm)

    // e1 components
    T vx1 = -vy;
    T vy1 = vx;
    T vz1 = T(0);

    // e2 components
    T vx2 = -vz * vx;
    T vy2 = -vz * vy;
    T vz2 = vx*vx + vy*vy;

    // Squared norms of eigenvectors
    // SOURCE: lines 453-454
    T vnorm2_1 = vx1*vx1 + vy1*vy1 + vz1*vz1;  // = vx² + vy²
    T vnorm2_2 = vx2*vx2 + vy2*vy2 + vz2*vz2;  // = vz²(vx²+vy²) + (vx²+vy²)² = (vx²+vy²)(vx²+vy²+vz²)

    // Handle degenerate cases where |e1|² or |e2|² → 0
    // This happens when vx=vy=0 (flow purely in Z)
    // NOTE: Legacy doesn't explicitly handle this - only apply when nan_prevention=true
    if (nan_prevention) {
        const T min_norm2 = toll * toll;
        if (vnorm2_1 < min_norm2) vnorm2_1 = min_norm2;
        if (vnorm2_2 < min_norm2) vnorm2_2 = min_norm2;
    }

    // Gamma coefficients (legacy lines 456-458)
    // gamma_i = sqrt(lambda_i) / |e_i|²
    //
    // lambda_0 = alpha + beta*vnorm2 = alphaL*vnorm + Dm
    // lambda_1 = lambda_2 = alpha = alphaT*vnorm + Dm
    T gamma0 = sqrt(alpha + beta * vnorm2) / vnorm2;
    T gamma1 = sqrt(alpha) / vnorm2_1;
    T gamma2 = sqrt(alpha) / vnorm2_2;

    // Pre-coefficient (line 461)
    T coeff = sqrt(T(2) * dt);

    // Displacement matrix components (lines 464-484)
    // B = coeff * sum_i(gamma_i * e_i ⊗ e_i)
    //
    // B_jk = coeff * sum_i(gamma_i * e_i[j] * e_i[k])
    B00 = coeff * (gamma0*vx*vx   + gamma1*vx1*vx1 + gamma2*vx2*vx2);
    B11 = coeff * (gamma0*vy*vy   + gamma1*vy1*vy1 + gamma2*vy2*vy2);
    B22 = coeff * (gamma0*vz*vz   + gamma1*vz1*vz1 + gamma2*vz2*vz2);
    B01 = coeff * (gamma0*vx*vy   + gamma1*vx1*vy1 + gamma2*vx2*vy2);
    B02 = coeff * (gamma0*vx*vz   + gamma1*vx1*vz1 + gamma2*vx2*vz2);
    B12 = coeff * (gamma0*vy*vz   + gamma1*vy1*vz1 + gamma2*vy2*vz2);
}

// =============================================================================
// Pure Isotropic Diffusion (no velocity)
// =============================================================================

/**
 * @brief Compute B for pure isotropic molecular diffusion (|v|=0 or no dispersivity).
 *
 * When there's no velocity or no dispersivity (alphaL=alphaT=0):
 *   D = Dm * I
 *   B = sqrt(2*Dm*dt) * I
 */
template <typename T>
__host__ __device__ __forceinline__
void compute_displacement_matrix_isotropic(
    T Dm, T dt,
    T& B00, T& B11, T& B22, T& B01, T& B02, T& B12
) {
    T sigma = sqrt(T(2) * Dm * dt);
    B00 = B11 = B22 = sigma;
    B01 = B02 = B12 = T(0);
}

// =============================================================================
// Combined Dispatcher (respects config mode in future)
// =============================================================================

/**
 * @brief Compute displacement matrix with automatic handling of edge cases.
 *
 * This is the primary entry point for the kernel.
 *
 * @param nan_prevention Enable extra NaN guards (not in legacy).
 */
template <typename T>
__host__ __device__ __forceinline__
void compute_B_matrix(
    T vx, T vy, T vz,
    T Dm, T alphaL, T alphaT, T dt,
    bool nan_prevention,
    T& B00, T& B11, T& B22, T& B01, T& B02, T& B12
) {
    // Check if we have any dispersion or diffusion
    if (Dm <= T(0) && alphaL <= T(0) && alphaT <= T(0)) {
        // No dispersion at all - zero matrix
        B00 = B11 = B22 = B01 = B02 = B12 = T(0);
        return;
    }

    // Check if we only have molecular diffusion (no dispersivity)
    if (alphaL <= T(0) && alphaT <= T(0)) {
        compute_displacement_matrix_isotropic(Dm, dt, B00, B11, B22, B01, B02, B12);
        return;
    }

    // Full anisotropic dispersion
    compute_displacement_matrix_legacy(vx, vy, vz, Dm, alphaL, alphaT, dt, nan_prevention,
                                       B00, B11, B22, B01, B02, B12);
}

// =============================================================================
// Dispersion Tensor D Components (for drift correction)
// =============================================================================

/**
 * @brief Compute dispersion tensor D components.
 *
 * D_ij = (alphaT*|v| + Dm) * delta_ij + (alphaL - alphaT) * vi*vj / |v|
 *
 * @param vx,vy,vz Velocity components
 * @param vnorm Velocity magnitude (|v|)
 * @param Dm Molecular diffusion
 * @param alphaL Longitudinal dispersivity
 * @param alphaT Transverse dispersivity
 * @param[out] D11,D22,D33,D12,D13,D23 Tensor components (symmetric)
 */
template <typename T>
__host__ __device__ __forceinline__
void compute_D_tensor(
    T vx, T vy, T vz, T vnorm,
    T Dm, T alphaL, T alphaT,
    T& D11, T& D22, T& D33, T& D12, T& D13, T& D23
) {
    T isotropic = alphaT * vnorm + Dm;
    T aniso_factor = (alphaL - alphaT) / vnorm;

    D11 = isotropic + aniso_factor * vx * vx;
    D22 = isotropic + aniso_factor * vy * vy;
    D33 = isotropic + aniso_factor * vz * vz;
    D12 = aniso_factor * vx * vy;
    D13 = aniso_factor * vx * vz;
    D23 = aniso_factor * vy * vz;
}

} // namespace internal
} // namespace par2

#endif // PAR2_CORE_INTERNAL_DISPERSION_CUH
