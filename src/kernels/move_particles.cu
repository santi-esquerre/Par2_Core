/**
 * @file move_particles.cu
 * @brief GPU kernel for Random Walk Particle Tracking (RWPT) step.
 *
 * @copyright Par2_Core - GPU-native transport engine
 *            Based on PAR² by Calogero B. Rizzo (GPLv3)
 *
 * =============================================================================
 * RWPT SEMANTIC CONTRACT (M1) - SOURCE OF TRUTH: legacy/
 * =============================================================================
 *
 * This kernel implements the RWPT algorithm with **exact** semantic parity
 * to legacy PAR². Any deviation is explicitly documented and justified.
 *
 * ## Mathematical Formulation
 *
 * The particle displacement per timestep is:
 *
 *   Δx = (v_interp + v_drift) * dt + B * ξ
 *
 * where:
 *   - v_interp : Interpolated velocity at particle position
 *   - v_drift  : Drift correction = ∇·D (divergence of dispersion tensor)
 *   - B        : Displacement matrix satisfying B·B^T = 2·D·dt
 *   - ξ        : Vector of 3 independent N(0,1) random variates
 *   - dt       : Time step
 *
 * ## Dispersion Tensor D (source: legacy cornerfield)
 *
 *   D_ij = (α_T |v| + D_m) δ_ij + (α_L - α_T) v_i v_j / |v|
 *
 * where:
 *   - D_m   : Molecular diffusion coefficient [L²/T]
 *   - α_L   : Longitudinal dispersivity [L]
 *   - α_T   : Transverse dispersivity [L]
 *   - |v|   : Velocity magnitude
 *
 * Eigenvalues:
 *   - λ_L = α_L |v| + D_m  (longitudinal, along v)
 *   - λ_T = α_T |v| + D_m  (transverse, 2× degenerate, ⊥ v)
 *
 * =============================================================================
 * FUNCTION MAPPING: Par2_Core ↔ Legacy
 * =============================================================================
 *
 * | Par2_Core Function              | Legacy Function                       |
 * |---------------------------------|---------------------------------------|
 * | sample_velocity_facefield()     | facefield::in()                       |
 * | sample_velocity_cornerfield()   | cornerfield::in()                     |
 * | compute_displacement_matrix()   | cornerfield::displacementMatrix()     |
 * | compute_drift_trilinear()       | cornerfield::velocityCorrection()     |
 * | sample_drift_precomputed()      | cellfield lookup (cdatax/y/z)         |
 *
 * =============================================================================
 * 1. VELOCITY INTERPOLATION - InterpolationMode
 * =============================================================================
 *
 * ### InterpolationMode::Linear (facefield::in)
 *
 * Source: legacy/Geometry/FaceField.cuh, lines 144-175
 *
 * Each velocity component is interpolated independently on its staggered axis:
 *
 *   t_x = (px - cell_center_x) / dx + 0.5    ∈ [0,1]
 *   v_x = lerp(t_x, U[idx,idy,idz], U[idx+1,idy,idz])
 *
 * Similarly for v_y (faces ⊥ Y) and v_z (faces ⊥ Z).
 *
 * Face-field index:
 *   id = idz * (ny+1) * (nx+1) + idy * (nx+1) + idx
 *
 * ### InterpolationMode::Trilinear (cornerfield::in)
 *
 * Source: legacy/Geometry/CornerField.cuh, lines 146-195
 *
 * All 8 corner values are used for true trilinear interpolation:
 *
 *   v_x = trilinear(t, Uc[corners])
 *
 * where t = (p - cell_corner) / cell_size ∈ [0,1]³
 *
 * =============================================================================
 * 2. DRIFT CORRECTION - DriftCorrectionMode
 * =============================================================================
 *
 * ### DriftCorrectionMode::TrilinearOnFly (cornerfield::velocityCorrection)
 *
 * Source: legacy/Geometry/CornerField.cuh, lines 217-334
 *
 * Computes ∇·D on-the-fly using trilinear derivative interpolation:
 *
 *   v_drift_x = ∂D_xx/∂x + ∂D_xy/∂y + ∂D_xz/∂z
 *   v_drift_y = ∂D_xy/∂x + ∂D_yy/∂y + ∂D_yz/∂z
 *   v_drift_z = ∂D_xz/∂x + ∂D_yz/∂y + ∂D_zz/∂z
 *
 * The derivatives are computed via trilinearDevX/Y/Z() (Interpolation.cuh).
 *
 * **Zero-velocity handling** (legacy, lines 278-286):
 *   toll = 0.01 * Dm / alphaL
 *   if (vx < toll && vy < toll && vz < toll) vx = toll
 *
 * This prevents NaN from division by |v|=0.
 *
 * ### DriftCorrectionMode::Precomputed (cellfield lookup)
 *
 * Source: legacy/Geometry/CellField.cuh, computeDriftCorrection()
 *         legacy/Particles/MoveParticle.cuh, lines 147-150
 *
 * Drift is precomputed per cell using finite differences:
 *   - Central differences in interior cells
 *   - One-sided differences at boundaries
 *
 * In kernel: drift[cell_id] where cell_id = mergeId(idx, idy, idz)
 *
 * =============================================================================
 * 3. DISPLACEMENT MATRIX B
 * =============================================================================
 *
 * Source: legacy/Geometry/CornerField.cuh, displacementMatrix(), lines 413-487
 *
 * The matrix B satisfies B·B^T = 2·D·dt and is constructed via eigendecomposition:
 *
 *   B = √(2dt) Σ_i √(λ_i) (e_i ⊗ e_i) / |e_i|²
 *
 * where (e_0, e_1, e_2) form an orthogonal basis:
 *
 *   e_0 = v                             (along flow)
 *   e_1 = (-v_y, v_x, 0)                (⊥ in XY plane)
 *   e_2 = (-v_z v_x, -v_z v_y, v_x² + v_y²)  (⊥ to both)
 *
 * **Implementation detail** (legacy line 432):
 *   - Threshold vx when all components are small: vx = max(vx, toll)
 *   - This ensures e_1, e_2 norms are non-zero.
 *
 * The 6 unique symmetric components:
 *   B = | B00  B01  B02 |
 *       | B01  B11  B12 |
 *       | B02  B12  B22 |
 *
 * =============================================================================
 * 4. BOUNDARY CONDITIONS
 * =============================================================================
 *
 * Source: legacy/Particles/MoveParticle.cuh, lines 180-187
 *
 * ### BoundaryType::Closed (legacy behavior)
 *
 *   dp_x = validX(p_x + dp_x) ? dp_x : 0
 *
 * where validX(x) = (grid.px < x < grid.px + nx*dx)
 *
 * The particle stays at current position if move would exit.
 *
 * ### BoundaryType::Periodic (Par2_Core extension)
 *
 *   p_x = wrap(p_x + dp_x, grid.px, grid.px + nx*dx)
 *
 * ### BoundaryType::Open (Par2_Core extension)
 *
 *   p_x += dp_x  // no constraint
 *   if (out_of_domain) status = Exited
 *
 * =============================================================================
 * 5. 2D SUPPORT (nz == 1)
 * =============================================================================
 *
 * Source: legacy/Particles/MoveParticle.cuh, lines 175-176, 186-187
 *
 * When grid.nz == 1:
 *   - dp_z is NOT computed (or forced to 0)
 *   - p_z remains unchanged
 *   - Drift z-derivatives are zero (single layer)
 *
 * =============================================================================
 * 6. RNG
 * =============================================================================
 *
 * Source: legacy/Particles/MoveParticle.cuh, lines 168-170
 *
 * Uses curand_normal_double() for N(0,1) variates.
 * Each particle has persistent curandState_t.
 * Three variates per step: ξ₀, ξ₁, ξ₂
 *
 * =============================================================================
 * ALGORITHM PSEUDOCODE
 * =============================================================================
 *
 *   for each particle i:
 *     1. (idx, idy, idz) = cell_from_position(p[i])
 *     2. valid = is_inside_grid(idx, idy, idz)
 *     3. v = sample_velocity(mode, p[i])
 *     4. v_drift = sample_drift(drift_mode, p[i])
 *     5. B = compute_displacement_matrix(v_corner, Dm, αL, αT, dt)
 *     6. ξ = (N(0,1), N(0,1), N(0,1))
 *     7. Δp = (v + v_drift)*dt + B*ξ
 *     8. p[i] += apply_boundary(Δp)
 *
 * =============================================================================
 */

#include "move_particles.cuh"
#include "../internal/cuda_check.cuh"
#include "../internal/math/vec3.cuh"
#include "../internal/grid/indexing.cuh"
#include "../internal/fields/facefield_accessor.cuh"
#include "../internal/fields/cornerfield_accessor.cuh"
#include "../internal/math/dispersion.cuh"
#include "../internal/boundary/boundary_helpers.cuh"

#include <curand_kernel.h>

namespace par2 {
namespace kernels {

// Import internal functions into kernel namespace for convenience
using internal::sample_velocity_facefield;
using internal::sample_velocity_facefield_2d_aware;
using internal::sample_velocity_cornerfield;
using internal::sample_velocity_cornerfield_2d_aware;
using internal::compute_drift_trilinear;
using internal::position_to_cell;
using internal::is_valid_cell;
using internal::cell_center;
using internal::facefield_index;
using internal::cellfield_index;
using internal::cornerfield_index;
using internal::lerp;
using internal::compute_B_matrix;

// =============================================================================
// RNG Initialization Kernel
// =============================================================================

__global__ void init_rng_states(
    curandState_t* states,
    int n,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Each thread gets a unique sequence
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// =============================================================================
// Helper Device Functions
// =============================================================================

namespace {

/// Check if position is valid in X
/// Source: legacy/Geometry/CartesianGrid.cuh, grid::validX()
template <typename T>
__device__ __forceinline__ bool valid_x(const GridDesc<T>& g, T x) {
    return x > g.px && x < g.px + T(g.nx) * g.dx;
}

/// Check if position is valid in Y
/// Source: legacy/Geometry/CartesianGrid.cuh, grid::validY()
template <typename T>
__device__ __forceinline__ bool valid_y(const GridDesc<T>& g, T y) {
    return y > g.py && y < g.py + T(g.ny) * g.dy;
}

/// Check if position is valid in Z
/// Source: legacy/Geometry/CartesianGrid.cuh, grid::validZ()
template <typename T>
__device__ __forceinline__ bool valid_z(const GridDesc<T>& g, T z) {
    return z > g.pz && z < g.pz + T(g.nz) * g.dz;
}

} // anonymous namespace

// =============================================================================
// Displacement Matrix Computation (LEGACY SEMANTICS)
// =============================================================================
// SOURCE OF TRUTH: legacy/Geometry/CornerField.cuh, displacementMatrix()
// Implementation delegated to internal::compute_B_matrix() for clean separation

/// Wrapper for displacement matrix computation
template <typename T>
__device__ __forceinline__ void compute_displacement_matrix(
    T vx, T vy, T vz,
    T Dm, T alphaL, T alphaT, T dt,
    bool nan_prevention,
    T& B00, T& B11, T& B22, T& B01, T& B02, T& B12
) {
    internal::compute_B_matrix(vx, vy, vz, Dm, alphaL, alphaT, dt, nan_prevention,
                               B00, B11, B22, B01, B02, B12);
}

// =============================================================================
// Move Particles Kernel (Full Mode Support)
// =============================================================================

/**
 * @brief Main RWPT kernel with full mode support and boundary conditions.
 *
 * M3 features:
 *   - Status gating: Exited/Inactive particles skip all computation
 *   - Closed BC (legacy): reject displacement if exits domain
 *   - Open BC: mark particle as Exited, stop moving
 *   - Periodic BC: wrap position, update wrap counter
 *
 * Supports:
 *   - InterpolationMode::Linear (face-centered, FaceField::in)
 *   - InterpolationMode::Trilinear (corner-centered, CornerField::in)
 *   - DriftCorrectionMode::None
 *   - DriftCorrectionMode::TrilinearOnFly (cornerfield::velocityCorrection)
 *   - DriftCorrectionMode::Precomputed (cellfield lookup)
 */
template <typename T>
__global__ void move_particles_kernel_full(
    const GridDesc<T> grid,
    const TransportParams<T> params,
    const AxisBoundary<T> bc_x,
    const AxisBoundary<T> bc_y,
    const AxisBoundary<T> bc_z,
    T dt,
    bool nan_prevention,  // Enable extra NaN guards (not in legacy)
    // Face-centered velocity (for Linear interpolation)
    const T* __restrict__ U,
    const T* __restrict__ V,
    const T* __restrict__ W,
    // Corner-centered velocity (for Trilinear interpolation + drift)
    const T* __restrict__ Uc,
    const T* __restrict__ Vc,
    const T* __restrict__ Wc,
    // Precomputed drift correction (if Precomputed mode)
    const T* __restrict__ drift_x,
    const T* __restrict__ drift_y,
    const T* __restrict__ drift_z,
    // Particles
    T* __restrict__ px,
    T* __restrict__ py,
    T* __restrict__ pz,
    uint8_t* __restrict__ status,  // May be nullptr
    int32_t* __restrict__ wrapX,   // May be nullptr
    int32_t* __restrict__ wrapY,   // May be nullptr
    int32_t* __restrict__ wrapZ,   // May be nullptr
    curandState_t* __restrict__ rng_states,
    int n,
    // Mode configuration
    InterpolationMode interp_mode,
    DriftCorrectionMode drift_mode
) {
    const T Dm = params.molecular_diffusion;
    const T alphaL = params.alpha_l;
    const T alphaT = params.alpha_t;

    // Domain bounds
    const T x_lo = grid.px;
    const T x_hi = grid.x_max();
    const T y_lo = grid.py;
    const T y_hi = grid.y_max();
    const T z_lo = grid.pz;
    const T z_hi = grid.z_max();

    // Grid-stride loop
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n;
         tid += blockDim.x * gridDim.x)
    {
        // =====================================================================
        // STEP 0: Status gating (M3-T1)
        // =====================================================================
        // Early-out for non-Active particles
        if (status != nullptr) {
            uint8_t s = status[tid];
            if (s != static_cast<uint8_t>(ParticleStatus::Active)) {
                continue;  // Exited or Inactive - skip entirely
            }
        }

        // Load particle position
        T x = px[tid];
        T y = py[tid];
        T z = pz[tid];

        // Get cell indices
        int idx, idy, idz;
        position_to_cell(grid, x, y, z, idx, idy, idz);
        bool valid = is_valid_cell(grid, idx, idy, idz);

        if (!valid) {
            // Particle outside grid - no movement
            continue;
        }

        // =====================================================================
        // STEP 1: Velocity interpolation
        // =====================================================================
        T vx, vy, vz;

        if (interp_mode == InterpolationMode::Linear) {
            // SOURCE: legacy/Geometry/FaceField.cuh, facefield::in()
            sample_velocity_facefield_2d_aware(U, V, W, grid, idx, idy, idz, valid, x, y, z, vx, vy, vz);
        } else {
            // InterpolationMode::Trilinear
            // SOURCE: legacy/Geometry/CornerField.cuh, cornerfield::in()
            sample_velocity_cornerfield_2d_aware(Uc, Vc, Wc, grid, idx, idy, idz, valid, x, y, z, vx, vy, vz);
        }

        // =====================================================================
        // STEP 2: Drift correction
        // =====================================================================
        T vdx = T(0), vdy = T(0), vdz = T(0);

        if (drift_mode == DriftCorrectionMode::TrilinearOnFly) {
            // SOURCE: legacy/Geometry/CornerField.cuh, cornerfield::velocityCorrection()
            compute_drift_trilinear(
                Uc, Vc, Wc, grid, idx, idy, idz, valid,
                x, y, z, Dm, alphaL, alphaT,
                vdx, vdy, vdz
            );
        } else if (drift_mode == DriftCorrectionMode::Precomputed && drift_x != nullptr) {
            // Precomputed mode: direct cell lookup (no interpolation, piecewise constant)
            // SOURCE: legacy/Particles/MoveParticle.cuh, lines 167-169
            // Legacy: vcx = idValid ? cdatax[id] : 0;
            int cell_id = cellfield_index(grid, idx, idy, idz);
            vdx = drift_x[cell_id];
            vdy = drift_y[cell_id];
            vdz = drift_z[cell_id];
        }
        // else: DriftCorrectionMode::None - leave drift at zero

        // =====================================================================
        // STEP 3: Displacement matrix B
        // =====================================================================
        // SOURCE: legacy/Geometry/CornerField.cuh, displacementMatrix()
        // For B matrix, we need corner velocity (trilinear) as per legacy
        T vx_B, vy_B, vz_B;
        if (interp_mode == InterpolationMode::Trilinear && Uc != nullptr) {
            vx_B = vx; vy_B = vy; vz_B = vz;
        } else if (Uc != nullptr) {
            // Use corner velocity for B matrix even in Linear mode
            sample_velocity_cornerfield_2d_aware(Uc, Vc, Wc, grid, idx, idy, idz, valid, x, y, z, vx_B, vy_B, vz_B);
        } else {
            // Fallback: use face-centered velocity
            vx_B = vx; vy_B = vy; vz_B = vz;
        }

        T B00, B11, B22, B01, B02, B12;
        compute_displacement_matrix(
            vx_B, vy_B, vz_B,
            Dm, alphaL, alphaT, dt,
            nan_prevention,
            B00, B11, B22, B01, B02, B12
        );

        // =====================================================================
        // STEP 4: Random variate generation
        // =====================================================================
        curandState_t local_state = rng_states[tid];
        T xi0 = curand_normal_double(&local_state);
        T xi1 = curand_normal_double(&local_state);
        T xi2 = curand_normal_double(&local_state);
        rng_states[tid] = local_state;

        // =====================================================================
        // STEP 5: Compute displacement
        // =====================================================================
        // Δx = (v_interp + v_drift) * dt + B * ξ
        T dpx = (vx + vdx) * dt + B00*xi0 + B01*xi1 + B02*xi2;
        T dpy = (vy + vdy) * dt + B01*xi0 + B11*xi1 + B12*xi2;
        T dpz = (vz + vdz) * dt + B02*xi0 + B12*xi1 + B22*xi2;

        // =====================================================================
        // STEP 6: Apply boundary conditions (M3-T2/T3/T4)
        // =====================================================================
        bool particle_exited = false;
        T new_x, new_y, new_z;
        int32_t kx = 0, ky = 0, kz = 0;

        // X-axis BC
        auto bcr_x = internal::apply_bc_axis(bc_x.lo, bc_x.hi, x, dpx, x_lo, x_hi);
        new_x = bcr_x.new_pos;
        kx = bcr_x.wrap_k;
        particle_exited |= bcr_x.exited;

        // DEBUG: Boundary tracing (disable for production)
        #if 0  // DISABLED - was used to diagnose NaN drift bug
        if (tid == 0 && idx == 99) {
            int id_xm = internal::facefield_index(grid, idx, idy, idz);
            int id_xp = internal::facefield_index(grid, idx+1, idy, idz);
            T face_xm = U[id_xm];
            T face_xp = U[id_xp];
            T cx_cell, cy_cell, cz_cell;
            internal::cell_center(grid, idx, idy, idz, cx_cell, cy_cell, cz_cell);
            T tx = (x - cx_cell) / grid.dx + T(0.5);
            bool vdx_nan = isnan(vdx);
            bool dpx_nan = isnan(dpx);
            printf("DEBUG cell99 tid=0: x=%.6f y=%.6f idx=%d idy=%d idz=%d\n"
                   "  face_xm=%.6f face_xp=%.6f tx=%.6f vx=%.6f\n"
                   "  vdx=%.6e vdx_nan=%d dpx=%.10f dpx_nan=%d new_x=%.10f\n",
                   x, y, idx, idy, idz,
                   face_xm, face_xp, tx, vx,
                   vdx, vdx_nan ? 1 : 0, dpx, dpx_nan ? 1 : 0, new_x);
        }
        #endif

        // Y-axis BC
        auto bcr_y = internal::apply_bc_axis(bc_y.lo, bc_y.hi, y, dpy, y_lo, y_hi);
        new_y = bcr_y.new_pos;
        ky = bcr_y.wrap_k;
        particle_exited |= bcr_y.exited;

        // Z-axis BC (2D case: force dpz=0)
        if (grid.nz > 1) {
            auto bcr_z = internal::apply_bc_axis(bc_z.lo, bc_z.hi, z, dpz, z_lo, z_hi);
            new_z = bcr_z.new_pos;
            kz = bcr_z.wrap_k;
            particle_exited |= bcr_z.exited;
        } else {
            new_z = z;  // 2D: no z-displacement
            kz = 0;
        }

        // =====================================================================
        // STEP 7: Update particle state
        // =====================================================================
        // Update position
        px[tid] = new_x;
        py[tid] = new_y;
        pz[tid] = new_z;

        // Update status if exited (Open BC)
        if (particle_exited && status != nullptr) {
            status[tid] = static_cast<uint8_t>(ParticleStatus::Exited);
        }

        // Update wrap counters (Periodic BC)
        if (wrapX != nullptr && kx != 0) {
            wrapX[tid] += kx;
        }
        if (wrapY != nullptr && ky != 0) {
            wrapY[tid] += ky;
        }
        if (wrapZ != nullptr && kz != 0) {
            wrapZ[tid] += kz;
        }
    }
}

// Legacy kernel (backward compatibility)
template <typename T>
__global__ void move_particles_kernel(
    const GridDesc<T> grid,
    const TransportParams<T> params,
    T dt,
    const T* __restrict__ U,
    const T* __restrict__ V,
    const T* __restrict__ W,
    T* __restrict__ px,
    T* __restrict__ py,
    T* __restrict__ pz,
    curandState_t* __restrict__ rng_states,
    int n
) {
    // Grid-stride loop
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < n;
         tid += blockDim.x * gridDim.x)
    {
        // Load particle position
        T x = px[tid];
        T y = py[tid];
        T z = pz[tid];

        // Get cell indices
        // SOURCE: legacy/Geometry/CartesianGrid.cuh, grid::idPoint()
        int idx, idy, idz;
        position_to_cell(grid, x, y, z, idx, idy, idz);

        bool valid = is_valid_cell(grid, idx, idy, idz);

        // Interpolate velocity (InterpolationMode::Linear)
        // SOURCE: legacy/Geometry/FaceField.cuh, facefield::in()
        T vx, vy, vz;
        sample_velocity_facefield_2d_aware(U, V, W, grid, idx, idy, idz, valid, x, y, z, vx, vy, vz);

        // Compute displacement matrix
        T B00, B11, B22, B01, B02, B12;
        compute_displacement_matrix(
            vx, vy, vz,
            params.molecular_diffusion, params.alpha_l, params.alpha_t, dt,
            B00, B11, B22, B01, B02, B12
        );

        // Generate random numbers
        curandState_t local_state = rng_states[tid];
        T xi0 = curand_normal_double(&local_state);
        T xi1 = curand_normal_double(&local_state);
        T xi2 = curand_normal_double(&local_state);
        rng_states[tid] = local_state;

        // Compute displacement
        // Note: drift correction (div(D)) is TODO for Tarea 4-5
        T dpx = valid ? (vx * dt + B00*xi0 + B01*xi1 + B02*xi2) : T(0);
        T dpy = valid ? (vy * dt + B01*xi0 + B11*xi1 + B12*xi2) : T(0);
        T dpz = valid ? (vz * dt + B02*xi0 + B12*xi1 + B22*xi2) : T(0);

        // Apply boundary conditions (closed - legacy)
        // SOURCE: legacy/Particles/MoveParticle.cuh, lines 180-187
        dpx = internal::apply_bc_closed(x, dpx, grid.px, grid.x_max());
        dpy = internal::apply_bc_closed(y, dpy, grid.py, grid.y_max());
        if (grid.nz > 1) {
            dpz = internal::apply_bc_closed(z, dpz, grid.pz, grid.z_max());
        } else {
            dpz = T(0);  // 2D case: no z-displacement
        }

        // Update position
        px[tid] = x + dpx;
        py[tid] = y + dpy;
        pz[tid] = z + dpz;
    }
}

// =============================================================================
// Inject Box Kernel
// =============================================================================

template <typename T>
__global__ void inject_box_kernel(
    T* __restrict__ px,
    T* __restrict__ py,
    T* __restrict__ pz,
    int n,
    T x0, T y0, T z0,
    T x1, T y1, T z1,
    unsigned long long seed,
    int offset
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Initialize local RNG
    curandState_t state;
    curand_init(seed, tid + offset, 0, &state);

    // Generate uniform positions in box
    T u = curand_uniform_double(&state);
    T v = curand_uniform_double(&state);
    T w = curand_uniform_double(&state);

    px[tid] = x0 + u * (x1 - x0);
    py[tid] = y0 + v * (y1 - y0);
    pz[tid] = z0 + w * (z1 - z0);
}

// =============================================================================
// Launch Functions (explicit instantiation)
// =============================================================================

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
) {
    // Use the full-featured kernel with BC support
    move_particles_kernel_full<<<num_blocks, block_size, 0, stream>>>(
        grid,
        params,
        boundary.x,
        boundary.y,
        boundary.z,
        dt,
        config.nan_prevention,
        // Face-centered velocity
        velocity.U,
        velocity.V,
        velocity.W,
        // Corner-centered velocity (may be nullptr)
        corner_velocity.Uc,
        corner_velocity.Vc,
        corner_velocity.Wc,
        // Precomputed drift (may be nullptr)
        drift_correction.dcx,
        drift_correction.dcy,
        drift_correction.dcz,
        // Particles
        particles.x,
        particles.y,
        particles.z,
        particles.status,
        particles.wrapX,
        particles.wrapY,
        particles.wrapZ,
        rng_states,
        particles.n,
        // Mode configuration
        config.interpolation_mode,
        config.drift_mode
    );
}

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
) {
    inject_box_kernel<<<num_blocks, block_size, 0, stream>>>(
        x, y, z, n,
        x0, y0, z0, x1, y1, z1,
        seed, offset
    );
}

// Explicit instantiations
template void launch_move_particles<float>(
    const GridDesc<float>&, const TransportParams<float>&,
    const BoundaryConfig<float>&, const EngineConfig&, float,
    const VelocityView<float>&, const CornerVelocityView<float>&,
    const ParticlesView<float>&, const DriftCorrectionView<float>&,
    curandState_t*, int, int, cudaStream_t);

template void launch_move_particles<double>(
    const GridDesc<double>&, const TransportParams<double>&,
    const BoundaryConfig<double>&, const EngineConfig&, double,
    const VelocityView<double>&, const CornerVelocityView<double>&,
    const ParticlesView<double>&, const DriftCorrectionView<double>&,
    curandState_t*, int, int, cudaStream_t);

template void launch_inject_box<float>(
    float*, float*, float*, int,
    float, float, float, float, float, float,
    unsigned long long, int, int, int, cudaStream_t);

template void launch_inject_box<double>(
    double*, double*, double*, int,
    double, double, double, double, double, double,
    unsigned long long, int, int, int, cudaStream_t);

} // namespace kernels
} // namespace par2
