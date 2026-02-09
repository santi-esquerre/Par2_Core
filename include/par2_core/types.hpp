/**
 * @file types.hpp
 * @brief Core types for Par2_Core transport engine.
 *
 * This header defines the fundamental types used throughout the API:
 * - Boundary conditions (per-axis)
 * - Transport parameters (diffusion, dispersivity)
 * - Engine configuration
 *
 * @note This header is intentionally lightweight - no CUDA runtime,
 *       no Thrust, no heavy dependencies.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_TYPES_HPP
#define PAR2_CORE_TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace par2 {

// =============================================================================
// Boundary Types
// =============================================================================

/**
 * @brief Boundary condition type for a single axis.
 *
 * Applied per-face (lo/hi) via AxisBoundary.  See boundary.hpp for full config.
 *
 * @see AxisBoundary, BoundaryConfig
 */
enum class BoundaryType : uint8_t {
    /**
     * @brief Reflective boundary (displacement rejection).
     *
     * If the proposed new position exits the domain (strict inequality:
     * lo < x < hi), the **entire displacement is zeroed** and the particle
     * stays at its current position.  This is a rejection scheme, not a
     * bounce/reflection scheme.
     */
    Closed,

    /**
     * @brief Wrap-around: position wraps modulo domain length.
     *
     * Both lo and hi faces of the axis must be set to Periodic.
     * A wrap counter (int32_t) tracks net domain crossings.
     * Wrapped domain is [lo, hi).
     *
     * @note Requires status/wrap arrays to be allocated (via
     *       ensure_tracking_arrays() or prepare()) for wrap counting.
     */
    Periodic,

    /**
     * @brief Allow exit: particle flagged as Exited.
     *
     * If the new position exits the domain, the particle retains its
     * current position and is marked ParticleStatus::Exited.  Exited
     * particles are skipped in subsequent step() calls.
     *
     * @note Requires status array to be allocated (auto-allocated by
     *       prepare() / ensure_tracking_arrays()).
     */
    Open
};

// =============================================================================
// Transport Parameters
// =============================================================================

/**
 * @brief Physical transport parameters for RWPT.
 *
 * @tparam T Floating point type (float or double)
 *
 * @note Units must be consistent across all inputs:
 *       - If grid is in meters, diffusion is m²/s, dispersivity is m
 *       - Velocity field must have units of length/time
 */
template <typename T>
struct TransportParams {
    T molecular_diffusion = T(0);  ///< Effective molecular diffusion coefficient [L²/T]
    T alpha_l = T(0);              ///< Longitudinal dispersivity [L]
    T alpha_t = T(0);              ///< Transverse dispersivity [L]

    /// Check if any dispersion is enabled
    constexpr bool has_dispersion() const noexcept {
        return molecular_diffusion > T(0) || alpha_l > T(0) || alpha_t > T(0);
    }
};

// =============================================================================
// Engine Configuration
// =============================================================================

/**
 * @brief Velocity interpolation mode.
 *
 * Controls how velocity is sampled at particle positions.
 *
 * - **Linear**: Face-centered bilinear interpolation (faster)
 * - **Trilinear**: Corner-based trilinear interpolation (smoother)
 *
 * @note Legacy PAR² uses Linear interpolation ALWAYS. The YAML "interpolation"
 *       field in legacy configs controls the DRIFT mode, not velocity interpolation.
 */
enum class InterpolationMode : uint8_t {
    Linear,     ///< Face-centered linear interpolation (faster)
    Trilinear   ///< Corner-based trilinear interpolation (smoother)
};

/**
 * @brief Drift correction computation mode.
 *
 * The drift correction term (div(D)) arises from the Fokker-Planck equation
 * and compensates for the spatial variation of the dispersion tensor D.
 * Without this term, particles would accumulate in low-dispersion regions.
 *
 * **The dispersion tensor D is defined as:**
 * @f[
 *   D_{ij} = (\alpha_T |v| + D_m) \delta_{ij} + (\alpha_L - \alpha_T) \frac{v_i v_j}{|v|}
 * @f]
 *
 * **The drift correction term is:**
 * @f[
 *   v_{drift,i} = \sum_j \frac{\partial D_{ij}}{\partial x_j}
 * @f]
 *
 * ## Mode Descriptions
 *
 * ### None
 * No drift correction. Only valid when dispersion parameters are zero.
 *
 * ### Precomputed (legacy: "finite difference")
 * Computes div(D) once per timestep at cell centers using finite differences:
 * - Central differences in interior cells: (D[i+1] - D[i-1]) / (2·dx)
 * - One-sided differences at boundaries: (D[i+1] - D[i]) / dx or (D[i] - D[i-1]) / dx
 *
 * **Advantages:** Can be computed once if velocity is steady.
 * **Storage:** Requires 3×num_cells floats for drift + 6×num_cells for D tensor.
 *
 * ### TrilinearOnFly (legacy: "trilinear")
 * Computes div(D) at each particle position using trilinear derivatives
 * of corner velocities. This is the default legacy PAR² mode.
 *
 * The implementation samples 8 corner velocities and uses their analytical
 * trilinear derivatives to compute ∂D/∂x, ∂D/∂y, ∂D/∂z.
 *
 * **Advantages:** No precomputation, works with time-varying velocity.
 * **Cost:** ~8 corner velocity reads + math per particle per step.
 *
 * ## Legacy PAR² Mapping
 *
 * | Legacy YAML "interpolation" | Par2_Core drift_mode  |
 * |-----------------------------|-----------------------|
 * | "trilinear"                 | TrilinearOnFly        |
 * | "finite difference"         | Precomputed           |
 *
 * @note Legacy PAR² always uses Linear velocity interpolation.
 *
 * @see cornerfield::velocityCorrection() for TrilinearOnFly implementation
 * @see launch_compute_drift_precomputed() for Precomputed implementation
 */
enum class DriftCorrectionMode : uint8_t {
    None,            ///< No drift correction (only valid if dispersion is zero)
    Precomputed,     ///< Use precomputed cell-centered div(D) field (legacy: "finite difference")
    TrilinearOnFly   ///< Compute div(D) on-the-fly via trilinear derivatives (legacy: "trilinear")
};

/**
 * @brief Engine runtime configuration.
 *
 * Configures the algorithm modes for RWPT simulation:
 *
 * | interpolation_mode | drift_mode      | Description                    |
 * |--------------------|-----------------|--------------------------------|
 * | Linear             | Precomputed     | Fast, good for steady flow     |
 * | Linear             | TrilinearOnFly  | Accurate drift, linear v       |
 * | Trilinear          | TrilinearOnFly  | Full legacy parity (smoothest) |
 * | Trilinear          | Precomputed     | Valid but uncommon             |
 *
 * ## Debug/Release Policies
 *
 * For HPC pipelines, debug checks should be controlled at **compile-time**
 * to avoid any overhead in production. Set `PAR2_DEBUG_LEVEL` via CMake:
 *
 * ```cmake
 * target_compile_definitions(myapp PRIVATE PAR2_DEBUG_LEVEL=0)  # Release
 * target_compile_definitions(myapp PRIVATE PAR2_DEBUG_LEVEL=1)  # Basic debug
 * ```
 *
 * The runtime `debug_checks` flag is provided for convenience during development
 * but may add per-launch overhead. For zero-overhead HPC builds, leave
 * `debug_checks=false` and compile with `PAR2_DEBUG_LEVEL=0`.
 *
 * @see debug_policy.hpp for compile-time debug configuration.
 */
struct EngineConfig {
    InterpolationMode interpolation_mode = InterpolationMode::Linear;
    DriftCorrectionMode drift_mode = DriftCorrectionMode::TrilinearOnFly;

    /**
     * @brief Seed for curand XORWOW initialization.
     *
     * Each particle gets independent curandState_t initialized via
     * curand_init(seed, particle_index, 0, &state).
     *
     * ## Reproducibility guarantees
     *
     * - Same GPU + same seed + same particle count + same step
     *   sequence → **bitwise identical** results.
     * - Changing particle count changes thread-to-state mapping →
     *   different random streams even with same seed.
     * - Cross-GPU reproducibility is **not** guaranteed (different
     *   warp scheduling may affect curand internal state).
     * - float vs double instantiations produce different results.
     */
    uint64_t rng_seed = 12345ULL;

    /// Block size for CUDA kernels (0 = auto-select)
    int kernel_block_size = 256;

    /**
     * @brief Enable runtime debug checks (default: false).
     *
     * When true, the engine checks for CUDA errors after kernel launches.
     * This adds a cudaPeekAtLastError() call per launch (no sync, but overhead).
     *
     * For HPC builds, prefer compile-time control via PAR2_DEBUG_LEVEL.
     *
     * @deprecated Use PAR2_DEBUG_LEVEL compile-time flag for zero-overhead debug.
     */
    bool debug_checks = false;

    /**
     * @brief Enable extra NaN-prevention guards (default: true = safe).
     *
     * When true (default), applies tolerance checks in places where legacy PAR²
     * could produce NaN due to division by zero (e.g., D tensor computation in
     * Precomputed drift mode when |v| ≈ 0).
     *
     * When false (legacy-strict), disables these guards to match legacy behavior
     * exactly. Use only for regression testing against original PAR².
     *
     * @note This creates a branch per particle in the kernel. For maximum HPC
     *       performance with known-safe velocity fields, set to false.
     *
     * @note The tolerance in TrilinearOnFly mode exists in legacy and is NOT
     *       affected by this flag.
     */
    bool nan_prevention = true;
};

// =============================================================================
// Particle Status (for tracking active/inactive)
// =============================================================================

/**
 * @brief Status flags for particles.
 */
enum class ParticleStatus : uint8_t {
    Active = 0,      ///< Particle is active in simulation
    Exited = 1,      ///< Particle exited via open boundary
    Inactive = 255   ///< Particle slot is unused
};

// =============================================================================
// Statistics / Output
// =============================================================================

/**
 * @brief Summary statistics from simulation step.
 */
template <typename T>
struct StepStats {
    int active_particles = 0;
    int exited_particles = 0;
    T max_displacement = T(0);
};

} // namespace par2

#endif // PAR2_CORE_TYPES_HPP
