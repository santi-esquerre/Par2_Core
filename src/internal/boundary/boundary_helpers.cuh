/**
 * @file boundary_helpers.cuh
 * @brief Device inline functions for boundary condition handling.
 *
 * Implements:
 * - Closed BC (legacy: reject displacement if exits domain)
 * - Open BC (mark Exited, stop moving)
 * - Periodic BC (wrap position, update wrap counter)
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_INTERNAL_BOUNDARY_HELPERS_CUH
#define PAR2_INTERNAL_BOUNDARY_HELPERS_CUH

#include <par2_core/types.hpp>
#include <cmath>  // for floor

namespace par2 {
namespace internal {

// =============================================================================
// Domain bounds helpers
// =============================================================================

/**
 * @brief Check if position is within domain bounds (exclusive).
 *
 * Legacy semantics: pos in (lo, hi), strict inequalities.
 *
 * Source: legacy/Geometry/CartesianGrid.cuh, grid::validX/Y/Z()
 */
template <typename T>
__device__ __forceinline__ bool in_domain(T pos, T lo, T hi) {
    return pos > lo && pos < hi;
}

// =============================================================================
// Closed BC (legacy semantics)
// =============================================================================

/**
 * @brief Apply Closed boundary condition to displacement.
 *
 * Legacy semantics: If the new position would exit the domain,
 * the displacement is zeroed (particle stays in place).
 *
 * @param pos Current position
 * @param dp Proposed displacement
 * @param lo Domain minimum
 * @param hi Domain maximum
 * @return Adjusted displacement (0 if would exit, dp otherwise)
 *
 * Source: legacy/Particles/MoveParticle.cuh, lines 180-187
 * Legacy uses STRICT inequality: lo < x && x < hi
 *
 * FIX: Added epsilon buffer to handle floating-point edge cases.
 */
template <typename T>
__device__ __forceinline__ T apply_bc_closed(T pos, T dp, T lo, T hi) {
    T new_pos = pos + dp;
    // LEGACY EXACT: grid::validX returns (lo < x && x < hi)
    // Use epsilon buffer for robust floating-point comparison
    const T L = hi - lo;
    const T eps = L * T(1e-14);
    // Valid if strictly inside domain with epsilon margin
    bool valid = (new_pos > lo + eps) && (new_pos < hi - eps);
    return valid ? dp : T(0);
}

// =============================================================================
// Periodic BC with wrap counter
// =============================================================================

/**
 * @brief Wrap a position into domain [lo, hi) with robust floor-based algorithm.
 *
 * This handles arbitrary displacements, including multi-domain crossings.
 *
 * Algorithm:
 *   k = floor((pos - lo) / L)
 *   wrapped = pos - k * L
 *
 * If wrapped lands exactly on hi, adjust to lo.
 *
 * @param pos Position to wrap (may be outside domain)
 * @param lo Domain minimum
 * @param L Domain length (hi - lo)
 * @param[out] k Number of domain crossings (positive = crossed in + direction)
 * @return Wrapped position in [lo, hi)
 */
template <typename T>
__device__ __forceinline__ T wrap_periodic_robust(T pos, T lo, T L, int32_t& k) {
    T offset = pos - lo;
    T k_float = floor(offset / L);
    k = static_cast<int32_t>(k_float);
    T wrapped = pos - k_float * L;

    // Edge case: if wrapped == hi (floating point), snap to lo
    // This ensures wrapped ∈ [lo, hi)
    T hi = lo + L;
    if (wrapped >= hi) {
        wrapped = lo;
        // Don't increment k here - the floor already handled it correctly
    }
    // Similarly if wrapped < lo due to floating point
    if (wrapped < lo) {
        wrapped = lo;
    }

    return wrapped;
}

/**
 * @brief Fast periodic wrap assuming displacement crosses at most 1 domain.
 *
 * Use this when |dp| << L (typical for RWPT). Falls back to robust if needed.
 *
 * @param pos Current position (assumed in domain)
 * @param dp Displacement
 * @param lo Domain minimum
 * @param hi Domain maximum
 * @param L Domain length
 * @param[out] k Wrap delta (+1, -1, or 0)
 * @return Wrapped new position
 */
template <typename T>
__device__ __forceinline__ T wrap_periodic_fast(
    T pos, T dp, T lo, T hi, T L, int32_t& k
) {
    T new_pos = pos + dp;
    k = 0;

    if (new_pos >= hi) {
        new_pos -= L;
        k = 1;
        // Check if still outside (multi-domain crossing)
        if (new_pos >= hi || new_pos < lo) {
            // Fall back to robust
            return wrap_periodic_robust(pos + dp, lo, L, k);
        }
    } else if (new_pos < lo) {
        new_pos += L;
        k = -1;
        // Check if still outside
        if (new_pos < lo || new_pos >= hi) {
            return wrap_periodic_robust(pos + dp, lo, L, k);
        }
    }
    // else: still inside, k=0

    return new_pos;
}

// =============================================================================
// Combined BC application per axis
// =============================================================================

/**
 * @brief Result of applying boundary condition to one axis.
 */
template <typename T>
struct BCResult {
    T new_pos;      ///< New position after BC
    int32_t wrap_k; ///< Wrap delta (only for Periodic)
    bool exited;    ///< True if particle exited (Open BC)
};

/**
 * @brief Apply boundary condition for one axis.
 *
 * @param bc_lo BC type at low boundary
 * @param bc_hi BC type at high boundary
 * @param pos Current position
 * @param dp Proposed displacement
 * @param lo Domain minimum
 * @param hi Domain maximum
 * @return BCResult with new position, wrap count, and exit flag
 *
 * Logic:
 * - If both sides Periodic: wrap position, update wrap counter
 * - If Closed at exit side: reject displacement (legacy)
 * - If Open at exit side: allow exit, flag particle as exited
 *
 * CRITICAL: Legacy uses STRICT inequalities: lo < x && x < hi
 * This prevents particles from ever landing exactly on the boundary.
 * SOURCE: legacy/Geometry/CartesianGrid.cuh validX/validY/validZ
 * SOURCE: legacy/Particles/MoveParticle.cuh lines 197-200
 */
template <typename T>
__device__ __forceinline__ BCResult<T> apply_bc_axis(
    BoundaryType bc_lo,
    BoundaryType bc_hi,
    T pos, T dp,
    T lo, T hi
) {
    BCResult<T> result;
    result.wrap_k = 0;
    result.exited = false;

    T new_pos = pos + dp;
    T L = hi - lo;

    // Case 1: Periodic on both sides (symmetric periodic)
    // Note: Mixed periodic (one side periodic, other not) is unusual but
    // we treat full-axis periodic as the common case
    if (bc_lo == BoundaryType::Periodic && bc_hi == BoundaryType::Periodic) {
        result.new_pos = wrap_periodic_fast(pos, dp, lo, hi, L, result.wrap_k);
        return result;
    }

    // =========================================================================
    // EXACT LEGACY SEMANTICS - Do not modify without careful analysis
    // =========================================================================
    // Legacy code (MoveParticle.cuh:197):
    //   thrust::get<0>(p) += (grid::validX(grid, thrust::get<0>(p) + dpx) ? dpx : 0);
    // 
    // Legacy validX (CartesianGrid.cuh:91):
    //   return g.px < x && x < g.px + g.dx*g.nx;
    //
    // This means: accept new_pos IFF (lo < new_pos AND new_pos < hi)
    // If invalid, do NOT apply displacement (return original pos)
    // =========================================================================

    // Check validity using STRICT inequalities (exactly like legacy)
    bool valid = (lo < new_pos) && (new_pos < hi);

    // DEBUG: Print when we're near the high boundary
    #if 0  // Enable for debugging
    if (new_pos > hi - T(1.0) && new_pos < hi + T(1.0)) {
        printf("BC DEBUG: pos=%.6f dp=%.6f new_pos=%.6f hi=%.6f valid=%d\n",
               pos, dp, new_pos, hi, valid ? 1 : 0);
    }
    #endif

    if (valid) {
        // Inside domain with strict inequalities - accept the new position
        result.new_pos = new_pos;
        return result;
    }

    // Invalid position - determine which boundary was crossed
    if (new_pos <= lo) {
        // Hit or crossed low boundary
        if (bc_lo == BoundaryType::Open) {
            result.new_pos = pos;  // Keep old position for exit tracking
            result.exited = true;
        } else {
            // Closed: reject displacement, keep old position (legacy behavior)
            result.new_pos = pos;
        }
    } else {
        // Hit or crossed high boundary (new_pos >= hi)
        if (bc_hi == BoundaryType::Open) {
            result.new_pos = pos;  // Keep old position for exit tracking
            result.exited = true;
        } else {
            // Closed: reject displacement, keep old position (legacy behavior)
            result.new_pos = pos;
        }
    }

    return result;
}

} // namespace internal
} // namespace par2

#endif // PAR2_INTERNAL_BOUNDARY_HELPERS_CUH
