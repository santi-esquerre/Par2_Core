/**
 * @file boundary.hpp
 * @brief Boundary condition configuration for Par2_Core.
 *
 * Defines per-axis boundary conditions for particle tracking.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_BOUNDARY_HPP
#define PAR2_CORE_BOUNDARY_HPP

#include "types.hpp"

namespace par2 {

/**
 * @brief Boundary conditions for a single axis.
 *
 * @tparam T Floating point type
 *
 * Specifies behavior at both the low (min) and high (max) boundaries.
 * For example, a pipe might have Periodic in X, Closed in Y and Z.
 *
 * @warning For Periodic BC, **both** lo and hi must be Periodic
 *          (symmetric periodic).  Mixed periodic (e.g., lo=Periodic,
 *          hi=Closed) is treated as **non-periodic** for that axis.
 *
 * @note Domain validity uses strict inequalities: lo < x < hi
 *       (matching legacy PAR² CartesianGrid::validX/Y/Z).
 */
template <typename T>
struct AxisBoundary {
    BoundaryType lo = BoundaryType::Closed;  ///< Behavior at min boundary
    BoundaryType hi = BoundaryType::Closed;  ///< Behavior at max boundary

    /// Factory for symmetric boundary (same on both sides)
    static constexpr AxisBoundary symmetric(BoundaryType type) noexcept {
        return AxisBoundary{type, type};
    }

    /// Factory for closed (reflective) boundaries
    static constexpr AxisBoundary closed() noexcept {
        return symmetric(BoundaryType::Closed);
    }

    /// Factory for periodic boundaries
    static constexpr AxisBoundary periodic() noexcept {
        return symmetric(BoundaryType::Periodic);
    }

    /// Factory for open boundaries (particles can exit)
    static constexpr AxisBoundary open() noexcept {
        return symmetric(BoundaryType::Open);
    }
};

/**
 * @brief Complete boundary configuration for 3D domain.
 *
 * @tparam T Floating point type
 *
 * Combines boundary conditions for all three axes (six faces total).
 * Default is closed (reflective) on all boundaries, matching legacy
 * PAR² behavior.
 *
 * ## Tracking array requirements
 *
 * - If any axis is **Open**, a status array (uint8_t per particle)
 *   is required.  Auto-allocated by prepare() / ensure_tracking_arrays().
 * - If any axis is **Periodic**, wrap counters (int32_t per particle
 *   per periodic axis) are required.  Auto-allocated similarly.
 *
 * @see TransportEngine::prepare(), TransportEngine::ensure_tracking_arrays()
 */
template <typename T>
struct BoundaryConfig {
    AxisBoundary<T> x = AxisBoundary<T>::closed();  ///< X-axis boundaries
    AxisBoundary<T> y = AxisBoundary<T>::closed();  ///< Y-axis boundaries
    AxisBoundary<T> z = AxisBoundary<T>::closed();  ///< Z-axis boundaries

    // =========================================================================
    // Factory methods for common configurations
    // =========================================================================

    /// All boundaries closed (reflective) - legacy default
    static constexpr BoundaryConfig all_closed() noexcept {
        return BoundaryConfig{
            AxisBoundary<T>::closed(),
            AxisBoundary<T>::closed(),
            AxisBoundary<T>::closed()
        };
    }

    /// All boundaries periodic
    static constexpr BoundaryConfig all_periodic() noexcept {
        return BoundaryConfig{
            AxisBoundary<T>::periodic(),
            AxisBoundary<T>::periodic(),
            AxisBoundary<T>::periodic()
        };
    }

    /// Periodic in X, closed in Y and Z (typical for channel flow)
    static constexpr BoundaryConfig channel_x() noexcept {
        return BoundaryConfig{
            AxisBoundary<T>::periodic(),
            AxisBoundary<T>::closed(),
            AxisBoundary<T>::closed()
        };
    }

    /// Open inlet (X-), open outlet (X+), closed lateral
    static constexpr BoundaryConfig open_x() noexcept {
        return BoundaryConfig{
            AxisBoundary<T>::open(),
            AxisBoundary<T>::closed(),
            AxisBoundary<T>::closed()
        };
    }

    // =========================================================================
    // Query methods
    // =========================================================================

    /// Check if any boundary is periodic (affects domain wrapping)
    constexpr bool has_periodic() const noexcept {
        return x.lo == BoundaryType::Periodic || x.hi == BoundaryType::Periodic ||
               y.lo == BoundaryType::Periodic || y.hi == BoundaryType::Periodic ||
               z.lo == BoundaryType::Periodic || z.hi == BoundaryType::Periodic;
    }

    /// Check if any boundary is open (need to track exited particles)
    constexpr bool has_open() const noexcept {
        return x.lo == BoundaryType::Open || x.hi == BoundaryType::Open ||
               y.lo == BoundaryType::Open || y.hi == BoundaryType::Open ||
               z.lo == BoundaryType::Open || z.hi == BoundaryType::Open;
    }
};

} // namespace par2

#endif // PAR2_CORE_BOUNDARY_HPP
