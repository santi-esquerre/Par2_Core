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
 * Combines boundary conditions for all three axes.
 * Default is closed (reflective) on all boundaries, matching legacy PAR² behavior.
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
