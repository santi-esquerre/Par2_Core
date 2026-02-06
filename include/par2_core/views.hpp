/**
 * @file views.hpp
 * @brief Lightweight view types for GPU data (zero-copy binding).
 *
 * These structs hold device pointers and sizes but do NOT own the memory.
 * They enable the engine to operate on user-allocated GPU buffers directly,
 * without copying data.
 *
 * @note NO Thrust headers here - just raw pointers and sizes.
 * @note All pointers must point to device memory (cudaMalloc, etc.)
 *
 * @see velocity_layout.hpp for detailed velocity field layout documentation.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_VIEWS_HPP
#define PAR2_CORE_VIEWS_HPP

#include "velocity_layout.hpp"  // FaceFieldView, CornerFieldView, VelocityView, CornerVelocityView

#include <cstddef>
#include <cstdint>
#include <cassert>

namespace par2 {

// =============================================================================
// DeviceSpan - Generic contiguous GPU memory view
// =============================================================================

/**
 * @brief Non-owning view of a contiguous GPU array.
 *
 * @tparam T Element type (may be const-qualified for read-only views)
 *
 * Similar to std::span but for device memory. Does not manage lifetime.
 */
template <typename T>
struct DeviceSpan {
    T* data = nullptr;   ///< Pointer to device memory
    size_t size = 0;     ///< Number of elements

    /// Check if the span is empty
    constexpr bool empty() const noexcept { return size == 0; }

    /// Check if the span is valid (non-null when size > 0)
    constexpr bool valid() const noexcept {
        return size == 0 || data != nullptr;
    }
};

/// Convenience alias for const views
template <typename T>
using ConstDeviceSpan = DeviceSpan<const T>;

// NOTE: VelocityView and CornerVelocityView are defined in velocity_layout.hpp
// They are imported into this namespace via that include.
// - VelocityView<T> = FaceFieldView<T> (staggered face velocities)
// - CornerVelocityView<T> = CornerFieldView<T> (corner-centered velocities)

// =============================================================================
// ParticlesView - SoA layout for particle positions
// =============================================================================

/**
 * @brief View of particle positions in Structure-of-Arrays (SoA) layout.
 *
 * @tparam T Floating point type (float or double)
 *
 * Particle positions are stored as three separate arrays for better
 * memory coalescing on GPU:
 * - x[i], y[i], z[i] = position of particle i
 *
 * Optionally includes:
 * - status: particle active/inactive flags
 * - wrapX/Y/Z: wrap counters for periodic boundary (net crossings)
 *
 * @note Memory is NOT owned by this view - user is responsible for allocation.
 * @note For inject_box, arrays must be writable (non-const pointers).
 *
 * ## Wrap counters (Periodic BC)
 *
 * When an axis has Periodic BC, particles wrap around the domain.
 * To recover the continuous (unwrapped) position for statistics/export:
 *
 *   x_unwrap = x + wrapX * Lx
 *
 * The wrap counters track net domain crossings (can be negative).
 * They are optional - if nullptr, wrapping still works but unwrap-on-demand
 * is not available for that axis.
 */
template <typename T>
struct ParticlesView {
    T* x = nullptr;         ///< X positions [device ptr, writable]
    T* y = nullptr;         ///< Y positions [device ptr, writable]
    T* z = nullptr;         ///< Z positions [device ptr, writable]
    int n = 0;              ///< Number of particles

    // Status tracking (required for Open BC)
    uint8_t* status = nullptr;  ///< Particle status flags [device ptr, optional]

    // Wrap counters for periodic BC (optional)
    int32_t* wrapX = nullptr;   ///< X wrap count [device ptr, optional]
    int32_t* wrapY = nullptr;   ///< Y wrap count [device ptr, optional]
    int32_t* wrapZ = nullptr;   ///< Z wrap count [device ptr, optional]

    /// Check basic validity
    constexpr bool valid() const noexcept {
        return (n == 0) || (x != nullptr && y != nullptr && z != nullptr);
    }

    /// Check if status tracking is enabled
    constexpr bool has_status() const noexcept {
        return status != nullptr;
    }

    /// Check if wrap tracking is available for an axis
    constexpr bool has_wrap_x() const noexcept { return wrapX != nullptr; }
    constexpr bool has_wrap_y() const noexcept { return wrapY != nullptr; }
    constexpr bool has_wrap_z() const noexcept { return wrapZ != nullptr; }

    /// Check if any wrap tracking is enabled
    constexpr bool has_any_wrap() const noexcept {
        return has_wrap_x() || has_wrap_y() || has_wrap_z();
    }
};

/**
 * @brief Read-only view of particle positions.
 *
 * Use this when you only need to read particle data (e.g., for statistics).
 */
template <typename T>
struct ConstParticlesView {
    const T* x = nullptr;
    const T* y = nullptr;
    const T* z = nullptr;
    int n = 0;
    const uint8_t* status = nullptr;
    const int32_t* wrapX = nullptr;
    const int32_t* wrapY = nullptr;
    const int32_t* wrapZ = nullptr;

    constexpr bool valid() const noexcept {
        return (n == 0) || (x != nullptr && y != nullptr && z != nullptr);
    }

    /// Implicit conversion from mutable view
    ConstParticlesView() = default;
    ConstParticlesView(const ParticlesView<T>& v)
        : x(v.x), y(v.y), z(v.z), n(v.n), status(v.status),
          wrapX(v.wrapX), wrapY(v.wrapY), wrapZ(v.wrapZ) {}
};

// =============================================================================
// UnwrappedPositionsView - Output for unwrapped position computation
// =============================================================================

/**
 * @brief View of unwrapped (continuous) particle positions.
 *
 * @tparam T Floating point type
 *
 * Used as output for compute_unwrapped_positions():
 *   x_u = x + wrapX * Lx
 *   y_u = y + wrapY * Ly
 *   z_u = z + wrapZ * Lz
 *
 * The user must allocate these arrays before calling compute_unwrapped_positions.
 * Size must be >= num_particles.
 */
template <typename T>
struct UnwrappedPositionsView {
    T* x_u = nullptr;  ///< Unwrapped X positions [device ptr, writable]
    T* y_u = nullptr;  ///< Unwrapped Y positions [device ptr, writable]
    T* z_u = nullptr;  ///< Unwrapped Z positions [device ptr, writable]
    int capacity = 0;  ///< Allocated size (must be >= n_particles)

    constexpr bool valid() const noexcept {
        return (capacity == 0) || (x_u != nullptr && y_u != nullptr && z_u != nullptr);
    }
};

// =============================================================================
// DriftCorrectionView - Precomputed div(D) field
// =============================================================================

/**
 * @brief View of precomputed drift correction field.
 *
 * @tparam T Floating point type
 *
 * This is the divergence of the dispersion tensor: div(D).
 * Can be cell-centered or corner-centered depending on configuration.
 */
template <typename T>
struct DriftCorrectionView {
    const T* dcx = nullptr;  ///< X-component of div(D) [device ptr]
    const T* dcy = nullptr;  ///< Y-component of div(D) [device ptr]
    const T* dcz = nullptr;  ///< Z-component of div(D) [device ptr]
    size_t size = 0;         ///< Number of elements per component

    constexpr bool valid() const noexcept {
        return (size == 0) || (dcx != nullptr && dcy != nullptr && dcz != nullptr);
    }
};

} // namespace par2

#endif // PAR2_CORE_VIEWS_HPP
