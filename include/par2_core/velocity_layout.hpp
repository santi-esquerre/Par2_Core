/**
 * @file velocity_layout.hpp
 * @brief Velocity field layout contracts and views.
 *
 * This header defines the **canonical velocity field layouts** for Par2_Core,
 * documenting their memory organization, indexing schemes, and expected usage.
 *
 * ## Terminology
 *
 * | Term          | Description                                              |
 * |---------------|----------------------------------------------------------|
 * | Face Field    | Values at cell face centers (staggered MAC grid)         |
 * | Corner Field  | Values at grid vertices/corners                          |
 * | Merged Index  | Linear index for 3D array: `iz*(ny+1)*(nx+1) + iy*(nx+1) + ix` |
 *
 * ## Legacy PAR² Layout (what we inherit)
 *
 * Legacy PAR² uses a "FaceField" with **size (nx+1)×(ny+1)×(nz+1)** for each
 * velocity component (U, V, W). Despite the name, this is actually the size
 * needed for a **corner field** on an nx×ny×nz cell grid.
 *
 * The indexing uses `mergeId(ix, iy, iz)`:
 * ```
 * index = iz * (ny+1) * (nx+1) + iy * (nx+1) + ix
 * ```
 *
 * Face velocity at cell (ix, iy, iz) is accessed as:
 * - U at right face (+X): `U[mergeId(ix+1, iy, iz)]`
 * - U at left face  (-X): `U[mergeId(ix,   iy, iz)]`
 * - V at front face (+Y): `V[mergeId(ix, iy+1, iz)]`
 * - etc.
 *
 * ## Par2_Core Naming Convention
 *
 * To avoid confusion, we use explicit names:
 *
 * | View Type               | Size           | Meaning                         |
 * |-------------------------|----------------|----------------------------------|
 * | `FaceFieldView<T>`      | (nx+1)(ny+1)(nz+1) | Staggered face velocities (legacy layout) |
 * | `CornerFieldView<T>`    | (nx+1)(ny+1)(nz+1) | Corner-centered velocities      |
 *
 * The existing `VelocityView<T>` is an alias for `FaceFieldView<T>`.
 * The existing `CornerVelocityView<T>` is an alias for `CornerFieldView<T>`.
 *
 * @note Both have the same array size but **different semantic meaning**:
 *       - FaceField: value at face center (U at x-faces, V at y-faces, W at z-faces)
 *       - CornerField: value interpolated to the corner/vertex
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_VELOCITY_LAYOUT_HPP
#define PAR2_CORE_VELOCITY_LAYOUT_HPP

#include "grid.hpp"
#include <cstddef>
#include <cstdint>

namespace par2 {

// =============================================================================
// Layout Constants and Utilities
// =============================================================================

/**
 * @brief Compute the merged linear index for a 3D point in legacy layout.
 *
 * @param ix X index (0 to nx inclusive)
 * @param iy Y index (0 to ny inclusive)
 * @param iz Z index (0 to nz inclusive)
 * @param nx Number of cells in X
 * @param ny Number of cells in Y
 * @return Linear index for arrays of size (nx+1)*(ny+1)*(nz+1)
 *
 * This matches legacy PAR² `facefield::mergeId()`:
 * ```
 * index = iz * (ny+1) * (nx+1) + iy * (nx+1) + ix
 * ```
 */
constexpr inline size_t merge_id(int ix, int iy, int iz, int nx, int ny) noexcept {
    return static_cast<size_t>(iz) * static_cast<size_t>(ny + 1) * static_cast<size_t>(nx + 1)
         + static_cast<size_t>(iy) * static_cast<size_t>(nx + 1)
         + static_cast<size_t>(ix);
}

/**
 * @brief Compute the expected array size for a face/corner field.
 *
 * @param nx Number of cells in X
 * @param ny Number of cells in Y
 * @param nz Number of cells in Z
 * @return (nx+1) * (ny+1) * (nz+1)
 */
constexpr inline size_t field_size(int nx, int ny, int nz) noexcept {
    return static_cast<size_t>(nx + 1) * static_cast<size_t>(ny + 1) * static_cast<size_t>(nz + 1);
}

/**
 * @brief Compute expected field size from grid descriptor.
 */
template <typename T>
constexpr inline size_t field_size(const GridDesc<T>& grid) noexcept {
    return field_size(grid.nx, grid.ny, grid.nz);
}

// =============================================================================
// FaceFieldView - Staggered velocity on faces (legacy layout)
// =============================================================================

/**
 * @brief Non-owning view of a staggered velocity field (MAC grid layout).
 *
 * @tparam T Floating point type (float or double)
 *
 * ## Memory Layout
 *
 * Each component (U, V, W) is stored as a flat array with:
 * - **Size:** `(nx+1) * (ny+1) * (nz+1)` elements
 * - **Indexing:** `merge_id(ix, iy, iz, nx, ny)`
 *
 * ## Semantic Meaning
 *
 * Despite using corner-point indexing, these represent **face-centered** values:
 *
 * | Component | Physical Location                    | Access for cell (cx, cy, cz) |
 * |-----------|--------------------------------------|------------------------------|
 * | U         | X-face centers (yz plane)            | `U[merge_id(cx, cy, cz)]` = left, `U[merge_id(cx+1, cy, cz)]` = right |
 * | V         | Y-face centers (xz plane)            | `V[merge_id(cx, cy, cz)]` = front, `V[merge_id(cx, cy+1, cz)]` = back |
 * | W         | Z-face centers (xy plane)            | `W[merge_id(cx, cy, cz)]` = bottom, `W[merge_id(cx, cy, cz+1)]` = top |
 *
 * ## Valid Index Ranges
 *
 * - ix: 0 to nx (inclusive) - nx+1 values per row
 * - iy: 0 to ny (inclusive) - ny+1 values per column
 * - iz: 0 to nz (inclusive) - nz+1 values per layer
 *
 * ## Usage Example
 *
 * ```cpp
 * // Allocate on GPU
 * size_t sz = par2::field_size(grid);
 * T* d_U, *d_V, *d_W;
 * cudaMalloc(&d_U, sz * sizeof(T));
 * cudaMalloc(&d_V, sz * sizeof(T));
 * cudaMalloc(&d_W, sz * sizeof(T));
 *
 * // Create view
 * par2::FaceFieldView<T> vel{d_U, d_V, d_W, sz};
 *
 * // Bind to engine
 * engine.bind_velocity(vel);
 * ```
 *
 * @note This is the **input format** expected by `TransportEngine::bind_velocity()`.
 * @note Legacy PAR² calls this "FaceField" and uses `facefield::mergeId()`.
 */
template <typename T>
struct FaceFieldView {
    const T* U = nullptr;  ///< X-component (face-centered on YZ planes) [device ptr]
    const T* V = nullptr;  ///< Y-component (face-centered on XZ planes) [device ptr]
    const T* W = nullptr;  ///< Z-component (face-centered on XY planes) [device ptr]
    size_t size = 0;       ///< Number of elements per component: (nx+1)(ny+1)(nz+1)

    /// Check if all pointers are set and size is valid
    constexpr bool valid() const noexcept {
        return (size == 0) || (U != nullptr && V != nullptr && W != nullptr);
    }

    /// Check if size matches expected for given grid
    template <typename GridT>
    constexpr bool matches_grid(const GridDesc<GridT>& grid) const noexcept {
        return size == field_size(grid);
    }
};

// =============================================================================
// CornerFieldView - Velocity at grid corners
// =============================================================================

/**
 * @brief Non-owning view of a corner-centered velocity field.
 *
 * @tparam T Floating point type
 *
 * ## Memory Layout
 *
 * Same as FaceFieldView:
 * - **Size:** `(nx+1) * (ny+1) * (nz+1)` elements
 * - **Indexing:** `merge_id(ix, iy, iz, nx, ny)`
 *
 * ## Semantic Meaning
 *
 * All components (Uc, Vc, Wc) represent velocity **at grid vertices/corners**:
 *
 * | Index (ix, iy, iz) | Physical Location               |
 * |--------------------|----------------------------------|
 * | (0, 0, 0)          | Origin corner                   |
 * | (nx, ny, nz)       | Far corner                      |
 * | (ix, iy, iz)       | Corner at (x0+ix*dx, y0+iy*dy, z0+iz*dz) |
 *
 * ## Computation from FaceField
 *
 * Corner velocity is typically computed by averaging adjacent face values:
 * ```
 * Uc[ix,iy,iz] = average of U values at 4 adjacent X-faces
 * Vc[ix,iy,iz] = average of V values at 4 adjacent Y-faces
 * Wc[ix,iy,iz] = average of W values at 4 adjacent Z-faces
 * ```
 *
 * The engine computes this automatically via `update_derived_fields()`.
 * Users can also provide pre-computed corner velocities via `bind_corner_velocity()`.
 *
 * ## Usage
 *
 * - **Internal:** Engine computes corner velocities from face field
 * - **External:** User provides pre-computed corner velocities (e.g., from MODFLOW)
 *
 * @note This is used for **trilinear velocity interpolation** within cells.
 */
template <typename T>
struct CornerFieldView {
    const T* Uc = nullptr;  ///< X-component at corners [device ptr]
    const T* Vc = nullptr;  ///< Y-component at corners [device ptr]
    const T* Wc = nullptr;  ///< Z-component at corners [device ptr]
    size_t size = 0;        ///< Number of elements per component

    constexpr bool valid() const noexcept {
        return (size == 0) || (Uc != nullptr && Vc != nullptr && Wc != nullptr);
    }

    template <typename GridT>
    constexpr bool matches_grid(const GridDesc<GridT>& grid) const noexcept {
        return size == field_size(grid);
    }
};

// =============================================================================
// Type Aliases for API Compatibility
// =============================================================================

/**
 * @brief Alias: VelocityView is a FaceFieldView.
 *
 * The name `VelocityView` is kept for backwards compatibility.
 * New code should prefer `FaceFieldView` for clarity.
 *
 * @deprecated Use FaceFieldView<T> in new code.
 */
template <typename T>
using VelocityView = FaceFieldView<T>;

/**
 * @brief Alias: CornerVelocityView is a CornerFieldView.
 *
 * @deprecated Use CornerFieldView<T> in new code.
 */
template <typename T>
using CornerVelocityView = CornerFieldView<T>;

// =============================================================================
// Validation Helpers (debug builds only)
// =============================================================================

#ifdef PAR2_CORE_DEBUG

/**
 * @brief Validate face field view against grid (debug only).
 *
 * @throws std::invalid_argument if validation fails
 */
template <typename T, typename GridT>
inline void validate_face_field(const FaceFieldView<T>& field, const GridDesc<GridT>& grid) {
    if (!field.valid()) {
        throw std::invalid_argument("FaceFieldView: invalid (null pointers with non-zero size)");
    }
    if (!field.matches_grid(grid)) {
        throw std::invalid_argument("FaceFieldView: size mismatch with grid");
    }
}

/**
 * @brief Validate corner field view against grid (debug only).
 */
template <typename T, typename GridT>
inline void validate_corner_field(const CornerFieldView<T>& field, const GridDesc<GridT>& grid) {
    if (!field.valid()) {
        throw std::invalid_argument("CornerFieldView: invalid (null pointers with non-zero size)");
    }
    if (!field.matches_grid(grid)) {
        throw std::invalid_argument("CornerFieldView: size mismatch with grid");
    }
}

#else

// No-op in release builds
template <typename T, typename GridT>
inline void validate_face_field(const FaceFieldView<T>&, const GridDesc<GridT>&) {}

template <typename T, typename GridT>
inline void validate_corner_field(const CornerFieldView<T>&, const GridDesc<GridT>&) {}

#endif // PAR2_CORE_DEBUG

} // namespace par2

#endif // PAR2_CORE_VELOCITY_LAYOUT_HPP
