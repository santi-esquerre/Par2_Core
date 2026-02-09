/**
 * @file grid.hpp
 * @brief Grid descriptor for Par2_Core.
 *
 * Defines the computational grid geometry without storing any data.
 * This is a lightweight POD struct that can be passed by value to kernels.
 *
 * @note Layout convention follows legacy PAR²:
 *       - Cell indexing: id = idz * (ny * nx) + idy * nx + idx
 *       - Face/corner fields: (nx+1) * (ny+1) * (nz+1) elements
 *       - Origin (px,py,pz) is the corner of cell (0,0,0), not its center
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_GRID_HPP
#define PAR2_CORE_GRID_HPP

#include <cstdint>

namespace par2 {

/**
 * @brief Cartesian grid descriptor.
 *
 * @tparam T Floating point type (float or double)
 *
 * Describes a uniform Cartesian grid with:
 * - nx × ny × nz cells
 * - Cell sizes dx, dy, dz (uniform per axis)
 * - Origin at (px, py, pz) — this is the **corner** of cell (0,0,0),
 *   not the cell center
 *
 * The domain spans:
 * - X: [px, px + nx*dx]
 * - Y: [py, py + ny*dy]
 * - Z: [pz, pz + nz*dz]
 *
 * For 2D simulations, set nz=1.  The kernel forces Z displacement to zero
 * and skips Z boundary checks when nz==1.
 *
 * @note This is a lightweight POD struct that can be passed by value to
 *       CUDA kernels (no pointers, no dynamic memory).
 */
template <typename T>
struct GridDesc {
    // Cell counts
    int nx = 1;  ///< Number of cells in X direction
    int ny = 1;  ///< Number of cells in Y direction
    int nz = 1;  ///< Number of cells in Z direction (1 for 2D)

    // Cell sizes
    T dx = T(1);  ///< Cell size in X direction
    T dy = T(1);  ///< Cell size in Y direction
    T dz = T(1);  ///< Cell size in Z direction

    // Origin (corner of cell [0,0,0])
    T px = T(0);  ///< X-coordinate of origin
    T py = T(0);  ///< Y-coordinate of origin
    T pz = T(0);  ///< Z-coordinate of origin

    // =========================================================================
    // Convenience methods (all constexpr, usable on host and device)
    // =========================================================================

    /// Total number of cells
    constexpr int num_cells() const noexcept {
        return nx * ny * nz;
    }

    /// Size of face/corner field arrays: (nx+1)*(ny+1)*(nz+1)
    constexpr int num_corners() const noexcept {
        return (nx + 1) * (ny + 1) * (nz + 1);
    }

    /// Domain extent in X
    constexpr T length_x() const noexcept { return T(nx) * dx; }

    /// Domain extent in Y
    constexpr T length_y() const noexcept { return T(ny) * dy; }

    /// Domain extent in Z
    constexpr T length_z() const noexcept { return T(nz) * dz; }

    /// Volume of a single cell
    constexpr T cell_volume() const noexcept { return dx * dy * dz; }

    /// Total domain volume
    constexpr T domain_volume() const noexcept {
        return length_x() * length_y() * length_z();
    }

    /// Check if this is a 2D grid (nz == 1)
    constexpr bool is_2d() const noexcept { return nz == 1; }

    /// Maximum coordinate in X
    constexpr T x_max() const noexcept { return px + length_x(); }

    /// Maximum coordinate in Y
    constexpr T y_max() const noexcept { return py + length_y(); }

    /// Maximum coordinate in Z
    constexpr T z_max() const noexcept { return pz + length_z(); }
};

// =============================================================================
// Factory functions
// =============================================================================

/**
 * @brief Create a uniform grid descriptor.
 *
 * @param nx,ny,nz Number of cells in each direction
 * @param dx,dy,dz Cell sizes
 * @param px,py,pz Origin coordinates (default: 0,0,0)
 */
template <typename T>
constexpr GridDesc<T> make_grid(
    int nx, int ny, int nz,
    T dx, T dy, T dz,
    T px = T(0), T py = T(0), T pz = T(0)
) noexcept {
    return GridDesc<T>{nx, ny, nz, dx, dy, dz, px, py, pz};
}

/**
 * @brief Create a uniform grid with same cell size in all directions.
 */
template <typename T>
constexpr GridDesc<T> make_uniform_grid(
    int nx, int ny, int nz,
    T cell_size,
    T px = T(0), T py = T(0), T pz = T(0)
) noexcept {
    return GridDesc<T>{nx, ny, nz, cell_size, cell_size, cell_size, px, py, pz};
}

} // namespace par2

#endif // PAR2_CORE_GRID_HPP
