/**
 * @file par2_core.hpp
 * @brief Par2_Core umbrella header - includes all public API.
 *
 * Include this single header to access the complete Par2_Core API.
 *
 * @copyright Par2_Core - GPU-native transport engine
 *            Based on PAR² by Calogero B. Rizzo
 *
 * @example
 * ```cpp
 * #include <par2_core/par2_core.hpp>
 *
 * int main() {
 *     // Create grid: 100x100x10 cells, 1m spacing
 *     auto grid = par2::make_grid<double>(100, 100, 10, 1.0, 1.0, 1.0);
 *
 *     // Transport parameters
 *     par2::TransportParams<double> params{1e-9, 0.1, 0.01};
 *
 *     // Boundary conditions (closed on all sides)
 *     auto bc = par2::BoundaryConfig<double>::all_closed();
 *
 *     // Create engine
 *     par2::TransportEngine<double> engine(grid, params, bc);
 *
 *     // Bind GPU data and run simulation...
 * }
 * ```
 */

#ifndef PAR2_CORE_HPP
#define PAR2_CORE_HPP

// Core types
#include "types.hpp"
#include "grid.hpp"
#include "velocity_layout.hpp"  // Velocity field layout contracts (FaceFieldView, CornerFieldView)
#include "views.hpp"            // Other views (ParticlesView, DeviceSpan, etc.)
#include "boundary.hpp"

// Main engine
#include "transport_engine.hpp"

// Utilities
#include "injectors.hpp"

// I/O helpers (optional, for export/download)
#include "io.hpp"

// Statistics (optional, for analysis)
#include "stats.hpp"

// Version information
#include "version.hpp"

#endif // PAR2_CORE_HPP
