/**
 * @file indexing.cuh
 * @brief Grid indexing utilities for CUDA kernels.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_INTERNAL_GRID_INDEXING_CUH
#define PAR2_INTERNAL_GRID_INDEXING_CUH

#include <par2_core/grid.hpp>
#include <cuda_runtime.h>

namespace par2 {
namespace grid {

/**
 * @brief Compute linear cell index from 3D indices.
 */
template <typename T>
__host__ __device__ __forceinline__
int cell_index(const GridDesc<T>& g, int ix, int iy, int iz) {
    return iz * g.ny * g.nx + iy * g.nx + ix;
}

/**
 * @brief Compute corner/face field index.
 */
template <typename T>
__host__ __device__ __forceinline__
int corner_index(const GridDesc<T>& g, int ix, int iy, int iz) {
    return iz * (g.ny + 1) * (g.nx + 1) + iy * (g.nx + 1) + ix;
}

/**
 * @brief Get 3D cell indices from position.
 */
template <typename T>
__host__ __device__ __forceinline__
void position_to_cell(const GridDesc<T>& g, T x, T y, T z, int& ix, int& iy, int& iz) {
    ix = static_cast<int>((x - g.px) / g.dx);
    iy = static_cast<int>((y - g.py) / g.dy);
    iz = static_cast<int>((z - g.pz) / g.dz);
}

/**
 * @brief Check if cell indices are valid.
 */
template <typename T>
__host__ __device__ __forceinline__
bool valid_cell(const GridDesc<T>& g, int ix, int iy, int iz) {
    return ix >= 0 && ix < g.nx &&
           iy >= 0 && iy < g.ny &&
           iz >= 0 && iz < g.nz;
}

} // namespace grid
} // namespace par2

#endif // PAR2_INTERNAL_GRID_INDEXING_CUH
