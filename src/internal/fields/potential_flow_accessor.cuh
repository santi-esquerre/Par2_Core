/**
 * @file potential_flow_accessor.cuh
 * @brief KH potential-flow velocity reconstruction from cell-centered K/head.
 */

#ifndef PAR2_CORE_INTERNAL_POTENTIAL_FLOW_ACCESSOR_CUH
#define PAR2_CORE_INTERNAL_POTENTIAL_FLOW_ACCESSOR_CUH

#include <par2_core/grid.hpp>
#include <par2_core/views.hpp>

#include "cornerfield_accessor.cuh"

#include <cmath>

namespace par2 {
namespace internal {

template <typename T>
__host__ __device__ __forceinline__
int wrap_index(int i, int n) {
    int r = i % n;
    return r < 0 ? r + n : r;
}

template <typename T>
__host__ __device__ __forceinline__
int clamp_index(int i, int n) {
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

template <typename T>
__host__ __device__ __forceinline__
int scalar_index(const GridDesc<T>& g, int i, int j, int k) {
    return i + g.nx * (j + g.ny * k);
}

template <typename T>
__host__ __device__ __forceinline__
bool is_periodic_axis(const ScalarAxisBoundary<T>& axis) {
    return axis.lo.type == ScalarBoundaryType::Periodic &&
           axis.hi.type == ScalarBoundaryType::Periodic;
}

template <typename T>
__host__ __device__ __forceinline__
void cell_interp_axis(T coord, T origin, T spacing, int n, bool periodic,
                      int& i0, int& i1, T& t) {
    const T lattice = (coord - origin) / spacing - T(0.5);
    int base = static_cast<int>(floor(lattice));
    t = lattice - static_cast<T>(base);

    if (periodic) {
        i0 = wrap_index<T>(base, n);
        i1 = wrap_index<T>(base + 1, n);
        return;
    }

    if (base < 0) {
        i0 = 0;
        i1 = 0;
        t = T(0);
    } else if (base >= n - 1) {
        i0 = n - 1;
        i1 = n - 1;
        t = T(0);
    } else {
        i0 = base;
        i1 = base + 1;
    }
}

template <typename T>
__device__ __forceinline__
T cell_value(const T* __restrict__ field, const GridDesc<T>& g, int i, int j, int k) {
    return field[scalar_index(g, i, j, k)];
}

template <typename T>
__device__ __forceinline__
T sample_cell_trilinear(const T* __restrict__ field, const GridDesc<T>& g,
                        const PotentialBoundaryConfig<T>& bc,
                        T x, T y, T z) {
    int i0, i1, j0, j1, k0, k1;
    T tx, ty, tz;
    cell_interp_axis(x, g.px, g.dx, g.nx, is_periodic_axis(bc.x), i0, i1, tx);
    cell_interp_axis(y, g.py, g.dy, g.ny, is_periodic_axis(bc.y), j0, j1, ty);
    cell_interp_axis(z, g.pz, g.dz, g.nz, is_periodic_axis(bc.z), k0, k1, tz);

    return trilinear(tx, ty, tz,
                     cell_value(field, g, i0, j0, k0),
                     cell_value(field, g, i1, j0, k0),
                     cell_value(field, g, i0, j1, k0),
                     cell_value(field, g, i1, j1, k0),
                     cell_value(field, g, i0, j0, k1),
                     cell_value(field, g, i1, j0, k1),
                     cell_value(field, g, i0, j1, k1),
                     cell_value(field, g, i1, j1, k1));
}

template <typename T>
__device__ __forceinline__
T grad_x_cell(const PotentialFlowView<T>& pf, const GridDesc<T>& g, int i, int j, int k) {
    const bool periodic = is_periodic_axis(pf.head_bc.x);
    const T* H = pf.head;

    if (g.nx == 1) {
        if (pf.head_bc.x.lo.type == ScalarBoundaryType::Dirichlet &&
            pf.head_bc.x.hi.type == ScalarBoundaryType::Dirichlet) {
            return (pf.head_bc.x.hi.value - pf.head_bc.x.lo.value) / g.dx;
        }
        return T(0);
    }

    if (periodic) {
        const int im = wrap_index<T>(i - 1, g.nx);
        const int ip = wrap_index<T>(i + 1, g.nx);
        return (cell_value(H, g, ip, j, k) - cell_value(H, g, im, j, k)) / (T(2) * g.dx);
    }

    if (i == 0) {
        if (pf.head_bc.x.lo.type == ScalarBoundaryType::Dirichlet) {
            return (cell_value(H, g, i, j, k) - pf.head_bc.x.lo.value) / (T(0.5) * g.dx);
        }
        if (pf.head_bc.x.lo.type == ScalarBoundaryType::Neumann) {
            return T(0);
        }
        return (cell_value(H, g, i + 1, j, k) - cell_value(H, g, i, j, k)) / g.dx;
    }

    if (i == g.nx - 1) {
        if (pf.head_bc.x.hi.type == ScalarBoundaryType::Dirichlet) {
            return (pf.head_bc.x.hi.value - cell_value(H, g, i, j, k)) / (T(0.5) * g.dx);
        }
        if (pf.head_bc.x.hi.type == ScalarBoundaryType::Neumann) {
            return T(0);
        }
        return (cell_value(H, g, i, j, k) - cell_value(H, g, i - 1, j, k)) / g.dx;
    }

    return (cell_value(H, g, i + 1, j, k) - cell_value(H, g, i - 1, j, k)) / (T(2) * g.dx);
}

template <typename T>
__device__ __forceinline__
T grad_y_cell(const PotentialFlowView<T>& pf, const GridDesc<T>& g, int i, int j, int k) {
    const bool periodic = is_periodic_axis(pf.head_bc.y);
    const T* H = pf.head;

    if (g.ny == 1) {
        return T(0);
    }

    if (periodic) {
        const int jm = wrap_index<T>(j - 1, g.ny);
        const int jp = wrap_index<T>(j + 1, g.ny);
        return (cell_value(H, g, i, jp, k) - cell_value(H, g, i, jm, k)) / (T(2) * g.dy);
    }

    if (j == 0) {
        if (pf.head_bc.y.lo.type == ScalarBoundaryType::Dirichlet) {
            return (cell_value(H, g, i, j, k) - pf.head_bc.y.lo.value) / (T(0.5) * g.dy);
        }
        if (pf.head_bc.y.lo.type == ScalarBoundaryType::Neumann) {
            return T(0);
        }
        return (cell_value(H, g, i, j + 1, k) - cell_value(H, g, i, j, k)) / g.dy;
    }

    if (j == g.ny - 1) {
        if (pf.head_bc.y.hi.type == ScalarBoundaryType::Dirichlet) {
            return (pf.head_bc.y.hi.value - cell_value(H, g, i, j, k)) / (T(0.5) * g.dy);
        }
        if (pf.head_bc.y.hi.type == ScalarBoundaryType::Neumann) {
            return T(0);
        }
        return (cell_value(H, g, i, j, k) - cell_value(H, g, i, j - 1, k)) / g.dy;
    }

    return (cell_value(H, g, i, j + 1, k) - cell_value(H, g, i, j - 1, k)) / (T(2) * g.dy);
}

template <typename T>
__device__ __forceinline__
T grad_z_cell(const PotentialFlowView<T>& pf, const GridDesc<T>& g, int i, int j, int k) {
    const bool periodic = is_periodic_axis(pf.head_bc.z);
    const T* H = pf.head;

    if (g.nz == 1) {
        return T(0);
    }

    if (periodic) {
        const int km = wrap_index<T>(k - 1, g.nz);
        const int kp = wrap_index<T>(k + 1, g.nz);
        return (cell_value(H, g, i, j, kp) - cell_value(H, g, i, j, km)) / (T(2) * g.dz);
    }

    if (k == 0) {
        if (pf.head_bc.z.lo.type == ScalarBoundaryType::Dirichlet) {
            return (cell_value(H, g, i, j, k) - pf.head_bc.z.lo.value) / (T(0.5) * g.dz);
        }
        if (pf.head_bc.z.lo.type == ScalarBoundaryType::Neumann) {
            return T(0);
        }
        return (cell_value(H, g, i, j, k + 1) - cell_value(H, g, i, j, k)) / g.dz;
    }

    if (k == g.nz - 1) {
        if (pf.head_bc.z.hi.type == ScalarBoundaryType::Dirichlet) {
            return (pf.head_bc.z.hi.value - cell_value(H, g, i, j, k)) / (T(0.5) * g.dz);
        }
        if (pf.head_bc.z.hi.type == ScalarBoundaryType::Neumann) {
            return T(0);
        }
        return (cell_value(H, g, i, j, k) - cell_value(H, g, i, j, k - 1)) / g.dz;
    }

    return (cell_value(H, g, i, j, k + 1) - cell_value(H, g, i, j, k - 1)) / (T(2) * g.dz);
}

template <typename T>
__device__ __forceinline__
T sample_grad_x_trilinear(const PotentialFlowView<T>& pf, const GridDesc<T>& g,
                          T x, T y, T z) {
    int i0, i1, j0, j1, k0, k1;
    T tx, ty, tz;
    cell_interp_axis(x, g.px, g.dx, g.nx, is_periodic_axis(pf.head_bc.x), i0, i1, tx);
    cell_interp_axis(y, g.py, g.dy, g.ny, is_periodic_axis(pf.head_bc.y), j0, j1, ty);
    cell_interp_axis(z, g.pz, g.dz, g.nz, is_periodic_axis(pf.head_bc.z), k0, k1, tz);

    return trilinear(tx, ty, tz,
                     grad_x_cell(pf, g, i0, j0, k0),
                     grad_x_cell(pf, g, i1, j0, k0),
                     grad_x_cell(pf, g, i0, j1, k0),
                     grad_x_cell(pf, g, i1, j1, k0),
                     grad_x_cell(pf, g, i0, j0, k1),
                     grad_x_cell(pf, g, i1, j0, k1),
                     grad_x_cell(pf, g, i0, j1, k1),
                     grad_x_cell(pf, g, i1, j1, k1));
}

template <typename T>
__device__ __forceinline__
T sample_grad_y_trilinear(const PotentialFlowView<T>& pf, const GridDesc<T>& g,
                          T x, T y, T z) {
    int i0, i1, j0, j1, k0, k1;
    T tx, ty, tz;
    cell_interp_axis(x, g.px, g.dx, g.nx, is_periodic_axis(pf.head_bc.x), i0, i1, tx);
    cell_interp_axis(y, g.py, g.dy, g.ny, is_periodic_axis(pf.head_bc.y), j0, j1, ty);
    cell_interp_axis(z, g.pz, g.dz, g.nz, is_periodic_axis(pf.head_bc.z), k0, k1, tz);

    return trilinear(tx, ty, tz,
                     grad_y_cell(pf, g, i0, j0, k0),
                     grad_y_cell(pf, g, i1, j0, k0),
                     grad_y_cell(pf, g, i0, j1, k0),
                     grad_y_cell(pf, g, i1, j1, k0),
                     grad_y_cell(pf, g, i0, j0, k1),
                     grad_y_cell(pf, g, i1, j0, k1),
                     grad_y_cell(pf, g, i0, j1, k1),
                     grad_y_cell(pf, g, i1, j1, k1));
}

template <typename T>
__device__ __forceinline__
T sample_grad_z_trilinear(const PotentialFlowView<T>& pf, const GridDesc<T>& g,
                          T x, T y, T z) {
    int i0, i1, j0, j1, k0, k1;
    T tx, ty, tz;
    cell_interp_axis(x, g.px, g.dx, g.nx, is_periodic_axis(pf.head_bc.x), i0, i1, tx);
    cell_interp_axis(y, g.py, g.dy, g.ny, is_periodic_axis(pf.head_bc.y), j0, j1, ty);
    cell_interp_axis(z, g.pz, g.dz, g.nz, is_periodic_axis(pf.head_bc.z), k0, k1, tz);

    return trilinear(tx, ty, tz,
                     grad_z_cell(pf, g, i0, j0, k0),
                     grad_z_cell(pf, g, i1, j0, k0),
                     grad_z_cell(pf, g, i0, j1, k0),
                     grad_z_cell(pf, g, i1, j1, k0),
                     grad_z_cell(pf, g, i0, j0, k1),
                     grad_z_cell(pf, g, i1, j0, k1),
                     grad_z_cell(pf, g, i0, j1, k1),
                     grad_z_cell(pf, g, i1, j1, k1));
}

template <typename T>
__device__ __forceinline__
void sample_velocity_kh_potential(const PotentialFlowView<T>& pf, const GridDesc<T>& g,
                                  T x, T y, T z, T& vx, T& vy, T& vz) {
    if (pf.K == nullptr || pf.head == nullptr || pf.size != static_cast<size_t>(g.num_cells()) ||
        x < g.px || x >= g.x_max()) {
        vx = vy = vz = T(0);
        return;
    }

    const T Kx = sample_cell_trilinear(pf.K, g, pf.head_bc, x, y, z);
    const T dhdx = sample_grad_x_trilinear(pf, g, x, y, z);
    const T dhdy = sample_grad_y_trilinear(pf, g, x, y, z);
    const T dhdz = sample_grad_z_trilinear(pf, g, x, y, z);

    vx = -Kx * dhdx;
    vy = -Kx * dhdy;
    vz = -Kx * dhdz;
}

} // namespace internal
} // namespace par2

#endif // PAR2_CORE_INTERNAL_POTENTIAL_FLOW_ACCESSOR_CUH
