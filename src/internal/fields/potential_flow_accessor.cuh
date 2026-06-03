/**
 * @file potential_flow_accessor.cuh
 * @brief KH potential-flow velocity reconstruction from cell-centered K/head.
 */

#ifndef PAR2_CORE_INTERNAL_POTENTIAL_FLOW_ACCESSOR_CUH
#define PAR2_CORE_INTERNAL_POTENTIAL_FLOW_ACCESSOR_CUH

#include <par2_core/grid.hpp>
#include <par2_core/types.hpp>
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
T sample_grad_x_trilinear(const PotentialFlowView<T>& pf, const GridDesc<T>& g,
                          T x, T y, T z);

template <typename T>
__device__ __forceinline__
T sample_grad_y_trilinear(const PotentialFlowView<T>& pf, const GridDesc<T>& g,
                          T x, T y, T z);

template <typename T>
__device__ __forceinline__
T sample_grad_z_trilinear(const PotentialFlowView<T>& pf, const GridDesc<T>& g,
                          T x, T y, T z);

struct CubicAxisStencil {
    int idx[4];
    double w[4];
    double dw[4];
    bool valid = false;
};

__host__ __device__ __forceinline__
void lagrange4_weights(double u, double* w, double* dw_du) {
    constexpr double nodes[4] = {0.0, 1.0, 2.0, 3.0};
    for (int a = 0; a < 4; ++a) {
        double wa = 1.0;
        for (int b = 0; b < 4; ++b) {
            if (b == a)
                continue;
            wa *= (u - nodes[b]) / (nodes[a] - nodes[b]);
        }
        w[a] = wa;

        double dwa = 0.0;
        for (int m = 0; m < 4; ++m) {
            if (m == a)
                continue;
            double term = 1.0 / (nodes[a] - nodes[m]);
            for (int b = 0; b < 4; ++b) {
                if (b == a || b == m)
                    continue;
                term *= (u - nodes[b]) / (nodes[a] - nodes[b]);
            }
            dwa += term;
        }
        dw_du[a] = dwa;
    }
}

template <typename T>
__host__ __device__ __forceinline__
CubicAxisStencil make_cubic_axis_stencil(T coord, T origin, T spacing, int n, bool periodic) {
    CubicAxisStencil stencil{};
    if (n < 4) {
        return stencil;
    }

    const double s = (static_cast<double>(coord) - static_cast<double>(origin)) /
                         static_cast<double>(spacing) -
                     0.5;
    int anchor = static_cast<int>(floor(s)) - 1;
    if (periodic) {
        for (int m = 0; m < 4; ++m) {
            stencil.idx[m] = wrap_index<T>(anchor + m, n);
        }
    } else {
        if (anchor < 0)
            anchor = 0;
        const int max_anchor = n - 4;
        if (anchor > max_anchor)
            anchor = max_anchor;
        for (int m = 0; m < 4; ++m) {
            stencil.idx[m] = anchor + m;
        }
    }

    const double u = s - static_cast<double>(anchor);
    double dw_du[4];
    lagrange4_weights(u, stencil.w, dw_du);
    const double inv_spacing = 1.0 / static_cast<double>(spacing);
    for (int m = 0; m < 4; ++m) {
        stencil.dw[m] = dw_du[m] * inv_spacing;
    }
    stencil.valid = true;
    return stencil;
}

template <typename SampleFn>
__device__ __forceinline__
double sample_cubic_value(const CubicAxisStencil& sx, const CubicAxisStencil& sy,
                          const CubicAxisStencil& sz, SampleFn&& sample_fn) {
    double accum = 0.0;
    for (int kk = 0; kk < 4; ++kk) {
        for (int jj = 0; jj < 4; ++jj) {
            for (int ii = 0; ii < 4; ++ii) {
                accum += sx.w[ii] * sy.w[jj] * sz.w[kk] *
                         sample_fn(sx.idx[ii], sy.idx[jj], sz.idx[kk]);
            }
        }
    }
    return accum;
}

template <typename SampleFn>
__device__ __forceinline__
double sample_cubic_dx(const CubicAxisStencil& sx, const CubicAxisStencil& sy,
                       const CubicAxisStencil& sz, SampleFn&& sample_fn) {
    double accum = 0.0;
    for (int kk = 0; kk < 4; ++kk) {
        for (int jj = 0; jj < 4; ++jj) {
            for (int ii = 0; ii < 4; ++ii) {
                accum += sx.dw[ii] * sy.w[jj] * sz.w[kk] *
                         sample_fn(sx.idx[ii], sy.idx[jj], sz.idx[kk]);
            }
        }
    }
    return accum;
}

template <typename SampleFn>
__device__ __forceinline__
double sample_cubic_dy(const CubicAxisStencil& sx, const CubicAxisStencil& sy,
                       const CubicAxisStencil& sz, SampleFn&& sample_fn) {
    double accum = 0.0;
    for (int kk = 0; kk < 4; ++kk) {
        for (int jj = 0; jj < 4; ++jj) {
            for (int ii = 0; ii < 4; ++ii) {
                accum += sx.w[ii] * sy.dw[jj] * sz.w[kk] *
                         sample_fn(sx.idx[ii], sy.idx[jj], sz.idx[kk]);
            }
        }
    }
    return accum;
}

template <typename SampleFn>
__device__ __forceinline__
double sample_cubic_dz(const CubicAxisStencil& sx, const CubicAxisStencil& sy,
                       const CubicAxisStencil& sz, SampleFn&& sample_fn) {
    double accum = 0.0;
    for (int kk = 0; kk < 4; ++kk) {
        for (int jj = 0; jj < 4; ++jj) {
            for (int ii = 0; ii < 4; ++ii) {
                accum += sx.w[ii] * sy.w[jj] * sz.dw[kk] *
                         sample_fn(sx.idx[ii], sy.idx[jj], sz.idx[kk]);
            }
        }
    }
    return accum;
}

template <typename T>
__device__ __forceinline__
double sample_cell_cubic_lagrange(const T* __restrict__ field, const GridDesc<T>& g,
                                  const PotentialBoundaryConfig<T>& bc, T x, T y, T z) {
    const CubicAxisStencil sx =
        make_cubic_axis_stencil(x, g.px, g.dx, g.nx, is_periodic_axis(bc.x));
    const CubicAxisStencil sy =
        make_cubic_axis_stencil(y, g.py, g.dy, g.ny, is_periodic_axis(bc.y));
    const CubicAxisStencil sz =
        make_cubic_axis_stencil(z, g.pz, g.dz, g.nz, is_periodic_axis(bc.z));
    if (!sx.valid || !sy.valid || !sz.valid) {
        return static_cast<double>(sample_cell_trilinear(field, g, bc, x, y, z));
    }

    auto sample_fn = [&](int i, int j, int k) -> double {
        return static_cast<double>(cell_value(field, g, i, j, k));
    };
    return sample_cubic_value(sx, sy, sz, sample_fn);
}

template <typename T>
__device__ __forceinline__
double safe_positive_for_log(T value) {
    const double v = static_cast<double>(value);
    return v > 1.0e-300 ? v : 1.0e-300;
}

template <typename T>
__device__ __forceinline__
double sample_conductivity_kh_linear(const PotentialFlowView<T>& pf, const GridDesc<T>& g, T x,
                                     T y, T z) {
    return static_cast<double>(sample_cell_trilinear(pf.K, g, pf.head_bc, x, y, z));
}

template <typename T>
__device__ __forceinline__
double sample_conductivity_kh_cubic(const PotentialFlowView<T>& pf, const GridDesc<T>& g, T x, T y,
                                    T z) {
    return sample_cell_cubic_lagrange(pf.K, g, pf.head_bc, x, y, z);
}

template <typename T>
__device__ __forceinline__
double sample_logk_kh_cubic(const PotentialFlowView<T>& pf, const GridDesc<T>& g, T x, T y, T z) {
    const CubicAxisStencil sx =
        make_cubic_axis_stencil(x, g.px, g.dx, g.nx, is_periodic_axis(pf.head_bc.x));
    const CubicAxisStencil sy =
        make_cubic_axis_stencil(y, g.py, g.dy, g.ny, is_periodic_axis(pf.head_bc.y));
    const CubicAxisStencil sz =
        make_cubic_axis_stencil(z, g.pz, g.dz, g.nz, is_periodic_axis(pf.head_bc.z));

    if (!sx.valid || !sy.valid || !sz.valid) {
        return log(safe_positive_for_log(sample_cell_trilinear(pf.K, g, pf.head_bc, x, y, z)));
    }

    auto sample_fn = [&](int i, int j, int k) -> double {
        return log(safe_positive_for_log(cell_value(pf.K, g, i, j, k)));
    };
    return sample_cubic_value(sx, sy, sz, sample_fn);
}

template <typename T>
__device__ __forceinline__
double sample_conductivity_kh_logk_cubic(const PotentialFlowView<T>& pf, const GridDesc<T>& g, T x,
                                         T y, T z, double* logk_interp = nullptr) {
    const double logk = sample_logk_kh_cubic(pf, g, x, y, z);
    if (logk_interp != nullptr) {
        *logk_interp = logk;
    }
    return exp(logk);
}

template <typename T>
__device__ __forceinline__
double sample_conductivity_potential_backend(const PotentialFlowView<T>& pf, const GridDesc<T>& g,
                                             VelocityEvalMode mode, T x, T y, T z,
                                             double* logk_interp = nullptr) {
    if (logk_interp != nullptr) {
        *logk_interp = 0.0;
    }

    switch (mode) {
    case VelocityEvalMode::KhLinear:
        return sample_conductivity_kh_linear(pf, g, x, y, z);
    case VelocityEvalMode::KhCubicPotentialReconstruction:
        return sample_conductivity_kh_cubic(pf, g, x, y, z);
    case VelocityEvalMode::KhLogKCubicPotentialReconstruction:
        return sample_conductivity_kh_logk_cubic(pf, g, x, y, z, logk_interp);
    case VelocityEvalMode::FaceTrilinear:
    default:
        return 0.0;
    }
}

template <typename T>
__device__ __forceinline__
void sample_head_cubic_lagrange_with_gradient(const PotentialFlowView<T>& pf, const GridDesc<T>& g,
                                              T x, T y, T z, double& dhdx, double& dhdy,
                                              double& dhdz) {
    const CubicAxisStencil sx =
        make_cubic_axis_stencil(x, g.px, g.dx, g.nx, is_periodic_axis(pf.head_bc.x));
    const CubicAxisStencil sy =
        make_cubic_axis_stencil(y, g.py, g.dy, g.ny, is_periodic_axis(pf.head_bc.y));
    const CubicAxisStencil sz =
        make_cubic_axis_stencil(z, g.pz, g.dz, g.nz, is_periodic_axis(pf.head_bc.z));
    if (!sx.valid || !sy.valid || !sz.valid) {
        dhdx = static_cast<double>(sample_grad_x_trilinear(pf, g, x, y, z));
        dhdy = static_cast<double>(sample_grad_y_trilinear(pf, g, x, y, z));
        dhdz = static_cast<double>(sample_grad_z_trilinear(pf, g, x, y, z));
        return;
    }

    auto sample_fn = [&](int i, int j, int k) -> double {
        return static_cast<double>(cell_value(pf.head, g, i, j, k));
    };
    dhdx = sample_cubic_dx(sx, sy, sz, sample_fn);
    dhdy = sample_cubic_dy(sx, sy, sz, sample_fn);
    dhdz = sample_cubic_dz(sx, sy, sz, sample_fn);
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

    const T Kx = static_cast<T>(sample_conductivity_kh_linear(pf, g, x, y, z));
    const T dhdx = sample_grad_x_trilinear(pf, g, x, y, z);
    const T dhdy = sample_grad_y_trilinear(pf, g, x, y, z);
    const T dhdz = sample_grad_z_trilinear(pf, g, x, y, z);

    vx = -Kx * dhdx;
    vy = -Kx * dhdy;
    vz = -Kx * dhdz;
}

template <typename T>
__device__ __forceinline__
void sample_velocity_kh_cubic_potential(const PotentialFlowView<T>& pf, const GridDesc<T>& g,
                                        T x, T y, T z, T& vx, T& vy, T& vz) {
    if (pf.K == nullptr || pf.head == nullptr || pf.size != static_cast<size_t>(g.num_cells()) ||
        x < g.px || x >= g.x_max()) {
        vx = vy = vz = T(0);
        return;
    }

    const double Kx = sample_conductivity_kh_cubic(pf, g, x, y, z);
    double dhdx = 0.0;
    double dhdy = 0.0;
    double dhdz = 0.0;
    sample_head_cubic_lagrange_with_gradient(pf, g, x, y, z, dhdx, dhdy, dhdz);

    vx = static_cast<T>(-Kx * dhdx);
    vy = static_cast<T>(-Kx * dhdy);
    vz = static_cast<T>(-Kx * dhdz);
}

template <typename T>
__device__ __forceinline__
void sample_velocity_kh_logk_cubic_potential(const PotentialFlowView<T>& pf, const GridDesc<T>& g,
                                             T x, T y, T z, T& vx, T& vy, T& vz) {
    if (pf.K == nullptr || pf.head == nullptr || pf.size != static_cast<size_t>(g.num_cells()) ||
        x < g.px || x >= g.x_max()) {
        vx = vy = vz = T(0);
        return;
    }

    double dhdx = 0.0;
    double dhdy = 0.0;
    double dhdz = 0.0;
    sample_head_cubic_lagrange_with_gradient(pf, g, x, y, z, dhdx, dhdy, dhdz);

    const double Kx = sample_conductivity_kh_logk_cubic(pf, g, x, y, z);
    vx = static_cast<T>(-Kx * dhdx);
    vy = static_cast<T>(-Kx * dhdy);
    vz = static_cast<T>(-Kx * dhdz);
}

template <typename T>
__device__ __forceinline__
void sample_velocity_potential_backend(const PotentialFlowView<T>& pf, const GridDesc<T>& g,
                                       VelocityEvalMode mode, T x, T y, T z, T& vx, T& vy, T& vz) {
    switch (mode) {
    case VelocityEvalMode::KhLinear:
        sample_velocity_kh_potential(pf, g, x, y, z, vx, vy, vz);
        return;
    case VelocityEvalMode::KhCubicPotentialReconstruction:
        sample_velocity_kh_cubic_potential(pf, g, x, y, z, vx, vy, vz);
        return;
    case VelocityEvalMode::KhLogKCubicPotentialReconstruction:
        sample_velocity_kh_logk_cubic_potential(pf, g, x, y, z, vx, vy, vz);
        return;
    case VelocityEvalMode::FaceTrilinear:
    default:
        vx = vy = vz = T(0);
        return;
    }
}

} // namespace internal
} // namespace par2

#endif // PAR2_CORE_INTERNAL_POTENTIAL_FLOW_ACCESSOR_CUH
