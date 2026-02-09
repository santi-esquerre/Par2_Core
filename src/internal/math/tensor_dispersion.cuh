/**
 * @file tensor_dispersion.cuh
 * @brief Dispersion tensor utilities.
 *
 * Functions for computing the anisotropic dispersion tensor D and its
 * Cholesky-like factor B for the RWPT algorithm.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_INTERNAL_MATH_TENSOR_DISPERSION_CUH
#define PAR2_INTERNAL_MATH_TENSOR_DISPERSION_CUH

#include <cuda_runtime.h>

namespace par2 {
namespace math {

/**
 * @brief Compute the dispersion tensor components.
 *
 * The dispersion tensor D is:
 *   D_ij = (alphaT * |v| + Dm) * delta_ij + (alphaL - alphaT) * vi*vj / |v|
 *
 * @param vx,vy,vz Velocity components
 * @param Dm Molecular diffusion coefficient
 * @param alphaL Longitudinal dispersivity
 * @param alphaT Transverse dispersivity
 * @param[out] D Symmetric 3x3 tensor stored as [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
 */
template <typename T>
__host__ __device__
void compute_dispersion_tensor(
    T vx, T vy, T vz,
    T Dm, T alphaL, T alphaT,
    T D[6]
) {
    T vnorm2 = vx*vx + vy*vy + vz*vz;
    T vnorm = sqrt(vnorm2);

    const T tol = T(1e-10);
    if (vnorm < tol) {
        // Isotropic diffusion only
        D[0] = D[3] = D[5] = Dm;  // Dxx, Dyy, Dzz
        D[1] = D[2] = D[4] = T(0);  // Dxy, Dxz, Dyz
        return;
    }

    T alpha = alphaT * vnorm + Dm;
    T beta = (alphaL - alphaT) / vnorm;

    D[0] = alpha + beta * vx * vx;  // Dxx
    D[1] = beta * vx * vy;          // Dxy
    D[2] = beta * vx * vz;          // Dxz
    D[3] = alpha + beta * vy * vy;  // Dyy
    D[4] = beta * vy * vz;          // Dyz
    D[5] = alpha + beta * vz * vz;  // Dzz
}

} // namespace math
} // namespace par2

#endif // PAR2_INTERNAL_MATH_TENSOR_DISPERSION_CUH
