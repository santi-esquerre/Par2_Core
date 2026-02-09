/**
 * @file vec3.cuh
 * @brief 3D vector math utilities for CUDA kernels.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_INTERNAL_MATH_VEC3_CUH
#define PAR2_INTERNAL_MATH_VEC3_CUH

#include <cuda_runtime.h>

namespace par2 {
namespace math {

/**
 * @brief Compute squared norm of a 3D vector.
 */
template <typename T>
__host__ __device__ __forceinline__
T norm2(T x, T y, T z) {
    return x*x + y*y + z*z;
}

/**
 * @brief Compute Euclidean norm of a 3D vector.
 */
template <typename T>
__host__ __device__ __forceinline__
T norm(T x, T y, T z) {
    return sqrt(norm2(x, y, z));
}

/**
 * @brief Dot product of two 3D vectors.
 */
template <typename T>
__host__ __device__ __forceinline__
T dot(T ax, T ay, T az, T bx, T by, T bz) {
    return ax*bx + ay*by + az*bz;
}

} // namespace math
} // namespace par2

#endif // PAR2_INTERNAL_MATH_VEC3_CUH
