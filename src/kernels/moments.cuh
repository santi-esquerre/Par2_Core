/**
 * @file moments.cuh
 * @brief GPU reduction kernel for computing position moments (mean, variance).
 *
 * Computes sum(x_u), sum(x_u^2) for unwrapped positions, enabling calculation
 * of mean and variance without host copies.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_KERNELS_MOMENTS_CUH
#define PAR2_KERNELS_MOMENTS_CUH

#include <par2_core/views.hpp>
#include <par2_core/types.hpp>
#include <cuda_runtime.h>

namespace par2 {

// =============================================================================
// Device Moments struct
// =============================================================================

/**
 * @brief Accumulated position moments (device memory).
 *
 * @tparam T Floating point type
 *
 * Stores partial sums for computing mean and variance:
 * - sum_x, sum_y, sum_z: Sum of positions (for mean)
 * - sum_x2, sum_y2, sum_z2: Sum of squared positions (for variance)
 * - count: Number of active particles included
 *
 * After reduction:
 *   mean_x = sum_x / count
 *   var_x = sum_x2/count - mean_x^2
 */
template <typename T>
struct DeviceMoments {
    T sum_x = T(0);
    T sum_y = T(0);
    T sum_z = T(0);
    T sum_x2 = T(0);
    T sum_y2 = T(0);
    T sum_z2 = T(0);
    int count = 0;
};

/**
 * @brief Host-side moments with convenience methods.
 */
template <typename T>
struct HostMoments {
    T sum_x = T(0);
    T sum_y = T(0);
    T sum_z = T(0);
    T sum_x2 = T(0);
    T sum_y2 = T(0);
    T sum_z2 = T(0);
    int count = 0;

    /// Mean position X
    T mean_x() const { return count > 0 ? sum_x / count : T(0); }
    T mean_y() const { return count > 0 ? sum_y / count : T(0); }
    T mean_z() const { return count > 0 ? sum_z / count : T(0); }

    /// Variance X (population variance)
    T var_x() const {
        if (count == 0) return T(0);
        T mx = mean_x();
        return sum_x2 / count - mx * mx;
    }
    T var_y() const {
        if (count == 0) return T(0);
        T my = mean_y();
        return sum_y2 / count - my * my;
    }
    T var_z() const {
        if (count == 0) return T(0);
        T mz = mean_z();
        return sum_z2 / count - mz * mz;
    }

    /// Standard deviation
    T std_x() const { return std::sqrt(var_x()); }
    T std_y() const { return std::sqrt(var_y()); }
    T std_z() const { return std::sqrt(var_z()); }
};

namespace kernels {

// =============================================================================
// Atomic add for double (compatible with older architectures)
// =============================================================================

/**
 * @brief Atomic add for double using atomicCAS (works on all architectures).
 */
__device__ __forceinline__ double atomicAddDouble(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

/**
 * @brief Atomic add for float (uses native atomicAdd).
 */
__device__ __forceinline__ float atomicAddFloat(float* address, float val) {
    return atomicAdd(address, val);
}

// Overloaded helper that dispatches to correct version
template <typename T>
__device__ __forceinline__ T atomicAddT(T* address, T val);

template <>
__device__ __forceinline__ double atomicAddT<double>(double* address, double val) {
    return atomicAddDouble(address, val);
}

template <>
__device__ __forceinline__ float atomicAddT<float>(float* address, float val) {
    return atomicAddFloat(address, val);
}

// =============================================================================
// Block reduction for moments
// =============================================================================

/**
 * @brief Warp-level reduction using shuffle.
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief Block-level reduction.
 */
template <typename T>
__device__ T block_reduce_sum(T val, T* shared_data) {
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    // Warp reduction
    val = warp_reduce_sum(val);

    // First thread of each warp writes to shared memory
    if (lane == 0) {
        shared_data[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces all warp sums
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? shared_data[threadIdx.x] : T(0);
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

/**
 * @brief Kernel to compute position moments with block-level reduction.
 *
 * Each block computes partial sums, then atomically adds to global output.
 *
 * @param x_u, y_u, z_u Unwrapped positions
 * @param status Status array (nullptr to include all particles)
 * @param n Number of particles
 * @param out Output moments (device memory, must be zero-initialized)
 */
template <typename T>
__global__ void compute_moments_kernel(
    const T* __restrict__ x_u,
    const T* __restrict__ y_u,
    const T* __restrict__ z_u,
    const uint8_t* __restrict__ status,  // May be nullptr
    int n,
    DeviceMoments<T>* out
) {
    // Shared memory for block reduction (max 32 warps = 1024 threads)
    __shared__ T shared_x[32];
    __shared__ T shared_y[32];
    __shared__ T shared_z[32];
    __shared__ T shared_x2[32];
    __shared__ T shared_y2[32];
    __shared__ T shared_z2[32];
    __shared__ int shared_count[32];

    // Thread's partial sums
    T local_x = T(0), local_y = T(0), local_z = T(0);
    T local_x2 = T(0), local_y2 = T(0), local_z2 = T(0);
    int local_count = 0;

    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {

        // Skip non-active particles if status is provided
        if (status != nullptr) {
            // ParticleStatus::Active == 0
            if (status[i] != 0) continue;
        }

        T px = x_u[i];
        T py = y_u[i];
        T pz = z_u[i];

        local_x += px;
        local_y += py;
        local_z += pz;
        local_x2 += px * px;
        local_y2 += py * py;
        local_z2 += pz * pz;
        local_count += 1;
    }

    // Block reduction
    T block_x = block_reduce_sum(local_x, shared_x);
    T block_y = block_reduce_sum(local_y, shared_y);
    T block_z = block_reduce_sum(local_z, shared_z);
    T block_x2 = block_reduce_sum(local_x2, shared_x2);
    T block_y2 = block_reduce_sum(local_y2, shared_y2);
    T block_z2 = block_reduce_sum(local_z2, shared_z2);

    // Count reduction (use float version for simplicity)
    __shared__ int shared_cnt[32];
    int block_count = 0;
    {
        const int lane = threadIdx.x % warpSize;
        const int warp_id = threadIdx.x / warpSize;

        // Warp reduction for count
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);
        }
        if (lane == 0) shared_cnt[warp_id] = local_count;
        __syncthreads();

        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        int cnt = (threadIdx.x < num_warps) ? shared_cnt[threadIdx.x] : 0;
        if (warp_id == 0) {
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                cnt += __shfl_down_sync(0xFFFFFFFF, cnt, offset);
            }
        }
        block_count = cnt;
    }

    // First thread of block atomically adds to global output
    if (threadIdx.x == 0) {
        atomicAddT(&out->sum_x, block_x);
        atomicAddT(&out->sum_y, block_y);
        atomicAddT(&out->sum_z, block_z);
        atomicAddT(&out->sum_x2, block_x2);
        atomicAddT(&out->sum_y2, block_y2);
        atomicAddT(&out->sum_z2, block_z2);
        atomicAdd(&out->count, block_count);
    }
}

/**
 * @brief Launch moments computation.
 *
 * @param x_u, y_u, z_u Unwrapped positions (device)
 * @param status Status array (device, may be nullptr)
 * @param n Number of particles
 * @param out Output moments (device, caller must zero-init before call)
 * @param stream CUDA stream
 */
template <typename T>
inline void launch_compute_moments(
    const T* x_u, const T* y_u, const T* z_u,
    const uint8_t* status,
    int n,
    DeviceMoments<T>* out,
    int block_size,
    cudaStream_t stream
) {
    if (n <= 0) return;

    // Limit grid size to avoid excessive atomics
    const int max_blocks = 256;
    const int num_blocks = min((n + block_size - 1) / block_size, max_blocks);

    compute_moments_kernel<<<num_blocks, block_size, 0, stream>>>(
        x_u, y_u, z_u, status, n, out
    );
}

} // namespace kernels

// =============================================================================
// High-level stats interface
// =============================================================================

/**
 * @brief Compute position moments on GPU.
 *
 * @tparam T Floating point type
 *
 * @param unwrapped Unwrapped positions (device)
 * @param status Particle status (device, nullptr to include all)
 * @param n Number of particles
 * @param d_moments Output on device (must be pre-allocated)
 * @param stream CUDA stream
 * @param block_size Kernel block size
 *
 * The caller must:
 * 1. Zero-initialize d_moments before calling (cudaMemsetAsync)
 * 2. Synchronize stream before reading results
 *
 * ## Usage
 * ```cpp
 * DeviceMoments<double>* d_mom;
 * cudaMalloc(&d_mom, sizeof(DeviceMoments<double>));
 * cudaMemsetAsync(d_mom, 0, sizeof(DeviceMoments<double>), stream);
 *
 * compute_moments(unwrap, particles.status, n, d_mom, stream);
 *
 * HostMoments<double> h_mom;
 * cudaMemcpyAsync(&h_mom, d_mom, sizeof(DeviceMoments<double>),
 *                cudaMemcpyDeviceToHost, stream);
 * cudaStreamSynchronize(stream);
 *
 * std::cout << "Mean X: " << h_mom.mean_x() << std::endl;
 * ```
 */
template <typename T>
inline void compute_moments(
    const UnwrappedPositionsView<T>& unwrapped,
    const uint8_t* status,
    int n,
    DeviceMoments<T>* d_moments,
    cudaStream_t stream,
    int block_size = 256
) {
    kernels::launch_compute_moments(
        unwrapped.x_u, unwrapped.y_u, unwrapped.z_u,
        status, n, d_moments, block_size, stream
    );
}

} // namespace par2

#endif // PAR2_KERNELS_MOMENTS_CUH
