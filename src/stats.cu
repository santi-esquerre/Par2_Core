/**
 * @file stats.cu
 * @brief GPU statistics implementation for Par2_Core.
 *
 * Implements position moment computation (mean, variance, std) and
 * particle counting using CUB-based reductions.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#include <par2_core/stats.hpp>
#include <par2_core/types.hpp>

#include <cub/cub.cuh>
#include <cmath>
#include <stdexcept>

namespace par2 {

// =============================================================================
// Internal Helpers
// =============================================================================

namespace {

/**
 * @brief Structure for parallel mean + variance computation.
 *
 * Uses Welford's online algorithm adapted for parallel reduction.
 * Each element stores (sum, sumSq, count) which can be combined.
 */
template <typename T>
struct MomentAccum {
    T sum;      ///< Sum of values
    T sum_sq;   ///< Sum of squared values
    int count;  ///< Number of values

    __host__ __device__
    MomentAccum() : sum(T(0)), sum_sq(T(0)), count(0) {}

    __host__ __device__
    MomentAccum(T val, bool active) 
        : sum(active ? val : T(0))
        , sum_sq(active ? val * val : T(0))
        , count(active ? 1 : 0) {}

    __host__ __device__
    MomentAccum operator+(const MomentAccum& other) const {
        MomentAccum result;
        result.sum = sum + other.sum;
        result.sum_sq = sum_sq + other.sum_sq;
        result.count = count + other.count;
        return result;
    }
};

/**
 * @brief Transform operator for particle → MomentAccum.
 */
template <typename T>
struct ParticleToMoment {
    const T* x;
    const T* y;
    const T* z;
    const int32_t* wrapX;
    const int32_t* wrapY;
    const int32_t* wrapZ;
    const uint8_t* status;
    T Lx, Ly, Lz;
    bool use_unwrap;
    bool filter_active;
    int axis;  // 0=X, 1=Y, 2=Z

    __device__
    MomentAccum<T> operator()(int i) const {
        // Check if particle is active
        bool is_active = true;
        if (filter_active && status != nullptr) {
            is_active = (status[i] == static_cast<uint8_t>(ParticleStatus::Active));
        }

        if (!is_active) {
            return MomentAccum<T>();
        }

        // Get position for this axis
        T val;
        switch (axis) {
            case 0:
                val = x[i];
                if (use_unwrap && wrapX != nullptr) {
                    val += static_cast<T>(wrapX[i]) * Lx;
                }
                break;
            case 1:
                val = y[i];
                if (use_unwrap && wrapY != nullptr) {
                    val += static_cast<T>(wrapY[i]) * Ly;
                }
                break;
            case 2:
            default:
                val = z[i];
                if (use_unwrap && wrapZ != nullptr) {
                    val += static_cast<T>(wrapZ[i]) * Lz;
                }
                break;
        }

        return MomentAccum<T>(val, true);
    }
};

/**
 * @brief Kernel to compute MomentAccum for all particles.
 */
template <typename T>
__global__ void computeMomentsKernel(
    MomentAccum<T>* __restrict__ output,
    const T* __restrict__ x,
    const T* __restrict__ y,
    const T* __restrict__ z,
    const int32_t* __restrict__ wrapX,
    const int32_t* __restrict__ wrapY,
    const int32_t* __restrict__ wrapZ,
    const uint8_t* __restrict__ status,
    T Lx, T Ly, T Lz,
    bool use_unwrap,
    bool filter_active,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize accumulators for 3 axes
    MomentAccum<T> accum[3];

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        // Check if active
        bool is_active = true;
        if (filter_active && status != nullptr) {
            is_active = (status[i] == static_cast<uint8_t>(ParticleStatus::Active));
        }

        if (is_active) {
            T vx = x[i];
            T vy = y[i];
            T vz = z[i];

            if (use_unwrap) {
                if (wrapX != nullptr) vx += static_cast<T>(wrapX[i]) * Lx;
                if (wrapY != nullptr) vy += static_cast<T>(wrapY[i]) * Ly;
                if (wrapZ != nullptr) vz += static_cast<T>(wrapZ[i]) * Lz;
            }

            accum[0].sum += vx;
            accum[0].sum_sq += vx * vx;
            accum[0].count += 1;

            accum[1].sum += vy;
            accum[1].sum_sq += vy * vy;
            accum[1].count += 1;

            accum[2].sum += vz;
            accum[2].sum_sq += vz * vz;
            accum[2].count += 1;
        }
    }

    // Block-level reduction using shared memory
    using BlockReduce = cub::BlockReduce<MomentAccum<T>, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage[3];

    MomentAccum<T> block_result[3];
    block_result[0] = BlockReduce(temp_storage[0]).Reduce(accum[0], cub::Sum());
    __syncthreads();
    block_result[1] = BlockReduce(temp_storage[1]).Reduce(accum[1], cub::Sum());
    __syncthreads();
    block_result[2] = BlockReduce(temp_storage[2]).Reduce(accum[2], cub::Sum());

    // Thread 0 writes block result
    if (threadIdx.x == 0) {
        int bid = blockIdx.x;
        output[bid * 3 + 0] = block_result[0];
        output[bid * 3 + 1] = block_result[1];
        output[bid * 3 + 2] = block_result[2];
    }
}

/**
 * @brief Kernel to count particles by status.
 */
__global__ void countStatusKernel(
    int* __restrict__ counts,  // [active, exited, inactive]
    const uint8_t* __restrict__ status,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int local_active = 0;
    int local_exited = 0;
    int local_inactive = 0;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        uint8_t s = status[i];
        if (s == static_cast<uint8_t>(ParticleStatus::Active)) {
            local_active++;
        } else if (s == static_cast<uint8_t>(ParticleStatus::Exited)) {
            local_exited++;
        } else {
            local_inactive++;
        }
    }

    // Block reduction
    using BlockReduce = cub::BlockReduce<int, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int block_active = BlockReduce(temp_storage).Reduce(local_active, cub::Sum());
    __syncthreads();
    int block_exited = BlockReduce(temp_storage).Reduce(local_exited, cub::Sum());
    __syncthreads();
    int block_inactive = BlockReduce(temp_storage).Reduce(local_inactive, cub::Sum());

    if (threadIdx.x == 0) {
        atomicAdd(&counts[0], block_active);
        atomicAdd(&counts[1], block_exited);
        atomicAdd(&counts[2], block_inactive);
    }
}

/**
 * @brief Kernel to finalize moments from block results.
 */
template <typename T>
__global__ void finalizeMomentsKernel(
    T* __restrict__ mean,     // [3]
    T* __restrict__ var,      // [3]
    T* __restrict__ stddev,   // [3]
    const MomentAccum<T>* __restrict__ block_results,
    int num_blocks
) {
    int axis = threadIdx.x;
    if (axis >= 3) return;

    // Reduce block results for this axis
    MomentAccum<T> total;
    for (int b = 0; b < num_blocks; ++b) {
        total = total + block_results[b * 3 + axis];
    }

    if (total.count > 0) {
        T m = total.sum / static_cast<T>(total.count);
        mean[axis] = m;

        if (total.count > 1) {
            // Variance = E[X²] - E[X]² (with Bessel's correction)
            T e_x2 = total.sum_sq / static_cast<T>(total.count);
            T variance = (e_x2 - m * m) * static_cast<T>(total.count) / static_cast<T>(total.count - 1);
            var[axis] = variance;
            stddev[axis] = sqrt(variance);
        } else {
            var[axis] = T(0);
            stddev[axis] = T(0);
        }
    } else {
        mean[axis] = T(0);
        var[axis] = T(0);
        stddev[axis] = T(0);
    }
}

/**
 * @brief Check if position is in box.
 */
template <typename T>
struct InBoxPredicate {
    T x0, y0, z0, x1, y1, z1;
    const T* x;
    const T* y;
    const T* z;

    __device__
    int operator()(int i) const {
        return (x[i] >= x0 && x[i] <= x1 &&
                y[i] >= y0 && y[i] <= y1 &&
                z[i] >= z0 && z[i] <= z1) ? 1 : 0;
    }
};

/**
 * @brief Check if position past plane.
 */
template <typename T>
struct PastPlanePredicate {
    T threshold;
    const T* coords;

    __device__
    int operator()(int i) const {
        return (coords[i] > threshold) ? 1 : 0;
    }
};

} // anonymous namespace

// =============================================================================
// StatsComputer Implementation
// =============================================================================

template <typename T>
struct StatsComputer<T>::Impl {
    int max_particles;
    int num_blocks;

    // Device buffers
    MomentAccum<T>* d_block_results = nullptr;
    T* d_mean = nullptr;
    T* d_var = nullptr;
    T* d_std = nullptr;
    int* d_counts = nullptr;  // [active, exited, inactive]

    // Pinned host buffers for async copy
    T* h_mean = nullptr;
    T* h_var = nullptr;
    T* h_std = nullptr;
    int* h_counts = nullptr;

    // Result cache
    bool has_result = false;
    int last_total = 0;

    Impl(int max_n) : max_particles(max_n) {
        // Calculate number of blocks needed
        const int block_size = 256;
        num_blocks = (max_particles + block_size - 1) / block_size;
        if (num_blocks < 1) num_blocks = 1;

        // Allocate device buffers
        cudaMalloc(&d_block_results, num_blocks * 3 * sizeof(MomentAccum<T>));
        cudaMalloc(&d_mean, 3 * sizeof(T));
        cudaMalloc(&d_var, 3 * sizeof(T));
        cudaMalloc(&d_std, 3 * sizeof(T));
        cudaMalloc(&d_counts, 3 * sizeof(int));

        // Allocate pinned host buffers
        cudaMallocHost(&h_mean, 3 * sizeof(T));
        cudaMallocHost(&h_var, 3 * sizeof(T));
        cudaMallocHost(&h_std, 3 * sizeof(T));
        cudaMallocHost(&h_counts, 4 * sizeof(int));  // +1 for total
    }

    ~Impl() {
        if (d_block_results) cudaFree(d_block_results);
        if (d_mean) cudaFree(d_mean);
        if (d_var) cudaFree(d_var);
        if (d_std) cudaFree(d_std);
        if (d_counts) cudaFree(d_counts);

        if (h_mean) cudaFreeHost(h_mean);
        if (h_var) cudaFreeHost(h_var);
        if (h_std) cudaFreeHost(h_std);
        if (h_counts) cudaFreeHost(h_counts);
    }
};

template <typename T>
StatsComputer<T>::StatsComputer(int max_particles) {
    impl_ = new Impl(max_particles);
}

template <typename T>
StatsComputer<T>::~StatsComputer() {
    delete impl_;
}

template <typename T>
StatsComputer<T>::StatsComputer(StatsComputer&& other) noexcept
    : impl_(other.impl_) {
    other.impl_ = nullptr;
}

template <typename T>
StatsComputer<T>& StatsComputer<T>::operator=(StatsComputer&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

template <typename T>
cudaError_t StatsComputer<T>::compute_async(
    const ConstParticlesView<T>& particles,
    const GridDesc<T>& grid,
    const StatsConfig& config,
    cudaStream_t stream
) {
    if (!impl_) return cudaErrorInvalidValue;
    if (particles.n > impl_->max_particles) return cudaErrorInvalidValue;
    if (particles.n <= 0) {
        impl_->has_result = false;
        return cudaSuccess;
    }

    const int n = particles.n;
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    // Domain lengths for unwrap
    T Lx = grid.length_x();
    T Ly = grid.length_y();
    T Lz = grid.length_z();

    // Zero the counts
    cudaMemsetAsync(impl_->d_counts, 0, 3 * sizeof(int), stream);

    // Launch moments kernel
    computeMomentsKernel<T><<<num_blocks, block_size, 0, stream>>>(
        impl_->d_block_results,
        particles.x, particles.y, particles.z,
        particles.wrapX, particles.wrapY, particles.wrapZ,
        particles.status,
        Lx, Ly, Lz,
        config.use_unwrapped,
        config.filter_active_only,
        n
    );

    // Launch status count kernel (if status available)
    if (particles.status != nullptr) {
        countStatusKernel<<<num_blocks, block_size, 0, stream>>>(
            impl_->d_counts,
            particles.status,
            n
        );
    }

    // Finalize moments (single block, 3 threads)
    finalizeMomentsKernel<T><<<1, 3, 0, stream>>>(
        impl_->d_mean, impl_->d_var, impl_->d_std,
        impl_->d_block_results,
        num_blocks
    );

    // Async copy results to pinned host
    cudaMemcpyAsync(impl_->h_mean, impl_->d_mean, 3 * sizeof(T),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(impl_->h_var, impl_->d_var, 3 * sizeof(T),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(impl_->h_std, impl_->d_std, 3 * sizeof(T),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(impl_->h_counts, impl_->d_counts, 3 * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);

    impl_->last_total = n;
    impl_->has_result = true;

    return cudaGetLastError();
}

template <typename T>
StatsResult<T> StatsComputer<T>::fetch_result() const {
    StatsResult<T> result;
    if (!impl_ || !impl_->has_result) {
        return result;
    }

    // Copy from pinned host memory
    for (int i = 0; i < 3; ++i) {
        result.moments.mean[i] = impl_->h_mean[i];
        result.moments.var[i] = impl_->h_var[i];
        result.moments.std[i] = impl_->h_std[i];
    }

    result.counts.total = impl_->last_total;

    if (impl_->h_counts[0] == 0 && impl_->h_counts[1] == 0 && impl_->h_counts[2] == 0) {
        // No status array was provided - all particles are "active"
        result.counts.active = impl_->last_total;
        result.counts.exited = 0;
        result.counts.inactive = 0;
    } else {
        result.counts.active = impl_->h_counts[0];
        result.counts.exited = impl_->h_counts[1];
        result.counts.inactive = impl_->h_counts[2];
    }

    result.computed = true;
    return result;
}

template <typename T>
int StatsComputer<T>::capacity() const noexcept {
    return impl_ ? impl_->max_particles : 0;
}

// =============================================================================
// Legacy-Compatible Functions
// =============================================================================

// Explicit kernel for box counting
template <typename T>
__global__ void countInBoxKernel(
    int* count,
    const T* x, const T* y, const T* z,
    T x0, T y0, T z0, T x1, T y1, T z1,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_count = 0;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (x[i] >= x0 && x[i] <= x1 &&
            y[i] >= y0 && y[i] <= y1 &&
            z[i] >= z0 && z[i] <= z1) {
            local_count++;
        }
    }

    // Block reduction
    using BlockReduce = cub::BlockReduce<int, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int block_count = BlockReduce(temp_storage).Reduce(local_count, cub::Sum());

    if (threadIdx.x == 0) {
        atomicAdd(count, block_count);
    }
}

// Explicit kernel for plane counting
template <typename T>
__global__ void countPastPlaneKernel(
    int* count,
    const T* coords,
    T threshold,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_count = 0;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (coords[i] > threshold) {
            local_count++;
        }
    }

    using BlockReduce = cub::BlockReduce<int, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int block_count = BlockReduce(temp_storage).Reduce(local_count, cub::Sum());

    if (threadIdx.x == 0) {
        atomicAdd(count, block_count);
    }
}

// Re-implement concentration_box using explicit kernel
template <typename T>
double concentration_box_impl(
    const ConstParticlesView<T>& particles,
    T x0, T y0, T z0,
    T x1, T y1, T z1,
    cudaStream_t stream
) {
    if (particles.n <= 0) return 0.0;

    int* d_count = nullptr;
    int h_count = 0;

    cudaMalloc(&d_count, sizeof(int));
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);

    const int n = particles.n;
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    countInBoxKernel<T><<<num_blocks, block_size, 0, stream>>>(
        d_count,
        particles.x, particles.y, particles.z,
        x0, y0, z0, x1, y1, z1,
        n
    );

    cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_count);

    return static_cast<double>(h_count) / static_cast<double>(n);
}

// Specialize the public function
template <>
double concentration_box<float>(
    const ConstParticlesView<float>& particles,
    float x0, float y0, float z0,
    float x1, float y1, float z1,
    cudaStream_t stream
) {
    return concentration_box_impl<float>(particles, x0, y0, z0, x1, y1, z1, stream);
}

template <>
double concentration_box<double>(
    const ConstParticlesView<double>& particles,
    double x0, double y0, double z0,
    double x1, double y1, double z1,
    cudaStream_t stream
) {
    return concentration_box_impl<double>(particles, x0, y0, z0, x1, y1, z1, stream);
}

template <typename T>
double concentration_past_plane_impl(
    const ConstParticlesView<T>& particles,
    int axis,
    T threshold,
    cudaStream_t stream
) {
    if (particles.n <= 0) return 0.0;

    const T* coords = nullptr;
    switch (axis) {
        case 0: coords = particles.x; break;
        case 1: coords = particles.y; break;
        case 2: coords = particles.z; break;
        default: return 0.0;
    }

    int* d_count = nullptr;
    int h_count = 0;

    cudaMalloc(&d_count, sizeof(int));
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);

    const int n = particles.n;
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    countPastPlaneKernel<T><<<num_blocks, block_size, 0, stream>>>(
        d_count, coords, threshold, n
    );

    cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_count);

    return static_cast<double>(h_count) / static_cast<double>(n);
}

template <>
double concentration_past_plane<float>(
    const ConstParticlesView<float>& particles,
    int axis,
    float threshold,
    cudaStream_t stream
) {
    return concentration_past_plane_impl<float>(particles, axis, threshold, stream);
}

template <>
double concentration_past_plane<double>(
    const ConstParticlesView<double>& particles,
    int axis,
    double threshold,
    cudaStream_t stream
) {
    return concentration_past_plane_impl<double>(particles, axis, threshold, stream);
}

// Status count kernel
__global__ void countByStatusKernel(
    int* counts,  // [active, exited, inactive]
    const uint8_t* status,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_active = 0, local_exited = 0, local_inactive = 0;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        uint8_t s = status[i];
        if (s == 0) local_active++;       // Active
        else if (s == 1) local_exited++;  // Exited
        else local_inactive++;            // Inactive
    }

    using BlockReduce = cub::BlockReduce<int, 256>;
    __shared__ typename BlockReduce::TempStorage temp;

    int ba = BlockReduce(temp).Reduce(local_active, cub::Sum()); __syncthreads();
    int be = BlockReduce(temp).Reduce(local_exited, cub::Sum()); __syncthreads();
    int bi = BlockReduce(temp).Reduce(local_inactive, cub::Sum());

    if (threadIdx.x == 0) {
        atomicAdd(&counts[0], ba);
        atomicAdd(&counts[1], be);
        atomicAdd(&counts[2], bi);
    }
}

template <typename T>
ParticleCounts count_by_status(
    const ConstParticlesView<T>& particles,
    cudaStream_t stream
) {
    ParticleCounts result;
    result.total = particles.n;

    if (particles.n <= 0) return result;

    if (particles.status == nullptr) {
        // No status array - all are active
        result.active = particles.n;
        return result;
    }

    int* d_counts = nullptr;
    int h_counts[3] = {0, 0, 0};

    cudaMalloc(&d_counts, 3 * sizeof(int));
    cudaMemsetAsync(d_counts, 0, 3 * sizeof(int), stream);

    const int n = particles.n;
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    countByStatusKernel<<<num_blocks, block_size, 0, stream>>>(
        d_counts, particles.status, n
    );

    cudaMemcpyAsync(h_counts, d_counts, 3 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_counts);

    result.active = h_counts[0];
    result.exited = h_counts[1];
    result.inactive = h_counts[2];

    return result;
}

// Explicit instantiations
template class StatsComputer<float>;
template class StatsComputer<double>;

template ParticleCounts count_by_status<float>(const ConstParticlesView<float>&, cudaStream_t);
template ParticleCounts count_by_status<double>(const ConstParticlesView<double>&, cudaStream_t);

} // namespace par2
