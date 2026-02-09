/**
 * @file stats.hpp
 * @brief Statistics utilities for Par2_Core.
 *
 * GPU-accelerated computation of particle statistics:
 * - Position moments (mean, variance, standard deviation)
 * - Status counts (active, exited, inactive)
 * - Legacy-compatible functions (concentrationBox, concentrationAfterX)
 *
 * ## Design Principles
 *
 * 1. **Stream-first**: All async operations take cudaStream_t
 * 2. **No hidden allocs**: StatsComputer pre-allocates buffers
 * 3. **Explicit sync**: User controls synchronization
 * 4. **Legacy-compatible**: Matches legacy semantics where applicable
 *
 * ## Usage Example
 *
 * ```cpp
 * // Create stats computer (allocates buffers once)
 * par2::StatsComputer<double> stats(num_particles);
 *
 * // Compute stats async
 * par2::StatsConfig cfg;
 * cfg.use_unwrapped = true;  // For periodic BC
 * stats.compute_async(particles, grid, cfg, stream);
 *
 * // Wait and fetch
 * cudaStreamSynchronize(stream);
 * auto result = stats.fetch_result();
 *
 * std::cout << "Mean X: " << result.moments.mean[0] << std::endl;
 * std::cout << "Active: " << result.counts.active << std::endl;
 * ```
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_STATS_HPP
#define PAR2_CORE_STATS_HPP

#include "views.hpp"
#include "grid.hpp"
#include <cuda_runtime.h>

namespace par2 {

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief 3D moment statistics (mean, variance, std).
 *
 * @tparam T Floating point type
 *
 * For unwrapped positions under periodic BC:
 *   x_unwrap = x + wrapX * Lx
 *
 * Variance uses the unbiased estimator (N-1 denominator).
 */
template <typename T>
struct Moments3 {
    T mean[3] = {T(0), T(0), T(0)};   ///< Mean position (x̄, ȳ, z̄)
    T var[3] = {T(0), T(0), T(0)};    ///< Variance (σ²_x, σ²_y, σ²_z)
    T std[3] = {T(0), T(0), T(0)};    ///< Standard deviation (σ_x, σ_y, σ_z)

    /// Check if any component is NaN
    bool has_nan() const noexcept {
        for (int i = 0; i < 3; ++i) {
            if (mean[i] != mean[i] || var[i] != var[i] || std[i] != std[i]) {
                return true;
            }
        }
        return false;
    }
};

/**
 * @brief Particle count statistics by status.
 *
 * - If no status array is provided, all particles are counted as "active"
 * - Counts should sum to total (active + exited + inactive == total)
 */
struct ParticleCounts {
    int total = 0;      ///< Total number of particles
    int active = 0;     ///< Particles with Status::Active (or no status)
    int exited = 0;     ///< Particles with Status::Exited
    int inactive = 0;   ///< Particles with Status::Inactive
};

/**
 * @brief Complete statistics result.
 *
 * @tparam T Floating point type
 */
template <typename T>
struct StatsResult {
    Moments3<T> moments;     ///< Position statistics
    ParticleCounts counts;   ///< Status counts
    bool computed = false;   ///< True if computation completed successfully

    /// Check validity (computed and no NaN)
    bool valid() const noexcept {
        return computed && !moments.has_nan();
    }
};

// =============================================================================
// Configuration
// =============================================================================

/**
 * @brief Configuration for statistics computation.
 */
struct StatsConfig {
    /**
     * @brief Use unwrapped positions for moment computation.
     *
     * When true, positions are computed as:
     *   x_unwrap = x + wrapX * Lx
     *
     * Requires ParticlesView to have wrapX/Y/Z set.
     * Essential for correct statistics under periodic BC.
     *
     * @note If wrapX/Y/Z are nullptr, this flag is ignored.
     */
    bool use_unwrapped = false;

    /**
     * @brief Compute moments only for Active particles.
     *
     * When true (default), Exited and Inactive particles are excluded
     * from mean/var/std computation.
     *
     * When false, all particles are included regardless of status.
     *
     * @note Legacy PAR² has no status concept, so all particles are
     *       effectively "active". Setting this to true matches legacy
     *       behavior when no status array is provided.
     */
    bool filter_active_only = true;
};

// =============================================================================
// StatsComputer Class
// =============================================================================

/**
 * @brief GPU-accelerated statistics computer with persistent buffers.
 *
 * @tparam T Floating point type (float or double)
 *
 * Computes position moments (mean, variance, std) and status counts
 * using GPU reduction kernels. Internal buffers are allocated once
 * and reused across calls.
 *
 * ## Thread Safety
 *
 * Not thread-safe. Use separate instances for different threads,
 * or serialize access with external synchronization.
 *
 * ## Memory Management
 *
 * - Device: Temporary reduction buffers (~8 * max_particles bytes)
 * - Host: Pinned memory for results (~100 bytes)
 */
template <typename T>
class StatsComputer {
public:
    /**
     * @brief Construct with capacity.
     *
     * @param max_particles Maximum particle count to support.
     *                      Larger values require more GPU memory.
     *
     * @throws std::bad_alloc if GPU allocation fails
     */
    explicit StatsComputer(int max_particles);

    /**
     * @brief Destructor - frees GPU and pinned memory.
     */
    ~StatsComputer();

    // Non-copyable
    StatsComputer(const StatsComputer&) = delete;
    StatsComputer& operator=(const StatsComputer&) = delete;

    // Movable
    StatsComputer(StatsComputer&& other) noexcept;
    StatsComputer& operator=(StatsComputer&& other) noexcept;

    /**
     * @brief Compute statistics asynchronously.
     *
     * Launches GPU kernels to compute moments and counts.
     * Results are written to pinned host memory asynchronously.
     *
     * @param particles Particle positions (device pointers)
     * @param grid Grid descriptor (for domain size in unwrap)
     * @param config Computation configuration
     * @param stream CUDA stream for async execution
     *
     * @return cudaSuccess on launch success, error code otherwise
     *
     * @note Does NOT synchronize. Caller must synchronize stream
     *       before calling fetch_result().
     *
     * @warning If particles.n > max_particles, returns cudaErrorInvalidValue.
     */
    cudaError_t compute_async(
        const ConstParticlesView<T>& particles,
        const GridDesc<T>& grid,
        const StatsConfig& config,
        cudaStream_t stream
    );

    /**
     * @brief Fetch results after stream synchronization.
     *
     * @return StatsResult with computed moments and counts.
     *
     * @pre Stream passed to compute_async() must be synchronized.
     * @pre compute_async() must have been called.
     *
     * @note If compute_async() was never called or returned an error,
     *       result.computed will be false.
     */
    StatsResult<T> fetch_result() const;

    /**
     * @brief Get the maximum capacity.
     */
    int capacity() const noexcept;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// =============================================================================
// Legacy-Compatible Convenience Functions
// =============================================================================

/**
 * @brief Count particles in a box region.
 *
 * Equivalent to legacy `concentrationBox()`.
 *
 * @tparam T Floating point type
 *
 * @param particles Particle positions (device)
 * @param x0,y0,z0 Minimum corner of box
 * @param x1,y1,z1 Maximum corner of box
 * @param stream CUDA stream (nullptr = default stream)
 *
 * @return Fraction of particles inside box [0.0, 1.0]
 *
 * @note This function SYNCHRONIZES the stream internally.
 * @warning This blocks the CPU until the stream completes.  Unsuitable
 *          for the hot path in HPC pipelines.  For async operation,
 *          use StatsComputer.
 *
 * @note Particles with any status are counted (no filtering).
 *       This matches legacy behavior.
 */
template <typename T>
double concentration_box(
    const ConstParticlesView<T>& particles,
    T x0, T y0, T z0,
    T x1, T y1, T z1,
    cudaStream_t stream = nullptr
);

/**
 * @brief Count particles past a plane.
 *
 * Equivalent to legacy `concentrationAfterX()` (when axis=0).
 *
 * @tparam T Floating point type
 *
 * @param particles Particle positions (device)
 * @param axis Which axis: 0=X, 1=Y, 2=Z
 * @param threshold Position of the plane
 * @param stream CUDA stream (nullptr = default stream)
 *
 * @return Fraction of particles with coord > threshold [0.0, 1.0]
 *
 * @note This function SYNCHRONIZES the stream internally.
 * @warning This blocks the CPU until the stream completes.  Unsuitable
 *          for the hot path in HPC pipelines.
 *
 * @note All particles counted regardless of status (legacy behavior).
 */
template <typename T>
double concentration_past_plane(
    const ConstParticlesView<T>& particles,
    int axis,
    T threshold,
    cudaStream_t stream = nullptr
);

/**
 * @brief Count particles by status.
 *
 * @tparam T Floating point type
 *
 * @param particles Particle view with status array
 * @param stream CUDA stream
 *
 * @return ParticleCounts with active/exited/inactive counts
 *
 * @note If particles.status is nullptr, all are counted as active.
 * @note This function SYNCHRONIZES the stream internally.
 */
template <typename T>
ParticleCounts count_by_status(
    const ConstParticlesView<T>& particles,
    cudaStream_t stream = nullptr
);

} // namespace par2

#endif // PAR2_CORE_STATS_HPP
