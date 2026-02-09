/**
 * @file workspace.cuh
 * @brief Persistent workspace for reusable GPU allocations.
 *
 * The Workspace manages GPU buffers that persist across simulation steps,
 * avoiding repeated allocations/deallocations in the hot path.
 *
 * ## M4 Design Principles
 *
 * 1. **Owner pattern**: Workspace owns all internal buffers
 * 2. **Growable**: Buffers resize only when capacity is insufficient
 * 3. **Stream-aware**: Async allocation where possible (cudaMallocAsync)
 * 4. **External priority**: User-provided buffers take precedence
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_INTERNAL_WORKSPACE_CUH
#define PAR2_INTERNAL_WORKSPACE_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

// Check for cudaMallocAsync support (CUDA 11.2+)
#if CUDART_VERSION >= 11020
#define PAR2_HAS_MALLOC_ASYNC 1
#else
#define PAR2_HAS_MALLOC_ASYNC 0
#endif

namespace par2 {
namespace workspace {

// =============================================================================
// Async-aware allocation helpers
// =============================================================================

/**
 * @brief Allocate GPU memory, preferring async allocation if available.
 *
 * Falls back to cudaMalloc if cudaMallocAsync is not supported.
 */
inline cudaError_t alloc_async(void** ptr, size_t size, cudaStream_t stream) {
#if PAR2_HAS_MALLOC_ASYNC
    return cudaMallocAsync(ptr, size, stream);
#else
    (void)stream;
    return cudaMalloc(ptr, size);
#endif
}

/**
 * @brief Free GPU memory, preferring async free if available.
 */
inline cudaError_t free_async(void* ptr, cudaStream_t stream) {
    if (ptr == nullptr) return cudaSuccess;
#if PAR2_HAS_MALLOC_ASYNC
    return cudaFreeAsync(ptr, stream);
#else
    (void)stream;
    return cudaFree(ptr);
#endif
}

// =============================================================================
// Workspace - Consolidated GPU buffer management
// =============================================================================

/**
 * @brief Persistent workspace for GPU allocations.
 *
 * Holds all internal buffers needed by TransportEngine:
 * - RNG states (per-particle curand states)
 * - Corner velocity field (computed from face field when needed)
 * - Status array (for particle tracking with Open BC)
 * - Wrap counters (for periodic BC tracking)
 *
 * ## Usage Pattern
 *
 * ```cpp
 * // Outside hot-path (in prepare() or after bind_particles):
 * workspace.ensure_rng(n_particles, stream);
 * workspace.ensure_corner(grid, stream);
 *
 * // In step(): no allocations - just use pointers
 * kernel<<<...>>>(workspace.rng_states, ...);
 * ```
 *
 * The workspace avoids allocations in the hot path (step()).
 */
template <typename T>
struct Workspace {
    // =========================================================================
    // Capacity tracking
    // =========================================================================
    int particle_capacity = 0;    ///< Allocated capacity for particle arrays
    int corner_capacity = 0;      ///< Allocated capacity for corner arrays

    // =========================================================================
    // RNG states (per-particle)
    // =========================================================================
    curandState_t* rng_states = nullptr;
    bool rng_initialized = false;  ///< True if RNG kernel has been run

    // =========================================================================
    // Corner velocity field (derived from face field)
    // =========================================================================
    T* corner_Uc = nullptr;
    T* corner_Vc = nullptr;
    T* corner_Wc = nullptr;

    bool corner_dirty = true;      ///< True if corner field needs recomputation
    bool corner_external = false;  ///< True if user provided external corner velocity

    // =========================================================================
    // Particle tracking (M3)
    // =========================================================================
    uint8_t* status = nullptr;
    bool status_external = false;

    int32_t* wrapX = nullptr;
    int32_t* wrapY = nullptr;
    int32_t* wrapZ = nullptr;
    bool wrap_external = false;

    // =========================================================================
    // ensure_* methods (call OUTSIDE hot-path)
    // =========================================================================

    /**
     * @brief Ensure RNG states buffer has sufficient capacity.
     *
     * @param n_particles Required number of particles
     * @param stream CUDA stream for async allocation
     * @return true if allocation succeeded or capacity already sufficient
     *
     * @note Does NOT initialize RNG states. Call init_rng() separately.
     */
    bool ensure_rng(int n_particles, cudaStream_t stream = nullptr) {
        if (n_particles <= particle_capacity && rng_states != nullptr) {
            return true;  // Already sufficient
        }

        // Free old buffer
        free_async(rng_states, stream);
        rng_states = nullptr;
        rng_initialized = false;

        if (n_particles <= 0) {
            particle_capacity = 0;
            return true;
        }

        // Allocate new (with some growth factor to avoid frequent reallocs)
        int new_cap = (n_particles * 5 / 4) + 64;  // 25% growth + minimum
        cudaError_t err = alloc_async(
            reinterpret_cast<void**>(&rng_states),
            new_cap * sizeof(curandState_t),
            stream
        );

        if (err != cudaSuccess) {
            particle_capacity = 0;
            return false;
        }

        particle_capacity = new_cap;
        return true;
    }

    /**
     * @brief Ensure corner velocity buffers have sufficient capacity.
     *
     * @param num_corners Number of corners = (nx+1)*(ny+1)*(nz+1)
     * @param stream CUDA stream for async allocation
     * @return true if allocation succeeded
     */
    bool ensure_corner(int num_corners, cudaStream_t stream = nullptr) {
        if (corner_external) return true;  // User provided

        if (num_corners <= corner_capacity &&
            corner_Uc && corner_Vc && corner_Wc) {
            return true;  // Already sufficient
        }

        // Free old buffers
        free_async(corner_Uc, stream);
        free_async(corner_Vc, stream);
        free_async(corner_Wc, stream);
        corner_Uc = corner_Vc = corner_Wc = nullptr;
        corner_capacity = 0;

        if (num_corners <= 0) return true;

        cudaError_t err;
        err = alloc_async(reinterpret_cast<void**>(&corner_Uc),
                         num_corners * sizeof(T), stream);
        if (err != cudaSuccess) return false;

        err = alloc_async(reinterpret_cast<void**>(&corner_Vc),
                         num_corners * sizeof(T), stream);
        if (err != cudaSuccess) {
            free_async(corner_Uc, stream);
            corner_Uc = nullptr;
            return false;
        }

        err = alloc_async(reinterpret_cast<void**>(&corner_Wc),
                         num_corners * sizeof(T), stream);
        if (err != cudaSuccess) {
            free_async(corner_Uc, stream);
            free_async(corner_Vc, stream);
            corner_Uc = corner_Vc = nullptr;
            return false;
        }

        corner_capacity = num_corners;
        corner_dirty = true;  // New buffers need computation
        return true;
    }

    /**
     * @brief Ensure status array has sufficient capacity.
     *
     * @param n_particles Required number of particles
     * @param stream CUDA stream
     * @return true if allocation succeeded
     */
    bool ensure_status(int n_particles, cudaStream_t stream = nullptr) {
        if (status_external) return true;

        // Check if we need to grow (status shares particle_capacity logic)
        if (n_particles <= particle_capacity && status != nullptr) {
            return true;
        }

        free_async(status, stream);
        status = nullptr;

        if (n_particles <= 0) return true;

        int cap = (n_particles * 5 / 4) + 64;
        cudaError_t err = alloc_async(
            reinterpret_cast<void**>(&status),
            cap * sizeof(uint8_t),
            stream
        );

        if (err != cudaSuccess) return false;

        // Initialize to Active (0) - async memset
        cudaMemsetAsync(status, 0, cap * sizeof(uint8_t), stream);
        return true;
    }

    /**
     * @brief Ensure wrap counter arrays have sufficient capacity.
     *
     * @param n_particles Required number of particles
     * @param need_x, need_y, need_z Which axes need wrap counters
     * @param stream CUDA stream
     * @return true if allocation succeeded
     */
    bool ensure_wrap(int n_particles, bool need_x, bool need_y, bool need_z,
                     cudaStream_t stream = nullptr) {
        if (wrap_external) return true;

        // Check current capacity
        bool ok = true;
        if (need_x && (wrapX == nullptr || n_particles > particle_capacity)) ok = false;
        if (need_y && (wrapY == nullptr || n_particles > particle_capacity)) ok = false;
        if (need_z && (wrapZ == nullptr || n_particles > particle_capacity)) ok = false;
        if (ok) return true;

        // Free old if internal
        if (!wrap_external) {
            free_async(wrapX, stream);
            free_async(wrapY, stream);
            free_async(wrapZ, stream);
        }
        wrapX = wrapY = wrapZ = nullptr;

        if (n_particles <= 0) return true;

        int cap = (n_particles * 5 / 4) + 64;
        cudaError_t err;

        if (need_x) {
            err = alloc_async(reinterpret_cast<void**>(&wrapX),
                             cap * sizeof(int32_t), stream);
            if (err != cudaSuccess) return false;
            cudaMemsetAsync(wrapX, 0, cap * sizeof(int32_t), stream);
        }

        if (need_y) {
            err = alloc_async(reinterpret_cast<void**>(&wrapY),
                             cap * sizeof(int32_t), stream);
            if (err != cudaSuccess) {
                free_async(wrapX, stream); wrapX = nullptr;
                return false;
            }
            cudaMemsetAsync(wrapY, 0, cap * sizeof(int32_t), stream);
        }

        if (need_z) {
            err = alloc_async(reinterpret_cast<void**>(&wrapZ),
                             cap * sizeof(int32_t), stream);
            if (err != cudaSuccess) {
                free_async(wrapX, stream); wrapX = nullptr;
                free_async(wrapY, stream); wrapY = nullptr;
                return false;
            }
            cudaMemsetAsync(wrapZ, 0, cap * sizeof(int32_t), stream);
        }

        return true;
    }

    // =========================================================================
    // Drift correction field (Precomputed mode)
    // =========================================================================
    T* drift_x = nullptr;     ///< X-component of div(D) [cell-centered]
    T* drift_y = nullptr;     ///< Y-component of div(D) [cell-centered]
    T* drift_z = nullptr;     ///< Z-component of div(D) [cell-centered]
    
    // Temporary buffers for D tensor computation
    T* temp_D11 = nullptr;    ///< D_xx component [cell-centered]
    T* temp_D22 = nullptr;    ///< D_yy component [cell-centered]
    T* temp_D33 = nullptr;    ///< D_zz component [cell-centered]
    T* temp_D12 = nullptr;    ///< D_xy component [cell-centered]
    T* temp_D13 = nullptr;    ///< D_xz component [cell-centered]
    T* temp_D23 = nullptr;    ///< D_yz component [cell-centered]
    
    int drift_capacity = 0;       ///< Allocated capacity for drift arrays
    bool drift_dirty = true;      ///< True if drift needs recomputation
    bool drift_external = false;  ///< True if user provided external drift

    /**
     * @brief Ensure drift correction buffers have sufficient capacity.
     *
     * @param num_cells Number of cells = nx*ny*nz
     * @param stream CUDA stream for async allocation
     * @return true if allocation succeeded
     */
    bool ensure_drift(int num_cells, cudaStream_t stream = nullptr) {
        if (drift_external) return true;  // User provided

        if (num_cells <= drift_capacity &&
            drift_x && drift_y && drift_z &&
            temp_D11 && temp_D22 && temp_D33 &&
            temp_D12 && temp_D13 && temp_D23) {
            return true;  // Already sufficient
        }

        // Free old buffers
        free_async(drift_x, stream);
        free_async(drift_y, stream);
        free_async(drift_z, stream);
        free_async(temp_D11, stream);
        free_async(temp_D22, stream);
        free_async(temp_D33, stream);
        free_async(temp_D12, stream);
        free_async(temp_D13, stream);
        free_async(temp_D23, stream);
        drift_x = drift_y = drift_z = nullptr;
        temp_D11 = temp_D22 = temp_D33 = nullptr;
        temp_D12 = temp_D13 = temp_D23 = nullptr;
        drift_capacity = 0;

        if (num_cells <= 0) return true;

        cudaError_t err;
        
        // Allocate drift output buffers
        err = alloc_async(reinterpret_cast<void**>(&drift_x),
                         num_cells * sizeof(T), stream);
        if (err != cudaSuccess) return false;

        err = alloc_async(reinterpret_cast<void**>(&drift_y),
                         num_cells * sizeof(T), stream);
        if (err != cudaSuccess) { free_drift(stream); return false; }

        err = alloc_async(reinterpret_cast<void**>(&drift_z),
                         num_cells * sizeof(T), stream);
        if (err != cudaSuccess) { free_drift(stream); return false; }

        // Allocate temporary D tensor buffers
        err = alloc_async(reinterpret_cast<void**>(&temp_D11),
                         num_cells * sizeof(T), stream);
        if (err != cudaSuccess) { free_drift(stream); return false; }

        err = alloc_async(reinterpret_cast<void**>(&temp_D22),
                         num_cells * sizeof(T), stream);
        if (err != cudaSuccess) { free_drift(stream); return false; }

        err = alloc_async(reinterpret_cast<void**>(&temp_D33),
                         num_cells * sizeof(T), stream);
        if (err != cudaSuccess) { free_drift(stream); return false; }

        err = alloc_async(reinterpret_cast<void**>(&temp_D12),
                         num_cells * sizeof(T), stream);
        if (err != cudaSuccess) { free_drift(stream); return false; }

        err = alloc_async(reinterpret_cast<void**>(&temp_D13),
                         num_cells * sizeof(T), stream);
        if (err != cudaSuccess) { free_drift(stream); return false; }

        err = alloc_async(reinterpret_cast<void**>(&temp_D23),
                         num_cells * sizeof(T), stream);
        if (err != cudaSuccess) { free_drift(stream); return false; }

        drift_capacity = num_cells;
        drift_dirty = true;  // New buffers need computation
        return true;
    }

    /// Helper to free all drift buffers
    void free_drift(cudaStream_t stream = nullptr) {
        free_async(drift_x, stream);
        free_async(drift_y, stream);
        free_async(drift_z, stream);
        free_async(temp_D11, stream);
        free_async(temp_D22, stream);
        free_async(temp_D33, stream);
        free_async(temp_D12, stream);
        free_async(temp_D13, stream);
        free_async(temp_D23, stream);
        drift_x = drift_y = drift_z = nullptr;
        temp_D11 = temp_D22 = temp_D33 = nullptr;
        temp_D12 = temp_D13 = temp_D23 = nullptr;
        drift_capacity = 0;
    }

    // =========================================================================
    // Query methods
    // =========================================================================

    /// Check if RNG states are allocated and initialized
    bool has_rng() const noexcept {
        return rng_states != nullptr && rng_initialized;
    }

    /// Check if corner buffers are allocated
    bool has_corner() const noexcept {
        return corner_external ||
               (corner_Uc && corner_Vc && corner_Wc && corner_capacity > 0);
    }

    /// Check if status buffer is ready
    bool has_status() const noexcept {
        return status_external || status != nullptr;
    }

    /// Check if drift buffers are allocated and computed
    bool has_drift() const noexcept {
        return drift_external ||
               (drift_x && drift_y && drift_z && drift_capacity > 0 && !drift_dirty);
    }

    // =========================================================================
    // Mark as external (user provided)
    // =========================================================================

    void set_corner_external(bool external) {
        corner_external = external;
        if (external) corner_dirty = false;
    }

    void set_status_external(bool external) {
        status_external = external;
    }

    void set_wrap_external(bool external) {
        wrap_external = external;
    }

    void set_drift_external(bool external) {
        drift_external = external;
        if (external) drift_dirty = false;
    }

    // =========================================================================
    // Cleanup
    // =========================================================================

    ~Workspace() {
        free_all(nullptr);
    }

    /**
     * @brief Free all internal GPU allocations.
     *
     * @param stream CUDA stream for async free (nullptr = sync free)
     */
    void free_all(cudaStream_t stream) {
        free_async(rng_states, stream);
        rng_states = nullptr;
        rng_initialized = false;
        particle_capacity = 0;

        if (!corner_external) {
            free_async(corner_Uc, stream);
            free_async(corner_Vc, stream);
            free_async(corner_Wc, stream);
        }
        corner_Uc = corner_Vc = corner_Wc = nullptr;
        corner_capacity = 0;
        corner_dirty = true;
        corner_external = false;

        if (!status_external) {
            free_async(status, stream);
        }
        status = nullptr;
        status_external = false;

        if (!wrap_external) {
            free_async(wrapX, stream);
            free_async(wrapY, stream);
            free_async(wrapZ, stream);
        }
        wrapX = wrapY = wrapZ = nullptr;
        wrap_external = false;

        // Free drift buffers
        if (!drift_external) {
            free_drift(stream);
        }
        drift_external = false;
        drift_dirty = true;
    }

    // Prevent copying
    Workspace() = default;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace(Workspace&&) = default;
    Workspace& operator=(Workspace&&) = default;
};

// Legacy compatibility alias
template <typename T>
bool ensure_corner_allocated(Workspace<T>& ws, int num_corners) {
    return ws.ensure_corner(num_corners, nullptr);
}

} // namespace workspace
} // namespace par2

#endif // PAR2_INTERNAL_WORKSPACE_CUH
