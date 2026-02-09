/**
 * @file transport_engine.hpp
 * @brief TransportEngine - GPU particle tracking engine.
 *
 * This is the main public interface for Par2_Core. The engine:
 * - Binds to user-provided GPU velocity fields (no copy)
 * - Binds to user-provided GPU particle arrays (no copy)
 * - Executes RWPT steps asynchronously on a CUDA stream
 * - Provides particle injection utilities
 *
 * @note All device pointers passed to bind_* must remain valid for the
 *       lifetime of the binding (until next bind or engine destruction).
 *
 * @note The engine does NOT call cudaDeviceSynchronize() in the hot path.
 *       Synchronization is the caller's responsibility.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_TRANSPORT_ENGINE_HPP
#define PAR2_CORE_TRANSPORT_ENGINE_HPP

#include "grid.hpp"
#include "types.hpp"
#include "views.hpp"
#include "boundary.hpp"

#include <cuda_runtime.h>
#include <memory>

namespace par2 {

// Forward declaration of implementation
namespace detail {
template <typename T> struct EngineImpl;
}

/**
 * @brief GPU-native particle tracking engine.
 *
 * @tparam T Floating point type (float or double)
 *
 * ## Thread Safety
 *
 * TransportEngine is **NOT thread-safe** for concurrent host calls.
 * All bind_*, step(), synchronize(), etc. must be serialized from the
 * host side.  Multiple engines on different CUDA streams can run
 * concurrently (one host thread per engine, or explicit serialization).
 *
 * ## Usage Example
 *
 * ```cpp
 * // Setup
 * auto grid = par2::make_grid<double>(100, 100, 10, 1.0, 1.0, 1.0);
 * par2::TransportParams<double> params{1e-9, 0.1, 0.01};
 * par2::BoundaryConfig<double> bc = par2::BoundaryConfig<double>::all_closed();
 * par2::EngineConfig config;
 *
 * par2::TransportEngine<double> engine(grid, params, bc, config);
 *
 * // Bind velocity field (already on GPU)
 * engine.bind_velocity({d_U, d_V, d_W, grid.num_corners()});
 *
 * // Bind particles (already on GPU)
 * engine.bind_particles({d_x, d_y, d_z, num_particles});
 *
 * // Inject particles in a box
 * engine.inject_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
 *
 * // Run simulation
 * double dt = 0.1;
 * for (int i = 0; i < 1000; ++i) {
 *     engine.step(dt);  // Async, no sync inside
 * }
 *
 * cudaStreamSynchronize(engine.stream());  // Sync when needed
 * ```
 */
template <typename T>
class TransportEngine {
public:
    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    /**
     * @brief Construct a transport engine.
     *
     * @param grid Grid descriptor (geometry)
     * @param params Transport parameters (diffusion, dispersivity)
     * @param bc Boundary conditions
     * @param config Engine configuration
     *
     * @throws std::runtime_error if CUDA initialization fails
     */
    TransportEngine(
        const GridDesc<T>& grid,
        const TransportParams<T>& params,
        const BoundaryConfig<T>& bc,
        const EngineConfig& config = EngineConfig{}
    );

    /// Destructor - releases CUDA resources
    ~TransportEngine();

    // Non-copyable, movable
    TransportEngine(const TransportEngine&) = delete;
    TransportEngine& operator=(const TransportEngine&) = delete;
    TransportEngine(TransportEngine&&) noexcept;
    TransportEngine& operator=(TransportEngine&&) noexcept;

    // =========================================================================
    // Stream Management (HPC Pipeline Integration)
    // =========================================================================

    /**
     * @brief Set the CUDA stream for all operations.
     *
     * @param stream CUDA stream (nullptr for default stream)
     *
     * All subsequent operations (step, inject, etc.) will use this stream.
     * The engine does NOT own the stream - caller must manage its lifetime.
     *
     * ## HPC Pipeline Usage
     *
     * For multi-solver pipelines, use a dedicated stream:
     * ```cpp
     * cudaStream_t transport_stream;
     * cudaStreamCreate(&transport_stream);
     * engine.set_stream(transport_stream);
     *
     * // Now engine operations are independent of other streams
     * engine.step(dt);  // Enqueued on transport_stream
     * ```
     *
     * @note The engine never calls cudaDeviceSynchronize() internally.
     *       Use synchronize() or events for coordination.
     */
    void set_stream(cudaStream_t stream) noexcept;

    /**
     * @brief Get the current CUDA stream.
     */
    cudaStream_t stream() const noexcept;

    // =========================================================================
    // Binding (zero-copy attachment to user data)
    // =========================================================================

    /**
     * @brief Bind velocity field (device pointers).
     *
     * @param vel Velocity view with device pointers to U, V, W arrays
     *
     * @pre vel.size == grid.num_corners()  (= (nx+1)*(ny+1)*(nz+1))
     * @pre All pointers are valid device memory
     * @post needs_corner_update() == true
     * @post needs_drift_update() == true (if using Precomputed drift)
     *
     * The engine does NOT copy this data — operates directly on user buffers.
     * The buffers must remain valid until the next bind_velocity() call or
     * engine destruction.
     *
     * @warning After calling this, you must call update_derived_fields()
     *          before the next step() if using Trilinear interpolation or
     *          on-the-fly drift correction.
     * @warning Clears any previous bind_corner_velocity() external binding.
     */
    void bind_velocity(const VelocityView<T>& vel);

    /**
     * @brief Bind external corner velocity field (for trilinear mode).
     *
     * @param cvel Corner velocity view with (nx+1)×(ny+1)×(nz+1) grid
     *
     * Use this when you have **pre-computed** corner velocities (e.g., from
     * MODFLOW/MT3DMS output). When bound, `update_derived_fields()` will
     * skip the internal corner computation and use these values instead.
     *
     * If not bound, call `update_derived_fields()` after `bind_velocity()`
     * to compute corner velocities from the face field automatically.
     *
     * @note External binding takes priority over internal computation.
     *       Call `bind_velocity()` again to clear external binding and
     *       revert to internal computation.
     */
    void bind_corner_velocity(const CornerVelocityView<T>& cvel);

    /**
     * @brief Bind particle arrays (device pointers).
     *
     * @param particles Particle view with device pointers to x, y, z arrays
     *
     * @pre particles.n <= allocated size of arrays
     * @pre All pointers are valid device memory
     * @post is_prepared() == false — you must call prepare() before step()
     *
     * The engine writes to these arrays during step() and inject_*().
     *
     * @warning The particle arrays must remain valid until the next
     *          bind_particles() call or engine destruction.
     * @warning If particles.status / wrapX/Y/Z are nullptr and the
     *          boundary config requires them, call ensure_tracking_arrays()
     *          or prepare() to auto-allocate.
     */
    void bind_particles(const ParticlesView<T>& particles);

    /**
     * @brief Bind precomputed drift correction field.
     *
     * @param dc Drift correction view
     *
     * Only used when config.drift_correction == DriftCorrectionMode::Precomputed.
     */
    void bind_drift_correction(const DriftCorrectionView<T>& dc);

    // =========================================================================
    // Derived Fields
    // =========================================================================

    /**
     * @brief Update derived fields (corner velocity) from face velocity.
     *
     * @param stream CUDA stream for the computation (default: engine stream)
     *
     * **IMPORTANT**: Call this after changing U/V/W when using:
     *   - InterpolationMode::Trilinear
     *   - DriftCorrectionMode::TrilinearOnFly
     *
     * If you bound an external corner velocity via bind_corner_velocity(),
     * this call is a no-op (external binding takes priority).
     *
     * This is ASYNC: kernel launches on stream, returns immediately.
     * No cudaDeviceSynchronize() inside.
     */
    void update_derived_fields(cudaStream_t stream = nullptr);

    /**
     * @brief Check if corner velocity needs update.
     *
     * @return true if update_derived_fields() should be called
     */
    bool needs_corner_update() const noexcept;

    /**
     * @brief Check if precomputed drift needs update.
     *
     * @return true if update_derived_fields() should be called (Precomputed mode)
     */
    bool needs_drift_update() const noexcept;

    // =========================================================================
    // Simulation
    // =========================================================================

    /**
     * @brief Execute one RWPT step.
     *
     * @param dt Time step size
     *
     * @pre has_velocity() == true
     * @pre has_particles() == true
     * @pre is_prepared() == true (call prepare() first)
     * @pre needs_corner_update() == false
     * @pre needs_drift_update() == false
     * @pre dt > 0
     *
     * This is ASYNC: the kernel is launched on the configured stream
     * and returns immediately. NO cudaDeviceSynchronize() inside.
     * NO allocations occur — all workspace was set up in prepare().
     *
     * After return, particle positions are updated (on GPU).
     * Particles with ParticleStatus::Exited are skipped.
     *
     * @warning Calling step() without prepare() causes an assertion
     *          failure in debug builds or undefined behavior in release.
     */
    void step(T dt);

    /**
     * @brief Execute multiple RWPT steps.
     *
     * @param dt Time step size per step
     * @param num_steps Number of steps to execute
     *
     * @pre Same preconditions as step()
     *
     * Equivalent to calling step(dt) num_steps times.
     * All steps run on the same stream without intermediate syncs.
     * No allocations, no synchronization during the loop.
     */
    void advance(T dt, int num_steps);

    // =========================================================================
    // Particle Injection
    // =========================================================================

    /**
     * @brief Initialize particles uniformly in a box.
     *
     * @param x0, y0, z0 Minimum corner of box
     * @param x1, y1, z1 Maximum corner of box
     * @param first_particle Starting index (default: 0)
     * @param count Number of particles to initialize (default: all bound)
     *
     * Particles are distributed uniformly within [x0,x1] x [y0,y1] x [z0,z1].
     * This uses a deterministic pattern based on particle index.
     *
     * @pre Particles are bound
     * @pre Box is within domain (no validation - user responsibility)
     *
     * This is ASYNC: kernel launched on configured stream.
     */
    void inject_box(
        T x0, T y0, T z0,
        T x1, T y1, T z1,
        int first_particle = 0,
        int count = -1  // -1 = all particles
    );

    // =========================================================================
    // Queries
    // =========================================================================

    /// Get the grid descriptor
    const GridDesc<T>& grid() const noexcept;

    /// Get the transport parameters
    const TransportParams<T>& params() const noexcept;

    /// Get the boundary configuration
    const BoundaryConfig<T>& boundary() const noexcept;

    /// Get the engine configuration
    const EngineConfig& config() const noexcept;

    /// Get number of bound particles
    int num_particles() const noexcept;

    /// Check if velocity is bound
    bool has_velocity() const noexcept;

    /// Check if particles are bound
    bool has_particles() const noexcept;

    // =========================================================================
    // M3: Tracking Support
    // =========================================================================

    /**
     * @brief Ensure tracking arrays are allocated for boundary conditions.
     *
     * Call this after bind_particles() if using:
     * - Open BC: allocates status array if not provided by user
     * - Periodic BC: allocates wrapX/Y/Z if not provided by user
     *
     * @pre has_particles() == true
     *
     * This is NOT called automatically to avoid hidden allocations.
     * Safe to call multiple times — no-op if already allocated.
     * User-provided buffers in ParticlesView take priority (not overwritten).
     *
     * @note Also called internally by prepare().
     *
     * @throws std::runtime_error if allocation fails
     */
    void ensure_tracking_arrays();

    // =========================================================================
    // M4: Pipeline Integration (stream/event semantics)
    // =========================================================================

    /**
     * @brief Prepare engine for simulation (allocate + initialize workspace).
     *
     * **MUST be called after bind_particles() and before step()/advance()**.
     *
     * @pre has_particles() == true
     * @post is_prepared() == true
     * @post step() and advance() can be called without allocations
     *
     * This method:
     * - Allocates RNG states if needed (one-time or grow)
     * - Initializes RNG states with kernel (seed from EngineConfig::rng_seed)
     * - Calls ensure_tracking_arrays() for Open/Periodic BC
     * - Computes corner velocity if trilinear mode and face velocity bound
     *
     * All allocations use cudaMallocAsync when available (CUDA 11.2+).
     * After prepare(), step()/advance() will NOT allocate.
     *
     * @param stream CUDA stream for allocation/init kernels (default: engine stream)
     *
     * @throws std::runtime_error if allocation fails
     *
     * ## Usage Pattern (Pipeline)
     * ```cpp
     * engine.bind_velocity(vel);
     * engine.bind_particles(particles);
     * engine.prepare();  // <-- All allocations happen here
     *
     * // Hot loop - no allocations, fully async
     * for (int i = 0; i < 1000; ++i) {
     *     engine.step(dt);
     * }
     *
     * engine.synchronize();  // Wait for completion
     * ```
     */
    void prepare(cudaStream_t stream = nullptr);

    /**
     * @brief Check if engine is prepared for stepping.
     *
     * @return true if prepare() has been called and RNG states are ready
     */
    bool is_prepared() const noexcept;

    /**
     * @brief Synchronize the engine stream (wait for all pending work).
     *
     * Equivalent to cudaStreamSynchronize(stream()), but preferred for clarity.
     * Use this to wait for step()/advance() to complete.
     *
     * @throws std::runtime_error if synchronization fails
     *
     * @note This is the ONLY synchronization point the engine exposes.
     *       The engine NEVER calls cudaDeviceSynchronize() internally.
     */
    void synchronize();

    /**
     * @brief Record a CUDA event on the engine stream.
     *
     * @param event Pre-created CUDA event
     *
     * Use this for fine-grained pipeline synchronization without blocking:
     * ```cpp
     * engine.step(dt);
     * engine.record_event(done_event);
     * // Other solver can cudaStreamWaitEvent(its_stream, done_event, 0);
     * ```
     *
     * @throws std::runtime_error if event recording fails
     */
    void record_event(cudaEvent_t event);

    /**
     * @brief Wait for an external event before continuing.
     *
     * @param event External CUDA event to wait on
     *
     * Makes the engine stream wait for an event from another stream:
     * ```cpp
     * // Flow solver records event after updating velocity
     * cudaEventRecord(vel_ready, flow_stream);
     * // Transport waits for velocity before stepping
     * engine.wait_event(vel_ready);
     * engine.step(dt);
     * ```
     *
     * @note This does NOT block the CPU - the wait happens on the GPU.
     */
    void wait_event(cudaEvent_t event);

    // =========================================================================
    // HPC Pipeline Integration: Multi-Stream Example
    // =========================================================================
    //
    // The engine is designed as a "good citizen" in multi-solver pipelines:
    //
    // ```cpp
    // // Create streams for different solvers
    // cudaStream_t flow_stream, transport_stream;
    // cudaStreamCreate(&flow_stream);
    // cudaStreamCreate(&transport_stream);
    //
    // // Create synchronization events
    // cudaEvent_t velocity_ready, transport_done;
    // cudaEventCreateWithFlags(&velocity_ready, cudaEventDisableTiming);
    // cudaEventCreateWithFlags(&transport_done, cudaEventDisableTiming);
    //
    // // Configure engine
    // engine.set_stream(transport_stream);
    //
    // // Coupled simulation loop (no CPU blocking!)
    // for (int t = 0; t < num_timesteps; ++t) {
    //     // Flow solver updates velocity on flow_stream
    //     flow_kernel<<<...>>>(d_velocity, ..., flow_stream);
    //     cudaEventRecord(velocity_ready, flow_stream);
    //
    //     // Transport waits for velocity, then steps
    //     engine.wait_event(velocity_ready);
    //     engine.step(dt);
    //     engine.record_event(transport_done);
    //
    //     // Flow solver can wait for transport if needed
    //     cudaStreamWaitEvent(flow_stream, transport_done, 0);
    // }
    //
    // // Only sync at end (or for output)
    // engine.synchronize();
    // ```
    //

    // =========================================================================
    // M5: Data Access (positions, unwrap, stats)
    // =========================================================================

    /**
     * @brief Get read-only view of current particle data (device pointers).
     *
     * Returns the bound particle view with:
     * - x, y, z: wrapped positions on device
     * - status: particle status (if tracking enabled)
     * - wrapX/Y/Z: wrap counters (if periodic BC with tracking)
     *
     * **Important**: The returned pointers are device memory. To read on CPU:
     * 1. Call engine.synchronize() to ensure step() completed
     * 2. Use cudaMemcpy to copy to host
     *
     * Or use compute_unwrapped_positions() + download helpers for full workflow.
     *
     * @return ConstParticlesView with device pointers (read-only)
     */
    ConstParticlesView<T> particles() const noexcept;

    /**
     * @brief Compute unwrapped (continuous) positions on-demand.
     *
     * For periodic BC, particles wrap around the domain. This kernel computes
     * the continuous positions:
     *
     *   x_u = x + wrapX * Lx
     *   y_u = y + wrapY * Ly
     *   z_u = z + wrapZ * Lz
     *
     * If an axis is not periodic or wrapCount is nullptr, the wrapped
     * position is copied directly (x_u = x).
     *
     * @param out Output view (user-allocated device arrays, capacity >= n)
     * @param stream CUDA stream (default: engine stream)
     *
     * @pre out.capacity >= num_particles()
     * @pre All output pointers are valid device memory
     *
     * This is ASYNC: kernel launches on stream, returns immediately.
     * No allocations, no sync.
     *
     * ## Usage
     * ```cpp
     * // Allocate output once
     * UnwrappedPositionsView<double> unwrap{d_xu, d_yu, d_zu, n};
     *
     * // After each step (or when needed):
     * engine.compute_unwrapped_positions(unwrap);
     * engine.synchronize();  // if need to read on CPU
     * ```
     */
    void compute_unwrapped_positions(UnwrappedPositionsView<T> out,
                                     cudaStream_t stream = nullptr);

private:
    std::unique_ptr<detail::EngineImpl<T>> impl_;
};

// Explicit instantiations declared (defined in .cu)
extern template class TransportEngine<float>;
extern template class TransportEngine<double>;

} // namespace par2

#endif // PAR2_CORE_TRANSPORT_ENGINE_HPP
