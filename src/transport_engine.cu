/**
 * @file transport_engine.cu
 * @brief Implementation of TransportEngine GPU particle tracking.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#include <par2_core/transport_engine.hpp>
#include "internal/cuda_check.cuh"
#include "kernels/move_particles.cuh"
#include "kernels/cornerfield.cuh"
#include "kernels/drift_correction.cuh"
#include "kernels/unwrap_positions.cuh"
#include "workspace/workspace.cuh"

#include <curand_kernel.h>
#include <stdexcept>
#include <cassert>

namespace par2 {

// =============================================================================
// Internal Helper: Debug check after kernel launch
// =============================================================================

/**
 * @brief Check for CUDA kernel launch errors.
 *
 * Compile-time check when PAR2_DEBUG_LEVEL >= 1.
 * Runtime fallback when debug_checks=true and compile-time disabled.
 *
 * @note Does NOT synchronize - only detects launch config errors.
 */
#if PAR2_ENABLE_CUDA_CHECKS
#  define PAR2_CHECK_KERNEL_LAUNCH(config) PAR2_CUDA_CHECK_LAST()
#else
#  define PAR2_CHECK_KERNEL_LAUNCH(config) \
      do { if ((config).debug_checks) ::par2::detail::cuda_check_last(__FILE__, __LINE__); } while(0)
#endif

namespace detail {

// =============================================================================
// Engine Implementation (PIMPL)
// =============================================================================

template <typename T>
struct EngineImpl {
    // Configuration (immutable after construction)
    GridDesc<T> grid;
    TransportParams<T> params;
    BoundaryConfig<T> boundary;
    EngineConfig config;

    // CUDA stream
    cudaStream_t stream_ = nullptr;

    // Bound views (non-owning pointers to user data)
    VelocityView<T> velocity;
    CornerVelocityView<T> corner_velocity;
    ParticlesView<T> particles;
    DriftCorrectionView<T> drift_correction;

    // Workspace owns ALL internal allocations (M4: consolidated owner)
    workspace::Workspace<T> workspace;

    // State flags
    bool corner_dirty = true;
    bool corner_external = false;  // True if user bound corner velocity
    bool prepared = false;         // True if prepare() was called successfully

    // Constructor
    EngineImpl(
        const GridDesc<T>& g,
        const TransportParams<T>& p,
        const BoundaryConfig<T>& bc,
        const EngineConfig& cfg
    ) : grid(g), params(p), boundary(bc), config(cfg) {}

    // Destructor - workspace handles its own cleanup
    ~EngineImpl() = default;

    // Initialize RNG states for n particles (uses workspace)
    void init_rng_states(int n, cudaStream_t stream) {
        if (n <= 0) return;

        // Ensure capacity
        if (!workspace.ensure_rng(n, stream)) {
            throw std::runtime_error("Failed to allocate RNG states");
        }

        // Skip init if already done for this capacity
        if (workspace.rng_initialized && n <= workspace.particle_capacity) {
            return;
        }

        // Initialize states with kernel
        const int block_size = config.kernel_block_size;
        const int num_blocks = (n + block_size - 1) / block_size;
        kernels::init_rng_states<<<num_blocks, block_size, 0, stream>>>(
            workspace.rng_states, n, config.rng_seed
        );

        PAR2_CHECK_KERNEL_LAUNCH(config);

        workspace.rng_initialized = true;
    }
};

} // namespace detail

// =============================================================================
// TransportEngine Implementation
// =============================================================================

template <typename T>
TransportEngine<T>::TransportEngine(
    const GridDesc<T>& grid,
    const TransportParams<T>& params,
    const BoundaryConfig<T>& bc,
    const EngineConfig& config
) : impl_(std::make_unique<detail::EngineImpl<T>>(grid, params, bc, config)) {
    // Validate configuration
    if (grid.nx < 1 || grid.ny < 1 || grid.nz < 1) {
        throw std::invalid_argument("Grid dimensions must be >= 1");
    }
    if (grid.dx <= T(0) || grid.dy <= T(0) || grid.dz <= T(0)) {
        throw std::invalid_argument("Grid cell sizes must be > 0");
    }
}

template <typename T>
TransportEngine<T>::~TransportEngine() = default;

template <typename T>
TransportEngine<T>::TransportEngine(TransportEngine&&) noexcept = default;

template <typename T>
TransportEngine<T>& TransportEngine<T>::operator=(TransportEngine&&) noexcept = default;

// Stream management
template <typename T>
void TransportEngine<T>::set_stream(cudaStream_t stream) noexcept {
    impl_->stream_ = stream;
}

template <typename T>
cudaStream_t TransportEngine<T>::stream() const noexcept {
    return impl_->stream_;
}

// Binding
template <typename T>
void TransportEngine<T>::bind_velocity(const VelocityView<T>& vel) {
    assert(vel.valid() && "Invalid velocity view");
    assert(vel.size == static_cast<size_t>(impl_->grid.num_corners()) &&
           "Velocity array size must match grid.num_corners()");
    impl_->velocity = vel;
    impl_->corner_dirty = true;  // Face velocity changed -> corner needs recompute
    impl_->workspace.drift_dirty = true;  // Face velocity changed -> drift needs recompute
}

template <typename T>
void TransportEngine<T>::bind_corner_velocity(const CornerVelocityView<T>& cvel) {
    assert(cvel.valid() && "Invalid corner velocity view");
    impl_->corner_velocity = cvel;
    impl_->corner_external = true;   // User provided external corner velocity
    impl_->corner_dirty = false;     // No need to recompute
}

template <typename T>
void TransportEngine<T>::bind_particles(const ParticlesView<T>& particles) {
    assert(particles.valid() && "Invalid particles view");
    impl_->particles = particles;
    impl_->prepared = false;  // Need to call prepare() again
}

template <typename T>
void TransportEngine<T>::bind_drift_correction(const DriftCorrectionView<T>& dc) {
    assert(dc.valid() && "Invalid drift correction view");
    impl_->drift_correction = dc;
}

// Derived fields
template <typename T>
void TransportEngine<T>::update_derived_fields(cudaStream_t stream) {
    // Use engine stream if not specified
    if (stream == nullptr) {
        stream = impl_->stream_;
    }

    // =========================================================================
    // Part 1: Corner velocity (for Trilinear interpolation or TrilinearOnFly drift)
    // =========================================================================
    const bool needs_corner =
        impl_->config.interpolation_mode == InterpolationMode::Trilinear ||
        impl_->config.drift_mode == DriftCorrectionMode::TrilinearOnFly;

    // Skip corner update if external or not dirty
    if (needs_corner && !impl_->corner_external && impl_->corner_dirty) {
        // Need face velocity to compute corner velocity
        assert(has_velocity() && "Cannot update corner field: face velocity not bound");

        // Ensure workspace has corner buffers allocated (stream-async)
        const int num_corners = impl_->grid.num_corners();
        if (!impl_->workspace.ensure_corner(num_corners, stream)) {
            throw std::runtime_error("Failed to allocate corner velocity buffers");
        }

        // Launch kernel to compute corner velocities
        const int block_size = impl_->config.kernel_block_size;
        const int num_blocks = (num_corners + block_size - 1) / block_size;

        kernels::launch_compute_corner_velocities(
            impl_->grid,
            impl_->velocity,
            impl_->workspace.corner_Uc,
            impl_->workspace.corner_Vc,
            impl_->workspace.corner_Wc,
            num_blocks,
            block_size,
            stream
        );

        // Update internal corner velocity view to point to workspace buffers
        impl_->corner_velocity.Uc = impl_->workspace.corner_Uc;
        impl_->corner_velocity.Vc = impl_->workspace.corner_Vc;
        impl_->corner_velocity.Wc = impl_->workspace.corner_Wc;
        impl_->corner_velocity.size = num_corners;

        // Clear dirty flag
        impl_->corner_dirty = false;
        impl_->workspace.corner_dirty = false;
    }

    // =========================================================================
    // Part 2: Precomputed drift correction (for Precomputed mode)
    // =========================================================================
    const bool needs_drift = 
        impl_->config.drift_mode == DriftCorrectionMode::Precomputed;

    // Skip drift update if external, not dirty, or not needed
    if (needs_drift && !impl_->workspace.drift_external && impl_->workspace.drift_dirty) {
        // Need face velocity to compute drift
        assert(has_velocity() && "Cannot compute drift: face velocity not bound");

        // Ensure workspace has drift buffers allocated (stream-async)
        const int num_cells = impl_->grid.num_cells();
        if (!impl_->workspace.ensure_drift(num_cells, stream)) {
            throw std::runtime_error("Failed to allocate drift correction buffers");
        }

        // Launch kernel to compute precomputed drift
        const int block_size = impl_->config.kernel_block_size;
        const int num_blocks = (num_cells + block_size - 1) / block_size;

        kernels::launch_compute_drift_precomputed(
            impl_->grid,
            impl_->velocity,
            impl_->params,
            impl_->config.nan_prevention,  // Pass NaN-prevention flag
            impl_->workspace.drift_x,
            impl_->workspace.drift_y,
            impl_->workspace.drift_z,
            impl_->workspace.temp_D11,
            impl_->workspace.temp_D22,
            impl_->workspace.temp_D33,
            impl_->workspace.temp_D12,
            impl_->workspace.temp_D13,
            impl_->workspace.temp_D23,
            num_blocks,
            block_size,
            stream
        );

        // Update internal drift correction view to point to workspace buffers
        impl_->drift_correction.dcx = impl_->workspace.drift_x;
        impl_->drift_correction.dcy = impl_->workspace.drift_y;
        impl_->drift_correction.dcz = impl_->workspace.drift_z;
        impl_->drift_correction.size = num_cells;

        // Clear dirty flag
        impl_->workspace.drift_dirty = false;
    }

    PAR2_CHECK_KERNEL_LAUNCH(impl_->config);
}

template <typename T>
bool TransportEngine<T>::needs_corner_update() const noexcept {
    if (impl_->corner_external) return false;
    if (!impl_->corner_dirty) return false;

    return impl_->config.interpolation_mode == InterpolationMode::Trilinear ||
           impl_->config.drift_mode == DriftCorrectionMode::TrilinearOnFly;
}

template <typename T>
bool TransportEngine<T>::needs_drift_update() const noexcept {
    if (impl_->workspace.drift_external) return false;
    if (!impl_->workspace.drift_dirty) return false;

    return impl_->config.drift_mode == DriftCorrectionMode::Precomputed;
}

// Simulation
template <typename T>
void TransportEngine<T>::step(T dt) {
    assert(has_velocity() && "Velocity not bound");
    assert(has_particles() && "Particles not bound");
    assert(dt > T(0) && "dt must be positive");
    assert(impl_->prepared && "Engine not prepared. Call prepare() before step().");

    // Check if corner velocity is needed but not ready
    // In debug builds, assert. In release, we could auto-update but that
    // violates "no allocations in step()". Better to require explicit call.
    if (needs_corner_update()) {
        assert(false && "Corner velocity needed but dirty. "
                       "Call update_derived_fields() after changing velocity.");
    }

    // Check if precomputed drift is needed but not ready
    if (needs_drift_update()) {
        assert(false && "Precomputed drift needed but dirty. "
                       "Call update_derived_fields() after changing velocity.");
    }

    const int n = impl_->particles.n;
    if (n == 0) return;

    const int block_size = impl_->config.kernel_block_size;
    const int num_blocks = (n + block_size - 1) / block_size;

    // Launch move kernel (uses workspace.rng_states)
    kernels::launch_move_particles(
        impl_->grid,
        impl_->params,
        impl_->boundary,
        impl_->config,
        dt,
        impl_->velocity,
        impl_->corner_velocity,
        impl_->particles,
        impl_->drift_correction,
        impl_->workspace.rng_states,  // RNG from workspace
        num_blocks,
        block_size,
        impl_->stream_
    );

    PAR2_CHECK_KERNEL_LAUNCH(impl_->config);
}

template <typename T>
void TransportEngine<T>::advance(T dt, int num_steps) {
    for (int i = 0; i < num_steps; ++i) {
        step(dt);
    }
}

// Injection
template <typename T>
void TransportEngine<T>::inject_box(
    T x0, T y0, T z0,
    T x1, T y1, T z1,
    int first_particle,
    int count
) {
    assert(has_particles() && "Particles not bound");

    const int n = impl_->particles.n;
    if (count < 0) count = n - first_particle;
    if (count <= 0 || first_particle >= n) return;

    // Clamp count
    if (first_particle + count > n) {
        count = n - first_particle;
    }

    const int block_size = impl_->config.kernel_block_size;
    const int num_blocks = (count + block_size - 1) / block_size;

    // Launch injection kernel
    kernels::launch_inject_box(
        impl_->particles.x + first_particle,
        impl_->particles.y + first_particle,
        impl_->particles.z + first_particle,
        count,
        x0, y0, z0,
        x1, y1, z1,
        impl_->config.rng_seed,
        first_particle,
        num_blocks,
        block_size,
        impl_->stream_
    );

    PAR2_CHECK_KERNEL_LAUNCH(impl_->config);
}

// Queries
template <typename T>
const GridDesc<T>& TransportEngine<T>::grid() const noexcept {
    return impl_->grid;
}

template <typename T>
const TransportParams<T>& TransportEngine<T>::params() const noexcept {
    return impl_->params;
}

template <typename T>
const BoundaryConfig<T>& TransportEngine<T>::boundary() const noexcept {
    return impl_->boundary;
}

template <typename T>
const EngineConfig& TransportEngine<T>::config() const noexcept {
    return impl_->config;
}

template <typename T>
int TransportEngine<T>::num_particles() const noexcept {
    return impl_->particles.n;
}

template <typename T>
bool TransportEngine<T>::has_velocity() const noexcept {
    return impl_->velocity.valid() && impl_->velocity.size > 0;
}

template <typename T>
bool TransportEngine<T>::has_particles() const noexcept {
    return impl_->particles.valid() && impl_->particles.n > 0;
}

// =============================================================================
// M3: Tracking support
// =============================================================================

template <typename T>
void TransportEngine<T>::ensure_tracking_arrays() {
    if (!has_particles()) return;

    const int n = impl_->particles.n;
    cudaStream_t stream = impl_->stream_;

    // Allocate status if user didn't provide one and we have open BC
    if (impl_->particles.status == nullptr && impl_->boundary.has_open()) {
        if (!impl_->workspace.ensure_status(n, stream)) {
            throw std::runtime_error("Failed to allocate status array");
        }
        impl_->particles.status = impl_->workspace.status;
    }

    // Allocate wrap counters if user didn't provide and we have periodic BC
    if (impl_->boundary.has_periodic()) {
        // Check which axes are periodic
        bool need_x = (impl_->boundary.x.lo == BoundaryType::Periodic &&
                       impl_->boundary.x.hi == BoundaryType::Periodic);
        bool need_y = (impl_->boundary.y.lo == BoundaryType::Periodic &&
                       impl_->boundary.y.hi == BoundaryType::Periodic);
        bool need_z = (impl_->boundary.z.lo == BoundaryType::Periodic &&
                       impl_->boundary.z.hi == BoundaryType::Periodic);

        // Only allocate if user didn't provide
        if ((need_x && impl_->particles.wrapX == nullptr) ||
            (need_y && impl_->particles.wrapY == nullptr) ||
            (need_z && impl_->particles.wrapZ == nullptr))
        {
            if (!impl_->workspace.ensure_wrap(n, need_x, need_y, need_z, stream)) {
                throw std::runtime_error("Failed to allocate wrap counter arrays");
            }

            // Point particles view to workspace arrays
            if (need_x && impl_->particles.wrapX == nullptr) {
                impl_->particles.wrapX = impl_->workspace.wrapX;
            }
            if (need_y && impl_->particles.wrapY == nullptr) {
                impl_->particles.wrapY = impl_->workspace.wrapY;
            }
            if (need_z && impl_->particles.wrapZ == nullptr) {
                impl_->particles.wrapZ = impl_->workspace.wrapZ;
            }
        }
    }
}

// =============================================================================
// M4: Pipeline Integration
// =============================================================================

template <typename T>
void TransportEngine<T>::prepare(cudaStream_t stream) {
    // Use engine stream if not specified
    if (stream == nullptr) {
        stream = impl_->stream_;
    }

    // Must have particles bound
    if (!has_particles()) {
        throw std::runtime_error("prepare() requires particles to be bound");
    }

    const int n = impl_->particles.n;

    // 1. Allocate and initialize RNG states
    impl_->init_rng_states(n, stream);

    // 2. Ensure tracking arrays (status, wrap) if needed
    ensure_tracking_arrays();

    // 3. Update derived fields (corner velocity) if needed
    if (needs_corner_update()) {
        update_derived_fields(stream);
    }

    impl_->prepared = true;
}

template <typename T>
bool TransportEngine<T>::is_prepared() const noexcept {
    return impl_->prepared && impl_->workspace.has_rng();
}

template <typename T>
void TransportEngine<T>::synchronize() {
    cudaError_t err = cudaStreamSynchronize(impl_->stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Stream synchronization failed: ") + cudaGetErrorString(err)
        );
    }
}

template <typename T>
void TransportEngine<T>::record_event(cudaEvent_t event) {
    cudaError_t err = cudaEventRecord(event, impl_->stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Event record failed: ") + cudaGetErrorString(err)
        );
    }
}

template <typename T>
void TransportEngine<T>::wait_event(cudaEvent_t event) {
    cudaError_t err = cudaStreamWaitEvent(impl_->stream_, event, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Stream wait event failed: ") + cudaGetErrorString(err)
        );
    }
}

// =============================================================================
// M5: Data Access
// =============================================================================

template <typename T>
ConstParticlesView<T> TransportEngine<T>::particles() const noexcept {
    return ConstParticlesView<T>(impl_->particles);
}

template <typename T>
void TransportEngine<T>::compute_unwrapped_positions(
    UnwrappedPositionsView<T> out,
    cudaStream_t stream
) {
    assert(out.valid() && "Invalid output view");
    assert(out.capacity >= impl_->particles.n && "Output capacity too small");

    if (stream == nullptr) {
        stream = impl_->stream_;
    }

    const int n = impl_->particles.n;
    if (n == 0) return;

    const int block_size = impl_->config.kernel_block_size;
    const int num_blocks = (n + block_size - 1) / block_size;

    kernels::launch_unwrap_positions(
        impl_->grid,
        ConstParticlesView<T>(impl_->particles),
        out.x_u, out.y_u, out.z_u,
        num_blocks,
        block_size,
        stream
    );

    PAR2_CHECK_KERNEL_LAUNCH(impl_->config);
}

// Explicit instantiations
template class TransportEngine<float>;
template class TransportEngine<double>;

} // namespace par2
