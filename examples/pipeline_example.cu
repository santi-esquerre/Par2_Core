/**
 * @file pipeline_example.cu
 * @brief Par2_Core — Canonical pipeline integration example.
 *
 * This is the **single reference example** for Par2_Core.  It shows how to
 * embed TransportEngine in a multi-solver HPC pipeline using CUDA streams
 * and events for fully asynchronous, zero-sync-in-the-hot-loop operation.
 *
 * ## Key patterns demonstrated
 *
 *  1. **Separate streams** — flow solver and transport each get their own
 *     cudaStream_t so they can overlap on the GPU.
 *  2. **Event-based sync** — cudaStreamWaitEvent replaces any global
 *     cudaDeviceSynchronize; the CPU never blocks inside the loop.
 *  3. **prepare() vs step()** — prepare() may allocate workspace; step()
 *     is allocation-free and safe to call at high frequency.
 *  4. **Explicit synchronize** — only called once, at the end, when we
 *     need to copy results to host for I/O.
 *
 * ## Build & run
 *
 *   cmake -S . -B build -DPAR2_BUILD_EXAMPLES=ON
 *   cmake --build build --target par2_pipeline_example
 *   ./build/examples/par2_pipeline_example
 *
 * No external files (MODFLOW, YAML, legacy) required.
 *
 * @copyright Par2_Core — GPU-native transport engine
 *            Based on PAR² by Calogero B. Rizzo
 */

#include <par2_core/par2_core.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// =============================================================================
// Simulated Flow Solver (updates velocity field)
// =============================================================================

/**
 * @brief Kernel that simulates a flow solver updating velocity.
 *
 * In a real pipeline, this would be your MODFLOW, MT3DMS, or custom flow solver.
 * Here we just do a simple time-varying uniform flow for demonstration.
 */
template <typename T>
__global__ void update_velocity_kernel(
    T* U, T* V, T* W,
    int num_corners,
    T time,
    T base_vx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_corners) return;

    // Simple sinusoidal variation in x-velocity
    T vx = base_vx * (T(1.0) + T(0.1) * sin(time));
    U[idx] = vx;
    V[idx] = T(0);
    W[idx] = T(0);
}

// =============================================================================
// Main: Pipeline Integration Demo
// =============================================================================

int main() {
    using T = double;

    std::cout << "=== Par2_Core Pipeline Integration Example ===\n";
    std::cout << "Version: " << PAR2_CORE_VERSION_STRING << "\n\n";

    // =========================================================================
    // Step 1: Create separate CUDA streams
    // =========================================================================
    std::cout << "Creating CUDA streams...\n";

    cudaStream_t flow_stream, transport_stream;
    cudaStreamCreate(&flow_stream);
    cudaStreamCreate(&transport_stream);

    // =========================================================================
    // Step 2: Create synchronization events (disable timing for performance)
    // =========================================================================
    std::cout << "Creating synchronization events...\n";

    cudaEvent_t velocity_ready, transport_done, loop_start, loop_end;
    cudaEventCreateWithFlags(&velocity_ready, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&transport_done, cudaEventDisableTiming);
    cudaEventCreate(&loop_start);  // These have timing for performance measurement
    cudaEventCreate(&loop_end);

    // =========================================================================
    // Step 3: Setup grid and engine
    // =========================================================================
    const int nx = 50, ny = 50, nz = 1;
    const T dx = 1.0, dy = 1.0, dz = 1.0;
    const int num_particles = 10000;
    const int num_corners = (nx + 1) * (ny + 1) * (nz + 1);

    auto grid = par2::make_grid<T>(nx, ny, nz, dx, dy, dz);
    std::cout << "Grid: " << nx << "x" << ny << "x" << nz << " cells\n";

    par2::TransportParams<T> params;
    params.molecular_diffusion = 0.01;
    params.alpha_l = 0.1;
    params.alpha_t = 0.01;

    par2::BoundaryConfig<T> bc = par2::BoundaryConfig<T>::all_closed();
    par2::EngineConfig config;
    config.debug_checks = false;  // Disable for HPC performance
    config.nan_prevention = false;  // Safe velocity field, skip checks

    par2::TransportEngine<T> engine(grid, params, bc, config);

    // =========================================================================
    // Step 4: Set engine to use transport stream
    // =========================================================================
    engine.set_stream(transport_stream);
    std::cout << "Engine configured with dedicated stream\n";

    // =========================================================================
    // Step 5: Allocate GPU memory
    // =========================================================================
    T *d_U, *d_V, *d_W;
    T *d_x, *d_y, *d_z;

    cudaMalloc(&d_U, num_corners * sizeof(T));
    cudaMalloc(&d_V, num_corners * sizeof(T));
    cudaMalloc(&d_W, num_corners * sizeof(T));
    cudaMalloc(&d_x, num_particles * sizeof(T));
    cudaMalloc(&d_y, num_particles * sizeof(T));
    cudaMalloc(&d_z, num_particles * sizeof(T));

    // Initialize velocity (uniform flow in x)
    const T base_vx = 0.5;
    int block = 256;
    int grid_size = (num_corners + block - 1) / block;
    update_velocity_kernel<<<grid_size, block, 0, flow_stream>>>(
        d_U, d_V, d_W, num_corners, T(0), base_vx
    );
    cudaStreamSynchronize(flow_stream);  // Initial sync only

    // =========================================================================
    // Step 6: Bind and prepare engine
    // =========================================================================
    engine.bind_velocity({d_U, d_V, d_W, static_cast<size_t>(num_corners)});
    engine.bind_particles({d_x, d_y, d_z, num_particles});

    // Inject particles
    engine.inject_box(5.0, 20.0, 0.0, 15.0, 30.0, dz);
    engine.prepare();

    std::cout << "Injected " << num_particles << " particles\n\n";

    // =========================================================================
    // Step 7: Coupled simulation loop (THE HOT PATH)
    // =========================================================================
    const int num_timesteps = 100;
    const T dt = 0.1;

    std::cout << "Running " << num_timesteps << " coupled timesteps...\n";
    std::cout << "(Flow solver on stream 1, Transport on stream 2)\n\n";

    // Record start time
    cudaEventRecord(loop_start, transport_stream);

    for (int t = 0; t < num_timesteps; ++t) {
        T time = t * dt;

        // -----------------------------------------------------------------
        // Flow solver: Update velocity on flow_stream
        // -----------------------------------------------------------------
        update_velocity_kernel<<<grid_size, block, 0, flow_stream>>>(
            d_U, d_V, d_W, num_corners, time, base_vx
        );

        // Record event: "velocity is ready"
        cudaEventRecord(velocity_ready, flow_stream);

        // -----------------------------------------------------------------
        // Transport: Wait for velocity, then step
        // -----------------------------------------------------------------
        engine.wait_event(velocity_ready);  // GPU-side wait (no CPU block!)
        engine.step(dt);

        // Record event: "transport is done"
        engine.record_event(transport_done);

        // -----------------------------------------------------------------
        // Flow solver could wait for transport_done here if needed
        // (e.g., for reactive transport feedback)
        // cudaStreamWaitEvent(flow_stream, transport_done, 0);
        // -----------------------------------------------------------------
    }

    // Record end time
    cudaEventRecord(loop_end, transport_stream);

    // =========================================================================
    // Step 8: Synchronize only at end (for timing/output)
    // =========================================================================
    engine.synchronize();

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, loop_start, loop_end);

    std::cout << "Pipeline completed!\n";
    std::cout << "  Total time: " << elapsed_ms << " ms\n";
    std::cout << "  Time per step: " << elapsed_ms / num_timesteps << " ms\n";
    std::cout << "  Particles/sec: "
              << (num_particles * num_timesteps) / (elapsed_ms / 1000.0) << "\n\n";

    // =========================================================================
    // Step 9: Verify results (optional)
    // =========================================================================
    // Download final positions for verification
    std::vector<T> h_x(num_particles);
    cudaMemcpy(h_x.data(), d_x, num_particles * sizeof(T), cudaMemcpyDeviceToHost);

    T mean_x = 0;
    for (int i = 0; i < num_particles; ++i) mean_x += h_x[i];
    mean_x /= num_particles;

    std::cout << "Final mean x-position: " << mean_x << "\n";
    std::cout << "Expected drift: ~" << base_vx * num_timesteps * dt << " m\n";

    // =========================================================================
    // Cleanup
    // =========================================================================
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    cudaEventDestroy(velocity_ready);
    cudaEventDestroy(transport_done);
    cudaEventDestroy(loop_start);
    cudaEventDestroy(loop_end);

    cudaStreamDestroy(flow_stream);
    cudaStreamDestroy(transport_stream);

    std::cout << "\nPipeline integration example completed successfully!\n";
    return 0;
}
