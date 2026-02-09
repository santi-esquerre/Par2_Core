/**
 * @file pipeline_example.cu
 * @brief Par2_Core — Production-quality pipeline integration example.
 *
 * This is the **canonical reference** for embedding Par2_Core in a multi-solver
 * HPC pipeline.  It demonstrates every pattern needed to couple an external
 * flow solver (time-dependent velocity) with the RWPT transport engine:
 *
 * ## Key patterns
 *
 *  1. **Two CUDA streams** — `stream_flow` (flow solver) and `stream_transport`
 *     (Par2_Core engine) run concurrently on the GPU.
 *
 *  2. **Event-based sync** — `cudaStreamWaitEvent` replaces any global
 *     `cudaDeviceSynchronize()`; the CPU never blocks inside the hot loop.
 *
 *  3. **prepare() vs step()** — `prepare()` allocates workspace once;
 *     `step()` is allocation-free and safe for high-frequency calls.
 *
 *  4. **update_derived_fields()** — called after velocity changes and before
 *     `step()` when using Trilinear interpolation or Precomputed drift.
 *
 *  5. **Stats (GPU reduction)** — mean/var/std computed on-GPU via
 *     `StatsComputer`; only 6 doubles copied to host per stats tick.
 *
 *  6. **Snapshots** — pinned-buffer async download + CSV, with an explicit
 *     stream-only sync ("I/O barrier") — never `cudaDeviceSynchronize()`.
 *
 * ## Build & run
 *
 *   cmake -S . -B build -DPAR2_BUILD_EXAMPLES=ON
 *   cmake --build build --target par2_pipeline_example
 *   ./build/examples/par2_pipeline_example
 *
 * No external files (MODFLOW, YAML, legacy) required.
 * Output:  output/stats.csv  +  output/snapshot_step*.csv
 *
 * @copyright Par2_Core — GPU-native transport engine
 *            Based on PAR² by Calogero B. Rizzo
 */

#include <par2_core/par2_core.hpp>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>   // mkdir (POSIX)

// =============================================================================
// CUDA_CHECK — lightweight error macro (no cudaDeviceSynchronize)
// =============================================================================
// Uses cudaPeekAtLastError for async kernel error detection.
// Fatal on any error — appropriate for an example; in production code you may
// prefer returning error codes.

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(err__));        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t err__ = cudaPeekAtLastError();                             \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA kernel error at %s:%d — %s\n",         \
                         __FILE__, __LINE__, cudaGetErrorString(err__));        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// =============================================================================
// Simulation parameters (edit these)
// =============================================================================

// Grid  (2-D sheet: nz=1)
static constexpr int    NX = 50;
static constexpr int    NY = 50;
static constexpr int    NZ = 1;
static constexpr double DX = 1.0;   // [m]
static constexpr double DY = 1.0;
static constexpr double DZ = 1.0;

// Particles
static constexpr int    N_PARTICLES = 10000;

// Time stepping
static constexpr double DT           = 0.1;    // [s]
static constexpr int    NUM_STEPS    = 200;

// Output
static constexpr int    STATS_EVERY    = 10;    // stats row every N steps
static constexpr int    SNAPSHOT_EVERY = 50;    // full snapshot every N steps
static constexpr int    CSV_PRECISION  = 15;    // digits in CSV

// Physics
static constexpr double BASE_VX     = 0.5;     // mean x-velocity [m/s]
static constexpr double OMEGA       = 0.5;     // angular freq for time variation
static constexpr double AMP         = 0.10;    // amplitude of variation (10 %)
static constexpr double MOL_DIFF    = 0.01;    // molecular diffusion [m²/s]
static constexpr double ALPHA_L     = 0.10;    // longitudinal dispersivity [m]
static constexpr double ALPHA_T     = 0.01;    // transverse dispersivity [m]

// Injection box (particles placed here at t=0)
static constexpr double INJ_X0 =  5.0, INJ_Y0 =  0.0, INJ_Z0 = 0.0;
static constexpr double INJ_X1 = 20.0, INJ_Y1 = 15.0, INJ_Z1 = DZ;

// RNG
static constexpr unsigned long long RNG_SEED = 42ULL;

// =============================================================================
// Simulated flow solver kernel (time-dependent velocity field)
// =============================================================================
// In a real pipeline this would be MODFLOW, PFLOTRAN, your CFD code, etc.
// It writes U,V,W directly in device memory — no host copies.

template <typename T>
__global__ void update_velocity_kernel(
    T* __restrict__ U,
    T* __restrict__ V,
    T* __restrict__ W,
    int   num_corners,
    T     time,
    T     base_vx,
    T     omega,
    T     amplitude)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_corners) return;

    // Sinusoidal variation around base_vx:
    //   U(t) = base_vx * (1 + amplitude * sin(omega * t))
    T vx = base_vx * (T(1) + amplitude * sin(omega * time));
    U[idx] = vx;
    V[idx] = T(0);
    W[idx] = T(0);
}

// =============================================================================
// Helpers
// =============================================================================

/// Create directory (POSIX).  No-op if it already exists.
static void ensure_dir(const char* path) {
    struct stat st{};
    if (stat(path, &st) != 0) {
        mkdir(path, 0755);
    }
}

/// Format snapshot filename: output/snapshot_step000042.csv
static std::string snapshot_filename(int step) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "output/snapshot_step%06d.csv", step);
    return std::string(buf);
}

/// Write a snapshot from host buffers to CSV.
static void write_snapshot_csv(
    const std::string& path,
    const double* hx, const double* hy, const double* hz,
    int n, double time, int step)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "  [WARN] Cannot open " << path << "\n";
        return;
    }
    f << std::fixed << std::setprecision(CSV_PRECISION);
    f << "step,t,id,x,y,z\n";
    for (int i = 0; i < n; ++i) {
        f << step << ',' << time << ',' << i << ','
          << hx[i] << ',' << hy[i] << ',' << hz[i] << '\n';
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    using T = double;

    std::cout << "=== Par2_Core Pipeline Integration Example ===\n"
              << "Version: " << PAR2_CORE_VERSION_STRING << "\n\n";

    ensure_dir("output");

    // =========================================================================
    // 1.  CUDA streams & events (created ONCE, reused every step)
    // =========================================================================
    // Two streams allow flow-solver and transport work to overlap on the GPU.
    // Events provide GPU-side ordering without blocking the CPU.

    cudaStream_t stream_flow, stream_transport;
    CUDA_CHECK(cudaStreamCreate(&stream_flow));
    CUDA_CHECK(cudaStreamCreate(&stream_transport));

    // vel_ready_evt:      flow solver records after writing U/V/W
    // transport_done_evt: transport records after step() — another solver
    //                     could wait on this for reactive feedback.
    // timing events:      loop_start / loop_end for wall-clock measurement.
    cudaEvent_t vel_ready_evt, transport_done_evt, loop_start_evt, loop_end_evt;
    CUDA_CHECK(cudaEventCreateWithFlags(&vel_ready_evt,      cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&transport_done_evt, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&loop_start_evt));
    CUDA_CHECK(cudaEventCreate(&loop_end_evt));

    std::cout << "[1] Streams & events created.\n";

    // =========================================================================
    // 2.  Grid + engine configuration
    // =========================================================================
    auto grid = par2::make_grid<T>(NX, NY, NZ, DX, DY, DZ);
    const int num_corners = grid.num_corners();

    par2::TransportParams<T> params;
    params.molecular_diffusion = MOL_DIFF;
    params.alpha_l             = ALPHA_L;
    params.alpha_t             = ALPHA_T;

    par2::BoundaryConfig<T> bc = par2::BoundaryConfig<T>::all_closed();

    par2::EngineConfig config;
    config.interpolation_mode = par2::InterpolationMode::Linear;
    config.drift_mode         = par2::DriftCorrectionMode::TrilinearOnFly;
    config.rng_seed           = RNG_SEED;
    config.debug_checks       = false;   // HPC: off
    config.nan_prevention     = false;   // safe field → skip branch

    par2::TransportEngine<T> engine(grid, params, bc, config);
    engine.set_stream(stream_transport);

    // Do we need to recompute corner / drift fields after each velocity update?
    const bool needs_derived = engine.needs_corner_update()
                            || engine.needs_drift_update();

    std::cout << "[2] Grid " << NX << "x" << NY << "x" << NZ
              << " (" << grid.num_cells() << " cells, "
              << num_corners << " corners)\n"
              << "    Derived fields needed each step: "
              << (needs_derived ? "YES" : "NO") << "\n";

    // =========================================================================
    // 3.  Allocate device buffers  (velocity + particles)
    // =========================================================================
    // Par2_Core is zero-copy: the engine operates on YOUR device buffers.
    // You own allocation & lifetime; the engine just binds views.

    T *d_U, *d_V, *d_W;
    CUDA_CHECK(cudaMalloc(&d_U, num_corners * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_V, num_corners * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_W, num_corners * sizeof(T)));

    T *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, N_PARTICLES * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_y, N_PARTICLES * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_z, N_PARTICLES * sizeof(T)));

    std::cout << "[3] Device buffers allocated.\n";

    // =========================================================================
    // 4.  Bind views + inject + prepare  (allocations happen HERE)
    // =========================================================================
    // bind_velocity / bind_particles attach device pointers — no copies.
    // inject_box fills positions on-GPU.
    // prepare() allocates RNG states and internal workspace ONCE.
    // After prepare(), step() will NEVER allocate.

    // Initial velocity (t=0) — launch on flow stream, sync before bind.
    const int block = 256;
    const int vel_grid = (num_corners + block - 1) / block;
    update_velocity_kernel<<<vel_grid, block, 0, stream_flow>>>(
        d_U, d_V, d_W, num_corners, T(0), T(BASE_VX), T(OMEGA), T(AMP));
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(stream_flow));  // one-time init sync

    engine.bind_velocity({d_U, d_V, d_W, static_cast<size_t>(num_corners)});
    engine.bind_particles({d_x, d_y, d_z, N_PARTICLES});
    engine.inject_box(INJ_X0, INJ_Y0, INJ_Z0, INJ_X1, INJ_Y1, INJ_Z1);
    engine.prepare();   // ← all workspace allocs happen here

    std::cout << "[4] Velocity bound, " << N_PARTICLES
              << " particles injected, engine prepared.\n";

    // =========================================================================
    // 5.  Pre-allocate I/O resources (stats computer + pinned snapshot buffer)
    // =========================================================================
    // StatsComputer keeps persistent GPU reduction buffers — no allocs per call.
    // Pinned host memory enables async D2H overlap with compute.

    par2::StatsComputer<T> stats(N_PARTICLES);

    par2::StatsConfig stats_cfg;
    stats_cfg.use_unwrapped     = false;  // Closed BC → no wrapping
    stats_cfg.filter_active_only = true;

    // Pinned host buffers for snapshot download
    T *h_x, *h_y, *h_z;
    CUDA_CHECK(cudaMallocHost(&h_x, N_PARTICLES * sizeof(T)));
    CUDA_CHECK(cudaMallocHost(&h_y, N_PARTICLES * sizeof(T)));
    CUDA_CHECK(cudaMallocHost(&h_z, N_PARTICLES * sizeof(T)));

    // Open stats CSV and write header
    std::ofstream stats_file("output/stats.csv");
    if (!stats_file.is_open()) {
        std::cerr << "Cannot create output/stats.csv\n";
        return EXIT_FAILURE;
    }
    stats_file << "step,t,mean_x,var_x,std_x,mean_y,var_y,std_y,mean_z,var_z,std_z\n";
    stats_file << std::fixed << std::setprecision(CSV_PRECISION);

    std::cout << "[5] Stats computer & pinned buffers allocated.\n\n";

    // =========================================================================
    // 6.  Initial snapshot + stats  (step 0)
    // =========================================================================
    {
        engine.synchronize();   // ensure inject_box completed

        // Stats at t=0
        auto pview = engine.particles();
        stats.compute_async(pview, grid, stats_cfg, stream_transport);
        CUDA_CHECK(cudaStreamSynchronize(stream_transport));
        auto r = stats.fetch_result();
        stats_file << 0 << ',' << 0.0 << ','
                   << r.moments.mean[0] << ',' << r.moments.var[0] << ',' << r.moments.std[0] << ','
                   << r.moments.mean[1] << ',' << r.moments.var[1] << ',' << r.moments.std[1] << ','
                   << r.moments.mean[2] << ',' << r.moments.var[2] << ',' << r.moments.std[2] << '\n';

        // Snapshot at step 0
        CUDA_CHECK(cudaMemcpyAsync(h_x, pview.x, N_PARTICLES * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream_transport));
        CUDA_CHECK(cudaMemcpyAsync(h_y, pview.y, N_PARTICLES * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream_transport));
        CUDA_CHECK(cudaMemcpyAsync(h_z, pview.z, N_PARTICLES * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream_transport));
        CUDA_CHECK(cudaStreamSynchronize(stream_transport));
        write_snapshot_csv(snapshot_filename(0), h_x, h_y, h_z, N_PARTICLES, 0.0, 0);

        std::cout << "Step 0 — snapshot + stats recorded.\n";
    }

    // =========================================================================
    // 7.  MAIN SIMULATION LOOP  (the "hot path")
    // =========================================================================
    //
    //  ┌─ stream_flow ───────────────────────────────────────────┐
    //  │  update_velocity_kernel  →  record(vel_ready_evt)       │
    //  └─────────────────────────────────────────────────────────┘
    //           ↓ GPU-side wait (no CPU block)
    //  ┌─ stream_transport ──────────────────────────────────────┐
    //  │  wait(vel_ready_evt)  →  update_derived_fields()        │
    //  │  →  step(dt)  →  record(transport_done_evt)             │
    //  └─────────────────────────────────────────────────────────┘
    //           ↓ (optional: another solver waits transport_done)
    //
    // The CPU posts work to both streams and moves on immediately.
    // Synchronization only happens:
    //   • at stats_every  steps  (stream_transport sync to read 6 doubles)
    //   • at snapshot_every steps (stream_transport sync to download positions)

    std::cout << "\nRunning " << NUM_STEPS << " steps (dt=" << DT << ")...\n";
    CUDA_CHECK(cudaEventRecord(loop_start_evt, stream_transport));

    for (int step = 1; step <= NUM_STEPS; ++step) {
        const T time = step * DT;

        // -----------------------------------------------------------------
        // (a) Flow solver: update U/V/W on stream_flow
        // -----------------------------------------------------------------
        update_velocity_kernel<<<vel_grid, block, 0, stream_flow>>>(
            d_U, d_V, d_W, num_corners, time, T(BASE_VX), T(OMEGA), T(AMP));
        CUDA_CHECK_LAST();

        // Record: "velocity field is ready"
        CUDA_CHECK(cudaEventRecord(vel_ready_evt, stream_flow));

        // -----------------------------------------------------------------
        // (b) Transport: wait for velocity, optionally recompute derived
        //     fields, then step.  All on stream_transport, no CPU block.
        // -----------------------------------------------------------------
        engine.wait_event(vel_ready_evt);

        // If the engine uses corner velocities or precomputed drift,
        // we must recompute them after U/V/W changed.
        // update_derived_fields() is async (same stream), zero-alloc.
        if (needs_derived) {
            engine.update_derived_fields();
        }

        engine.step(DT);

        // Record: "transport step done" — another solver could wait on this
        // for reactive-transport feedback (see commented example below).
        engine.record_event(transport_done_evt);

        // -----------------------------------------------------------------
        // (c) Stats (periodic) — GPU reduction, only 6 doubles to host
        // -----------------------------------------------------------------
        if (step % STATS_EVERY == 0) {
            // compute_async enqueues reduction kernels on stream_transport.
            // We sync ONLY stream_transport to read the pinned result.
            auto pview = engine.particles();
            stats.compute_async(pview, grid, stats_cfg, stream_transport);
            CUDA_CHECK(cudaStreamSynchronize(stream_transport));  // sync transport only
            auto r = stats.fetch_result();

            stats_file << step << ',' << time << ','
                       << r.moments.mean[0] << ',' << r.moments.var[0] << ',' << r.moments.std[0] << ','
                       << r.moments.mean[1] << ',' << r.moments.var[1] << ',' << r.moments.std[1] << ','
                       << r.moments.mean[2] << ',' << r.moments.var[2] << ',' << r.moments.std[2] << '\n';
        }

        // -----------------------------------------------------------------
        // (d) Snapshot (periodic) — async D2H via pinned buffer
        // -----------------------------------------------------------------
        if (step % SNAPSHOT_EVERY == 0) {
            // I/O barrier: sync stream_transport so positions are final.
            CUDA_CHECK(cudaStreamSynchronize(stream_transport));

            auto pview = engine.particles();
            CUDA_CHECK(cudaMemcpyAsync(h_x, pview.x, N_PARTICLES * sizeof(T),
                                       cudaMemcpyDeviceToHost, stream_transport));
            CUDA_CHECK(cudaMemcpyAsync(h_y, pview.y, N_PARTICLES * sizeof(T),
                                       cudaMemcpyDeviceToHost, stream_transport));
            CUDA_CHECK(cudaMemcpyAsync(h_z, pview.z, N_PARTICLES * sizeof(T),
                                       cudaMemcpyDeviceToHost, stream_transport));
            CUDA_CHECK(cudaStreamSynchronize(stream_transport));

            write_snapshot_csv(snapshot_filename(step), h_x, h_y, h_z,
                               N_PARTICLES, time, step);
        }

        // -----------------------------------------------------------------
        // (optional) Another solver could wait for transport:
        //   cudaStreamWaitEvent(stream_reactive, transport_done_evt, 0);
        // This keeps everything GPU-side, zero CPU blocking.
        // -----------------------------------------------------------------
    }

    CUDA_CHECK(cudaEventRecord(loop_end_evt, stream_transport));

    // =========================================================================
    // 8.  Final sync + timing
    // =========================================================================
    engine.synchronize();

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, loop_start_evt, loop_end_evt));

    std::cout << "\n=== Results ===\n"
              << "  Steps:          " << NUM_STEPS << "\n"
              << "  Total time:     " << elapsed_ms << " ms\n"
              << "  Time/step:      " << elapsed_ms / NUM_STEPS << " ms\n"
              << "  Particles/sec:  " << std::scientific << std::setprecision(3)
              << (double(N_PARTICLES) * NUM_STEPS) / (elapsed_ms / 1000.0)
              << std::fixed << "\n";

    // =========================================================================
    // 9.  Final snapshot + final stats row
    // =========================================================================
    {
        auto pview = engine.particles();

        // Stats (only if final step wasn't already a stats tick)
        if (NUM_STEPS % STATS_EVERY != 0) {
            stats.compute_async(pview, grid, stats_cfg, stream_transport);
            CUDA_CHECK(cudaStreamSynchronize(stream_transport));
            auto r = stats.fetch_result();
            stats_file << NUM_STEPS << ',' << NUM_STEPS * DT << ','
                       << r.moments.mean[0] << ',' << r.moments.var[0] << ',' << r.moments.std[0] << ','
                       << r.moments.mean[1] << ',' << r.moments.var[1] << ',' << r.moments.std[1] << ','
                       << r.moments.mean[2] << ',' << r.moments.var[2] << ',' << r.moments.std[2] << '\n';
        }

        // Print final stats (always — read from last compute)
        stats.compute_async(pview, grid, stats_cfg, stream_transport);
        CUDA_CHECK(cudaStreamSynchronize(stream_transport));
        auto r = stats.fetch_result();
        std::cout << "\n  Final mean  x: " << r.moments.mean[0]
                  << "\n  Final std   x: " << r.moments.std[0]
                  << "\n  Expected drift ~  " << BASE_VX * NUM_STEPS * DT << " m\n";

        // Snapshot  (only if last step wasn't already a snapshot tick)
        if (NUM_STEPS % SNAPSHOT_EVERY != 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_x, pview.x, N_PARTICLES * sizeof(T),
                                       cudaMemcpyDeviceToHost, stream_transport));
            CUDA_CHECK(cudaMemcpyAsync(h_y, pview.y, N_PARTICLES * sizeof(T),
                                       cudaMemcpyDeviceToHost, stream_transport));
            CUDA_CHECK(cudaMemcpyAsync(h_z, pview.z, N_PARTICLES * sizeof(T),
                                       cudaMemcpyDeviceToHost, stream_transport));
            CUDA_CHECK(cudaStreamSynchronize(stream_transport));
            write_snapshot_csv(snapshot_filename(NUM_STEPS), h_x, h_y, h_z,
                               N_PARTICLES, NUM_STEPS * DT, NUM_STEPS);
        }
    }
    stats_file.close();

    // =========================================================================
    // 10.  Cleanup
    // =========================================================================
    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_y));
    CUDA_CHECK(cudaFreeHost(h_z));

    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    CUDA_CHECK(cudaEventDestroy(vel_ready_evt));
    CUDA_CHECK(cudaEventDestroy(transport_done_evt));
    CUDA_CHECK(cudaEventDestroy(loop_start_evt));
    CUDA_CHECK(cudaEventDestroy(loop_end_evt));

    CUDA_CHECK(cudaStreamDestroy(stream_flow));
    CUDA_CHECK(cudaStreamDestroy(stream_transport));

    std::cout << "\nOutput files:\n"
              << "  output/stats.csv                 — mean/var/std vs time\n"
              << "  output/snapshot_step*.csv         — particle positions\n"
              << "\nDone.\n";
    return EXIT_SUCCESS;
}

// =============================================================================
//  ADVANCED PATTERN: Double-buffered velocity (optional, not implemented here)
// =============================================================================
//
//  For true overlap of flow-solve(t_{n+1}) with transport(t_n), you would use
//  two sets of velocity buffers and alternate:
//
//    T *d_U[2], *d_V[2], *d_W[2];
//    int cur = 0;
//
//    for (int step = 0; step < N; ++step) {
//        int nxt = 1 - cur;
//
//        // Flow writes NEXT velocity on stream_flow
//        update_velocity<<<..., stream_flow>>>(d_U[nxt], d_V[nxt], ...);
//        cudaEventRecord(vel_ready_evt, stream_flow);
//
//        // Transport uses CURRENT velocity on stream_transport
//        engine.wait_event(vel_ready_evt);   // ensures prev write done
//        engine.bind_velocity({d_U[cur], d_V[cur], d_W[cur], num_corners});
//        engine.update_derived_fields();
//        engine.step(dt);
//
//        cur = nxt;
//    }
//
//  This maximises GPU occupancy when both solvers are compute-heavy.
// =============================================================================
