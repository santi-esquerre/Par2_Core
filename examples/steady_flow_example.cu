/**
 * @file steady_flow_example.cu
 * @brief Par2_Core — Steady (time-independent) velocity field example.
 *
 * Companion to `pipeline_example.cu`.  This variant shows the simpler case
 * where the velocity field is **constant in time**: set once, derive corner
 * fields once, then run the step loop with zero per-step overhead.
 *
 * ## Differences vs pipeline_example.cu
 *
 *  | Aspect                    | pipeline_example        | steady_flow_example     |
 *  |---------------------------|-------------------------|-------------------------|
 *  | Velocity                  | time-dependent U(t)     | constant U              |
 *  | Flow stream               | YES (stream_flow)       | not needed              |
 *  | vel_ready event           | every step              | not needed              |
 *  | update_derived_fields()   | every step              | ONCE after prepare()    |
 *  | Hot-loop body             | wait → derive → step    | step() only             |
 *
 * ## Build & run
 *
 *   cmake -S . -B build -DPAR2_BUILD_EXAMPLES=ON
 *   cmake --build build --target par2_steady_flow_example
 *   ./build/examples/par2_steady_flow_example
 *
 * No external files required.
 * Output:  output/steady_stats.csv  +  output/steady_snapshot_step*.csv
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
#include <sys/stat.h>

// =============================================================================
// CUDA_CHECK
// =============================================================================

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
// Simulation parameters
// =============================================================================

static constexpr int    NX = 50;
static constexpr int    NY = 50;
static constexpr int    NZ = 1;
static constexpr double DX = 1.0;
static constexpr double DY = 1.0;
static constexpr double DZ = 1.0;

static constexpr int    N_PARTICLES = 10000;

static constexpr double DT           = 0.1;
static constexpr int    NUM_STEPS    = 200;

static constexpr int    STATS_EVERY    = 10;
static constexpr int    SNAPSHOT_EVERY = 50;
static constexpr int    CSV_PRECISION  = 15;

// Steady velocity: uniform flow in +X
static constexpr double VX = 0.5;   // [m/s]
static constexpr double VY = 0.0;
static constexpr double VZ = 0.0;

static constexpr double MOL_DIFF = 0.01;
static constexpr double ALPHA_L  = 0.10;
static constexpr double ALPHA_T  = 0.01;

static constexpr double INJ_X0 =  5.0, INJ_Y0 =  0.0, INJ_Z0 = 0.0;
static constexpr double INJ_X1 = 20.0, INJ_Y1 = 15.0, INJ_Z1 = DZ;

static constexpr unsigned long long RNG_SEED = 42ULL;

// =============================================================================
// Kernel: fill velocity field ONCE (constant)
// =============================================================================

template <typename T>
__global__ void fill_velocity_kernel(
    T* __restrict__ U,
    T* __restrict__ V,
    T* __restrict__ W,
    int num_corners,
    T vx, T vy, T vz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_corners) return;
    U[idx] = vx;
    V[idx] = vy;
    W[idx] = vz;
}

// =============================================================================
// Helpers
// =============================================================================

static void ensure_dir(const char* path) {
    struct stat st{};
    if (stat(path, &st) != 0) mkdir(path, 0755);
}

static std::string snapshot_filename(int step) {
    char buf[80];
    std::snprintf(buf, sizeof(buf), "output/steady_snapshot_step%06d.csv", step);
    return std::string(buf);
}

static void write_snapshot_csv(
    const std::string& path,
    const double* hx, const double* hy, const double* hz,
    int n, double time, int step)
{
    std::ofstream f(path);
    if (!f.is_open()) { std::cerr << "  [WARN] Cannot open " << path << "\n"; return; }
    f << std::fixed << std::setprecision(CSV_PRECISION);
    f << "step,t,id,x,y,z\n";
    for (int i = 0; i < n; ++i)
        f << step << ',' << time << ',' << i << ','
          << hx[i] << ',' << hy[i] << ',' << hz[i] << '\n';
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    using T = double;

    std::cout << "=== Par2_Core Steady-Flow Example ===\n"
              << "Version: " << PAR2_CORE_VERSION_STRING << "\n\n";

    ensure_dir("output");

    // =========================================================================
    // 1.  Single CUDA stream (no flow stream needed for steady field)
    // =========================================================================
    // With a constant velocity field there is no producer to synchronize
    // against, so one stream is enough.  We still avoid cudaDeviceSynchronize()
    // and use stream-scoped sync only for I/O.

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Timing events
    cudaEvent_t loop_start_evt, loop_end_evt;
    CUDA_CHECK(cudaEventCreate(&loop_start_evt));
    CUDA_CHECK(cudaEventCreate(&loop_end_evt));

    std::cout << "[1] Stream created.\n";

    // =========================================================================
    // 2.  Grid + engine
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
    config.debug_checks       = false;
    config.nan_prevention     = false;

    par2::TransportEngine<T> engine(grid, params, bc, config);
    engine.set_stream(stream);

    std::cout << "[2] Grid " << NX << "x" << NY << "x" << NZ
              << " (" << grid.num_cells() << " cells, "
              << num_corners << " corners)\n";

    // =========================================================================
    // 3.  Allocate device buffers
    // =========================================================================
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
    // 4.  Fill velocity ONCE, bind, inject, prepare
    // =========================================================================
    // Key difference vs time-dependent case: the velocity kernel runs once
    // and update_derived_fields() is called once after prepare().
    // In the step loop there is NO velocity update and NO derived recompute.

    const int block = 256;
    const int vel_grid = (num_corners + block - 1) / block;
    fill_velocity_kernel<<<vel_grid, block, 0, stream>>>(
        d_U, d_V, d_W, num_corners, T(VX), T(VY), T(VZ));
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(stream));  // one-time init sync

    engine.bind_velocity({d_U, d_V, d_W, static_cast<size_t>(num_corners)});
    engine.bind_particles({d_x, d_y, d_z, N_PARTICLES});
    engine.inject_box(INJ_X0, INJ_Y0, INJ_Z0, INJ_X1, INJ_Y1, INJ_Z1);
    engine.prepare();

    // Corner / drift fields derived ONCE — velocity won't change.
    if (engine.needs_corner_update() || engine.needs_drift_update()) {
        engine.update_derived_fields();
        std::cout << "    Derived fields computed (one-time).\n";
    }

    std::cout << "[4] Steady velocity (" << VX << ", " << VY << ", " << VZ
              << ") bound, " << N_PARTICLES << " particles injected.\n";

    // =========================================================================
    // 5.  I/O resources
    // =========================================================================
    par2::StatsComputer<T> stats(N_PARTICLES);

    par2::StatsConfig stats_cfg;
    stats_cfg.use_unwrapped      = false;
    stats_cfg.filter_active_only = true;

    T *h_x, *h_y, *h_z;
    CUDA_CHECK(cudaMallocHost(&h_x, N_PARTICLES * sizeof(T)));
    CUDA_CHECK(cudaMallocHost(&h_y, N_PARTICLES * sizeof(T)));
    CUDA_CHECK(cudaMallocHost(&h_z, N_PARTICLES * sizeof(T)));

    std::ofstream stats_file("output/steady_stats.csv");
    if (!stats_file.is_open()) {
        std::cerr << "Cannot create output/steady_stats.csv\n";
        return EXIT_FAILURE;
    }
    stats_file << "step,t,mean_x,var_x,std_x,mean_y,var_y,std_y,mean_z,var_z,std_z\n";
    stats_file << std::fixed << std::setprecision(CSV_PRECISION);

    std::cout << "[5] Stats computer & pinned buffers ready.\n\n";

    // =========================================================================
    // 6.  Step-0 snapshot + stats
    // =========================================================================
    {
        engine.synchronize();

        auto pview = engine.particles();
        stats.compute_async(pview, grid, stats_cfg, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto r = stats.fetch_result();
        stats_file << 0 << ',' << 0.0 << ','
                   << r.moments.mean[0] << ',' << r.moments.var[0] << ',' << r.moments.std[0] << ','
                   << r.moments.mean[1] << ',' << r.moments.var[1] << ',' << r.moments.std[1] << ','
                   << r.moments.mean[2] << ',' << r.moments.var[2] << ',' << r.moments.std[2] << '\n';

        CUDA_CHECK(cudaMemcpyAsync(h_x, pview.x, N_PARTICLES * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_y, pview.y, N_PARTICLES * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_z, pview.z, N_PARTICLES * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        write_snapshot_csv(snapshot_filename(0), h_x, h_y, h_z, N_PARTICLES, 0.0, 0);

        std::cout << "Step 0 — snapshot + stats recorded.\n";
    }

    // =========================================================================
    // 7.  SIMULATION LOOP (steady field → minimal hot path)
    // =========================================================================
    //
    //  The loop body is just:
    //      engine.step(dt)
    //
    //  No velocity update, no derived-field recompute, no event waits.
    //  Sync only when we need to pull stats or snapshots to the host.

    std::cout << "\nRunning " << NUM_STEPS << " steps (dt=" << DT << ")...\n";
    CUDA_CHECK(cudaEventRecord(loop_start_evt, stream));

    for (int step = 1; step <= NUM_STEPS; ++step) {
        const T time = step * DT;

        // ---- The entire hot path: one async kernel launch ----
        engine.step(DT);

        // Stats (periodic)
        if (step % STATS_EVERY == 0) {
            auto pview = engine.particles();
            stats.compute_async(pview, grid, stats_cfg, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto r = stats.fetch_result();
            stats_file << step << ',' << time << ','
                       << r.moments.mean[0] << ',' << r.moments.var[0] << ',' << r.moments.std[0] << ','
                       << r.moments.mean[1] << ',' << r.moments.var[1] << ',' << r.moments.std[1] << ','
                       << r.moments.mean[2] << ',' << r.moments.var[2] << ',' << r.moments.std[2] << '\n';
        }

        // Snapshot (periodic)
        if (step % SNAPSHOT_EVERY == 0) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto pview = engine.particles();
            CUDA_CHECK(cudaMemcpyAsync(h_x, pview.x, N_PARTICLES * sizeof(T), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_y, pview.y, N_PARTICLES * sizeof(T), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_z, pview.z, N_PARTICLES * sizeof(T), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            write_snapshot_csv(snapshot_filename(step), h_x, h_y, h_z, N_PARTICLES, time, step);
        }
    }

    CUDA_CHECK(cudaEventRecord(loop_end_evt, stream));

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
    // 9.  Final stats / snapshot
    // =========================================================================
    {
        auto pview = engine.particles();

        if (NUM_STEPS % STATS_EVERY != 0) {
            stats.compute_async(pview, grid, stats_cfg, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto r = stats.fetch_result();
            stats_file << NUM_STEPS << ',' << NUM_STEPS * DT << ','
                       << r.moments.mean[0] << ',' << r.moments.var[0] << ',' << r.moments.std[0] << ','
                       << r.moments.mean[1] << ',' << r.moments.var[1] << ',' << r.moments.std[1] << ','
                       << r.moments.mean[2] << ',' << r.moments.var[2] << ',' << r.moments.std[2] << '\n';
        }

        stats.compute_async(pview, grid, stats_cfg, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto r = stats.fetch_result();
        std::cout << "\n  Final mean  x: " << r.moments.mean[0]
                  << "\n  Final std   x: " << r.moments.std[0]
                  << "\n  Expected drift = " << VX * NUM_STEPS * DT << " m\n";

        if (NUM_STEPS % SNAPSHOT_EVERY != 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_x, pview.x, N_PARTICLES * sizeof(T), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_y, pview.y, N_PARTICLES * sizeof(T), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_z, pview.z, N_PARTICLES * sizeof(T), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
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

    CUDA_CHECK(cudaEventDestroy(loop_start_evt));
    CUDA_CHECK(cudaEventDestroy(loop_end_evt));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "\nOutput files:\n"
              << "  output/steady_stats.csv               — mean/var/std vs time\n"
              << "  output/steady_snapshot_step*.csv       — particle positions\n"
              << "\nDone.\n";
    return EXIT_SUCCESS;
}
