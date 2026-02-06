/**
 * @file export_csv_and_stats.cu
 * @brief Example demonstrating CSV export and statistics computation.
 *
 * This example shows how to:
 * 1. Run a simple RWPT simulation
 * 2. Compute statistics (mean, variance, std) at intervals
 * 3. Export particle snapshots to CSV files
 * 4. Use both legacy-compatible and extended CSV formats
 *
 * ## Output Files
 *
 * - snapshot_legacy_0000.csv - Legacy format (id, x coord, y coord, z coord)
 * - snapshot_extended_0000.csv - Extended format with status and time
 * - stats_timeseries.csv - Mean/variance over time
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#include <par2_core/par2_core.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

// Simple CUDA error check macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

/**
 * @brief Format step number with zero padding.
 */
std::string format_step(int step, int width = 4) {
    std::ostringstream ss;
    ss << std::setw(width) << std::setfill('0') << step;
    return ss.str();
}

int main() {
    std::cout << "=== Par2_Core Export & Stats Example ===" << std::endl;
    std::cout << "Version: " << par2::Version::string << std::endl;

    // =========================================================================
    // 1. Simulation Setup
    // =========================================================================
    
    // Grid: 20x20x1 cells (2D), 1.0 m spacing
    const int nx = 20, ny = 20, nz = 1;
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);

    std::cout << "\nGrid: " << nx << "x" << ny << "x" << nz << " cells" << std::endl;
    std::cout << "Domain: [0, " << grid.length_x() << "] x [0, " << grid.length_y() << "]" << std::endl;

    // Transport parameters (relatively high dispersion for visible effect)
    par2::TransportParams<double> params;
    params.molecular_diffusion = 0.1;   // m²/s
    params.alpha_l = 0.5;               // m
    params.alpha_t = 0.1;               // m

    std::cout << "Transport: Dm=" << params.molecular_diffusion
              << ", αL=" << params.alpha_l
              << ", αT=" << params.alpha_t << std::endl;

    // Boundary conditions: closed on all sides (legacy default)
    auto bc = par2::BoundaryConfig<double>::all_closed();

    // Engine config
    par2::EngineConfig config;
    config.rng_seed = 12345;  // Reproducible
    config.drift_mode = par2::DriftCorrectionMode::TrilinearOnFly;

    // Create engine
    par2::TransportEngine<double> engine(grid, params, bc, config);
    std::cout << "Engine created" << std::endl;

    // =========================================================================
    // 2. Allocate Velocity Field (uniform flow in X direction)
    // =========================================================================
    
    const size_t vel_size = grid.num_corners();
    double *d_U, *d_V, *d_W;

    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));

    // Uniform velocity: vx = 0.5 m/s
    std::vector<double> h_U(vel_size, 0.5);
    std::vector<double> h_V(vel_size, 0.0);
    std::vector<double> h_W(vel_size, 0.0);

    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));

    // Bind velocity to engine
    par2::VelocityView<double> vel_view;
    vel_view.U = d_U;
    vel_view.V = d_V;
    vel_view.W = d_W;
    vel_view.size = vel_size;
    engine.bind_velocity(vel_view);

    std::cout << "Velocity bound (uniform vx=0.5 m/s)" << std::endl;

    // =========================================================================
    // 3. Allocate Particles
    // =========================================================================
    
    const int num_particles = 1000;
    double *d_x, *d_y, *d_z;
    uint8_t *d_status;

    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_status, num_particles * sizeof(uint8_t)));

    // Initialize status to Active
    CUDA_CHECK(cudaMemset(d_status, 0, num_particles * sizeof(uint8_t)));

    par2::ParticlesView<double> particles;
    particles.x = d_x;
    particles.y = d_y;
    particles.z = d_z;
    particles.status = d_status;
    particles.n = num_particles;

    engine.bind_particles(particles);

    // Inject particles in a box at left side of domain
    engine.inject_box(
        2.0, 8.0, 0.0,   // min corner
        4.0, 12.0, dz,   // max corner
        0,               // first_particle
        num_particles    // count
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Injected " << num_particles << " particles in box [2,4] x [8,12]" << std::endl;

    // Prepare engine (allocates RNG states, etc.)
    engine.prepare();
    std::cout << "Engine prepared for stepping" << std::endl;

    // =========================================================================
    // 4. Setup Statistics and CSV Writers
    // =========================================================================
    
    // Statistics computer
    par2::StatsComputer<double> stats_computer(num_particles);
    par2::StatsConfig stats_config;
    stats_config.use_unwrapped = false;  // No periodic BC here
    stats_config.filter_active_only = true;

    // CSV writer - legacy format
    par2::io::CsvSnapshotConfig legacy_csv_cfg;
    legacy_csv_cfg.legacy_format = true;
    legacy_csv_cfg.include_id = true;
    legacy_csv_cfg.include_time = false;
    legacy_csv_cfg.include_status = false;
    par2::io::CsvSnapshotWriter<double> legacy_writer(num_particles, legacy_csv_cfg);

    // CSV writer - extended format
    par2::io::CsvSnapshotConfig extended_csv_cfg;
    extended_csv_cfg.legacy_format = false;
    extended_csv_cfg.include_id = true;
    extended_csv_cfg.include_time = true;
    extended_csv_cfg.include_status = true;
    extended_csv_cfg.stride = 10;  // Sample every 10th particle
    par2::io::CsvSnapshotWriter<double> extended_writer(num_particles, extended_csv_cfg);

    // Open stats time-series file
    std::ofstream stats_file("stats_timeseries.csv");
    stats_file << std::setprecision(6) << std::fixed;
    stats_file << "step,time,mean_x,mean_y,mean_z,std_x,std_y,std_z,active,exited" << std::endl;

    // =========================================================================
    // 5. Simulation Loop with Export
    // =========================================================================
    
    const int total_steps = 100;
    const double dt = 0.1;  // time step
    const int snapshot_interval = 20;
    const int stats_interval = 10;

    std::cout << "\nRunning " << total_steps << " steps (dt=" << dt << "s)..." << std::endl;
    std::cout << "Snapshot interval: " << snapshot_interval << " steps" << std::endl;
    std::cout << "Stats interval: " << stats_interval << " steps" << std::endl;

    cudaStream_t stream = nullptr;  // Use default stream

    for (int step = 0; step <= total_steps; ++step) {
        double t = step * dt;

        // --- Compute and log statistics ---
        if (step % stats_interval == 0) {
            stats_computer.compute_async(
                par2::ConstParticlesView<double>(particles),
                grid,
                stats_config,
                stream
            );
            CUDA_CHECK(cudaStreamSynchronize(stream));

            auto result = stats_computer.fetch_result();
            if (result.computed) {
                stats_file << step << ","
                          << t << ","
                          << result.moments.mean[0] << ","
                          << result.moments.mean[1] << ","
                          << result.moments.mean[2] << ","
                          << result.moments.std[0] << ","
                          << result.moments.std[1] << ","
                          << result.moments.std[2] << ","
                          << result.counts.active << ","
                          << result.counts.exited << std::endl;

                if (step % 50 == 0) {
                    std::cout << "Step " << step << " (t=" << t << "): "
                              << "mean_x=" << result.moments.mean[0]
                              << ", std_x=" << result.moments.std[0]
                              << ", active=" << result.counts.active << std::endl;
                }
            }
        }

        // --- Export snapshots ---
        if (step % snapshot_interval == 0) {
            // Legacy format
            std::string legacy_file = "snapshot_legacy_" + format_step(step) + ".csv";
            legacy_writer.write_snapshot(
                par2::ConstParticlesView<double>(particles),
                legacy_file,
                t,
                stream
            );

            // Extended format (sampled)
            std::string extended_file = "snapshot_extended_" + format_step(step) + ".csv";
            extended_writer.write_snapshot(
                par2::ConstParticlesView<double>(particles),
                extended_file,
                t,
                stream
            );

            std::cout << "Wrote snapshots at step " << step << std::endl;
        }

        // --- Advance simulation ---
        if (step < total_steps) {
            engine.step(dt);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    stats_file.close();

    // =========================================================================
    // 6. Final Statistics
    // =========================================================================
    
    std::cout << "\n=== Final Statistics ===" << std::endl;

    // Use convenience functions (legacy-style)
    double conc_past_10 = par2::concentration_past_plane<double>(
        par2::ConstParticlesView<double>(particles),
        0,     // X axis
        10.0,  // threshold
        stream
    );
    std::cout << "Fraction past x=10m: " << conc_past_10 << std::endl;

    double conc_in_box = par2::concentration_box<double>(
        par2::ConstParticlesView<double>(particles),
        15.0, 5.0, 0.0,   // min
        20.0, 15.0, 1.0,  // max
        stream
    );
    std::cout << "Fraction in box [15,20]x[5,15]: " << conc_in_box << std::endl;

    auto counts = par2::count_by_status<double>(
        par2::ConstParticlesView<double>(particles),
        stream
    );
    std::cout << "Status counts: active=" << counts.active
              << ", exited=" << counts.exited
              << ", inactive=" << counts.inactive << std::endl;

    // =========================================================================
    // 7. Verify Output Files
    // =========================================================================
    
    std::cout << "\n=== Output Files ===" << std::endl;

    // Read first few lines of a snapshot to show format
    std::ifstream legacy_check("snapshot_legacy_0000.csv");
    if (legacy_check.is_open()) {
        std::cout << "\nLegacy CSV format (snapshot_legacy_0000.csv):" << std::endl;
        std::string line;
        for (int i = 0; i < 5 && std::getline(legacy_check, line); ++i) {
            std::cout << "  " << line << std::endl;
        }
        std::cout << "  ..." << std::endl;
        legacy_check.close();
    }

    std::ifstream extended_check("snapshot_extended_0000.csv");
    if (extended_check.is_open()) {
        std::cout << "\nExtended CSV format (snapshot_extended_0000.csv):" << std::endl;
        std::string line;
        for (int i = 0; i < 5 && std::getline(extended_check, line); ++i) {
            std::cout << "  " << line << std::endl;
        }
        std::cout << "  ..." << std::endl;
        extended_check.close();
    }

    std::ifstream stats_check("stats_timeseries.csv");
    if (stats_check.is_open()) {
        std::cout << "\nStats time-series (stats_timeseries.csv):" << std::endl;
        std::string line;
        for (int i = 0; i < 5 && std::getline(stats_check, line); ++i) {
            std::cout << "  " << line << std::endl;
        }
        std::cout << "  ..." << std::endl;
        stats_check.close();
    }

    // =========================================================================
    // 8. Cleanup
    // =========================================================================
    
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_status));

    std::cout << "\nExample completed successfully!" << std::endl;
    return 0;
}
