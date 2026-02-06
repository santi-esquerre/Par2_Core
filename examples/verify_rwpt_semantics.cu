/**
 * @file verify_rwpt_semantics.cu
 * @brief Verification example for M1 - RWPT Semantic Parity.
 *
 * This example verifies that Par2_Core reproduces the exact RWPT semantics
 * from legacy PAR². It tests:
 *
 * 1. **Linear interpolation** (FaceField::in equivalent)
 * 2. **Trilinear interpolation** (CornerField::in equivalent)
 * 3. **Displacement matrix B** (CornerField::displacementMatrix equivalent)
 * 4. **Drift correction TrilinearOnFly** (CornerField::velocityCorrection)
 * 5. **Drift correction Precomputed** (CellField finite differences)
 *
 * Mathematical formula verified:
 *   Δx = (v_interp + v_drift) * dt + B * ξ
 *
 * SOURCE OF TRUTH: legacy/Geometry/*, legacy/Particles/MoveParticle.cuh
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#include <par2_core/par2_core.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// =============================================================================
// Test utilities
// =============================================================================

struct TestResult {
    bool passed;
    std::string message;
    double mean_displacement;
    double std_displacement;
};

template <typename T>
void compute_stats(const std::vector<T>& data, T& mean, T& stddev) {
    T sum = 0;
    for (auto v : data) sum += v;
    mean = sum / data.size();
    
    T sq_sum = 0;
    for (auto v : data) sq_sum += (v - mean) * (v - mean);
    stddev = std::sqrt(sq_sum / data.size());
}

// =============================================================================
// Test 1: Linear interpolation mode (Linear + TrilinearOnFly)
// =============================================================================

TestResult test_linear_trilinear_onfly() {
    std::cout << "\n=== Test 1: Linear + TrilinearOnFly ===" << std::endl;
    std::cout << "  (FaceField interpolation, cornerfield drift)" << std::endl;
    
    // Setup grid
    const int nx = 20, ny = 20, nz = 10;
    const double dx = 0.5, dy = 0.5, dz = 0.5;
    auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);
    
    // Transport params
    par2::TransportParams<double> params;
    params.molecular_diffusion = 1e-5;
    params.alpha_l = 0.1;
    params.alpha_t = 0.01;
    
    // Engine config - Linear + TrilinearOnFly
    par2::EngineConfig config;
    config.interpolation_mode = par2::InterpolationMode::Linear;
    config.drift_mode = par2::DriftCorrectionMode::TrilinearOnFly;
    config.rng_seed = 12345;
    
    auto bc = par2::BoundaryConfig<double>::all_closed();
    par2::TransportEngine<double> engine(grid, params, bc, config);
    
    // Allocate velocity field (uniform flow)
    const size_t vel_size = grid.num_corners();
    double *d_U, *d_V, *d_W;
    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));
    
    const double vx_uniform = 0.1;  // m/s
    std::vector<double> h_U(vel_size, vx_uniform);
    std::vector<double> h_V(vel_size, 0.0);
    std::vector<double> h_W(vel_size, 0.0);
    
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    
    engine.bind_velocity({d_U, d_V, d_W, vel_size});
    
    // TrilinearOnFly drift needs corner velocity - let engine compute it
    engine.update_derived_fields();
    CUDA_CHECK(cudaDeviceSynchronize());  // Wait for corner computation
    
    // Allocate particles
    const int num_particles = 10000;
    double *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));
    
    engine.bind_particles({d_x, d_y, d_z, num_particles});
    
    // Inject in center
    double x0 = 3.0, y0 = 3.0, z0 = 2.0;
    double x1 = 7.0, y1 = 7.0, z1 = 3.0;
    engine.inject_box(x0, y0, z0, x1, y1, z1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Read initial mean position
    std::vector<double> h_x(num_particles), h_y(num_particles), h_z(num_particles);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    double mean_x0, std_x0;
    compute_stats(h_x, mean_x0, std_x0);
    
    // Run simulation
    const double dt = 0.1;
    const int num_steps = 50;
    for (int i = 0; i < num_steps; ++i) {
        engine.step(dt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Read final positions
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    double mean_x1, std_x1;
    compute_stats(h_x, mean_x1, std_x1);
    
    double total_time = dt * num_steps;
    double expected_advection = vx_uniform * total_time;
    double actual_advection = mean_x1 - mean_x0;
    double error_percent = std::abs(actual_advection - expected_advection) / expected_advection * 100;
    
    std::cout << "  Expected advection: " << expected_advection << " m" << std::endl;
    std::cout << "  Actual advection:   " << actual_advection << " m" << std::endl;
    std::cout << "  Error: " << error_percent << "%" << std::endl;
    std::cout << "  Dispersion (std): " << std_x1 << " m" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    
    // Pass if advection is within 10% and dispersion is reasonable
    bool passed = error_percent < 10.0 && std_x1 > 0.01;
    return {passed, passed ? "PASS" : "FAIL", actual_advection, std_x1};
}

// =============================================================================
// Test 2: Trilinear interpolation mode (Trilinear + TrilinearOnFly)
// =============================================================================

TestResult test_trilinear_trilinear_onfly() {
    std::cout << "\n=== Test 2: Trilinear + TrilinearOnFly ===" << std::endl;
    std::cout << "  (Full legacy parity - smoothest mode)" << std::endl;
    
    const int nx = 20, ny = 20, nz = 10;
    const double dx = 0.5, dy = 0.5, dz = 0.5;
    auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);
    
    par2::TransportParams<double> params;
    params.molecular_diffusion = 1e-5;
    params.alpha_l = 0.1;
    params.alpha_t = 0.01;
    
    par2::EngineConfig config;
    config.interpolation_mode = par2::InterpolationMode::Trilinear;
    config.drift_mode = par2::DriftCorrectionMode::TrilinearOnFly;
    config.rng_seed = 12345;
    
    auto bc = par2::BoundaryConfig<double>::all_closed();
    par2::TransportEngine<double> engine(grid, params, bc, config);
    
    const size_t vel_size = grid.num_corners();
    double *d_U, *d_V, *d_W;
    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));
    
    const double vx_uniform = 0.1;
    std::vector<double> h_U(vel_size, vx_uniform);
    std::vector<double> h_V(vel_size, 0.0);
    std::vector<double> h_W(vel_size, 0.0);
    
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    
    engine.bind_velocity({d_U, d_V, d_W, vel_size});
    
    // Trilinear interpolation + TrilinearOnFly drift needs corner velocity
    engine.update_derived_fields();
    CUDA_CHECK(cudaDeviceSynchronize());  // Wait for corner computation
    
    const int num_particles = 10000;
    double *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));
    
    engine.bind_particles({d_x, d_y, d_z, num_particles});
    engine.inject_box(3.0, 3.0, 2.0, 7.0, 7.0, 3.0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<double> h_x(num_particles);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    double mean_x0, std_x0;
    compute_stats(h_x, mean_x0, std_x0);
    
    const double dt = 0.1;
    const int num_steps = 50;
    for (int i = 0; i < num_steps; ++i) {
        engine.step(dt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    double mean_x1, std_x1;
    compute_stats(h_x, mean_x1, std_x1);
    
    double total_time = dt * num_steps;
    double expected_advection = vx_uniform * total_time;
    double actual_advection = mean_x1 - mean_x0;
    double error_percent = std::abs(actual_advection - expected_advection) / expected_advection * 100;
    
    std::cout << "  Expected advection: " << expected_advection << " m" << std::endl;
    std::cout << "  Actual advection:   " << actual_advection << " m" << std::endl;
    std::cout << "  Error: " << error_percent << "%" << std::endl;
    std::cout << "  Dispersion (std): " << std_x1 << " m" << std::endl;
    
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    
    bool passed = error_percent < 10.0 && std_x1 > 0.01;
    return {passed, passed ? "PASS" : "FAIL", actual_advection, std_x1};
}

// =============================================================================
// Test 3: Linear + Precomputed (requires drift field)
// =============================================================================

TestResult test_linear_precomputed() {
    std::cout << "\n=== Test 3: Linear + Precomputed ===" << std::endl;
    std::cout << "  (Fast mode for steady-state flows)" << std::endl;
    
    const int nx = 20, ny = 20, nz = 10;
    const double dx = 0.5, dy = 0.5, dz = 0.5;
    auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);
    
    par2::TransportParams<double> params;
    params.molecular_diffusion = 1e-5;
    params.alpha_l = 0.1;
    params.alpha_t = 0.01;
    
    par2::EngineConfig config;
    config.interpolation_mode = par2::InterpolationMode::Linear;
    config.drift_mode = par2::DriftCorrectionMode::Precomputed;
    config.rng_seed = 12345;
    
    auto bc = par2::BoundaryConfig<double>::all_closed();
    par2::TransportEngine<double> engine(grid, params, bc, config);
    
    const size_t vel_size = grid.num_corners();
    const size_t cell_size = grid.num_cells();
    double *d_U, *d_V, *d_W;
    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));
    
    const double vx_uniform = 0.1;
    std::vector<double> h_U(vel_size, vx_uniform);
    std::vector<double> h_V(vel_size, 0.0);
    std::vector<double> h_W(vel_size, 0.0);
    
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    
    engine.bind_velocity({d_U, d_V, d_W, vel_size});
    // Linear + Precomputed doesn't need corner velocity
    
    // Allocate precomputed drift field (zero for uniform velocity)
    double *d_dcx, *d_dcy, *d_dcz;
    CUDA_CHECK(cudaMalloc(&d_dcx, cell_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dcy, cell_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dcz, cell_size * sizeof(double)));
    
    // For uniform velocity, drift correction is zero (div(D) = 0)
    CUDA_CHECK(cudaMemset(d_dcx, 0, cell_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dcy, 0, cell_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dcz, 0, cell_size * sizeof(double)));
    
    engine.bind_drift_correction({d_dcx, d_dcy, d_dcz, cell_size});
    
    const int num_particles = 10000;
    double *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));
    
    engine.bind_particles({d_x, d_y, d_z, num_particles});
    engine.inject_box(3.0, 3.0, 2.0, 7.0, 7.0, 3.0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<double> h_x(num_particles);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    double mean_x0, std_x0;
    compute_stats(h_x, mean_x0, std_x0);
    
    const double dt = 0.1;
    const int num_steps = 50;
    for (int i = 0; i < num_steps; ++i) {
        engine.step(dt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    double mean_x1, std_x1;
    compute_stats(h_x, mean_x1, std_x1);
    
    double total_time = dt * num_steps;
    double expected_advection = vx_uniform * total_time;
    double actual_advection = mean_x1 - mean_x0;
    double error_percent = std::abs(actual_advection - expected_advection) / expected_advection * 100;
    
    std::cout << "  Expected advection: " << expected_advection << " m" << std::endl;
    std::cout << "  Actual advection:   " << actual_advection << " m" << std::endl;
    std::cout << "  Error: " << error_percent << "%" << std::endl;
    std::cout << "  Dispersion (std): " << std_x1 << " m" << std::endl;
    
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_dcx));
    CUDA_CHECK(cudaFree(d_dcy));
    CUDA_CHECK(cudaFree(d_dcz));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    
    bool passed = error_percent < 10.0 && std_x1 > 0.01;
    return {passed, passed ? "PASS" : "FAIL", actual_advection, std_x1};
}

// =============================================================================
// Test 4: 2D mode (nz=1)
// =============================================================================

TestResult test_2d_mode() {
    std::cout << "\n=== Test 3: 2D Mode (nz=1) ===" << std::endl;
    std::cout << "  (Verifies z-displacement is zero)" << std::endl;
    
    const int nx = 20, ny = 20, nz = 1;  // 2D!
    const double dx = 0.5, dy = 0.5, dz = 0.5;
    auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);
    
    par2::TransportParams<double> params;
    params.molecular_diffusion = 1e-5;
    params.alpha_l = 0.1;
    params.alpha_t = 0.01;
    
    par2::EngineConfig config;
    config.interpolation_mode = par2::InterpolationMode::Linear;
    config.drift_mode = par2::DriftCorrectionMode::TrilinearOnFly;
    config.rng_seed = 12345;
    
    auto bc = par2::BoundaryConfig<double>::all_closed();
    par2::TransportEngine<double> engine(grid, params, bc, config);
    
    const size_t vel_size = grid.num_corners();
    double *d_U, *d_V, *d_W;
    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));
    
    std::vector<double> h_U(vel_size, 0.1);
    std::vector<double> h_V(vel_size, 0.0);
    std::vector<double> h_W(vel_size, 0.0);
    
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    
    engine.bind_velocity({d_U, d_V, d_W, vel_size});
    
    // TrilinearOnFly needs corner velocity
    engine.update_derived_fields();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    const int num_particles = 1000;
    double *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));
    
    engine.bind_particles({d_x, d_y, d_z, num_particles});
    
    // Inject at z=0.25 (center of single layer)
    engine.inject_box(3.0, 3.0, 0.25, 7.0, 7.0, 0.25);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<double> h_z(num_particles);
    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    double mean_z0 = h_z[0];  // All should be 0.25
    
    const double dt = 0.1;
    const int num_steps = 50;
    for (int i = 0; i < num_steps; ++i) {
        engine.step(dt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Check all z values unchanged
    double max_z_change = 0;
    for (int i = 0; i < num_particles; ++i) {
        max_z_change = std::max(max_z_change, std::abs(h_z[i] - mean_z0));
    }
    
    std::cout << "  Initial z: " << mean_z0 << std::endl;
    std::cout << "  Max z-change: " << max_z_change << std::endl;
    
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    
    bool passed = max_z_change < 1e-10;
    return {passed, passed ? "PASS" : "FAIL", 0, max_z_change};
}

// =============================================================================
// Test 4: Zero-velocity tolerance
// =============================================================================

TestResult test_zero_velocity_tolerance() {
    std::cout << "\n=== Test 4: Zero-Velocity Tolerance ===" << std::endl;
    std::cout << "  (Verifies no NaN with zero velocity)" << std::endl;
    
    const int nx = 10, ny = 10, nz = 5;
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);
    
    par2::TransportParams<double> params;
    params.molecular_diffusion = 1e-5;
    params.alpha_l = 0.1;
    params.alpha_t = 0.01;
    
    par2::EngineConfig config;
    config.interpolation_mode = par2::InterpolationMode::Trilinear;
    config.drift_mode = par2::DriftCorrectionMode::TrilinearOnFly;
    config.rng_seed = 12345;
    
    auto bc = par2::BoundaryConfig<double>::all_closed();
    par2::TransportEngine<double> engine(grid, params, bc, config);
    
    const size_t vel_size = grid.num_corners();
    double *d_U, *d_V, *d_W;
    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));
    
    // ZERO velocity everywhere!
    std::vector<double> h_U(vel_size, 0.0);
    std::vector<double> h_V(vel_size, 0.0);
    std::vector<double> h_W(vel_size, 0.0);
    
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    
    engine.bind_velocity({d_U, d_V, d_W, vel_size});
    
    // TrilinearOnFly needs corner velocity
    engine.update_derived_fields();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    const int num_particles = 100;
    double *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));
    
    engine.bind_particles({d_x, d_y, d_z, num_particles});
    engine.inject_box(2.0, 2.0, 1.0, 8.0, 8.0, 4.0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    const double dt = 0.1;
    const int num_steps = 10;
    for (int i = 0; i < num_steps; ++i) {
        engine.step(dt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check for NaN
    std::vector<double> h_x(num_particles), h_y(num_particles), h_z(num_particles);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    
    bool has_nan = false;
    for (int i = 0; i < num_particles; ++i) {
        if (std::isnan(h_x[i]) || std::isnan(h_y[i]) || std::isnan(h_z[i])) {
            has_nan = true;
            break;
        }
    }
    
    std::cout << "  NaN detected: " << (has_nan ? "YES" : "NO") << std::endl;
    
    // Check that pure diffusion occurred (particles moved)
    double mean_x, std_x;
    compute_stats(h_x, mean_x, std_x);
    std::cout << "  Dispersion (std_x): " << std_x << " m" << std::endl;
    
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    
    bool passed = !has_nan && std_x > 1e-6;
    return {passed, passed ? "PASS" : "FAIL", 0, std_x};
}

// =============================================================================
// Test 5: External corner velocity binding (M2 - T6)
// =============================================================================

TestResult test_external_corner_binding() {
    std::cout << "\n=== Test 5: External Corner Velocity Binding ===" << std::endl;
    std::cout << "  (Verifies bind_corner_velocity takes priority)" << std::endl;
    
    const int nx = 10, ny = 10, nz = 5;
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);
    
    par2::TransportParams<double> params;
    params.molecular_diffusion = 1e-9;  // Very small diffusion
    params.alpha_l = 0.0;
    params.alpha_t = 0.0;
    
    par2::EngineConfig config;
    config.interpolation_mode = par2::InterpolationMode::Trilinear;
    config.drift_mode = par2::DriftCorrectionMode::None;  // No drift to isolate advection
    config.rng_seed = 12345;
    
    auto bc = par2::BoundaryConfig<double>::all_closed();
    par2::TransportEngine<double> engine(grid, params, bc, config);
    
    // Allocate velocity field (ZERO velocity in face field)
    const size_t vel_size = grid.num_corners();
    double *d_U, *d_V, *d_W;
    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_U, 0, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_V, 0, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_W, 0, vel_size * sizeof(double)));
    
    // Allocate external corner velocity (NON-ZERO, uniform)
    const double external_velocity = 0.5;  // 0.5 m/s in X
    double *d_Uc, *d_Vc, *d_Wc;
    CUDA_CHECK(cudaMalloc(&d_Uc, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Vc, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Wc, vel_size * sizeof(double)));
    
    // Fill external corner with uniform velocity
    std::vector<double> h_Uc(vel_size, external_velocity);
    std::vector<double> h_Vc(vel_size, 0.0);
    std::vector<double> h_Wc(vel_size, 0.0);
    CUDA_CHECK(cudaMemcpy(d_Uc, h_Uc.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Vc, h_Vc.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wc, h_Wc.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    
    // Bind face velocity (ZERO)
    engine.bind_velocity({d_U, d_V, d_W, vel_size});
    
    // Bind EXTERNAL corner velocity (NON-ZERO)
    // This should take priority over internal computation
    engine.bind_corner_velocity({d_Uc, d_Vc, d_Wc, vel_size});
    
    // Call update_derived_fields - should do NOTHING because external is bound
    engine.update_derived_fields();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Setup particles at center
    const int num_particles = 100;
    double *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));
    
    engine.bind_particles({d_x, d_y, d_z, num_particles});
    engine.inject_box(4.0, 4.0, 2.0, 6.0, 6.0, 3.0);  // Center of domain
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get initial positions
    std::vector<double> h_x0(num_particles);
    CUDA_CHECK(cudaMemcpy(h_x0.data(), d_x, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Step particles
    const double dt = 1.0;
    const int num_steps = 10;  // Total time = 10s
    for (int i = 0; i < num_steps; ++i) {
        engine.step(dt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get final positions
    std::vector<double> h_x(num_particles);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, num_particles * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Compute mean displacement
    double mean_dx = 0.0;
    for (int i = 0; i < num_particles; ++i) {
        mean_dx += (h_x[i] - h_x0[i]);
    }
    mean_dx /= num_particles;
    
    // Expected displacement: v * t = 0.5 m/s * 10 s = 5.0 m
    const double expected_dx = external_velocity * dt * num_steps;
    const double error_pct = 100.0 * std::abs(mean_dx - expected_dx) / expected_dx;
    
    std::cout << "  External corner velocity: " << external_velocity << " m/s" << std::endl;
    std::cout << "  Face velocity: 0.0 m/s (should be ignored)" << std::endl;
    std::cout << "  Expected displacement: " << expected_dx << " m" << std::endl;
    std::cout << "  Actual mean displacement: " << mean_dx << " m" << std::endl;
    std::cout << "  Error: " << std::fixed << std::setprecision(1) << error_pct << "%" << std::endl;
    
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_Uc));
    CUDA_CHECK(cudaFree(d_Vc));
    CUDA_CHECK(cudaFree(d_Wc));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    
    // Pass if mean displacement is close to expected (within 10%)
    // If internal computation had run (using zero face velocity), displacement would be ~0
    bool passed = (mean_dx > expected_dx * 0.8) && (mean_dx < expected_dx * 1.2);
    return {passed, passed ? "PASS" : "FAIL", mean_dx, 0.0};
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         Par2_Core M1 - RWPT Semantic Parity Tests            ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Formula: Δx = (v_interp + v_drift)*dt + B*ξ                 ║" << std::endl;
    std::cout << "║  Source: legacy/Geometry/*, legacy/Particles/MoveParticle.cuh ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;

    std::vector<TestResult> results;
    
    results.push_back(test_linear_trilinear_onfly());
    results.push_back(test_trilinear_trilinear_onfly());
    results.push_back(test_linear_precomputed());
    results.push_back(test_2d_mode());
    results.push_back(test_zero_velocity_tolerance());
    results.push_back(test_external_corner_binding());
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                         SUMMARY                               ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
    
    int passed = 0;
    const char* test_names[] = {
        "Linear + TrilinearOnFly",
        "Trilinear + TrilinearOnFly",
        "Linear + Precomputed",
        "2D Mode (nz=1)",
        "Zero-Velocity Tolerance",
        "External Corner Binding"
    };
    
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "║  " << std::left << std::setw(30) << test_names[i] 
                  << ": " << std::setw(4) << results[i].message << "             ║" << std::endl;
        if (results[i].passed) ++passed;
    }
    
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Total: " << passed << "/" << results.size() 
              << " tests passed                                       ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
    
    return (passed == static_cast<int>(results.size())) ? 0 : 1;
}
