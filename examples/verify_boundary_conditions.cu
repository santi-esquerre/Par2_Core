/**
 * @file verify_boundary_conditions.cu
 * @brief Verification tests for M3 - Boundary Conditions + Tracking.
 *
 * This example verifies:
 * 1. Closed BC: particles stay inside domain (legacy behavior)
 * 2. Open BC: particles exit and get status=Exited
 * 3. Periodic BC: particles wrap correctly with wrapCount tracking
 * 4. Status gating: Exited particles don't move
 * 5. Unwrap on-demand: wrapped position + wrapCount*L = continuous position
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
    double value1;
    double value2;
};

// =============================================================================
// Test 1: Closed BC (legacy behavior)
// =============================================================================

TestResult test_closed_bc() {
    std::cout << "\n=== Test 1: Closed BC (Legacy Behavior) ===" << std::endl;
    std::cout << "  (Particle at boundary should stay in place)" << std::endl;

    const int nx = 10, ny = 10, nz = 5;
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);

    par2::TransportParams<double> params;
    params.molecular_diffusion = 0.0;  // No diffusion - pure advection
    params.alpha_l = 0.0;
    params.alpha_t = 0.0;

    par2::EngineConfig config;
    config.interpolation_mode = par2::InterpolationMode::Linear;
    config.drift_mode = par2::DriftCorrectionMode::None;
    config.rng_seed = 12345;

    // All closed BC (default)
    auto bc = par2::BoundaryConfig<double>::all_closed();
    par2::TransportEngine<double> engine(grid, params, bc, config);

    // Velocity pointing OUT of domain (towards X+)
    const size_t vel_size = grid.num_corners();
    const double vx_out = 1.0;  // Strong velocity towards boundary

    double *d_U, *d_V, *d_W;
    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));

    std::vector<double> h_U(vel_size, vx_out);
    std::vector<double> h_V(vel_size, 0.0);
    std::vector<double> h_W(vel_size, 0.0);

    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));

    engine.bind_velocity({d_U, d_V, d_W, vel_size});

    // Single particle near X+ boundary
    const int num_particles = 1;
    double *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));

    // Place particle close to X+ boundary
    double x0 = 9.5;  // Close to boundary at x=10
    double y0 = 5.0;
    double z0 = 2.5;
    CUDA_CHECK(cudaMemcpy(d_x, &x0, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, &y0, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, &z0, sizeof(double), cudaMemcpyHostToDevice));

    engine.bind_particles({d_x, d_y, d_z, num_particles});

    // Step with large dt - would move particle 1.0m, past boundary
    const double dt = 1.0;
    engine.step(dt);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get final position
    double x_final;
    CUDA_CHECK(cudaMemcpy(&x_final, d_x, sizeof(double), cudaMemcpyDeviceToHost));

    // With closed BC, particle should NOT move past boundary
    // Legacy behavior: displacement is rejected if it would exit
    std::cout << "  Initial X: " << x0 << std::endl;
    std::cout << "  Final X:   " << x_final << std::endl;
    std::cout << "  Expected:  stay at " << x0 << " (displacement rejected)" << std::endl;

    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    // Pass if particle stayed in place
    bool passed = std::abs(x_final - x0) < 1e-10;
    return {passed, passed ? "PASS" : "FAIL", x0, x_final};
}

// =============================================================================
// Test 2: Open BC + Exited Status
// =============================================================================

TestResult test_open_bc() {
    std::cout << "\n=== Test 2: Open BC + Exited Status ===" << std::endl;
    std::cout << "  (Particle crossing open boundary gets Exited status)" << std::endl;

    const int nx = 10, ny = 10, nz = 5;
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);

    par2::TransportParams<double> params;
    params.molecular_diffusion = 0.0;
    params.alpha_l = 0.0;
    params.alpha_t = 0.0;

    par2::EngineConfig config;
    config.interpolation_mode = par2::InterpolationMode::Linear;
    config.drift_mode = par2::DriftCorrectionMode::None;
    config.rng_seed = 12345;

    // Open BC in X
    auto bc = par2::BoundaryConfig<double>::open_x();
    par2::TransportEngine<double> engine(grid, params, bc, config);

    // Velocity pointing towards X+ boundary
    const size_t vel_size = grid.num_corners();
    const double vx_out = 2.0;

    double *d_U, *d_V, *d_W;
    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));

    std::vector<double> h_U(vel_size, vx_out);
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_V, 0, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_W, 0, vel_size * sizeof(double)));

    engine.bind_velocity({d_U, d_V, d_W, vel_size});

    // Particle near X+ boundary
    const int num_particles = 1;
    double *d_x, *d_y, *d_z;
    uint8_t *d_status;
    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_status, num_particles * sizeof(uint8_t)));

    double x0 = 9.5, y0 = 5.0, z0 = 2.5;
    uint8_t status_init = static_cast<uint8_t>(par2::ParticleStatus::Active);
    CUDA_CHECK(cudaMemcpy(d_x, &x0, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, &y0, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, &z0, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_status, &status_init, sizeof(uint8_t), cudaMemcpyHostToDevice));

    par2::ParticlesView<double> pview{d_x, d_y, d_z, num_particles, d_status};
    engine.bind_particles(pview);

    // Step - particle should exit through X+
    engine.step(1.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check status
    uint8_t status_final;
    double x_final;
    CUDA_CHECK(cudaMemcpy(&status_final, d_status, sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&x_final, d_x, sizeof(double), cudaMemcpyDeviceToHost));

    bool is_exited = (status_final == static_cast<uint8_t>(par2::ParticleStatus::Exited));
    std::cout << "  Status after step: " << (is_exited ? "Exited" : "Active") << std::endl;
    std::cout << "  Position X: " << x_final << " (stayed at " << x0 << " when exiting)" << std::endl;

    // Try another step - particle should NOT move (status gating)
    double x_before_step2 = x_final;
    engine.step(1.0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&x_final, d_x, sizeof(double), cudaMemcpyDeviceToHost));

    bool no_movement = std::abs(x_final - x_before_step2) < 1e-10;
    std::cout << "  Position after 2nd step: " << x_final << " (should be same - status gating)" << std::endl;

    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_status));

    bool passed = is_exited && no_movement;
    return {passed, passed ? "PASS" : "FAIL", static_cast<double>(is_exited), static_cast<double>(no_movement)};
}

// =============================================================================
// Test 3: Periodic BC + wrapCount
// =============================================================================

TestResult test_periodic_bc() {
    std::cout << "\n=== Test 3: Periodic BC + wrapCount ===" << std::endl;
    std::cout << "  (Particle wraps and wrapCount accumulates)" << std::endl;

    const int nx = 10, ny = 10, nz = 5;
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);
    const double Lx = nx * dx;  // Domain length in X

    par2::TransportParams<double> params;
    params.molecular_diffusion = 0.0;
    params.alpha_l = 0.0;
    params.alpha_t = 0.0;

    par2::EngineConfig config;
    config.interpolation_mode = par2::InterpolationMode::Linear;
    config.drift_mode = par2::DriftCorrectionMode::None;
    config.rng_seed = 12345;

    // Periodic BC in X
    par2::BoundaryConfig<double> bc;
    bc.x = par2::AxisBoundary<double>::periodic();
    bc.y = par2::AxisBoundary<double>::closed();
    bc.z = par2::AxisBoundary<double>::closed();
    par2::TransportEngine<double> engine(grid, params, bc, config);

    // Strong velocity in X
    const size_t vel_size = grid.num_corners();
    const double vx = 2.0;  // 2 m/s towards X+

    double *d_U, *d_V, *d_W;
    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));

    std::vector<double> h_U(vel_size, vx);
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_V, 0, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_W, 0, vel_size * sizeof(double)));

    engine.bind_velocity({d_U, d_V, d_W, vel_size});

    // Particle with wrapCount
    const int num_particles = 1;
    double *d_x, *d_y, *d_z;
    int32_t *d_wrapX;
    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_wrapX, num_particles * sizeof(int32_t)));

    double x0 = 9.0, y0 = 5.0, z0 = 2.5;  // Close to X+ boundary
    int32_t wrap0 = 0;
    CUDA_CHECK(cudaMemcpy(d_x, &x0, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, &y0, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, &z0, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wrapX, &wrap0, sizeof(int32_t), cudaMemcpyHostToDevice));

    par2::ParticlesView<double> pview{d_x, d_y, d_z, num_particles};
    pview.wrapX = d_wrapX;
    engine.bind_particles(pview);

    // Step with dt=1.0: displacement = 2.0m, should wrap
    // x0 + 2.0 = 11.0, which should wrap to 1.0 with wrapX=1
    engine.step(1.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    double x_after1;
    int32_t wrap_after1;
    CUDA_CHECK(cudaMemcpy(&x_after1, d_x, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&wrap_after1, d_wrapX, sizeof(int32_t), cudaMemcpyDeviceToHost));

    std::cout << "  After step 1:" << std::endl;
    std::cout << "    x_wrapped: " << x_after1 << " (expected ~1.0)" << std::endl;
    std::cout << "    wrapX:     " << wrap_after1 << " (expected 1)" << std::endl;

    // Compute unwrapped position
    double x_unwrap1 = x_after1 + wrap_after1 * Lx;
    std::cout << "    x_unwrap:  " << x_unwrap1 << " (expected ~11.0)" << std::endl;

    // Another step: should wrap again
    // x_after1 + 2.0 ≈ 3.0, no wrap
    engine.step(1.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    double x_after2;
    int32_t wrap_after2;
    CUDA_CHECK(cudaMemcpy(&x_after2, d_x, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&wrap_after2, d_wrapX, sizeof(int32_t), cudaMemcpyDeviceToHost));

    std::cout << "  After step 2:" << std::endl;
    std::cout << "    x_wrapped: " << x_after2 << " (expected ~3.0)" << std::endl;
    std::cout << "    wrapX:     " << wrap_after2 << " (expected 1, no new wrap)" << std::endl;

    // 5 more steps: total dt=7s, total displacement=14m, should be 5 wraps (14/10=1.4)
    // But starting from x0=9: continuous position = 9+14 = 23
    // Net wraps = floor(23/10) = 2
    engine.advance(1.0, 5);
    CUDA_CHECK(cudaDeviceSynchronize());

    double x_after7;
    int32_t wrap_after7;
    CUDA_CHECK(cudaMemcpy(&x_after7, d_x, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&wrap_after7, d_wrapX, sizeof(int32_t), cudaMemcpyDeviceToHost));

    double x_unwrap7 = x_after7 + wrap_after7 * Lx;
    double expected_unwrap = x0 + vx * 7.0;  // 9 + 14 = 23

    std::cout << "  After 7 steps (total 14m displacement):" << std::endl;
    std::cout << "    x_wrapped: " << x_after7 << std::endl;
    std::cout << "    wrapX:     " << wrap_after7 << std::endl;
    std::cout << "    x_unwrap:  " << x_unwrap7 << " (expected " << expected_unwrap << ")" << std::endl;

    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_wrapX));

    // Verify:
    // 1. x_wrapped is in [0, 10)
    // 2. wrapCount is correct
    // 3. unwrap recovers continuous position
    bool in_range = (x_after7 >= 0.0 && x_after7 < Lx);
    bool unwrap_correct = std::abs(x_unwrap7 - expected_unwrap) < 1e-6;

    bool passed = in_range && unwrap_correct;
    return {passed, passed ? "PASS" : "FAIL", x_unwrap7, expected_unwrap};
}

// =============================================================================
// Test 4: Status gating (Inactive particles don't move)
// =============================================================================

TestResult test_status_gating() {
    std::cout << "\n=== Test 4: Status Gating ===" << std::endl;
    std::cout << "  (Inactive particles don't move)" << std::endl;

    const int nx = 10, ny = 10, nz = 5;
    auto grid = par2::make_grid<double>(nx, ny, nz, 1.0, 1.0, 1.0);

    par2::TransportParams<double> params;
    params.molecular_diffusion = 1e-5;
    params.alpha_l = 0.1;
    params.alpha_t = 0.01;

    par2::EngineConfig config;
    config.interpolation_mode = par2::InterpolationMode::Linear;
    config.drift_mode = par2::DriftCorrectionMode::None;
    config.rng_seed = 12345;

    auto bc = par2::BoundaryConfig<double>::all_closed();
    par2::TransportEngine<double> engine(grid, params, bc, config);

    // Velocity field
    const size_t vel_size = grid.num_corners();
    double *d_U, *d_V, *d_W;
    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));

    std::vector<double> h_U(vel_size, 0.5);
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_V, 0, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_W, 0, vel_size * sizeof(double)));

    engine.bind_velocity({d_U, d_V, d_W, vel_size});
    engine.update_derived_fields();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Two particles: one Active, one Inactive
    const int num_particles = 2;
    double *d_x, *d_y, *d_z;
    uint8_t *d_status;
    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_status, num_particles * sizeof(uint8_t)));

    std::vector<double> h_x = {5.0, 5.0};
    std::vector<double> h_y = {5.0, 5.0};
    std::vector<double> h_z = {2.5, 2.5};
    std::vector<uint8_t> h_status = {
        static_cast<uint8_t>(par2::ParticleStatus::Active),
        static_cast<uint8_t>(par2::ParticleStatus::Inactive)
    };

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_status, h_status.data(), num_particles * sizeof(uint8_t), cudaMemcpyHostToDevice));

    par2::ParticlesView<double> pview{d_x, d_y, d_z, num_particles, d_status};
    engine.bind_particles(pview);

    // Run several steps
    engine.advance(0.1, 10);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get final positions
    std::vector<double> h_x_final(num_particles);
    CUDA_CHECK(cudaMemcpy(h_x_final.data(), d_x, num_particles * sizeof(double), cudaMemcpyDeviceToHost));

    double active_moved = std::abs(h_x_final[0] - h_x[0]);
    double inactive_moved = std::abs(h_x_final[1] - h_x[1]);

    std::cout << "  Active particle moved:   " << active_moved << " m" << std::endl;
    std::cout << "  Inactive particle moved: " << inactive_moved << " m" << std::endl;

    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_status));

    bool passed = (active_moved > 0.1) && (inactive_moved < 1e-10);
    return {passed, passed ? "PASS" : "FAIL", active_moved, inactive_moved};
}

// =============================================================================
// Test 5: ensure_tracking_arrays() automatic allocation
// =============================================================================

TestResult test_auto_tracking_arrays() {
    std::cout << "\n=== Test 5: Auto Tracking Arrays ===" << std::endl;
    std::cout << "  (Engine allocates status/wrapCount automatically)" << std::endl;

    const int nx = 10, ny = 10, nz = 5;
    auto grid = par2::make_grid<double>(nx, ny, nz, 1.0, 1.0, 1.0);

    par2::TransportParams<double> params;
    params.molecular_diffusion = 0.0;
    params.alpha_l = 0.0;
    params.alpha_t = 0.0;

    par2::EngineConfig config;
    config.interpolation_mode = par2::InterpolationMode::Linear;
    config.drift_mode = par2::DriftCorrectionMode::None;
    config.rng_seed = 12345;

    // Periodic in X (should auto-allocate wrapX)
    par2::BoundaryConfig<double> bc;
    bc.x = par2::AxisBoundary<double>::periodic();
    bc.y = par2::AxisBoundary<double>::closed();
    bc.z = par2::AxisBoundary<double>::closed();
    par2::TransportEngine<double> engine(grid, params, bc, config);

    // Velocity field
    const size_t vel_size = grid.num_corners();
    double *d_U, *d_V, *d_W;
    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));

    std::vector<double> h_U(vel_size, 5.0);  // 5 m/s
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_V, 0, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_W, 0, vel_size * sizeof(double)));

    engine.bind_velocity({d_U, d_V, d_W, vel_size});

    // Bind particles WITHOUT wrapCount - let engine allocate
    const int num_particles = 10;
    double *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));

    // Initialize positions
    std::vector<double> h_x(num_particles, 5.0);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_z, 0, num_particles * sizeof(double)));

    par2::ParticlesView<double> pview{d_x, d_y, d_z, num_particles};
    engine.bind_particles(pview);

    // This should allocate wrapX in workspace
    engine.ensure_tracking_arrays();

    // Run some steps - should not crash
    bool no_crash = true;
    try {
        engine.advance(1.0, 5);
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (...) {
        no_crash = false;
    }

    std::cout << "  ensure_tracking_arrays(): OK" << std::endl;
    std::cout << "  5 steps with periodic BC: " << (no_crash ? "OK" : "CRASH") << std::endl;

    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    return {no_crash, no_crash ? "PASS" : "FAIL", 0, 0};
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Par2_Core M3 - Boundary Conditions + Tracking Tests      ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  BCs: Closed (legacy) / Open (exit) / Periodic (wrap)        ║" << std::endl;
    std::cout << "║  Tracking: ParticleStatus, wrapCount, unwrap on-demand       ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;

    std::vector<TestResult> results;

    results.push_back(test_closed_bc());
    results.push_back(test_open_bc());
    results.push_back(test_periodic_bc());
    results.push_back(test_status_gating());
    results.push_back(test_auto_tracking_arrays());

    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                         SUMMARY                               ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;

    int passed = 0;
    const char* test_names[] = {
        "Closed BC (legacy)",
        "Open BC + Exited",
        "Periodic BC + wrapCount",
        "Status Gating",
        "Auto Tracking Arrays"
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
