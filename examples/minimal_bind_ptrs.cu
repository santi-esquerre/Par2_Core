/**
 * @file minimal_bind_ptrs.cu
 * @brief Minimal example demonstrating Par2_Core API.
 *
 * This example shows:
 * 1. Creating a grid and transport engine
 * 2. Allocating velocity and particle arrays on GPU
 * 3. Binding device pointers (zero-copy)
 * 4. Injecting particles in a box
 * 5. Running simulation steps
 * 6. Reading back results for verification
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#include <par2_core/par2_core.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

// Simple CUDA error check macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {
    std::cout << "=== Par2_Core Minimal Example ===" << std::endl;
    std::cout << "Version: " << par2::Version::string << std::endl;

    // =========================================================================
    // 1. Grid setup: 10x10x5 cells, 1.0 m spacing
    // =========================================================================
    const int nx = 10, ny = 10, nz = 5;
    const double dx = 1.0, dy = 1.0, dz = 1.0;

    auto grid = par2::make_grid<double>(nx, ny, nz, dx, dy, dz);

    std::cout << "\nGrid: " << nx << "x" << ny << "x" << nz
              << " cells, " << grid.num_corners() << " corners" << std::endl;

    // =========================================================================
    // 2. Transport parameters
    // =========================================================================
    par2::TransportParams<double> params;
    params.molecular_diffusion = 1e-5;  // m²/s
    params.alpha_l = 0.1;               // m (longitudinal dispersivity)
    params.alpha_t = 0.01;              // m (transverse dispersivity)

    std::cout << "Transport: Dm=" << params.molecular_diffusion
              << ", αL=" << params.alpha_l
              << ", αT=" << params.alpha_t << std::endl;

    // =========================================================================
    // 3. Boundary conditions (closed on all sides)
    // =========================================================================
    auto bc = par2::BoundaryConfig<double>::all_closed();

    // =========================================================================
    // 4. Engine configuration
    // =========================================================================
    par2::EngineConfig config;
    config.rng_seed = 42;
    config.debug_checks = true;

    // =========================================================================
    // 5. Create transport engine
    // =========================================================================
    par2::TransportEngine<double> engine(grid, params, bc, config);
    std::cout << "Engine created successfully" << std::endl;

    // =========================================================================
    // 6. Allocate velocity field on GPU (uniform flow in X direction)
    // =========================================================================
    const size_t vel_size = grid.num_corners();
    double *d_U, *d_V, *d_W;

    CUDA_CHECK(cudaMalloc(&d_U, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, vel_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, vel_size * sizeof(double)));

    // Initialize with uniform velocity: vx=0.1 m/s, vy=vz=0
    std::vector<double> h_U(vel_size, 0.1);  // 0.1 m/s in X
    std::vector<double> h_V(vel_size, 0.0);
    std::vector<double> h_W(vel_size, 0.0);

    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), vel_size * sizeof(double), cudaMemcpyHostToDevice));

    std::cout << "Velocity field allocated: " << vel_size << " values per component" << std::endl;

    // =========================================================================
    // 7. Bind velocity field (zero-copy - engine uses these pointers directly)
    // =========================================================================
    engine.bind_velocity({d_U, d_V, d_W, vel_size});
    std::cout << "Velocity bound (zero-copy)" << std::endl;

    // =========================================================================
    // 8. Allocate particle arrays on GPU
    // =========================================================================
    const int num_particles = 1000;
    double *d_x, *d_y, *d_z;

    CUDA_CHECK(cudaMalloc(&d_x, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, num_particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, num_particles * sizeof(double)));

    std::cout << "Particles allocated: " << num_particles << std::endl;

    // =========================================================================
    // 9. Bind particles (zero-copy)
    // =========================================================================
    engine.bind_particles({d_x, d_y, d_z, num_particles});
    std::cout << "Particles bound (zero-copy)" << std::endl;

    // =========================================================================
    // 10. Inject particles in a box (all on GPU, no host transfer)
    // =========================================================================
    // Inject in center region: [2,8] x [2,8] x [1,4]
    engine.inject_box(2.0, 2.0, 1.0,   // min corner
                      8.0, 8.0, 4.0);  // max corner

    // Synchronize to ensure injection is complete
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Particles injected in box [2,8]x[2,8]x[1,4]" << std::endl;

    // =========================================================================
    // 11. Read initial positions (just first 5 for display)
    // =========================================================================
    std::vector<double> h_x(5), h_y(5), h_z(5);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, 5 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, 5 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, 5 * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "\nInitial positions (first 5 particles):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        printf("  P[%d]: (%.4f, %.4f, %.4f)\n", i, h_x[i], h_y[i], h_z[i]);
    }

    // =========================================================================
    // 12. Run simulation steps
    // =========================================================================
    const double dt = 0.1;  // seconds
    const int num_steps = 10;

    std::cout << "\nRunning " << num_steps << " steps with dt=" << dt << "s..." << std::endl;

    for (int step = 0; step < num_steps; ++step) {
        engine.step(dt);  // Async, no sync inside
    }

    // Synchronize after all steps
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Simulation complete" << std::endl;

    // =========================================================================
    // 13. Read final positions
    // =========================================================================
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, 5 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, 5 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, 5 * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "\nFinal positions (first 5 particles):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        printf("  P[%d]: (%.4f, %.4f, %.4f)\n", i, h_x[i], h_y[i], h_z[i]);
    }

    // =========================================================================
    // 14. Verify movement (particles should have moved in +X direction)
    // =========================================================================
    // Expected advection distance: vx * dt * num_steps = 0.1 * 0.1 * 10 = 0.1 m
    // Plus random dispersion component
    std::cout << "\nExpected advection: ~" << (0.1 * dt * num_steps) << " m in X" << std::endl;

    // =========================================================================
    // 15. Cleanup
    // =========================================================================
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    std::cout << "\n=== Example completed successfully ===" << std::endl;

    return 0;
}
