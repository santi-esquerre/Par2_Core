/**
 * @file main.cu
 * @brief par2core_legacy_runner - CLI tool for legacy PAR² compatibility.
 *
 * Reads legacy PAR² YAML configuration and MODFLOW FTL files,
 * runs Par2_Core, and produces compatible output for regression testing.
 */

#include "legacy_config.hpp"
#include "ftl_importer.hpp"

#include <par2_core/transport_engine.hpp>
#include <par2_core/grid.hpp>
#include <par2_core/types.hpp>
#include <par2_core/boundary.hpp>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>

using namespace par2;
using namespace par2::legacy;

// Use double precision throughout (matches legacy PAR²)
using RealType = double;

// =============================================================================
// Statistics Kernels (implemented here since stats.hpp is a stub)
// =============================================================================

/**
 * @brief Count particles past a plane (x > threshold).
 */
template <typename T>
struct PastPlaneX {
    T threshold;
    __host__ __device__
    int operator()(T x) const {
        return (x > threshold) ? 1 : 0;
    }
};

/**
 * @brief Check if a point is inside a box.
 */
template <typename T>
struct InBox {
    T x0, y0, z0, x1, y1, z1;
    __host__ __device__
    int operator()(const thrust::tuple<T, T, T>& p) const {
        T px = thrust::get<0>(p);
        T py = thrust::get<1>(p);
        T pz = thrust::get<2>(p);
        return (px >= x0 && px <= x1 &&
                py >= y0 && py <= y1 &&
                pz >= z0 && pz <= z1) ? 1 : 0;
    }
};

// =============================================================================
// Output Helpers
// =============================================================================

void write_csv_header(std::ofstream& out, const LegacyConfig& cfg) {
    out << "step, time";
    for (const auto& item : cfg.output.csv.items) {
        out << ", " << item.label;
    }
    out << std::endl;
}

template <typename T>
double compute_csv_item(
    const CsvItem& item,
    const thrust::device_vector<T>& x,
    const thrust::device_vector<T>& y,
    const thrust::device_vector<T>& z,
    int N
) {
    if (item.type == CsvItemType::AfterX) {
        // Count particles with x > item.x
        int count = thrust::transform_reduce(
            x.begin(), x.end(),
            PastPlaneX<T>{static_cast<T>(item.x)},
            0,
            thrust::plus<int>()
        );
        return static_cast<double>(count) / N;
    } else {
        // Box: count particles inside
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(
            x.begin(), y.begin(), z.begin()
        ));
        auto end = thrust::make_zip_iterator(thrust::make_tuple(
            x.end(), y.end(), z.end()
        ));

        InBox<T> in_box{
            static_cast<T>(item.p1[0]), static_cast<T>(item.p1[1]), static_cast<T>(item.p1[2]),
            static_cast<T>(item.p2[0]), static_cast<T>(item.p2[1]), static_cast<T>(item.p2[2])
        };

        int count = thrust::transform_reduce(
            begin, end,
            in_box,
            0,
            thrust::plus<int>()
        );
        return static_cast<double>(count) / N;
    }
}

template <typename T>
void write_snapshot(
    const std::string& path,
    const thrust::device_vector<T>& x,
    const thrust::device_vector<T>& y,
    const thrust::device_vector<T>& z,
    int N
) {
    // Copy to host
    thrust::host_vector<T> hx = x;
    thrust::host_vector<T> hy = y;
    thrust::host_vector<T> hz = z;

    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Warning: Cannot write snapshot to " << path << "\n";
        return;
    }

    // Match original format: CSV with header "id,x coord,y coord,z coord"
    out << "id,x coord,y coord,z coord" << std::endl;
    out << std::setprecision(6);  // Match original precision
    for (int i = 0; i < N; ++i) {
        out << i << "," << hx[i] << "," << hy[i] << "," << hz[i] << std::endl;
    }
    out.close();
}

// =============================================================================
// Main Runner
// =============================================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml> [--legacy-strict]\n";
        std::cerr << "  --legacy-strict  Disable NaN-prevention guards for exact legacy matching\n";
        return 1;
    }

    std::string yaml_path = argv[1];

    // Check for --legacy-strict flag
    bool legacy_strict = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--legacy-strict") {
            legacy_strict = true;
        }
    }

    // =========================================================================
    // 1. Load configuration
    // =========================================================================
    std::cout << "Loading configuration: " << yaml_path << "\n";
    LegacyConfig cfg;
    try {
        cfg = load_legacy_yaml(yaml_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << "\n";
        return 1;
    }

    std::cout << "  Grid: " << cfg.grid.dimension[0] << "x"
              << cfg.grid.dimension[1] << "x"
              << cfg.grid.dimension[2] << "\n";
    std::cout << "  Particles: " << cfg.simulation.particles.N << "\n";
    std::cout << "  Steps: " << cfg.simulation.steps << "\n";
    std::cout << "  dt: " << cfg.simulation.dt << "\n";

    // =========================================================================
    // 2. Import velocity field from FTL
    // =========================================================================
    std::string ftl_path = cfg.base_path + cfg.physics.velocity.file;
    std::cout << "Loading velocity field: " << ftl_path << "\n";

    int nx = cfg.grid.dimension[0];
    int ny = cfg.grid.dimension[1];
    int nz = cfg.grid.dimension[2];
    double dx = cfg.grid.cell_size[0];
    double dy = cfg.grid.cell_size[1];
    double dz = cfg.grid.cell_size[2];

    std::vector<RealType> h_vx, h_vy, h_vz;
    try {
        import_ftl(ftl_path, nx, ny, nz, dx, dy, dz, cfg.physics.porosity,
                   h_vx, h_vy, h_vz);
    } catch (const std::exception& e) {
        std::cerr << "Error loading FTL: " << e.what() << "\n";
        return 1;
    }

    // Upload to GPU
    thrust::device_vector<RealType> d_vx(h_vx.begin(), h_vx.end());
    thrust::device_vector<RealType> d_vy(h_vy.begin(), h_vy.end());
    thrust::device_vector<RealType> d_vz(h_vz.begin(), h_vz.end());

    std::cout << "  Velocity field uploaded to GPU\n";

    // =========================================================================
    // 3. Create engine
    // =========================================================================
    GridDesc<RealType> grid = make_grid<RealType>(nx, ny, nz, dx, dy, dz);

    TransportParams<RealType> params;
    params.molecular_diffusion = static_cast<RealType>(cfg.physics.molecular_diffusion);
    params.alpha_l = static_cast<RealType>(cfg.physics.longitudinal_dispersivity);
    params.alpha_t = static_cast<RealType>(cfg.physics.transverse_dispersivity);

    // Legacy uses closed boundaries
    BoundaryConfig<RealType> bc = BoundaryConfig<RealType>::all_closed();

    EngineConfig engine_cfg;
    engine_cfg.rng_seed = static_cast<uint64_t>(cfg.simulation.seed);

    // NaN prevention: disable if running in legacy-strict mode
    engine_cfg.nan_prevention = !legacy_strict;
    if (legacy_strict) {
        std::cout << "  Mode: legacy-strict (NaN-prevention disabled)\n";
    }

    // LEGACY SEMANTICS:
    // - Velocity interpolation is ALWAYS face-centered (Linear mode)
    // - The YAML "interpolation" field controls DRIFT CORRECTION mode:
    //   "trilinear" → TrilinearOnFly (div(D) computed via trilinear derivatives)
    //   "finite difference" → Precomputed (div(D) from cell-centered finite differences)
    engine_cfg.interpolation_mode = InterpolationMode::Linear;

    if (cfg.physics.velocity.interpolation == "finite difference") {
        engine_cfg.drift_mode = DriftCorrectionMode::Precomputed;
        std::cout << "  Drift mode: Precomputed (finite difference)\n";
    } else {
        engine_cfg.drift_mode = DriftCorrectionMode::TrilinearOnFly;
        std::cout << "  Drift mode: TrilinearOnFly (trilinear)\n";
    }

    TransportEngine<RealType> engine(grid, params, bc, engine_cfg);

    // =========================================================================
    // 4. Allocate and bind particles
    // =========================================================================
    int N = cfg.simulation.particles.N;
    thrust::device_vector<RealType> d_x(N);
    thrust::device_vector<RealType> d_y(N);
    thrust::device_vector<RealType> d_z(N);

    // Bind velocity
    VelocityView<RealType> vel_view;
    vel_view.U = thrust::raw_pointer_cast(d_vx.data());
    vel_view.V = thrust::raw_pointer_cast(d_vy.data());
    vel_view.W = thrust::raw_pointer_cast(d_vz.data());
    vel_view.size = d_vx.size();
    engine.bind_velocity(vel_view);

    // Bind particles
    ParticlesView<RealType> particles_view;
    particles_view.x = thrust::raw_pointer_cast(d_x.data());
    particles_view.y = thrust::raw_pointer_cast(d_y.data());
    particles_view.z = thrust::raw_pointer_cast(d_z.data());
    particles_view.n = N;
    engine.bind_particles(particles_view);

    // Update derived fields (corner velocity)
    engine.update_derived_fields();

    // Inject particles in box
    auto& start = cfg.simulation.particles.start;
    engine.inject_box(
        static_cast<RealType>(start.p1[0]),
        static_cast<RealType>(start.p1[1]),
        static_cast<RealType>(start.p1[2]),
        static_cast<RealType>(start.p2[0]),
        static_cast<RealType>(start.p2[1]),
        static_cast<RealType>(start.p2[2])
    );

    // Prepare engine (RNG init)
    engine.prepare();

    std::cout << "  Engine prepared\n";

    // =========================================================================
    // 5. Setup output
    // =========================================================================
    std::ofstream csv_out;
    if (cfg.output.csv.enabled) {
        std::string csv_path = cfg.base_path + cfg.output.csv.file;
        csv_out.open(csv_path);
        if (!csv_out.is_open()) {
            std::cerr << "Warning: Cannot open CSV output: " << csv_path << "\n";
        } else {
            // Match original precision: 15 decimals, fixed format
            csv_out << std::setprecision(15) << std::fixed;
            write_csv_header(csv_out, cfg);
        }
    }

    // =========================================================================
    // 6. Simulation loop (LEGACY ORDER: output BEFORE step)
    // =========================================================================
    double dt = cfg.simulation.dt;
    int steps = cfg.simulation.steps;
    int csv_skip = cfg.output.csv.skip;

    std::cout << "Running simulation...\n";

    for (int step = 0; step <= steps; ++step) {
        double time = step * dt;

        // ---------------------------------------------------------------------
        // CSV output (BEFORE stepping, if step % skip == 0)
        // ---------------------------------------------------------------------
        if (cfg.output.csv.enabled && csv_out.is_open()) {
            if (step % csv_skip == 0) {
                engine.synchronize();  // Ensure previous step completed

                csv_out << step << ", " << time;
                for (const auto& item : cfg.output.csv.items) {
                    double frac = compute_csv_item(item, d_x, d_y, d_z, N);
                    csv_out << ", " << frac;
                }
                csv_out << std::endl;
            }
        }

        // ---------------------------------------------------------------------
        // Snapshot output (BEFORE stepping)
        // ---------------------------------------------------------------------
        if (cfg.output.snapshot.enabled) {
            bool do_snapshot = false;

            if (cfg.output.snapshot.use_skip) {
                do_snapshot = (step % cfg.output.snapshot.skip == 0);
            } else {
                // Check if step is in the list
                auto& steps_list = cfg.output.snapshot.steps;
                do_snapshot = std::find(steps_list.begin(), steps_list.end(), step) != steps_list.end();
            }

            if (do_snapshot) {
                engine.synchronize();
                std::string snap_path = cfg.base_path + expand_snapshot_path(cfg.output.snapshot.file, step);
                write_snapshot(snap_path, d_x, d_y, d_z, N);
            }
        }

        // ---------------------------------------------------------------------
        // Step (except after last iteration)
        // ---------------------------------------------------------------------
        if (step < steps) {
            engine.step(static_cast<RealType>(dt));
        }

        // Progress indicator
        if (step % 100 == 0 || step == steps) {
            std::cout << "\r  Step " << step << "/" << steps << std::flush;
        }
    }

    engine.synchronize();
    std::cout << "\n  Simulation complete\n";

    if (csv_out.is_open()) {
        csv_out.close();
    }

    return 0;
}
