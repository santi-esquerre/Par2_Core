/**
 * @file legacy_config.cpp
 * @brief YAML parser implementation for legacy PAR² configuration.
 */

#include "legacy_config.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <ctime>

namespace par2 {
namespace legacy {

namespace fs = std::filesystem;

LegacyConfig load_legacy_yaml(const std::string& yaml_path) {
    LegacyConfig cfg;

    // Get base path (directory containing the YAML file)
    fs::path yaml_fs_path(yaml_path);
    cfg.base_path = yaml_fs_path.parent_path().string();
    if (!cfg.base_path.empty() && cfg.base_path.back() != '/') {
        cfg.base_path += '/';
    }

    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("Failed to load YAML file: ") + yaml_path + " (" + e.what() + ")"
        );
    }

    // =========================================================================
    // Grid
    // =========================================================================
    if (!root["grid"]) {
        throw std::runtime_error("Missing 'grid' section in YAML");
    }

    auto grid_node = root["grid"];
    if (!grid_node["dimension"] || grid_node["dimension"].size() != 3) {
        throw std::runtime_error("Missing or invalid 'grid.dimension' (expected [Nx, Ny, Nz])");
    }
    cfg.grid.dimension = {
        grid_node["dimension"][0].as<int>(),
        grid_node["dimension"][1].as<int>(),
        grid_node["dimension"][2].as<int>()
    };

    if (!grid_node["cell size"] || grid_node["cell size"].size() != 3) {
        throw std::runtime_error("Missing or invalid 'grid.cell size' (expected [dx, dy, dz])");
    }
    cfg.grid.cell_size = {
        grid_node["cell size"][0].as<double>(),
        grid_node["cell size"][1].as<double>(),
        grid_node["cell size"][2].as<double>()
    };

    // =========================================================================
    // Physics
    // =========================================================================
    if (!root["physics"]) {
        throw std::runtime_error("Missing 'physics' section in YAML");
    }

    auto phys_node = root["physics"];
    cfg.physics.porosity = phys_node["porosity"].as<double>();
    cfg.physics.molecular_diffusion = phys_node["molecular diffusion"].as<double>();
    cfg.physics.longitudinal_dispersivity = phys_node["longitudinal dispersivity"].as<double>();
    cfg.physics.transverse_dispersivity = phys_node["transverse dispersivity"].as<double>();

    if (!phys_node["velocity"]) {
        throw std::runtime_error("Missing 'physics.velocity' section");
    }
    cfg.physics.velocity.type = phys_node["velocity"]["type"].as<std::string>();
    cfg.physics.velocity.file = phys_node["velocity"]["file"].as<std::string>();

    // Interpolation (optional, default trilinear)
    if (phys_node["velocity"]["interpolation"]) {
        cfg.physics.velocity.interpolation = phys_node["velocity"]["interpolation"].as<std::string>();
    } else if (root["simulation"]["interpolation"]) {
        // Legacy also supports it under simulation
        cfg.physics.velocity.interpolation = root["simulation"]["interpolation"].as<std::string>();
    } else {
        cfg.physics.velocity.interpolation = "trilinear";
    }

    // =========================================================================
    // Simulation
    // =========================================================================
    if (!root["simulation"]) {
        throw std::runtime_error("Missing 'simulation' section in YAML");
    }

    auto sim_node = root["simulation"];

    // Particles
    if (!sim_node["particles"]) {
        throw std::runtime_error("Missing 'simulation.particles' section");
    }
    cfg.simulation.particles.N = sim_node["particles"]["N"].as<int>();

    auto start_node = sim_node["particles"]["start"];
    cfg.simulation.particles.start.p1 = {
        start_node["p1"][0].as<double>(),
        start_node["p1"][1].as<double>(),
        start_node["p1"][2].as<double>()
    };
    cfg.simulation.particles.start.p2 = {
        start_node["p2"][0].as<double>(),
        start_node["p2"][1].as<double>(),
        start_node["p2"][2].as<double>()
    };

    cfg.simulation.dt = sim_node["dt"].as<double>();
    cfg.simulation.steps = sim_node["steps"].as<int>();

    // Seed (optional)
    if (sim_node["seed"]) {
        cfg.simulation.seed = sim_node["seed"].as<long>();
    } else {
        cfg.simulation.seed = static_cast<long>(std::time(nullptr));
    }

    // =========================================================================
    // Output
    // =========================================================================
    if (root["output"]) {
        auto out_node = root["output"];

        // CSV output
        if (out_node["csv"]) {
            cfg.output.csv.enabled = true;
            cfg.output.csv.file = out_node["csv"]["file"].as<std::string>();
            cfg.output.csv.skip = out_node["csv"]["skip"].as<int>();

            if (out_node["csv"]["items"]) {
                for (const auto& item_node : out_node["csv"]["items"]) {
                    CsvItem item;
                    item.label = item_node["label"].as<std::string>();
                    std::string type_str = item_node["type"].as<std::string>();

                    if (type_str == "after-x") {
                        item.type = CsvItemType::AfterX;
                        item.x = item_node["x"].as<double>();
                    } else if (type_str == "box") {
                        item.type = CsvItemType::Box;
                        item.p1 = {
                            item_node["p1"][0].as<double>(),
                            item_node["p1"][1].as<double>(),
                            item_node["p1"][2].as<double>()
                        };
                        item.p2 = {
                            item_node["p2"][0].as<double>(),
                            item_node["p2"][1].as<double>(),
                            item_node["p2"][2].as<double>()
                        };
                    } else {
                        throw std::runtime_error(
                            "Unknown CSV item type: " + type_str +
                            " (expected 'after-x' or 'box')"
                        );
                    }

                    cfg.output.csv.items.push_back(item);
                }
            }
        }

        // Snapshot output
        if (out_node["snapshot"]) {
            cfg.output.snapshot.enabled = true;
            cfg.output.snapshot.file = out_node["snapshot"]["file"].as<std::string>();

            if (out_node["snapshot"]["skip"]) {
                cfg.output.snapshot.use_skip = true;
                cfg.output.snapshot.skip = out_node["snapshot"]["skip"].as<int>();
            } else if (out_node["snapshot"]["steps"]) {
                cfg.output.snapshot.use_skip = false;
                for (const auto& step_node : out_node["snapshot"]["steps"]) {
                    cfg.output.snapshot.steps.push_back(step_node.as<int>());
                }
            }
        }
    }

    return cfg;
}

} // namespace legacy
} // namespace par2
