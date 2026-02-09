/**
 * @file legacy_config.hpp
 * @brief YAML parser for legacy PAR² configuration files.
 *
 * Reads exactly the legacy YAML schema (grid/physics/simulation/output)
 * for compatibility testing with Par2_Core.
 *
 * @note This is tool-only code, NOT part of the core library.
 */

#ifndef PAR2_LEGACY_CONFIG_HPP
#define PAR2_LEGACY_CONFIG_HPP

#include <string>
#include <vector>
#include <array>
#include <stdexcept>

namespace par2 {
namespace legacy {

// =============================================================================
// Configuration structs (mirrors legacy YAML schema exactly)
// =============================================================================

struct GridConfig {
    std::array<int, 3> dimension;     // [Nx, Ny, Nz]
    std::array<double, 3> cell_size;  // [dx, dy, dz]
};

struct VelocityConfig {
    std::string type;          // "modflow"
    std::string file;          // path to .ftl file
    std::string interpolation; // "trilinear" or "finite difference"
};

struct PhysicsConfig {
    double porosity;
    double molecular_diffusion;
    double longitudinal_dispersivity;
    double transverse_dispersivity;
    VelocityConfig velocity;
};

struct ParticleStartConfig {
    std::array<double, 3> p1;  // [p1x, p1y, p1z]
    std::array<double, 3> p2;  // [p2x, p2y, p2z]
};

struct ParticlesConfig {
    int N;
    ParticleStartConfig start;
};

struct SimulationConfig {
    ParticlesConfig particles;
    double dt;
    int steps;
    long seed;  // 0 = use time-based seed
};

// CSV output item types
enum class CsvItemType {
    AfterX,  // fraction of particles past x plane
    Box      // fraction of particles in box
};

struct CsvItem {
    std::string label;
    CsvItemType type;
    // For after-x
    double x;
    // For box
    std::array<double, 3> p1;
    std::array<double, 3> p2;
};

struct CsvOutputConfig {
    bool enabled = false;
    std::string file;
    int skip;
    std::vector<CsvItem> items;
};

struct SnapshotConfig {
    bool enabled = false;
    std::string file;           // path with '*' placeholder
    bool use_skip = false;
    int skip = 0;
    std::vector<int> steps;     // explicit steps list
};

struct OutputConfig {
    CsvOutputConfig csv;
    SnapshotConfig snapshot;
};

struct LegacyConfig {
    GridConfig grid;
    PhysicsConfig physics;
    SimulationConfig simulation;
    OutputConfig output;

    // Base path for resolving relative paths
    std::string base_path;
};

// =============================================================================
// Parser function
// =============================================================================

/**
 * @brief Load legacy YAML configuration file.
 *
 * @param yaml_path Path to the YAML file
 * @return LegacyConfig Parsed configuration
 * @throws std::runtime_error on parse errors
 *
 * Paths in the config (velocity file, csv output, snapshot) are relative
 * to the YAML file's directory, matching legacy behavior.
 */
LegacyConfig load_legacy_yaml(const std::string& yaml_path);

/**
 * @brief Expand snapshot path by replacing '*' with step number.
 */
inline std::string expand_snapshot_path(const std::string& pattern, int step) {
    std::string result = pattern;
    auto pos = result.find('*');
    if (pos != std::string::npos) {
        result.replace(pos, 1, std::to_string(step));
    }
    return result;
}

} // namespace legacy
} // namespace par2

#endif // PAR2_LEGACY_CONFIG_HPP
