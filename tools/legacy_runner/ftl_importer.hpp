/**
 * @file ftl_importer.hpp
 * @brief MODFLOW FTL file importer for legacy velocity fields.
 *
 * Parses MODFLOW-LMT output files containing face-centered volumetric flow
 * (QXX, QYY, QZZ) and converts to velocity using porosity and cell geometry.
 */

#pragma once

#include <array>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace par2 {
namespace legacy {

/**
 * @brief Import velocity field from a MODFLOW FTL file.
 *
 * Reads volumetric flow sections 'QXX', 'QYY', 'QZZ' and converts them
 * to velocity using: v = Q / (area * porosity).
 *
 * Face layout (staggered grid):
 *  - X faces: (nx+1) * (ny+1) * (nz+1) but legacy indexes starting at x=1
 *  - Y faces: similar, starting at y=1
 *  - Z faces: similar, starting at z=1
 *
 * @tparam T Floating point type (float or double)
 * @param file_path      Path to the FTL file
 * @param nx, ny, nz     Grid dimensions (number of cells)
 * @param dx, dy, dz     Cell sizes
 * @param porosity       Effective porosity for velocity conversion
 * @param[out] vx        X-component velocities (face-centered)
 * @param[out] vy        Y-component velocities (face-centered)
 * @param[out] vz        Z-component velocities (face-centered)
 */
template <typename T>
void import_ftl(
    const std::string& file_path,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double porosity,
    std::vector<T>& vx,
    std::vector<T>& vy,
    std::vector<T>& vz
) {
    // Allocate face arrays with legacy staggered dimensions
    std::size_t face_size = static_cast<std::size_t>((nx + 1) * (ny + 1) * (nz + 1));
    vx.assign(face_size, T(0));
    vy.assign(face_size, T(0));
    vz.assign(face_size, T(0));

    std::ifstream infile(file_path);
    if (!infile.is_open()) {
        throw std::runtime_error("Cannot open FTL file: " + file_path);
    }

    std::string line;
    bool is_2d = (nz == 1);

    auto find_section = [&](const std::string& prefix) {
        while (std::getline(infile, line)) {
            if (line.compare(0, prefix.size(), prefix) == 0) {
                return true;
            }
        }
        return false;
    };

    // =========================================================================
    // QXX section (X-component of volumetric flow)
    // =========================================================================
    if (!find_section(" 'QXX")) {
        throw std::runtime_error("FTL file missing 'QXX' section");
    }

    double area_x = dy * dz;
    double val;
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (!(infile >> val)) {
                    throw std::runtime_error("Unexpected EOF in QXX section");
                }
                // Legacy indexing: id = z*(nx+1)*(ny+1) + y*(nx+1) + (x+1)
                int id = z * (nx + 1) * (ny + 1) + y * (nx + 1) + (x + 1);
                vx[id] = static_cast<T>(val / (area_x * porosity));
            }
        }
    }

    // Reset stream position for next section search
    infile.clear();

    // =========================================================================
    // QYY section (Y-component of volumetric flow)
    // =========================================================================
    if (!find_section(" 'QYY")) {
        throw std::runtime_error("FTL file missing 'QYY' section");
    }

    double area_y = dx * dz;
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (!(infile >> val)) {
                    throw std::runtime_error("Unexpected EOF in QYY section");
                }
                // Legacy indexing: id = z*(nx+1)*(ny+1) + (y+1)*(nx+1) + x
                int id = z * (nx + 1) * (ny + 1) + (y + 1) * (nx + 1) + x;
                vy[id] = static_cast<T>(val / (area_y * porosity));
            }
        }
    }

    // =========================================================================
    // QZZ section (Z-component of volumetric flow, skip for 2D)
    // =========================================================================
    if (!is_2d) {
        infile.clear();
        if (!find_section(" 'QZZ")) {
            throw std::runtime_error("FTL file missing 'QZZ' section");
        }

        double area_z = dx * dy;
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    if (!(infile >> val)) {
                        throw std::runtime_error("Unexpected EOF in QZZ section");
                    }
                    // Legacy indexing: id = (z+1)*(nx+1)*(ny+1) + y*(nx+1) + x
                    int id = (z + 1) * (nx + 1) * (ny + 1) + y * (nx + 1) + x;
                    vz[id] = static_cast<T>(val / (area_z * porosity));
                }
            }
        }
    }

    infile.close();
}

/**
 * @brief Helper to compute flat index for face-centered velocity.
 *
 * Used for accessing the imported velocity arrays.
 */
inline int face_index(int x, int y, int z, int nx, int ny, int nz) {
    (void)nz; // Unused but kept for symmetry
    return z * (nx + 1) * (ny + 1) + y * (nx + 1) + x;
}

} // namespace legacy
} // namespace par2
