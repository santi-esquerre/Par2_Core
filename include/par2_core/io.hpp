/**
 * @file io.hpp
 * @brief I/O helpers for particle data (host-side, explicit operations).
 *
 * These utilities enable downloading particle data from GPU to CPU for
 * export, visualization, or external analysis. All operations are explicit
 * and stream-aware - no hidden copies.
 *
 * ## Design Principles
 *
 * 1. **Explicit**: User controls when data is copied.
 * 2. **Stream-aware**: Use cudaMemcpyAsync with user-provided stream.
 * 3. **No hidden allocations**: User provides host buffers (prefer pinned).
 * 4. **Optional**: This module is not needed for pure GPU pipelines.
 *
 * ## CSV Export
 *
 * Two approaches are available:
 *
 * 1. **Low-level** (`download_positions_async` + `write_particles_csv`):
 *    Manual control over buffer management and timing.
 *
 * 2. **High-level** (`CsvSnapshotWriter`):
 *    Manages pinned buffers internally, supports legacy format.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_IO_HPP
#define PAR2_CORE_IO_HPP

#include "views.hpp"
#include "grid.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include <fstream>
#include <vector>
#include <iomanip>

namespace par2 {
namespace io {

// =============================================================================
// Host buffer view
// =============================================================================

/**
 * @brief Host-side buffer view for particle positions.
 *
 * @tparam T Floating point type
 *
 * User must allocate these arrays before calling download functions.
 * For best performance, use cudaMallocHost for pinned memory.
 */
template <typename T>
struct HostParticlesBuffer {
    T* x = nullptr;         ///< X positions [host ptr]
    T* y = nullptr;         ///< Y positions [host ptr]
    T* z = nullptr;         ///< Z positions [host ptr]
    int capacity = 0;       ///< Allocated size

    // Optional: unwrapped positions (if user wants them downloaded too)
    T* x_u = nullptr;       ///< Unwrapped X [host ptr, optional]
    T* y_u = nullptr;       ///< Unwrapped Y [host ptr, optional]
    T* z_u = nullptr;       ///< Unwrapped Z [host ptr, optional]

    // Optional: status
    uint8_t* status = nullptr;  ///< Status flags [host ptr, optional]

    constexpr bool valid() const noexcept {
        return capacity > 0 && x != nullptr && y != nullptr && z != nullptr;
    }

    constexpr bool has_unwrapped() const noexcept {
        return x_u != nullptr && y_u != nullptr && z_u != nullptr;
    }
};

// =============================================================================
// Async download functions
// =============================================================================

/**
 * @brief Download wrapped particle positions asynchronously.
 *
 * @tparam T Floating point type
 *
 * @param host Destination host buffer (must have capacity >= n)
 * @param device Source device view
 * @param n Number of particles to download
 * @param stream CUDA stream for async copy
 *
 * @return cudaSuccess on success, error code otherwise
 *
 * **Important**: For pinned host memory, copy overlaps with computation.
 * After this call, synchronize the stream before reading host data.
 */
template <typename T>
inline cudaError_t download_positions_async(
    HostParticlesBuffer<T>& host,
    const ConstParticlesView<T>& device,
    int n,
    cudaStream_t stream
) {
    if (n <= 0) return cudaSuccess;
    if (!host.valid() || host.capacity < n) return cudaErrorInvalidValue;
    if (!device.valid()) return cudaErrorInvalidValue;

    cudaError_t err;

    err = cudaMemcpyAsync(host.x, device.x, n * sizeof(T),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(host.y, device.y, n * sizeof(T),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(host.z, device.z, n * sizeof(T),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;

    // Status (optional)
    if (host.status != nullptr && device.status != nullptr) {
        err = cudaMemcpyAsync(host.status, device.status, n * sizeof(uint8_t),
                             cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) return err;
    }

    return cudaSuccess;
}

/**
 * @brief Download unwrapped positions asynchronously.
 *
 * @tparam T Floating point type
 *
 * @param host Destination (must have x_u/y_u/z_u set)
 * @param device_unwrap Source device unwrapped view
 * @param n Number of particles
 * @param stream CUDA stream
 *
 * @return cudaSuccess on success
 *
 * Call after engine.compute_unwrapped_positions() to get continuous positions.
 */
template <typename T>
inline cudaError_t download_unwrapped_async(
    HostParticlesBuffer<T>& host,
    const UnwrappedPositionsView<T>& device_unwrap,
    int n,
    cudaStream_t stream
) {
    if (n <= 0) return cudaSuccess;
    if (!host.has_unwrapped() || host.capacity < n) return cudaErrorInvalidValue;

    cudaError_t err;

    err = cudaMemcpyAsync(host.x_u, device_unwrap.x_u, n * sizeof(T),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(host.y_u, device_unwrap.y_u, n * sizeof(T),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(host.z_u, device_unwrap.z_u, n * sizeof(T),
                         cudaMemcpyDeviceToHost, stream);

    return err;
}

// =============================================================================
// CSV Export (host-side, blocking on stream)
// =============================================================================

/**
 * @brief Write particle snapshot to CSV file.
 *
 * @tparam T Floating point type
 *
 * @param filename Output file path
 * @param host Host buffer with downloaded data
 * @param n Number of particles
 * @param time Current simulation time
 * @param include_unwrapped If true, include x_u,y_u,z_u columns
 * @param include_status If true, include status column
 * @param append If true, append to file; otherwise overwrite
 *
 * @return true on success, false on I/O error
 *
 * CSV format: t,id,x,y,z[,xu,yu,zu][,status]
 *
 * **Note**: This is a blocking host-side operation. Call after
 * cudaStreamSynchronize to ensure data is ready.
 */
template <typename T>
inline bool write_particles_csv(
    const std::string& filename,
    const HostParticlesBuffer<T>& host,
    int n,
    T time,
    bool include_unwrapped = true,
    bool include_status = true,
    bool append = true
) {
    std::ios_base::openmode mode = std::ios::out;
    if (append) mode |= std::ios::app;

    std::ofstream file(filename, mode);
    if (!file.is_open()) return false;

    // Write header only if not appending or file is empty
    if (!append) {
        file << "t,id,x,y,z";
        if (include_unwrapped && host.has_unwrapped()) {
            file << ",xu,yu,zu";
        }
        if (include_status && host.status != nullptr) {
            file << ",status";
        }
        file << "\n";
    }

    // Write data rows
    for (int i = 0; i < n; ++i) {
        file << time << "," << i << ","
             << host.x[i] << "," << host.y[i] << "," << host.z[i];

        if (include_unwrapped && host.has_unwrapped()) {
            file << "," << host.x_u[i] << "," << host.y_u[i] << "," << host.z_u[i];
        }

        if (include_status && host.status != nullptr) {
            file << "," << static_cast<int>(host.status[i]);
        }

        file << "\n";
    }

    return file.good();
}

/**
 * @brief RAII wrapper for pinned host buffer allocation.
 *
 * @tparam T Floating point type
 *
 * Allocates pinned memory for efficient async transfers.
 * Automatically frees on destruction.
 */
template <typename T>
class PinnedParticlesBuffer {
public:
    HostParticlesBuffer<T> buffer;

    PinnedParticlesBuffer() = default;

    /**
     * @brief Allocate pinned buffers.
     *
     * @param capacity Number of particles
     * @param with_unwrapped Allocate x_u/y_u/z_u
     * @param with_status Allocate status array
     */
    bool allocate(int capacity, bool with_unwrapped = true, bool with_status = true) {
        free();

        cudaError_t err;

        err = cudaMallocHost(&buffer.x, capacity * sizeof(T));
        if (err != cudaSuccess) return false;

        err = cudaMallocHost(&buffer.y, capacity * sizeof(T));
        if (err != cudaSuccess) { free(); return false; }

        err = cudaMallocHost(&buffer.z, capacity * sizeof(T));
        if (err != cudaSuccess) { free(); return false; }

        if (with_unwrapped) {
            err = cudaMallocHost(&buffer.x_u, capacity * sizeof(T));
            if (err != cudaSuccess) { free(); return false; }

            err = cudaMallocHost(&buffer.y_u, capacity * sizeof(T));
            if (err != cudaSuccess) { free(); return false; }

            err = cudaMallocHost(&buffer.z_u, capacity * sizeof(T));
            if (err != cudaSuccess) { free(); return false; }
        }

        if (with_status) {
            err = cudaMallocHost(&buffer.status, capacity * sizeof(uint8_t));
            if (err != cudaSuccess) { free(); return false; }
        }

        buffer.capacity = capacity;
        return true;
    }

    void free() {
        if (buffer.x) cudaFreeHost(buffer.x);
        if (buffer.y) cudaFreeHost(buffer.y);
        if (buffer.z) cudaFreeHost(buffer.z);
        if (buffer.x_u) cudaFreeHost(buffer.x_u);
        if (buffer.y_u) cudaFreeHost(buffer.y_u);
        if (buffer.z_u) cudaFreeHost(buffer.z_u);
        if (buffer.status) cudaFreeHost(buffer.status);
        buffer = HostParticlesBuffer<T>{};
    }

    ~PinnedParticlesBuffer() { free(); }

    // Non-copyable
    PinnedParticlesBuffer(const PinnedParticlesBuffer&) = delete;
    PinnedParticlesBuffer& operator=(const PinnedParticlesBuffer&) = delete;

    // Movable
    PinnedParticlesBuffer(PinnedParticlesBuffer&& other) noexcept
        : buffer(other.buffer) {
        other.buffer = HostParticlesBuffer<T>{};
    }
    PinnedParticlesBuffer& operator=(PinnedParticlesBuffer&& other) noexcept {
        if (this != &other) {
            free();
            buffer = other.buffer;
            other.buffer = HostParticlesBuffer<T>{};
        }
        return *this;
    }
};

// =============================================================================
// CSV Snapshot Configuration
// =============================================================================

/**
 * @brief Configuration for CSV snapshot export.
 *
 * Controls format, columns, and sampling for CSV particle snapshots.
 *
 * ## Legacy Format
 *
 * When `legacy_format = true` (default), output matches legacy PAR² exactly:
 * ```csv
 * id,x coord,y coord,z coord
 * 0,1.234567890123456,2.345678901234567,3.456789012345678
 * ```
 *
 * ## Extended Format
 *
 * When `legacy_format = false`, uses simpler column names and allows
 * optional columns for status, wrap counts, and unwrapped positions:
 * ```csv
 * id,x,y,z,xu,yu,zu,status,wrapX,wrapY,wrapZ
 * ```
 */
struct CsvSnapshotConfig {
    // =========================================================================
    // Format options
    // =========================================================================

    /**
     * @brief Use legacy PAR² column names.
     *
     * When true: "id,x coord,y coord,z coord"
     * When false: "id,x,y,z"
     *
     * Default: true (legacy compatible)
     */
    bool legacy_format = true;

    /**
     * @brief Include particle ID column (0-based index).
     *
     * Legacy always includes ID. Set false only for special exports.
     */
    bool include_id = true;

    /**
     * @brief Include time column at start of each row.
     *
     * Legacy snapshots do NOT include time. Set true for time-series exports.
     * Format: "t,id,x,y,z,..."
     */
    bool include_time = false;

    /**
     * @brief Include particle status column (0=Active, 1=Exited, 255=Inactive).
     *
     * Not in legacy. Useful for analyzing particle fate with Open BC.
     */
    bool include_status = false;

    /**
     * @brief Include wrap count columns (wrapX, wrapY, wrapZ).
     *
     * Not in legacy. Useful for debugging periodic BC.
     */
    bool include_wrap_counts = false;

    /**
     * @brief Include unwrapped position columns (xu, yu, zu).
     *
     * Not in legacy. For periodic BC, unwrapped = wrapped + wrap*L.
     * Requires unwrapped positions to be computed before snapshot.
     */
    bool include_unwrapped = false;

    // =========================================================================
    // Precision
    // =========================================================================

    /**
     * @brief Decimal digits for floating point values.
     *
     * Legacy uses 15 (full double precision). Reduce for smaller files.
     */
    int precision = 15;

    // =========================================================================
    // Sampling (for large particle counts)
    // =========================================================================

    /**
     * @brief Export every Nth particle.
     *
     * stride=1: all particles
     * stride=10: every 10th particle (10% sample)
     *
     * Useful for reducing output size with millions of particles.
     */
    int stride = 1;

    /**
     * @brief Maximum particles to export.
     *
     * -1 = no limit (export all after stride)
     * Positive = hard cap on output rows
     */
    int max_particles = -1;
};

// =============================================================================
// CSV Snapshot Writer
// =============================================================================

/**
 * @brief Efficient CSV snapshot writer with pinned buffers.
 *
 * @tparam T Floating point type (float or double)
 *
 * Manages pinned host memory for efficient D2H transfers and writes
 * particle snapshots to CSV files. Supports both legacy format and
 * extended format with additional columns.
 *
 * ## Usage Example
 *
 * ```cpp
 * // Create writer (allocates pinned buffers once)
 * par2::io::CsvSnapshotConfig config;
 * config.legacy_format = true;  // Match legacy PAR²
 * par2::io::CsvSnapshotWriter<double> writer(num_particles, config);
 *
 * // In simulation loop...
 * if (step % snapshot_interval == 0) {
 *     std::string filename = "snapshot_" + std::to_string(step) + ".csv";
 *     writer.write_snapshot(particles, filename, time, stream);
 * }
 * ```
 *
 * ## Thread Safety
 *
 * Not thread-safe. Use separate instances for concurrent exports.
 *
 * ## Memory
 *
 * Allocates pinned memory proportional to max_particles:
 * - Base: 3 * sizeof(T) * capacity (x, y, z)
 * - With unwrap: +3 * sizeof(T) * capacity (xu, yu, zu)
 * - With status: +1 * capacity (status)
 * - With wrap: +3 * sizeof(int32_t) * capacity (wrapX, wrapY, wrapZ)
 */
template <typename T>
class CsvSnapshotWriter {
public:
    /**
     * @brief Construct writer with capacity and configuration.
     *
     * @param max_particles Maximum particles to support
     * @param config Snapshot configuration
     *
     * @throws std::bad_alloc if pinned memory allocation fails
     */
    CsvSnapshotWriter(int max_particles, const CsvSnapshotConfig& config = {});

    /**
     * @brief Destructor - frees pinned memory.
     */
    ~CsvSnapshotWriter();

    // Non-copyable
    CsvSnapshotWriter(const CsvSnapshotWriter&) = delete;
    CsvSnapshotWriter& operator=(const CsvSnapshotWriter&) = delete;

    // Movable
    CsvSnapshotWriter(CsvSnapshotWriter&& other) noexcept;
    CsvSnapshotWriter& operator=(CsvSnapshotWriter&& other) noexcept;

    /**
     * @brief Write particle snapshot to CSV file.
     *
     * Performs the following steps:
     * 1. Copy positions from device to pinned host buffer (async)
     * 2. Copy optional data (status, wrap counts) if configured
     * 3. Synchronize the stream
     * 4. Write CSV file (CPU-side, blocking)
     *
     * @param particles Particle positions (device pointers)
     * @param filename Output file path (will be overwritten)
     * @param time Simulation time (only used if include_time=true)
     * @param stream CUDA stream for D2H transfers
     * @param unwrapped Optional unwrapped positions (device)
     *                  Required if include_unwrapped=true
     *
     * @return true on success, false on I/O error
     *
     * @note Synchronizes stream internally before file write.
     *       The D2H copy is overlapped with prior GPU work.
     *
     * @note For very large exports, consider using write_snapshot_chunked().
     */
    bool write_snapshot(
        const ConstParticlesView<T>& particles,
        const std::string& filename,
        T time,
        cudaStream_t stream,
        const UnwrappedPositionsView<T>* unwrapped = nullptr
    );

    /**
     * @brief Write snapshot in chunks (for very large particle counts).
     *
     * Same as write_snapshot but processes particles in chunks to
     * limit peak host memory usage. Uses internal buffer size as chunk.
     *
     * @param particles Full particle view
     * @param filename Output file path
     * @param time Simulation time
     * @param stream CUDA stream
     * @param unwrapped Optional unwrapped positions
     *
     * @return true on success
     *
     * @note Appends chunks to file, so slightly slower than single write.
     */
    bool write_snapshot_chunked(
        const ConstParticlesView<T>& particles,
        const std::string& filename,
        T time,
        cudaStream_t stream,
        const UnwrappedPositionsView<T>* unwrapped = nullptr
    );

    /**
     * @brief Access configuration.
     */
    const CsvSnapshotConfig& config() const noexcept;

    /**
     * @brief Get buffer capacity.
     */
    int capacity() const noexcept;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// =============================================================================
// Legacy-format convenience function
// =============================================================================

/**
 * @brief Write legacy-format CSV snapshot (one-shot, no reused buffers).
 *
 * @tparam T Floating point type
 *
 * Convenience function that matches legacy PAR² exportCSV() exactly:
 * - Columns: id, x coord, y coord, z coord
 * - Precision: 15 decimal digits
 * - No status, no wrap, no time
 *
 * @param particles Particle positions (device)
 * @param filename Output file path
 * @param stream CUDA stream
 *
 * @return true on success
 *
 * @note Allocates temporary pinned memory. For repeated exports,
 *       prefer CsvSnapshotWriter to reuse buffers.
 *
 * @note Synchronizes stream internally.
 */
template <typename T>
inline bool write_legacy_csv_snapshot(
    const ConstParticlesView<T>& particles,
    const std::string& filename,
    cudaStream_t stream = nullptr
) {
    if (particles.n <= 0) return true;  // Empty is success

    // Allocate temporary pinned buffers
    T* h_x = nullptr;
    T* h_y = nullptr;
    T* h_z = nullptr;
    const int n = particles.n;

    cudaError_t err;
    err = cudaMallocHost(&h_x, n * sizeof(T));
    if (err != cudaSuccess) return false;

    err = cudaMallocHost(&h_y, n * sizeof(T));
    if (err != cudaSuccess) { cudaFreeHost(h_x); return false; }

    err = cudaMallocHost(&h_z, n * sizeof(T));
    if (err != cudaSuccess) { cudaFreeHost(h_x); cudaFreeHost(h_y); return false; }

    // Async copy
    cudaMemcpyAsync(h_x, particles.x, n * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_y, particles.y, n * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_z, particles.z, n * sizeof(T), cudaMemcpyDeviceToHost, stream);

    // Wait for copy
    cudaStreamSynchronize(stream);

    // Write file (legacy format)
    std::ofstream out(filename);
    if (!out.is_open()) {
        cudaFreeHost(h_x); cudaFreeHost(h_y); cudaFreeHost(h_z);
        return false;
    }

    out << std::setprecision(15) << std::fixed;
    out << "id,x coord,y coord,z coord\n";

    for (int i = 0; i < n; ++i) {
        out << i << "," << h_x[i] << "," << h_y[i] << "," << h_z[i] << "\n";
    }

    out.close();

    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFreeHost(h_z);

    return out.good();
}

} // namespace io
} // namespace par2

#endif // PAR2_CORE_IO_HPP