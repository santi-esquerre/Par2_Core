/**
 * @file io.cu
 * @brief I/O implementation for Par2_Core.
 *
 * Implements CsvSnapshotWriter for efficient CSV particle snapshot export.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#include <par2_core/io.hpp>
#include <par2_core/types.hpp>

#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace par2 {
namespace io {

// =============================================================================
// CsvSnapshotWriter Implementation
// =============================================================================

template <typename T>
struct CsvSnapshotWriter<T>::Impl {
    CsvSnapshotConfig config;
    int max_particles;

    // Pinned host buffers
    T* h_x = nullptr;
    T* h_y = nullptr;
    T* h_z = nullptr;
    T* h_xu = nullptr;  // unwrapped
    T* h_yu = nullptr;
    T* h_zu = nullptr;
    uint8_t* h_status = nullptr;
    int32_t* h_wrapX = nullptr;
    int32_t* h_wrapY = nullptr;
    int32_t* h_wrapZ = nullptr;

    Impl(int capacity, const CsvSnapshotConfig& cfg)
        : config(cfg), max_particles(capacity)
    {
        // Allocate base buffers (always needed)
        cudaError_t err;

        err = cudaMallocHost(&h_x, capacity * sizeof(T));
        if (err != cudaSuccess) throw std::bad_alloc();

        err = cudaMallocHost(&h_y, capacity * sizeof(T));
        if (err != cudaSuccess) { cleanup(); throw std::bad_alloc(); }

        err = cudaMallocHost(&h_z, capacity * sizeof(T));
        if (err != cudaSuccess) { cleanup(); throw std::bad_alloc(); }

        // Optional: unwrapped positions
        if (config.include_unwrapped) {
            err = cudaMallocHost(&h_xu, capacity * sizeof(T));
            if (err != cudaSuccess) { cleanup(); throw std::bad_alloc(); }

            err = cudaMallocHost(&h_yu, capacity * sizeof(T));
            if (err != cudaSuccess) { cleanup(); throw std::bad_alloc(); }

            err = cudaMallocHost(&h_zu, capacity * sizeof(T));
            if (err != cudaSuccess) { cleanup(); throw std::bad_alloc(); }
        }

        // Optional: status
        if (config.include_status) {
            err = cudaMallocHost(&h_status, capacity * sizeof(uint8_t));
            if (err != cudaSuccess) { cleanup(); throw std::bad_alloc(); }
        }

        // Optional: wrap counts
        if (config.include_wrap_counts) {
            err = cudaMallocHost(&h_wrapX, capacity * sizeof(int32_t));
            if (err != cudaSuccess) { cleanup(); throw std::bad_alloc(); }

            err = cudaMallocHost(&h_wrapY, capacity * sizeof(int32_t));
            if (err != cudaSuccess) { cleanup(); throw std::bad_alloc(); }

            err = cudaMallocHost(&h_wrapZ, capacity * sizeof(int32_t));
            if (err != cudaSuccess) { cleanup(); throw std::bad_alloc(); }
        }
    }

    void cleanup() {
        if (h_x) cudaFreeHost(h_x);
        if (h_y) cudaFreeHost(h_y);
        if (h_z) cudaFreeHost(h_z);
        if (h_xu) cudaFreeHost(h_xu);
        if (h_yu) cudaFreeHost(h_yu);
        if (h_zu) cudaFreeHost(h_zu);
        if (h_status) cudaFreeHost(h_status);
        if (h_wrapX) cudaFreeHost(h_wrapX);
        if (h_wrapY) cudaFreeHost(h_wrapY);
        if (h_wrapZ) cudaFreeHost(h_wrapZ);

        h_x = h_y = h_z = nullptr;
        h_xu = h_yu = h_zu = nullptr;
        h_status = nullptr;
        h_wrapX = h_wrapY = h_wrapZ = nullptr;
    }

    ~Impl() {
        cleanup();
    }

    bool download_async(
        const ConstParticlesView<T>& particles,
        const UnwrappedPositionsView<T>* unwrapped,
        int n,
        cudaStream_t stream
    ) {
        if (n <= 0) return true;
        if (n > max_particles) return false;

        // Copy positions
        cudaMemcpyAsync(h_x, particles.x, n * sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_y, particles.y, n * sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_z, particles.z, n * sizeof(T), cudaMemcpyDeviceToHost, stream);

        // Copy unwrapped (if configured and available)
        if (config.include_unwrapped && unwrapped != nullptr) {
            cudaMemcpyAsync(h_xu, unwrapped->x_u, n * sizeof(T), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_yu, unwrapped->y_u, n * sizeof(T), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_zu, unwrapped->z_u, n * sizeof(T), cudaMemcpyDeviceToHost, stream);
        }

        // Copy status (if configured and available)
        if (config.include_status && particles.status != nullptr) {
            cudaMemcpyAsync(h_status, particles.status, n * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
        }

        // Copy wrap counts (if configured and available)
        if (config.include_wrap_counts) {
            if (particles.wrapX != nullptr) {
                cudaMemcpyAsync(h_wrapX, particles.wrapX, n * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            }
            if (particles.wrapY != nullptr) {
                cudaMemcpyAsync(h_wrapY, particles.wrapY, n * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            }
            if (particles.wrapZ != nullptr) {
                cudaMemcpyAsync(h_wrapZ, particles.wrapZ, n * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            }
        }

        return true;
    }

    void write_header(std::ofstream& out, bool has_unwrap, bool has_status, bool has_wrap) {
        // Column names
        if (config.include_time) {
            out << "t,";
        }

        if (config.include_id) {
            out << "id,";
        }

        if (config.legacy_format) {
            out << "x coord,y coord,z coord";
        } else {
            out << "x,y,z";
        }

        if (config.include_unwrapped && has_unwrap) {
            out << ",xu,yu,zu";
        }

        if (config.include_status && has_status) {
            out << ",status";
        }

        if (config.include_wrap_counts && has_wrap) {
            out << ",wrapX,wrapY,wrapZ";
        }

        out << "\n";
    }

    void write_row(std::ofstream& out, int i, T time, bool has_unwrap, bool has_status, bool has_wrap) {
        if (config.include_time) {
            out << time << ",";
        }

        if (config.include_id) {
            out << i << ",";
        }

        out << h_x[i] << "," << h_y[i] << "," << h_z[i];

        if (config.include_unwrapped && has_unwrap && h_xu != nullptr) {
            out << "," << h_xu[i] << "," << h_yu[i] << "," << h_zu[i];
        }

        if (config.include_status && has_status && h_status != nullptr) {
            out << "," << static_cast<int>(h_status[i]);
        }

        if (config.include_wrap_counts && has_wrap) {
            out << "," << (h_wrapX ? h_wrapX[i] : 0)
                << "," << (h_wrapY ? h_wrapY[i] : 0)
                << "," << (h_wrapZ ? h_wrapZ[i] : 0);
        }

        out << "\n";
    }

    bool write_file(
        const std::string& filename,
        int n,
        T time,
        bool has_unwrap,
        bool has_status,
        bool has_wrap
    ) {
        std::ofstream out(filename);
        if (!out.is_open()) return false;

        out << std::setprecision(config.precision) << std::fixed;

        write_header(out, has_unwrap, has_status, has_wrap);

        // Apply stride and max
        int stride = config.stride > 0 ? config.stride : 1;
        int max_out = (config.max_particles > 0) ? config.max_particles : n;
        int count = 0;

        for (int i = 0; i < n && count < max_out; i += stride, ++count) {
            write_row(out, i, time, has_unwrap, has_status, has_wrap);
        }

        out.close();
        return out.good();
    }

    bool write_file_chunked(
        const std::string& filename,
        const ConstParticlesView<T>& particles,
        const UnwrappedPositionsView<T>* unwrapped,
        T time,
        cudaStream_t stream
    ) {
        const int n = particles.n;
        if (n <= 0) return true;

        std::ofstream out(filename);
        if (!out.is_open()) return false;

        out << std::setprecision(config.precision) << std::fixed;

        bool has_unwrap = (unwrapped != nullptr);
        bool has_status = (particles.status != nullptr);
        bool has_wrap = (particles.wrapX != nullptr || particles.wrapY != nullptr || particles.wrapZ != nullptr);

        write_header(out, has_unwrap, has_status, has_wrap);

        // Process in chunks
        int stride = config.stride > 0 ? config.stride : 1;
        int max_out = (config.max_particles > 0) ? config.max_particles : n;
        int written = 0;

        for (int chunk_start = 0; chunk_start < n && written < max_out; ) {
            // Calculate chunk size
            int chunk_size = std::min(max_particles, n - chunk_start);

            // Create sub-views for this chunk
            ConstParticlesView<T> chunk_particles;
            chunk_particles.x = particles.x + chunk_start;
            chunk_particles.y = particles.y + chunk_start;
            chunk_particles.z = particles.z + chunk_start;
            chunk_particles.n = chunk_size;
            chunk_particles.status = particles.status ? particles.status + chunk_start : nullptr;
            chunk_particles.wrapX = particles.wrapX ? particles.wrapX + chunk_start : nullptr;
            chunk_particles.wrapY = particles.wrapY ? particles.wrapY + chunk_start : nullptr;
            chunk_particles.wrapZ = particles.wrapZ ? particles.wrapZ + chunk_start : nullptr;

            UnwrappedPositionsView<T> chunk_unwrap;
            UnwrappedPositionsView<T>* chunk_unwrap_ptr = nullptr;
            if (unwrapped) {
                chunk_unwrap.x_u = unwrapped->x_u + chunk_start;
                chunk_unwrap.y_u = unwrapped->y_u + chunk_start;
                chunk_unwrap.z_u = unwrapped->z_u + chunk_start;
                chunk_unwrap.capacity = chunk_size;
                chunk_unwrap_ptr = &chunk_unwrap;
            }

            // Download chunk
            if (!download_async(chunk_particles, chunk_unwrap_ptr, chunk_size, stream)) {
                return false;
            }
            cudaStreamSynchronize(stream);

            // Write chunk
            for (int i = 0; i < chunk_size && written < max_out; i += stride, ++written) {
                int global_idx = chunk_start + i;
                
                if (config.include_time) {
                    out << time << ",";
                }

                if (config.include_id) {
                    out << global_idx << ",";
                }

                out << h_x[i] << "," << h_y[i] << "," << h_z[i];

                if (config.include_unwrapped && has_unwrap && h_xu != nullptr) {
                    out << "," << h_xu[i] << "," << h_yu[i] << "," << h_zu[i];
                }

                if (config.include_status && has_status && h_status != nullptr) {
                    out << "," << static_cast<int>(h_status[i]);
                }

                if (config.include_wrap_counts && has_wrap) {
                    out << "," << (h_wrapX ? h_wrapX[i] : 0)
                        << "," << (h_wrapY ? h_wrapY[i] : 0)
                        << "," << (h_wrapZ ? h_wrapZ[i] : 0);
                }

                out << "\n";
            }

            chunk_start += chunk_size;
        }

        out.close();
        return out.good();
    }
};

// Constructor
template <typename T>
CsvSnapshotWriter<T>::CsvSnapshotWriter(int max_particles, const CsvSnapshotConfig& config) {
    impl_ = new Impl(max_particles, config);
}

// Destructor
template <typename T>
CsvSnapshotWriter<T>::~CsvSnapshotWriter() {
    delete impl_;
}

// Move constructor
template <typename T>
CsvSnapshotWriter<T>::CsvSnapshotWriter(CsvSnapshotWriter&& other) noexcept
    : impl_(other.impl_)
{
    other.impl_ = nullptr;
}

// Move assignment
template <typename T>
CsvSnapshotWriter<T>& CsvSnapshotWriter<T>::operator=(CsvSnapshotWriter&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

// Write snapshot
template <typename T>
bool CsvSnapshotWriter<T>::write_snapshot(
    const ConstParticlesView<T>& particles,
    const std::string& filename,
    T time,
    cudaStream_t stream,
    const UnwrappedPositionsView<T>* unwrapped
) {
    if (!impl_) return false;

    const int n = particles.n;
    if (n <= 0) {
        // Write empty file with header only
        std::ofstream out(filename);
        if (!out.is_open()) return false;
        impl_->write_header(out, false, false, false);
        return true;
    }

    if (n > impl_->max_particles) {
        // Use chunked write for large data
        return write_snapshot_chunked(particles, filename, time, stream, unwrapped);
    }

    // Download to pinned buffers
    if (!impl_->download_async(particles, unwrapped, n, stream)) {
        return false;
    }

    // Wait for transfers to complete
    cudaStreamSynchronize(stream);

    // Write file
    bool has_unwrap = (unwrapped != nullptr);
    bool has_status = (particles.status != nullptr);
    bool has_wrap = (particles.wrapX != nullptr || particles.wrapY != nullptr || particles.wrapZ != nullptr);

    return impl_->write_file(filename, n, time, has_unwrap, has_status, has_wrap);
}

// Write snapshot chunked
template <typename T>
bool CsvSnapshotWriter<T>::write_snapshot_chunked(
    const ConstParticlesView<T>& particles,
    const std::string& filename,
    T time,
    cudaStream_t stream,
    const UnwrappedPositionsView<T>* unwrapped
) {
    if (!impl_) return false;
    return impl_->write_file_chunked(filename, particles, unwrapped, time, stream);
}

// Access config
template <typename T>
const CsvSnapshotConfig& CsvSnapshotWriter<T>::config() const noexcept {
    static CsvSnapshotConfig empty;
    return impl_ ? impl_->config : empty;
}

// Get capacity
template <typename T>
int CsvSnapshotWriter<T>::capacity() const noexcept {
    return impl_ ? impl_->max_particles : 0;
}

// Explicit instantiations
template class CsvSnapshotWriter<float>;
template class CsvSnapshotWriter<double>;

} // namespace io
} // namespace par2
