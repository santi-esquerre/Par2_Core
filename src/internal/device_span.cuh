/**
 * @file device_span.cuh
 * @brief Device memory span utility.
 *
 * Internal helper similar to std::span but for device memory.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_INTERNAL_DEVICE_SPAN_CUH
#define PAR2_INTERNAL_DEVICE_SPAN_CUH

#include <cstddef>

namespace par2 {
namespace detail {

/**
 * @brief Non-owning view of contiguous device memory.
 *
 * This is an internal utility - public API uses DeviceSpan from views.hpp.
 */
template <typename T>
struct device_span {
    T* data;
    size_t size;

    __host__ __device__ T& operator[](size_t i) { return data[i]; }
    __host__ __device__ const T& operator[](size_t i) const { return data[i]; }
    __host__ __device__ bool empty() const { return size == 0; }
};

} // namespace detail
} // namespace par2

#endif // PAR2_INTERNAL_DEVICE_SPAN_CUH
