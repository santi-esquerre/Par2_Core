/**
 * @file rng_policy.cuh
 * @brief RNG policies for particle tracking kernels.
 *
 * This header defines the interface for different RNG implementations.
 * The default uses curand's XORWOW generator (curandState_t), but
 * other generators can be added for better performance or quality.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_INTERNAL_RNG_RNG_POLICY_CUH
#define PAR2_INTERNAL_RNG_RNG_POLICY_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace par2 {
namespace rng {

/**
 * @brief Default RNG policy using curand XORWOW.
 *
 * This is the simplest policy and matches legacy PAR² behavior.
 * Each particle has its own curandState_t.
 */
struct DefaultPolicy {
    using state_type = curandState_t;

    /// Initialize state for a particle
    __device__ static void init(state_type& state, unsigned long long seed, int tid) {
        curand_init(seed, tid, 0, &state);
    }

    /// Generate a normal random number (double precision)
    __device__ static double normal(state_type& state) {
        return curand_normal_double(&state);
    }

    /// Generate a uniform random number in (0,1]
    __device__ static double uniform(state_type& state) {
        return curand_uniform_double(&state);
    }
};

/**
 * @brief Philox RNG policy for better parallel performance.
 *
 * Philox is a counter-based PRNG that doesn't require per-thread state
 * to be stored between kernel invocations (if counter is tracked globally).
 *
 * @note TODO: Implement in future stages for improved performance.
 */
struct PhiloxPolicy {
    using state_type = curandStatePhilox4_32_10_t;

    __device__ static void init(state_type& state, unsigned long long seed, int tid) {
        curand_init(seed, tid, 0, &state);
    }

    __device__ static double normal(state_type& state) {
        return curand_normal_double(&state);
    }

    __device__ static double uniform(state_type& state) {
        return curand_uniform_double(&state);
    }
};

} // namespace rng
} // namespace par2

#endif // PAR2_INTERNAL_RNG_RNG_POLICY_CUH
