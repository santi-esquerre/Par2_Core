/**
 * @file debug_policy.hpp
 * @brief Compile-time debug/release policies for Par2_Core.
 *
 * This header defines compile-time macros that control debug features:
 * - PAR2_DEBUG_LEVEL: Overall debug level (0=release, 1=basic, 2=full)
 * - PAR2_ENABLE_NAN_GUARDS: Enable NaN/Inf checks in kernels
 * - PAR2_ENABLE_BOUNDS_CHECKS: Enable array bounds validation
 *
 * ## Design Philosophy
 *
 * **Zero overhead in release builds.** All debug checks use `#if` preprocessor
 * conditionals that eliminate code entirely in release builds. No runtime
 * branches in kernel hot paths.
 *
 * ## Usage
 *
 * Set these macros via CMake or compiler flags:
 * ```cmake
 * # Debug build with full checks
 * target_compile_definitions(myapp PRIVATE PAR2_DEBUG_LEVEL=2)
 *
 * # Release build (default, no checks)
 * # Nothing needed - defaults to PAR2_DEBUG_LEVEL=0
 * ```
 *
 * Or in code before including any Par2_Core headers:
 * ```cpp
 * #define PAR2_DEBUG_LEVEL 1
 * #include <par2_core/par2_core.hpp>
 * ```
 *
 * ## Debug Levels
 *
 * | Level | Description                              | Overhead        |
 * |-------|------------------------------------------|-----------------|
 * | 0     | Release: No checks, maximum performance  | None            |
 * | 1     | Basic: API validation, CUDA error checks | Minimal (async) |
 * | 2     | Full: + NaN guards, bounds checks        | Per-particle    |
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_DEBUG_POLICY_HPP
#define PAR2_CORE_DEBUG_POLICY_HPP

// =============================================================================
// Debug Level Configuration
// =============================================================================

/**
 * @brief Master debug level for Par2_Core.
 *
 * - 0: Release (default) - no debug overhead
 * - 1: Basic debug - CUDA error checks after launches (no sync)
 * - 2: Full debug - adds NaN guards and bounds checks in kernels
 */
#ifndef PAR2_DEBUG_LEVEL
#  if defined(NDEBUG)
#    define PAR2_DEBUG_LEVEL 0
#  elif defined(DEBUG) || defined(_DEBUG)
#    define PAR2_DEBUG_LEVEL 1
#  else
#    define PAR2_DEBUG_LEVEL 0
#  endif
#endif

// =============================================================================
// Derived Feature Macros
// =============================================================================

/**
 * @brief Enable CUDA error checks after kernel launches.
 *
 * When enabled, uses cudaPeekAtLastError() after kernel launches.
 * Does NOT synchronize - only catches launch config errors.
 */
#ifndef PAR2_ENABLE_CUDA_CHECKS
#  if PAR2_DEBUG_LEVEL >= 1
#    define PAR2_ENABLE_CUDA_CHECKS 1
#  else
#    define PAR2_ENABLE_CUDA_CHECKS 0
#  endif
#endif

/**
 * @brief Enable NaN/Inf guards in dispersion computations.
 *
 * When enabled, kernels check for division by zero and clamp values
 * to prevent NaN propagation. Adds per-particle overhead.
 */
#ifndef PAR2_ENABLE_NAN_GUARDS
#  if PAR2_DEBUG_LEVEL >= 2
#    define PAR2_ENABLE_NAN_GUARDS 1
#  else
#    define PAR2_ENABLE_NAN_GUARDS 0
#  endif
#endif

/**
 * @brief Enable array bounds checking in kernels.
 *
 * When enabled, kernels validate array indices before access.
 * Significant per-particle overhead.
 */
#ifndef PAR2_ENABLE_BOUNDS_CHECKS
#  if PAR2_DEBUG_LEVEL >= 2
#    define PAR2_ENABLE_BOUNDS_CHECKS 1
#  else
#    define PAR2_ENABLE_BOUNDS_CHECKS 0
#  endif
#endif

// =============================================================================
// Legacy Compatibility (map to new system)
// =============================================================================

/**
 * @brief Legacy macro for debug builds.
 *
 * @deprecated Use PAR2_DEBUG_LEVEL instead.
 */
#if PAR2_DEBUG_LEVEL >= 1
#  ifndef PAR2_CORE_DEBUG
#    define PAR2_CORE_DEBUG 1
#  endif
#endif

// =============================================================================
// Compile-time Assertions
// =============================================================================

// Sanity check: debug level must be 0, 1, or 2
static_assert(PAR2_DEBUG_LEVEL >= 0 && PAR2_DEBUG_LEVEL <= 2,
              "PAR2_DEBUG_LEVEL must be 0 (release), 1 (basic), or 2 (full)");

// =============================================================================
// Debug Utility Macros (for internal use)
// =============================================================================

/**
 * @brief Execute code only in debug builds.
 *
 * Usage:
 * ```cpp
 * PAR2_DEBUG_ONLY(printf("Debug: x=%f\n", x);)
 * ```
 */
#if PAR2_DEBUG_LEVEL >= 1
#  define PAR2_DEBUG_ONLY(code) do { code } while(0)
#else
#  define PAR2_DEBUG_ONLY(code) ((void)0)
#endif

/**
 * @brief Execute code only with NaN guards enabled.
 */
#if PAR2_ENABLE_NAN_GUARDS
#  define PAR2_NAN_GUARD(code) do { code } while(0)
#else
#  define PAR2_NAN_GUARD(code) ((void)0)
#endif

/**
 * @brief Device-side assertion (only in debug builds).
 *
 * Usage in kernels:
 * ```cpp
 * PAR2_DEVICE_ASSERT(idx < n, "Index out of bounds");
 * ```
 */
#if PAR2_DEBUG_LEVEL >= 2 && defined(__CUDA_ARCH__)
#  define PAR2_DEVICE_ASSERT(cond, msg) \
     do { if (!(cond)) { printf("PAR2 Assert: %s at %s:%d\n", msg, __FILE__, __LINE__); } } while(0)
#else
#  define PAR2_DEVICE_ASSERT(cond, msg) ((void)0)
#endif

#endif // PAR2_CORE_DEBUG_POLICY_HPP
