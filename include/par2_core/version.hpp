/**
 * @file version.hpp
 * @brief Par2_Core version information.
 *
 * @copyright Par2_Core - GPU-native transport engine
 */

#ifndef PAR2_CORE_VERSION_HPP
#define PAR2_CORE_VERSION_HPP

#define PAR2_CORE_VERSION_MAJOR 0
#define PAR2_CORE_VERSION_MINOR 1
#define PAR2_CORE_VERSION_PATCH 0

#define PAR2_CORE_VERSION_STRING "0.1.0"

namespace par2 {

struct Version {
    static constexpr int major = PAR2_CORE_VERSION_MAJOR;
    static constexpr int minor = PAR2_CORE_VERSION_MINOR;
    static constexpr int patch = PAR2_CORE_VERSION_PATCH;
    static constexpr const char* string = PAR2_CORE_VERSION_STRING;
};

} // namespace par2

#endif // PAR2_CORE_VERSION_HPP
