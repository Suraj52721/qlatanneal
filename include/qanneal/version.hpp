#pragma once

#include <string>

namespace qanneal {

inline constexpr int version_major = 2;
inline constexpr int version_minor = 0;
inline constexpr int version_patch = 0;

inline std::string version_string() {
    return std::to_string(version_major) + "." +
           std::to_string(version_minor) + "." +
           std::to_string(version_patch);
}

}
