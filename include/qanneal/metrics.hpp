#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "qanneal/state.hpp"

namespace qanneal {

inline double magnetization(const int8_t *spins, std::size_t n) {
    if (n == 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(spins[i]);
    }
    return sum / static_cast<double>(n);
}

inline double magnetization(const State &state) {
    return magnetization(state.spins.data(), state.size());
}

inline double overlap(const int8_t *a, const int8_t *b, std::size_t n) {
    if (n == 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum / static_cast<double>(n);
}

inline double overlap(const State &a, const State &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("State size mismatch for overlap.");
    }
    return overlap(a.spins.data(), b.spins.data(), a.size());
}

}
