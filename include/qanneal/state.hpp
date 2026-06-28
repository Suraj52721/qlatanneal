#pragma once

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace qanneal {

struct State {
    std::vector<int8_t> spins;  // values in {-1, +1}

    State() = default;

    explicit State(std::size_t n) : spins(n, 1) {}

    std::size_t size() const { return spins.size(); }

    int8_t &operator[](std::size_t idx) { return spins[idx]; }
    const int8_t &operator[](std::size_t idx) const { return spins[idx]; }

    template <class URNG>
    static State random(std::size_t n, URNG &rng) {
        State s(n);
        std::uniform_int_distribution<int> dist(0, 1);
        for (std::size_t i = 0; i < n; ++i) {
            s.spins[i] = dist(rng) ? 1 : -1;
        }
        return s;
    }
};

inline void validate_spins(const State &state) {
    for (auto v : state.spins) {
        if (v != -1 && v != 1) {
            throw std::invalid_argument("State spins must be -1 or +1.");
        }
    }
}

inline void validate_spins(const int8_t *spins, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const auto v = spins[i];
        if (v != -1 && v != 1) {
            throw std::invalid_argument("State spins must be -1 or +1.");
        }
    }
}

}
