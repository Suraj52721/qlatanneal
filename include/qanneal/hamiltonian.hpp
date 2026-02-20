#pragma once

#include <cstddef>
#include <cstdint>

#include "qanneal/state.hpp"

namespace qanneal {

class Hamiltonian {
public:
    virtual ~Hamiltonian() = default;

    virtual double energy(const int8_t *spins, std::size_t n) const = 0;
    virtual double delta_energy(const int8_t *spins, std::size_t n, std::size_t flip) const = 0;
    virtual std::size_t size() const = 0;

    double energy(const State &state) const {
        return energy(state.spins.data(), state.size());
    }

    double delta_energy(const State &state, std::size_t flip) const {
        return delta_energy(state.spins.data(), state.size(), flip);
    }
};

}
