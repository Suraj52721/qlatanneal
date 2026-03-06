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

    // Local field caching interface.
    // fields[i] = h[i] + sum_{j!=i} J[i,j] * s[j]
    // Enables O(1) delta: dE(flip) = -2 * spins[flip] * fields[flip]
    // Default: derive from delta_energy (works for any Hamiltonian, O(N^2) init).
    virtual void compute_local_fields(const int8_t *spins, std::size_t n,
                                      double *fields) const {
        for (std::size_t i = 0; i < n; ++i) {
            const double de = delta_energy(spins, n, i);
            const double si = static_cast<double>(spins[i]);
            fields[i] = (si != 0.0) ? (-de / (2.0 * si)) : 0.0;
        }
    }

    // Update fields in-place after flipping spin `flip` from old_spin to -old_spin.
    // new_spins is the state AFTER the flip.
    // Default: full recompute (slow fallback; subclasses override for O(degree) update).
    virtual void update_local_fields_after_flip(double *fields,
                                                const int8_t *new_spins,
                                                std::size_t n,
                                                std::size_t flip,
                                                int8_t old_spin) const {
        (void)flip; (void)old_spin;
        compute_local_fields(new_spins, n, fields);
    }
};

}
