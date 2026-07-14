#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "qanneal/hamiltonian.hpp"

namespace qanneal {

// ---------------------------------------------------------------------------
// Native higher-order Ising (HUBO) Hamiltonian.
//
// Energy convention (spins s_i in {-1, +1}):
//
//     E(s) = c + sum_i h_i s_i + sum_t J_t * prod_{i in vars(t)} s_i
//
// where each term t couples an arbitrary subset vars(t) of two or more spins.
// Degree-1 terms are folded into h_i, degree-0 terms into c.
//
// The class is fully STATELESS / const: like DenseIsing and SparseIsing it
// caches no per-configuration data on itself.  The annealer owns the
// `fields[]` buffer; every method here is a pure function of the spins passed
// in, so a single shared_ptr<const HigherOrderIsing> is safe to drive many
// replicas / Trotter slices concurrently.
//
// Local-field contract (same as the quadratic Hamiltonians):
//     field[i] = h_i + sum_{t : i in vars(t)} J_t * prod_{k in vars(t), k!=i} s_k
//     delta_energy(flip) = -2 * s[flip] * field[flip]
// This is what makes every existing solver (SA / SQA / SQA-PT / replica /
// CT-PIMC) work with HUBO unchanged: they only ever touch a model through the
// abstract Hamiltonian interface.
//
// Cost per single-spin flip is O(sum over terms containing `flip` of their
// arity) -- i.e. proportional to how often the flipped spin appears, not to N.
// For bounded-degree, sparse HUBO this is effectively O(1), matching the
// sparse-quadratic path.
// ---------------------------------------------------------------------------
class HigherOrderIsing final : public Hamiltonian {
public:
    // A raw term as supplied by the caller: (variable indices, coefficient).
    // Indices need not be sorted or unique; repeated indices are reduced by
    // spin parity (s_i^2 = 1) and identical terms are merged in the ctor.
    using RawTerm = std::pair<std::vector<std::size_t>, double>;

    HigherOrderIsing() = default;

    HigherOrderIsing(std::size_t n,
                     const std::vector<RawTerm> &terms,
                     double c = 0.0);

    using Hamiltonian::energy;
    using Hamiltonian::delta_energy;

    std::size_t size() const override { return n_; }
    double energy(const int8_t *spins, std::size_t n) const override;
    double delta_energy(const int8_t *spins, std::size_t n, std::size_t flip) const override;
    void compute_local_fields(const int8_t *spins, std::size_t n,
                              double *fields) const override;
    void update_local_fields_after_flip(double *fields,
                                        const int8_t *new_spins,
                                        std::size_t n,
                                        std::size_t flip,
                                        int8_t old_spin) const override;

    std::vector<double> linear_terms() const { return h_; }

    // Coupling (degree >= 2) part of the energy for a configuration:
    //     E(s) - c - sum_i h_i s_i = sum_t J_t prod s.
    // Handy for diagnostics and for observables that need the interaction
    // energy without the on-site/constant offset.
    double coupling_energy(const int8_t *spins, std::size_t n) const;

    // Introspection / reuse.
    std::size_t max_degree() const { return max_degree_; }
    std::size_t num_terms() const { return term_coeff_.size(); }
    const std::vector<double> &h() const { return h_; }
    double constant() const { return c_; }

private:
    // Higher-order terms (degree >= 2). Variables are stored contiguously in
    // `term_vars_`; term t owns the range [term_offset_[t], term_offset_[t+1]).
    std::vector<double> term_coeff_;        // J_t
    std::vector<std::size_t> term_offset_;  // size num_terms + 1 (CSR-style)
    std::vector<std::uint32_t> term_vars_;  // flattened, sorted per term

    // Incidence: for each variable, the ids of terms that contain it.
    std::vector<std::vector<std::uint32_t>> incidence_;

    std::vector<double> h_;  // degree-1 coefficients, size n_
    std::size_t n_ = 0;
    double c_ = 0.0;
    std::size_t max_degree_ = 0;

    // View onto the variables of term t.
    struct TermView {
        const std::uint32_t *vars;
        std::size_t count;
        double coeff;
    };
    TermView term(std::size_t t) const {
        const std::size_t begin = term_offset_[t];
        return TermView{term_vars_.data() + begin,
                        term_offset_[t + 1] - begin,
                        term_coeff_[t]};
    }
    std::size_t num_higher_terms() const { return term_coeff_.size(); }
};

}
