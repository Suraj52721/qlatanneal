// Math-verification tests for the native higher-order Ising (HUBO) Hamiltonian.
//
// Everything is checked against a brute-force ground truth computed straight
// from the definition E(s) = c + sum_i h_i s_i + sum_t J_t prod_{k in t} s_k:
//
//   (1) energy() matches the brute-force sum.
//   (2) delta_energy(flip) matches energy(after) - energy(before) for a flip.
//   (3) the local-field cache is self-consistent: compute_local_fields gives
//       fields with -2 s_i field_i == delta_energy(i) for every i, and an
//       incremental update_local_fields_after_flip stays identical to a full
//       recompute across a long random walk of accepted flips.
//   (4) coupling_energy() == energy() minus constant and linear parts.
//   (5) parity reduction (s_i^2 = 1) and duplicate-term merging are applied.
//
// These pin the fast incremental math to ground truth independently of any
// annealer, exactly as test_catalyst_math.cpp does for the catalyst.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "qanneal/higher_order_ising.hpp"

namespace {

using qanneal::HigherOrderIsing;

struct Term {
    std::vector<std::size_t> vars;
    double coeff;
};

// Brute-force energy from the raw (unreduced) term list.
double brute_energy(const std::vector<int8_t> &s,
                    const std::vector<double> &h,
                    const std::vector<Term> &terms,
                    double c) {
    double E = c;
    for (std::size_t i = 0; i < s.size(); ++i) {
        E += h[i] * static_cast<double>(s[i]);
    }
    for (const auto &t : terms) {
        double p = t.coeff;
        for (std::size_t v : t.vars) {
            p *= static_cast<double>(s[v]);
        }
        E += p;
    }
    return E;
}

std::vector<int8_t> random_state(std::size_t n, std::mt19937_64 &rng) {
    std::uniform_int_distribution<int> bd(0, 1);
    std::vector<int8_t> s(n);
    for (std::size_t i = 0; i < n; ++i) {
        s[i] = bd(rng) ? 1 : -1;
    }
    return s;
}

// Build a random HUBO with mixed degrees (1..4), plus terms with repeated
// indices and duplicate supports to exercise reduction/merging.
void build_random(std::size_t n, std::mt19937_64 &rng,
                  std::vector<double> &h, std::vector<Term> &terms, double &c) {
    std::uniform_real_distribution<double> ud(-1.5, 1.5);
    std::uniform_int_distribution<std::size_t> vd(0, n - 1);
    std::uniform_int_distribution<int> deg(1, 4);

    c = ud(rng);
    h.assign(n, 0.0);
    terms.clear();
    const int num_terms = 40;
    for (int t = 0; t < num_terms; ++t) {
        Term term;
        const int d = deg(rng);
        for (int k = 0; k < d; ++k) {
            term.vars.push_back(vd(rng));  // may repeat -> parity reduction
        }
        term.coeff = ud(rng);
        terms.push_back(term);
    }
    // A couple of deliberately duplicated supports (to test merging) and a
    // guaranteed even-multiplicity term (which must vanish).
    terms.push_back(Term{{0, 1, 2}, 0.7});
    terms.push_back(Term{{2, 1, 0}, -0.3});   // same support, different order
    terms.push_back(Term{{3, 3, 4, 4}, 9.9});  // fully cancels -> constant-ish
}

// Convert the raw term list into the HigherOrderIsing RawTerm form.
std::vector<HigherOrderIsing::RawTerm> to_raw(const std::vector<double> &h,
                                              const std::vector<Term> &terms) {
    std::vector<HigherOrderIsing::RawTerm> raw;
    for (std::size_t i = 0; i < h.size(); ++i) {
        if (h[i] != 0.0) {
            raw.push_back({{i}, h[i]});
        }
    }
    for (const auto &t : terms) {
        raw.push_back({t.vars, t.coeff});
    }
    return raw;
}

void test_instance(std::size_t n, std::mt19937_64 &rng) {
    std::vector<double> h;
    std::vector<Term> terms;
    double c = 0.0;
    build_random(n, rng, h, terms, c);

    const HigherOrderIsing ham(n, to_raw(h, terms), c);

    for (int trial = 0; trial < 100; ++trial) {
        std::vector<int8_t> s = random_state(n, rng);

        // (1) energy matches brute force.
        const double E = ham.energy(s.data(), n);
        const double E_ref = brute_energy(s, h, terms, c);
        assert(std::abs(E - E_ref) < 1e-9);

        // (2) delta_energy matches an explicit flip.
        for (std::size_t f = 0; f < n; ++f) {
            const double de = ham.delta_energy(s.data(), n, f);
            std::vector<int8_t> s2 = s;
            s2[f] = static_cast<int8_t>(-s2[f]);
            const double de_ref = brute_energy(s2, h, terms, c) - E_ref;
            assert(std::abs(de - de_ref) < 1e-9);
        }

        // (3a) local fields give the correct delta for every spin.
        std::vector<double> fields(n, 0.0);
        ham.compute_local_fields(s.data(), n, fields.data());
        for (std::size_t i = 0; i < n; ++i) {
            const double de_field = -2.0 * static_cast<double>(s[i]) * fields[i];
            assert(std::abs(de_field - ham.delta_energy(s.data(), n, i)) < 1e-9);
        }

        // (4) coupling_energy == energy - constant - linear, using the model's
        // own folded constant()/h() (the ctor folds reduced degree-0/1 terms
        // into c_/h_, so the raw input c/h no longer describe the split).
        double lin = 0.0;
        for (std::size_t i = 0; i < n; ++i) lin += ham.h()[i] * static_cast<double>(s[i]);
        assert(std::abs(ham.coupling_energy(s.data(), n) - (E - ham.constant() - lin)) < 1e-9);
    }

    // (3b) incremental field updates stay identical to a full recompute over a
    // long random walk of accepted flips.
    std::vector<int8_t> s = random_state(n, rng);
    std::vector<double> fields(n, 0.0);
    ham.compute_local_fields(s.data(), n, fields.data());
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    for (int step = 0; step < 2000; ++step) {
        const std::size_t f = pick(rng);
        const int8_t old = s[f];
        s[f] = static_cast<int8_t>(-old);
        ham.update_local_fields_after_flip(fields.data(), s.data(), n, f, old);

        std::vector<double> ref(n, 0.0);
        ham.compute_local_fields(s.data(), n, ref.data());
        for (std::size_t i = 0; i < n; ++i) {
            assert(std::abs(fields[i] - ref[i]) < 1e-7);
        }
    }
}

}  // namespace

int main() {
    std::mt19937_64 rng(0xC0FFEEu);
    for (std::size_t n : {5u, 8u, 12u, 20u}) {
        for (int rep = 0; rep < 5; ++rep) {
            test_instance(n, rng);
        }
    }
    return 0;
}
