#include "qanneal/higher_order_ising.hpp"

#include <algorithm>
#include <map>

#include "qanneal/state.hpp"

namespace qanneal {

HigherOrderIsing::HigherOrderIsing(std::size_t n,
                                   const std::vector<RawTerm> &terms,
                                   double c)
    : h_(n, 0.0), n_(n), c_(c) {
    if (n_ == 0) {
        throw std::invalid_argument("HigherOrderIsing size must be > 0.");
    }

    // Consolidate terms: reduce repeated variables by spin parity (s_i^2 = 1),
    // fold degree 0/1 into c_/h_, and merge identical higher-order supports by
    // summing coefficients so each distinct support appears at most once.
    std::map<std::vector<std::uint32_t>, double> merged;
    for (const auto &raw : terms) {
        const std::vector<std::size_t> &vars = raw.first;
        const double coeff = raw.second;
        if (coeff == 0.0) {
            continue;
        }

        // Parity reduction: a variable that appears an even number of times
        // cancels (s^2 = 1); an odd number of times collapses to one factor.
        std::vector<std::uint32_t> reduced;
        reduced.reserve(vars.size());
        std::vector<std::size_t> sorted(vars);
        for (std::size_t v : sorted) {
            if (v >= n_) {
                throw std::invalid_argument("HigherOrderIsing term variable index out of range.");
            }
        }
        std::sort(sorted.begin(), sorted.end());
        std::size_t i = 0;
        while (i < sorted.size()) {
            std::size_t j = i;
            while (j < sorted.size() && sorted[j] == sorted[i]) {
                ++j;
            }
            if (((j - i) & 1u) == 1u) {  // odd multiplicity survives
                reduced.push_back(static_cast<std::uint32_t>(sorted[i]));
            }
            i = j;
        }

        const std::size_t degree = reduced.size();
        if (degree == 0) {
            c_ += coeff;
        } else if (degree == 1) {
            h_[reduced[0]] += coeff;
        } else {
            merged[std::move(reduced)] += coeff;
        }
    }

    // Flatten the surviving higher-order terms into CSR-style storage.
    term_coeff_.reserve(merged.size());
    term_offset_.reserve(merged.size() + 1);
    term_offset_.push_back(0);
    incidence_.assign(n_, {});
    for (const auto &entry : merged) {
        const std::vector<std::uint32_t> &vars = entry.first;
        const double coeff = entry.second;
        if (coeff == 0.0) {  // exact cancellation after merge
            continue;
        }
        const std::uint32_t term_id = static_cast<std::uint32_t>(term_coeff_.size());
        term_coeff_.push_back(coeff);
        for (std::uint32_t v : vars) {
            term_vars_.push_back(v);
            incidence_[v].push_back(term_id);
        }
        term_offset_.push_back(term_vars_.size());
        max_degree_ = std::max(max_degree_, vars.size());
    }
    if (!h_.empty()) {
        for (double hv : h_) {
            if (hv != 0.0) {
                max_degree_ = std::max<std::size_t>(max_degree_, 1);
                break;
            }
        }
    }
}

double HigherOrderIsing::energy(const int8_t *spins, std::size_t n) const {
    if (n != n_) {
        throw std::invalid_argument("State size mismatch.");
    }
    validate_spins(spins, n_);
    double E = c_;
    for (std::size_t i = 0; i < n_; ++i) {
        E += h_[i] * static_cast<double>(spins[i]);
    }
    const std::size_t nt = num_higher_terms();
    for (std::size_t t = 0; t < nt; ++t) {
        const TermView tv = term(t);
        double prod = tv.coeff;
        for (std::size_t k = 0; k < tv.count; ++k) {
            prod *= static_cast<double>(spins[tv.vars[k]]);
        }
        E += prod;
    }
    return E;
}

double HigherOrderIsing::delta_energy(const int8_t *spins, std::size_t n, std::size_t flip) const {
    if (n != n_) {
        throw std::invalid_argument("State size mismatch.");
    }
    if (flip >= n_) {
        throw std::invalid_argument("Flip index out of range.");
    }
    const double s = static_cast<double>(spins[flip]);
    // local == field[flip] = h_flip + sum_{t ni flip} J_t * prod_{k != flip} s_k
    double local = h_[flip];
    for (std::uint32_t t : incidence_[flip]) {
        const TermView tv = term(t);
        double prod = tv.coeff;
        for (std::size_t k = 0; k < tv.count; ++k) {
            const std::uint32_t v = tv.vars[k];
            if (v == flip) {
                continue;
            }
            prod *= static_cast<double>(spins[v]);
        }
        local += prod;
    }
    return -2.0 * s * local;
}

void HigherOrderIsing::compute_local_fields(const int8_t *spins, std::size_t n,
                                            double *fields) const {
    if (n != n_) {
        throw std::invalid_argument("State size mismatch.");
    }
    for (std::size_t i = 0; i < n_; ++i) {
        fields[i] = h_[i];
    }
    const std::size_t nt = num_higher_terms();
    for (std::size_t t = 0; t < nt; ++t) {
        const TermView tv = term(t);
        // prod = J_t * prod_{k in vars} s_k ; contribution to field[v] is
        // prod / s_v == prod * s_v (spins are +-1).
        double prod = tv.coeff;
        for (std::size_t k = 0; k < tv.count; ++k) {
            prod *= static_cast<double>(spins[tv.vars[k]]);
        }
        for (std::size_t k = 0; k < tv.count; ++k) {
            const std::uint32_t v = tv.vars[k];
            fields[v] += prod * static_cast<double>(spins[v]);
        }
    }
}

// After flipping spin `flip` (new_spins already reflects the flip): only terms
// that contain `flip` change the fields of their *other* members.  For such a
// term the product over its variables has negated, so each other member v gets
//     fields[v] += 2 * J_t * prod_{k in vars} new_spins[k] * new_spins[v].
// fields[flip] itself is independent of s[flip] and stays unchanged.
void HigherOrderIsing::update_local_fields_after_flip(double *fields,
                                                      const int8_t *new_spins,
                                                      std::size_t n,
                                                      std::size_t flip,
                                                      int8_t /*old_spin*/) const {
    if (n != n_) {
        throw std::invalid_argument("State size mismatch.");
    }
    for (std::uint32_t t : incidence_[flip]) {
        const TermView tv = term(t);
        double newprod = tv.coeff;
        for (std::size_t k = 0; k < tv.count; ++k) {
            newprod *= static_cast<double>(new_spins[tv.vars[k]]);
        }
        for (std::size_t k = 0; k < tv.count; ++k) {
            const std::uint32_t v = tv.vars[k];
            if (v == flip) {
                continue;
            }
            fields[v] += 2.0 * newprod * static_cast<double>(new_spins[v]);
        }
    }
}

double HigherOrderIsing::coupling_energy(const int8_t *spins, std::size_t n) const {
    if (n != n_) {
        throw std::invalid_argument("State size mismatch.");
    }
    double E = 0.0;
    const std::size_t nt = num_higher_terms();
    for (std::size_t t = 0; t < nt; ++t) {
        const TermView tv = term(t);
        double prod = tv.coeff;
        for (std::size_t k = 0; k < tv.count; ++k) {
            prod *= static_cast<double>(spins[tv.vars[k]]);
        }
        E += prod;
    }
    return E;
}

}
