#include "qanneal/dense_ising.hpp"

#include <cmath>

#include "qanneal/state.hpp"

namespace qanneal {

DenseIsing::DenseIsing(std::vector<double> h,
                       std::vector<double> J,
                       std::size_t n,
                       double c)
    : h_(std::move(h)), J_(std::move(J)), n_(n), c_(c) {
    validate_sizes();
}

void DenseIsing::validate_sizes() const {
    if (n_ == 0) {
        throw std::invalid_argument("DenseIsing size must be > 0.");
    }
    if (h_.size() != n_) {
        throw std::invalid_argument("DenseIsing h size mismatch.");
    }
    if (J_.size() != n_ * n_) {
        throw std::invalid_argument("DenseIsing J size mismatch.");
    }
}

double DenseIsing::energy(const int8_t *spins, std::size_t n) const {
    if (n != n_) {
        throw std::invalid_argument("State size mismatch.");
    }
    validate_spins(spins, n_);
    double E = c_;
    for (std::size_t i = 0; i < n_; ++i) {
        E += h_[i] * static_cast<double>(spins[i]);
    }
    for (std::size_t i = 0; i < n_; ++i) {
        for (std::size_t j = i + 1; j < n_; ++j) {
            E += J_at(i, j) * static_cast<double>(spins[i]) * static_cast<double>(spins[j]);
        }
    }
    return E;
}

double DenseIsing::delta_energy(const int8_t *spins, std::size_t n, std::size_t flip) const {
    if (n != n_) {
        throw std::invalid_argument("State size mismatch.");
    }
    if (flip >= n_) {
        throw std::invalid_argument("Flip index out of range.");
    }
    const double s = static_cast<double>(spins[flip]);
    double local = h_[flip];
    for (std::size_t j = 0; j < n_; ++j) {
        if (j == flip) {
            continue;
        }
        local += J_at(flip, j) * static_cast<double>(spins[j]);
    }
    return -2.0 * s * local;
}

void DenseIsing::compute_local_fields(const int8_t *spins, std::size_t n,
                                       double *fields) const {
    if (n != n_) {
        throw std::invalid_argument("State size mismatch.");
    }
    // Branch-free inner loop: sum over all j (including i), then subtract
    // the self-coupling J[i,i]*s[i].  For standard Ising J[i,i]=0 so the
    // subtraction is a no-op; including it costs one FMA but removes the
    // conditional from the hot loop, allowing the compiler to emit SIMD
    // (AVX2 / NEON) instructions for the dot product.
    for (std::size_t i = 0; i < n_; ++i) {
        double f = h_[i];
        const double *row = J_.data() + i * n_;
        for (std::size_t j = 0; j < n_; ++j) {
            f += row[j] * static_cast<double>(spins[j]);
        }
        f -= row[i] * static_cast<double>(spins[i]);  // remove self-coupling
        fields[i] = f;
    }
}

// After flipping spin `flip` from old_spin to -old_spin:
// fields[j] += J[flip,j] * (-2 * old_spin)  for all j != flip
// fields[flip] is unchanged (depends only on other spins).
void DenseIsing::update_local_fields_after_flip(double *fields,
                                                 const int8_t * /*new_spins*/,
                                                 std::size_t n,
                                                 std::size_t flip,
                                                 int8_t old_spin) const {
    if (n != n_) {
        throw std::invalid_argument("State size mismatch.");
    }
    const double ds = -2.0 * static_cast<double>(old_spin);
    const double *row = J_.data() + flip * n_;
    // Branch-free inner loop: update all j, then undo the self-update for
    // fields[flip].  J[flip,flip] is typically 0 in Ising models so the
    // fixup is free; but including it makes the loop unconditional and
    // enables the compiler to auto-vectorize this DAXPY pattern.
    for (std::size_t j = 0; j < n_; ++j) {
        fields[j] += row[j] * ds;
    }
    fields[flip] -= row[flip] * ds;  // undo the self-contribution
}

}
