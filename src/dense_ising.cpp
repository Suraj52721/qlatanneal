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

}
