#include "qanneal/qubo.hpp"

#include <stdexcept>

namespace qanneal {

QUBO::QUBO(std::vector<double> q, std::size_t n)
    : q_(std::move(q)), n_(n) {
    if (n_ == 0) {
        throw std::invalid_argument("QUBO size must be > 0.");
    }
    if (q_.size() != n_ * n_) {
        throw std::invalid_argument("QUBO matrix size mismatch.");
    }
}

DenseIsing QUBO::to_ising() const {
    // Symmetrize Q
    std::vector<double> W(q_.size());
    for (std::size_t i = 0; i < n_; ++i) {
        for (std::size_t j = 0; j < n_; ++j) {
            const double v = 0.5 * (q_[i * n_ + j] + q_[j * n_ + i]);
            W[i * n_ + j] = v;
        }
    }

    std::vector<double> h(n_, 0.0);
    std::vector<double> J(n_ * n_, 0.0);
    double c = 0.0;

    // E = 1/4 * s^T W s + 1/2 * s^T W 1 + 1/4 * 1^T W 1
    for (std::size_t i = 0; i < n_; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < n_; ++j) {
            row_sum += W[i * n_ + j];
        }
        h[i] = 0.5 * row_sum;
    }

    for (std::size_t i = 0; i < n_; ++i) {
        for (std::size_t j = 0; j < n_; ++j) {
            J[i * n_ + j] = 0.25 * W[i * n_ + j];
            c += 0.25 * W[i * n_ + j];
        }
    }

    return DenseIsing(std::move(h), std::move(J), n_, c);
}

}
