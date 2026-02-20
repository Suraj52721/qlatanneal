#include "qanneal/sparse_ising.hpp"

#include <cmath>

#include "qanneal/state.hpp"

namespace qanneal {

SparseIsing::SparseIsing(std::vector<double> h,
                         std::vector<SparseEdge> edges,
                         std::size_t n,
                         double c)
    : h_(std::move(h)), edges_(std::move(edges)), n_(n), c_(c) {
    validate_sizes();
    build_adjacency();
}

void SparseIsing::validate_sizes() const {
    if (n_ == 0) {
        throw std::invalid_argument("SparseIsing size must be > 0.");
    }
    if (h_.size() != n_) {
        throw std::invalid_argument("SparseIsing h size mismatch.");
    }
    for (const auto &edge : edges_) {
        if (edge.i >= n_ || edge.j >= n_) {
            throw std::invalid_argument("SparseIsing edge index out of range.");
        }
        if (edge.i == edge.j) {
            throw std::invalid_argument("SparseIsing self-edge not allowed.");
        }
    }
}

void SparseIsing::build_adjacency() {
    adj_.assign(n_, {});
    for (const auto &edge : edges_) {
        adj_[edge.i].push_back(Neighbor{edge.j, edge.value});
        adj_[edge.j].push_back(Neighbor{edge.i, edge.value});
    }
}

double SparseIsing::energy(const int8_t *spins, std::size_t n) const {
    if (n != n_) {
        throw std::invalid_argument("State size mismatch.");
    }
    validate_spins(spins, n_);
    double E = c_;
    for (std::size_t i = 0; i < n_; ++i) {
        E += h_[i] * static_cast<double>(spins[i]);
    }
    for (const auto &edge : edges_) {
        E += edge.value * static_cast<double>(spins[edge.i]) * static_cast<double>(spins[edge.j]);
    }
    return E;
}

double SparseIsing::delta_energy(const int8_t *spins, std::size_t n, std::size_t flip) const {
    if (n != n_) {
        throw std::invalid_argument("State size mismatch.");
    }
    if (flip >= n_) {
        throw std::invalid_argument("Flip index out of range.");
    }
    const double s = static_cast<double>(spins[flip]);
    double local = h_[flip];
    for (const auto &neighbor : adj_[flip]) {
        local += neighbor.value * static_cast<double>(spins[neighbor.idx]);
    }
    return -2.0 * s * local;
}

}
