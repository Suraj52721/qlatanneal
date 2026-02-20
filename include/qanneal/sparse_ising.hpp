#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "qanneal/hamiltonian.hpp"

namespace qanneal {

struct SparseEdge {
    std::size_t i;
    std::size_t j;
    double value;
};

class SparseIsing final : public Hamiltonian {
public:
    SparseIsing() = default;

    SparseIsing(std::vector<double> h,
                std::vector<SparseEdge> edges,
                std::size_t n,
                double c = 0.0);

    using Hamiltonian::energy;
    using Hamiltonian::delta_energy;

    std::size_t size() const override { return n_; }
    double energy(const int8_t *spins, std::size_t n) const override;
    double delta_energy(const int8_t *spins, std::size_t n, std::size_t flip) const override;

    const std::vector<double> &h() const { return h_; }
    const std::vector<SparseEdge> &edges() const { return edges_; }
    double constant() const { return c_; }

private:
    struct Neighbor {
        std::size_t idx;
        double value;
    };

    std::vector<double> h_;
    std::vector<SparseEdge> edges_;
    std::vector<std::vector<Neighbor>> adj_;
    std::size_t n_ = 0;
    double c_ = 0.0;

    void build_adjacency();
    void validate_sizes() const;
};

}
