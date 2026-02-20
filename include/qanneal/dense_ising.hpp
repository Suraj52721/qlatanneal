#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "qanneal/hamiltonian.hpp"

namespace qanneal {

// Energy convention:
// E = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + c
class DenseIsing final : public Hamiltonian {
public:
    DenseIsing() = default;

    DenseIsing(std::vector<double> h,
               std::vector<double> J,
               std::size_t n,
               double c = 0.0);

    using Hamiltonian::energy;
    using Hamiltonian::delta_energy;

    std::size_t size() const override { return n_; }
    double energy(const int8_t *spins, std::size_t n) const override;
    double delta_energy(const int8_t *spins, std::size_t n, std::size_t flip) const override;

    const std::vector<double> &h() const { return h_; }
    const std::vector<double> &J() const { return J_; }
    double constant() const { return c_; }

private:
    std::vector<double> h_;
    std::vector<double> J_;  // row-major n x n
    std::size_t n_ = 0;
    double c_ = 0.0;

    double J_at(std::size_t r, std::size_t c) const {
        return J_[r * n_ + c];
    }

    void validate_sizes() const;
};

}
