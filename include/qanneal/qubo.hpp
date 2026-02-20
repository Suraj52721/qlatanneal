#pragma once

#include <cstddef>
#include <vector>

#include "qanneal/dense_ising.hpp"

namespace qanneal {

class QUBO {
public:
    QUBO() = default;

    QUBO(std::vector<double> q, std::size_t n);

    const std::vector<double> &matrix() const { return q_; }
    std::size_t size() const { return n_; }

    DenseIsing to_ising() const;

private:
    std::vector<double> q_;
    std::size_t n_ = 0;
};

}
