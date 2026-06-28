#include <cassert>
#include <cmath>

#include "qanneal/dense_ising.hpp"
#include "qanneal/parallel_tempering.hpp"

int main() {
    const std::size_t n = 3;
    std::vector<double> h = {0.1, -0.1, 0.2};
    std::vector<double> J = {
        0.0, 0.4, 0.0,
        0.4, 0.0, -0.2,
        0.0, -0.2, 0.0
    };

    qanneal::DenseIsing ham(h, J, n);
    std::vector<double> betas = {0.2, 0.6, 1.0};

    qanneal::ParallelTemperingAnnealer annealer(ham, betas);
    annealer.set_seed(7);
    auto result = annealer.run(3, 4, 1);

    assert(result.average_energy_trace.size() == 4);
    assert(result.swap_acceptance_trace.size() == 4);

    const double e = ham.energy(result.best_state);
    assert(std::abs(e - result.best_energy) < 1e-12);

    return 0;
}
