#include <cassert>
#include <cmath>

#include "qanneal/dense_ising.hpp"
#include "qanneal/sqa_parallel_tempering.hpp"

int main() {
    const std::size_t n = 3;
    std::vector<double> h = {0.0, 0.0, 0.0};
    std::vector<double> J = {
        0.0, 0.5, -0.2,
        0.5, 0.0, 0.3,
        -0.2, 0.3, 0.0,
    };

    qanneal::DenseIsing ham(h, J, n);
    std::vector<double> betas = {0.2, 0.5, 1.0};
    std::vector<double> gammas = {2.0, 1.0, 0.5};

    qanneal::SQAParallelTemperingAnnealer annealer(ham, betas, gammas, 8);
    annealer.set_seed(123);
    auto result = annealer.run(4, 1, 5, 1, 1, 0);

    assert(result.final_states.size() == betas.size());
    assert(result.final_energies.size() == betas.size());
    assert(result.average_energy_trace.size() == 5);
    assert(result.swap_acceptance_trace.size() == 5);

    const double e = ham.energy(result.best_state);
    assert(std::abs(e - result.best_energy) < 1e-12);

    return 0;
}
