#include <cassert>
#include <cmath>

#include "qanneal/dense_ising.hpp"
#include "qanneal/sqa_annealer.hpp"
#include "qanneal/sqa_schedule.hpp"

int main() {
    const std::size_t n = 3;
    std::vector<double> h = {0.0, 0.0, 0.0};
    std::vector<double> J = {
        0.0, 0.5, -0.3,
        0.5, 0.0, 0.2,
        -0.3, 0.2, 0.0
    };

    qanneal::DenseIsing ham(h, J, n);

    std::vector<double> betas = {0.2, 0.5, 1.0};
    std::vector<double> gammas = {2.0, 1.0, 0.5};
    auto sched = qanneal::SQASchedule::from_vectors(betas, gammas);

    qanneal::SQAAnnealer annealer(ham, sched, 8, 2);
    annealer.set_seed(42);
    auto result = annealer.run(5, 1);

    assert(result.energy_trace.size() == betas.size());
    const double e = ham.energy(result.best_state);
    assert(std::abs(e - result.best_energy) < 1e-12);

    return 0;
}
