#include <cassert>

#include "qanneal/dense_ising.hpp"
#include "qanneal/replica_annealer.hpp"
#include "qanneal/schedule.hpp"

int main() {
    const std::size_t n = 3;
    std::vector<double> h = {0.1, -0.1, 0.2};
    std::vector<double> J = {
        0.0, 0.4, 0.0,
        0.4, 0.0, -0.2,
        0.0, -0.2, 0.0
    };

    qanneal::DenseIsing ham(h, J, n);
    auto schedule = qanneal::AnnealSchedule::linear(0.2, 1.0, 4);

    qanneal::ReplicaAnnealer annealer(ham, schedule, 3);
    annealer.set_seed(123);
    auto result = annealer.run(5);

    assert(result.replicas.size() == 3);
    assert(result.average_energy_trace.size() == schedule.size());
    assert(result.average_magnetization_trace.size() == schedule.size());

    return 0;
}
