#include <iostream>

#include "qanneal/annealer.hpp"
#include "qanneal/dense_ising.hpp"
#include "qanneal/schedule.hpp"

int main() {
    const std::size_t n = 4;

    std::vector<double> h = {0.1, -0.2, 0.0, 0.3};
    std::vector<double> J(n * n, 0.0);
    J[0 * n + 1] = 0.5;
    J[1 * n + 0] = 0.5;
    J[2 * n + 3] = -0.4;
    J[3 * n + 2] = -0.4;

    qanneal::DenseIsing ham(h, J, n);
    auto schedule = qanneal::AnnealSchedule::linear(0.1, 2.0, 50);
    qanneal::Annealer annealer(ham, schedule);

    auto result = annealer.run(100);
    std::cout << "Best energy: " << result.best_energy << "\n";
    return 0;
}
