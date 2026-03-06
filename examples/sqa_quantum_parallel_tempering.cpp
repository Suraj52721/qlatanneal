#include <iostream>
#include <vector>

#include "qanneal/dense_ising.hpp"
#include "qanneal/sqa_parallel_tempering.hpp"

int main() {
    const std::size_t n = 6;
    std::vector<double> h(n, 0.0);
    std::vector<double> J(n * n, 0.0);

    auto set_sym = [&](std::size_t i, std::size_t j, double v) {
        J[i * n + j] = v;
        J[j * n + i] = v;
    };

    set_sym(0, 1, 0.7);
    set_sym(1, 2, -0.9);
    set_sym(2, 3, 0.4);
    set_sym(3, 4, -0.8);
    set_sym(4, 5, 0.6);
    set_sym(0, 5, -0.5);

    qanneal::DenseIsing ising(h, J, n, 0.0);

    std::vector<double> betas = {0.2, 0.35, 0.6, 0.95, 1.4, 2.0};
    std::vector<double> gammas = {3.0, 2.2, 1.6, 1.0, 0.45, 0.15};

    qanneal::SQAParallelTemperingAnnealer annealer(ising, betas, gammas, 24);
    annealer.set_seed(42);

    auto result = annealer.run(
        /*sweeps_per_step=*/24,
        /*worldline_sweeps=*/4,
        /*steps=*/80,
        /*swap_interval=*/1,
        /*cluster_sweeps=*/1,
        /*continuous_time_slices=*/0
    );

    std::cout << "Best energy: " << result.best_energy << "\n";
    std::cout << "Final swap acceptance: "
              << (result.swap_acceptance_trace.empty() ? 0.0 : result.swap_acceptance_trace.back())
              << "\n";
    std::cout << "Best spins: ";
    for (auto s : result.best_state.spins) {
        std::cout << static_cast<int>(s) << " ";
    }
    std::cout << "\n";
    return 0;
}
