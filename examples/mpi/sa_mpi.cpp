#include <iostream>

#include "qanneal/dense_ising.hpp"
#include "qanneal/mpi/mpi_context.hpp"
#include "qanneal/mpi/mpi_replica.hpp"
#include "qanneal/schedule.hpp"

int main(int argc, char **argv) {
    qanneal::mpi::MPIContext mpi(argc, argv);

    const std::size_t n = 4;
    std::vector<double> h = {0.1, -0.2, 0.0, 0.3};
    std::vector<double> J(n * n, 0.0);
    J[0 * n + 1] = 0.5;
    J[1 * n + 0] = 0.5;
    J[2 * n + 3] = -0.4;
    J[3 * n + 2] = -0.4;

    qanneal::DenseIsing ham(h, J, n);
    auto schedule = qanneal::AnnealSchedule::linear(0.1, 2.0, 20);

    auto summary = qanneal::mpi::run_replica_anneal(
        ham, schedule, /*sweeps_per_beta=*/50, /*replicas_per_rank=*/2, /*seed=*/42);

    if (summary.rank == 0) {
        std::cout << "Global best energy: " << summary.global_best_energy << "\n";
    }

    return 0;
}
