#include "qanneal/mpi/mpi_replica.hpp"

#include <limits>
#include <stdexcept>

namespace qanneal::mpi {

MPIReplicaSummary run_replica_anneal(const Hamiltonian &hamiltonian,
                                     AnnealSchedule schedule,
                                     std::size_t sweeps_per_beta,
                                     std::size_t replicas_per_rank,
                                     std::uint64_t seed,
                                     MPI_Comm comm) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    ReplicaAnnealer annealer(hamiltonian, std::move(schedule), replicas_per_rank);
    if (seed != 0) {
        annealer.set_seed(seed + static_cast<std::uint64_t>(rank));
    }
    const auto local_result = annealer.run(sweeps_per_beta);

    MPIReplicaSummary summary;
    summary.rank = rank;
    summary.size = size;
    summary.local_best_energy = local_result.global_best_energy;
    summary.local_best_state = local_result.global_best_state;

    struct {
        double energy;
        int rank;
    } local, global;

    local.energy = local_result.global_best_energy;
    local.rank = rank;

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);

    summary.global_best_energy = global.energy;
    summary.global_best_state = State(hamiltonian.size());

    if (rank == global.rank) {
        summary.global_best_state = local_result.global_best_state;
    }

    const int n = static_cast<int>(hamiltonian.size());
    MPI_Bcast(summary.global_best_state.spins.data(), n, MPI_SIGNED_CHAR, global.rank, comm);

    return summary;
}

} // namespace qanneal::mpi
