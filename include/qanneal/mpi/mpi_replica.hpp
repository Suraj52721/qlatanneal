#pragma once

#include <cstddef>
#include <cstdint>

#include <mpi.h>

#include "qanneal/replica_annealer.hpp"
#include "qanneal/state.hpp"

namespace qanneal::mpi {

struct MPIReplicaSummary {
    int rank = 0;
    int size = 1;
    double local_best_energy = 0.0;
    State local_best_state;
    double global_best_energy = 0.0;
    State global_best_state;
};

MPIReplicaSummary run_replica_anneal(const Hamiltonian &hamiltonian,
                                     AnnealSchedule schedule,
                                     std::size_t sweeps_per_beta,
                                     std::size_t replicas_per_rank,
                                     std::uint64_t seed = 0,
                                     MPI_Comm comm = MPI_COMM_WORLD);

} // namespace qanneal::mpi
