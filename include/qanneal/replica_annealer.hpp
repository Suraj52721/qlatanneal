#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "qanneal/backend.hpp"
#include "qanneal/metrics.hpp"
#include "qanneal/schedule.hpp"
#include "qanneal/state.hpp"

namespace qanneal {

struct ReplicaResult {
    State best_state;
    double best_energy = 0.0;
    std::vector<double> energy_trace;
    std::vector<double> magnetization_trace;
};

struct MultiAnnealResult {
    std::vector<ReplicaResult> replicas;
    State global_best_state;
    double global_best_energy = 0.0;
    std::vector<double> average_energy_trace;
    std::vector<double> average_magnetization_trace;
};

class ReplicaAnnealer {
public:
    ReplicaAnnealer(const Hamiltonian &hamiltonian,
                    AnnealSchedule schedule,
                    std::size_t replicas);
    ReplicaAnnealer(std::shared_ptr<Backend> backend,
                    AnnealSchedule schedule,
                    std::size_t replicas);

    void set_seed(std::uint64_t seed);

    MultiAnnealResult run(std::size_t sweeps_per_beta);

private:
    std::shared_ptr<Backend> backend_;
    AnnealSchedule schedule_;
    std::size_t replicas_ = 0;
    std::mt19937_64 rng_;
};

}
