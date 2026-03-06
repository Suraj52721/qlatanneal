#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "qanneal/backend.hpp"
#include "qanneal/state.hpp"

namespace qanneal {

struct SQAParallelTemperingResult {
    std::vector<State> final_states;
    std::vector<double> final_energies;
    State best_state;
    double best_energy = 0.0;
    std::vector<double> average_energy_trace;
    std::vector<double> swap_acceptance_trace;
};

class SQAParallelTemperingAnnealer {
public:
    SQAParallelTemperingAnnealer(const Hamiltonian &hamiltonian,
                                 std::vector<double> betas,
                                 std::vector<double> gammas,
                                 std::size_t trotter_slices);

    SQAParallelTemperingAnnealer(std::shared_ptr<Backend> backend,
                                 std::vector<double> betas,
                                 std::vector<double> gammas,
                                 std::size_t trotter_slices);

    void set_seed(std::uint64_t seed);

    SQAParallelTemperingResult run(std::size_t sweeps_per_step,
                                   std::size_t worldline_sweeps,
                                   std::size_t steps,
                                   std::size_t swap_interval = 1,
                                   std::size_t cluster_sweeps = 0,
                                   std::size_t continuous_time_slices = 0);

private:
    std::shared_ptr<Backend> backend_;
    std::vector<double> betas_;
    std::vector<double> gammas_;
    std::size_t slices_ = 0;
    std::mt19937_64 rng_;

    double trotter_coupling(double beta, double gamma, std::size_t slices) const;
};

} // namespace qanneal
