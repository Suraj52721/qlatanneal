#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "qanneal/backend.hpp"
#include "qanneal/state.hpp"

namespace qanneal {

struct ParallelTemperingResult {
    std::vector<State> final_states;
    std::vector<double> final_energies;
    State best_state;
    double best_energy = 0.0;
    std::vector<double> average_energy_trace;
    std::vector<double> swap_acceptance_trace;
};

class ParallelTemperingAnnealer {
public:
    ParallelTemperingAnnealer(const Hamiltonian &hamiltonian,
                              std::vector<double> betas);
    ParallelTemperingAnnealer(std::shared_ptr<Backend> backend,
                              std::vector<double> betas);

    void set_seed(std::uint64_t seed);

    ParallelTemperingResult run(std::size_t sweeps_per_step,
                                std::size_t steps,
                                std::size_t swap_interval = 1);

private:
    std::shared_ptr<Backend> backend_;
    std::vector<double> betas_;
    std::mt19937_64 rng_;
};

} // namespace qanneal
