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

    // Adaptive optimal schedule for SQAPT (and SQAPT+SW when cluster_sweeps > 0).
    // All replicas share a single j_perp that evolves via the local adiabaticity ODE.
    // Each replica keeps its own fixed beta; PT swaps reduce to a purely classical
    // criterion (Trotter terms cancel when j_perp is shared).
    // j_perp starts from trotter_coupling(betas_[0], gammas_[0], slices) and increases
    // toward j_perp_end.  alpha = z/(2-eta) + 1/2; default = 15/14 (1-D quantum Ising).
    SQAParallelTemperingResult run_optimal(std::size_t num_steps,
                                          std::size_t sweeps_per_step,
                                          std::size_t worldline_sweeps,
                                          double eps_tilde,
                                          double alpha = 15.0 / 14.0,
                                          double j_perp_end = 0.0,
                                          std::size_t cluster_sweeps = 0,
                                          std::size_t swap_interval = 1,
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
