#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "qanneal/backend.hpp"
#include "qanneal/sqa_observer.hpp"
#include "qanneal/sqa_schedule.hpp"
#include "qanneal/sqa_state.hpp"
#include "qanneal/state.hpp"

namespace qanneal {

struct SQAResult {
    State best_state;
    double best_energy = 0.0;
    std::vector<double> energy_trace;
    std::vector<double> j_perp_trace;  // populated by run_optimal; empty for run()
};

class SQAAnnealer {
public:
    SQAAnnealer(const Hamiltonian &hamiltonian,
                SQASchedule schedule,
                std::size_t trotter_slices,
                std::size_t replicas = 1);
    SQAAnnealer(std::shared_ptr<Backend> backend,
                SQASchedule schedule,
                std::size_t trotter_slices,
                std::size_t replicas = 1);

    void set_seed(std::uint64_t seed);

    SQAResult run(std::size_t sweeps_per_beta,
                  std::size_t worldline_sweeps,
                  std::size_t cluster_sweeps = 0,
                  std::size_t continuous_time_slices = 0,
                  SQAObserver *observer = nullptr);

    // Adaptive optimal schedule (Roland-Cerf SQA analogue from the local adiabaticity derivation).
    // j_perp increases from j_perp_start toward j_perp_end; beta is fixed throughout.
    // Step size: delta_j = eps_tilde * chi_B^(-alpha), where chi_B = Var(B) across replicas.
    // alpha = z/(2-eta) + 1/2; for 1-D quantum Ising universality class: alpha = 15/14.
    SQAResult run_optimal(double beta,
                          double j_perp_start,
                          double j_perp_end,
                          double eps_tilde,
                          double alpha,
                          std::size_t num_steps,
                          std::size_t sweeps_per_step,
                          std::size_t worldline_sweeps = 0,
                          std::size_t cluster_sweeps = 0);

private:
    std::shared_ptr<Backend> backend_;
    SQASchedule schedule_;
    std::size_t slices_ = 0;
    std::size_t replicas_ = 0;
    std::mt19937_64 rng_;

    double trotter_coupling(double beta, double gamma, std::size_t slices) const;
    double delta_trotter(const SQAState &state,
                         std::size_t replica,
                         std::size_t slice,
                         std::size_t spin,
                         double j_perp,
                         std::size_t slices) const;
};

}
