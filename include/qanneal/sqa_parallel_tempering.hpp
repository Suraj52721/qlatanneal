#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
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
    // Diagnostics for the optimal adaptive schedule (run_optimal only; empty otherwise).
    std::vector<double> j_perp_trace;  // realized J_perp at each step
    std::vector<double> chi_B_trace;   // measured chi_B = Var(B) at each step
    double calibrated_eps_tilde = 0.0; // eps_tilde actually used (post-calibration)
    double j_perp_start = 0.0;         // resolved starting J_perp
    double resolved_j_perp_end = 0.0;  // resolved target J_perp (after sentinel/fallback logic)
    double final_j_perp = 0.0;         // J_perp after the last step (reaches resolved end when calibrated)
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
    //
    // Budget calibration:  if eps_tilde <= 0 (the default), a short pilot pre-pass
    // measures chi_B(J_perp) at `calib_probes` points spanning [j_perp_start, j_perp_end]
    // (using `calib_sweeps` sweeps each) and sets eps_tilde so the schedule consumes its
    // full `num_steps` budget and reaches j_perp_end by construction:
    //     eps_tilde = (1/num_steps) * integral_{J0}^{Jend} chi_B(J)^alpha dJ.
    // A positive eps_tilde is treated as an explicit override and skips calibration.
    // If `debug_csv_path` is non-empty, per-step diagnostics
    // (phase,step_index,j_perp,chi_B,delta_j,eps_tilde,alpha,floor_hit) are written there.
    SQAParallelTemperingResult run_optimal(std::size_t num_steps,
                                          std::size_t sweeps_per_step,
                                          std::size_t worldline_sweeps,
                                          double eps_tilde,
                                          double alpha = 15.0 / 14.0,
                                          double j_perp_end = 0.0,
                                          std::size_t cluster_sweeps = 0,
                                          std::size_t swap_interval = 1,
                                          std::size_t continuous_time_slices = 0,
                                          std::size_t calib_probes = 12,
                                          std::size_t calib_sweeps = 10,
                                          const std::string &debug_csv_path = "");

private:
    std::shared_ptr<Backend> backend_;
    std::vector<double> betas_;
    std::vector<double> gammas_;
    std::size_t slices_ = 0;
    std::mt19937_64 rng_;

    double trotter_coupling(double beta, double gamma, std::size_t slices) const;
};

} // namespace qanneal
