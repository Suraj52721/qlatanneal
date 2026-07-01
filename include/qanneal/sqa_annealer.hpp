#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
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
    // Budget-calibration diagnostics (run_optimal with eps_tilde <= 0). Mirror SQAPT.
    std::vector<double> chi_B_trace;        // driving chi_B at each step
    double calibrated_eps_tilde = 0.0;      // eps_tilde actually used (post-calibration)
    double j_perp_start = 0.0;              // resolved start J_perp
    double resolved_j_perp_end = 0.0;       // resolved target J_perp
    double final_j_perp = 0.0;              // J_perp after the last update (measures reach)
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
    // alpha = z/(2-eta) + 1/2; for 1-D quantum Ising universality class: alpha = 15/14.
    //
    // eps_tilde <= 0 (recommended) triggers BUDGET CALIBRATION: a pilot pre-pass measures
    // chi_B(J) at `calib_probes` points (`calib_sweeps` sweeps each), sets eps_tilde =
    // (1/num_steps) * integral chi_B^alpha dJ, and drives j_perp along the precomputed
    // inverse-CDF trajectory so the run traverses [j_perp_start, j_perp_end] in exactly
    // num_steps (dwelling where chi_B is large). A positive eps_tilde keeps the legacy online
    // update delta_j = eps_tilde * chi_B^(-alpha). debug_csv_path (if set) writes per-step
    // diagnostics. This mirrors SQAParallelTemperingAnnealer::run_optimal.
    SQAResult run_optimal(double beta,
                          double j_perp_start,
                          double j_perp_end,
                          double eps_tilde,
                          double alpha,
                          std::size_t num_steps,
                          std::size_t sweeps_per_step,
                          std::size_t worldline_sweeps = 0,
                          std::size_t cluster_sweeps = 0,
                          std::size_t calib_probes = 12,
                          std::size_t calib_sweeps = 10,
                          const std::string &debug_csv_path = "",
                          // Two-phase protocol: phase 1 ramps beta from beta_ramp_start to `beta`
                          // (thermal anneal) at fixed j_perp_start for beta_ramp_fraction*num_steps
                          // steps; phase 2 fixes beta and runs the adaptive J_perp schedule for the
                          // rest. beta_ramp_fraction <= 0 recovers single-phase behaviour.
                          double beta_ramp_fraction = 0.3,
                          double beta_ramp_start = 0.1);

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
