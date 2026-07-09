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

// Result of run_surrogate: the bond-susceptibility surrogate schedule
// (chi_B-weighted time allocation; see Singh et al., "Bond Susceptibility as a
// Surrogate for Spectral Gaps in Quantum Annealing Schedule Design").
struct SQASurrogateResult {
    State best_state;
    double best_energy = 0.0;
    std::vector<double> energy_trace;      // per-step average energy of the main anneal
    // Resolved per-step schedule actually run (phase-1 beta ramp + phase-2 surrogate gammas).
    std::vector<double> beta_schedule;
    std::vector<double> gamma_schedule;
    std::vector<double> s_schedule;        // annealing parameter s = A0/(A0+gamma) per step
    // Pilot chi_B scan on a uniform s grid (the surrogate profile).
    std::vector<double> scan_s;
    std::vector<double> scan_gamma;
    std::vector<double> scan_chi_B;
    // Quantum-critical-point estimate from the chi_B peak.
    double s_star = 0.0;
    double gamma_star = 0.0;
    double j_perp_star = 0.0;
    double chi0 = 0.0;                     // regularization chi_0 actually used
    double driver_A0 = 0.0;                // resolved driver scale A0
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

    // Bond-susceptibility SURROGATE schedule (two-stage, from the paper):
    //   Stage 1 (pilot): scan chi_B = Var(B) on a uniform grid of the annealing parameter
    //     s = A0/(A0+gamma) between s(gamma_start) and s(gamma_end), carrying the worldline
    //     state along the scan.  chi_B peaks at the quantum critical point, playing the role
    //     of 1/Delta(s)^2 in the Roland-Cerf condition.
    //   Stage 2 (schedule): allocate the num_steps time budget with weight
    //     w(s) = chi_B(s) + chi_0,  tau(s) = int_0^s w / int_0^1 w,  s_k = tau^{-1}(k/(n-1)),
    //     then run a fresh standard SQA anneal along gamma_k = A0*(1-s_k)/s_k at fixed beta.
    //   chi_0 = chi0_fraction * max(chi_B) keeps the velocity finite away from the peak
    //   (the paper's regularization); driver_A0 <= 0 resolves to gamma_start.
    // A phase-1 thermal ramp (beta_ramp_start -> beta at gamma_start, beta_ramp_fraction of
    // the budget) precedes the surrogate gamma schedule, mirroring run_optimal's protocol.
    SQASurrogateResult run_surrogate(double beta,
                                     double gamma_start,
                                     double gamma_end,
                                     std::size_t num_steps,
                                     std::size_t sweeps_per_step,
                                     std::size_t worldline_sweeps = 0,
                                     std::size_t cluster_sweeps = 0,
                                     std::size_t scan_points = 16,
                                     std::size_t scan_sweeps = 30,
                                     std::size_t scan_burn = 10,
                                     double chi0_fraction = 0.05,
                                     double driver_A0 = 0.0,
                                     double beta_ramp_fraction = 0.3,
                                     double beta_ramp_start = 0.1,
                                     const std::string &debug_csv_path = "");

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
