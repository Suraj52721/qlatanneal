#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "qanneal/backend.hpp"
#include "qanneal/sqa_state.hpp"
#include "qanneal/state.hpp"

namespace qanneal {

// Result of SQAChiAnnealer::run_chi: the worldline-magnetization susceptibility
// schedule (method "sqa_chi"). See sqa_chi_annealer.hpp for the algorithm.
struct SQAChiResult {
    State best_state;
    double best_energy = 0.0;
    std::vector<double> energy_trace;      // per-step average energy of the main anneal
    // Resolved per-step schedule actually run (phase-1 beta ramp + phase-2 chi-weighted gammas).
    std::vector<double> beta_schedule;
    std::vector<double> gamma_schedule;
    std::vector<double> s_schedule;        // annealing parameter s = A0/(A0+gamma) per step
    // Pilot chi_B scan on a uniform s grid.
    std::vector<double> scan_s;
    std::vector<double> scan_gamma;
    std::vector<double> scan_chi_B;        // n*M*(<m^2> - <|m|>^2), pooled over replicas
    // Quantum-critical-point estimate from the chi_B peak.
    double s_star = 0.0;
    double gamma_star = 0.0;
    double j_perp_star = 0.0;
    double chi_floor = 0.0;                // floor actually applied to chi_B before integrating
    double driver_A0 = 0.0;                // resolved driver scale A0
};

// SQAChiAnnealer: worldline (path-integral) QMC solver whose annealing
// trajectory is built from the susceptibility chi_B of the worldline
// magnetization order parameter, m = (1/(n*M)) * sum_{i,k} sigma_{i,k}.
//
// This is method "sqa_chi" -- a standalone solver alongside "sa"/"sqa"/"sqapt",
// generalized from a TFIM validation script (dense J, hardcoded checkerboard
// update) to qanneal's Backend abstraction (dense or sparse Ising, any graph).
//
// Suzuki-Trotter mapping (TFIM at inverse temperature beta, M Trotter slices):
//   J_perp = (1/2) ln(1/tanh(beta*Gamma/M))                         (trotter_coupling)
//   E_cl   = (beta/M) * sum_k H_classical(sigma_k) - J_perp * sum_{i,k} sigma_{i,k} sigma_{i,k+1}
// Order parameter and susceptibility (peaks at the quantum critical point,
// mirroring 1/Delta(s)^2 in the Roland-Cerf adiabatic condition):
//   m     = (1/(n*M)) sum_{i,k} sigma_{i,k}
//   chi_B = n*M * ( <m^2> - <|m|>^2 )
// Schedule: t(s)/T = integral_0^s (chi_B(s') + floor) ds' / integral_0^1 (...) ds'.
//
// Update kernel: a *parity-parallel checkerboard* sweep over Trotter slices.
// Slices of the same parity (even/odd index) never interact directly (the
// Trotter coupling only links slice k to k+/-1), so all replica x slice cells
// of one parity can be swept fully independently and in parallel; each cell
// runs an exact sequential single-spin Metropolis sweep over its own n spins
// via the generalized Backend interface (dense or sparse J), using cached
// local fields for O(1) (dense) / O(degree) (sparse) delta-energies. This is
// an exact-detailed-balance generalization of the reference script's
// same-slice parallel (Gibbs-style) update, which is only exact for bipartite
// spatial couplings; the per-slice sequential update here is exact for any
// coupling graph while still parallelizing across (replica, slice) cells.
class SQAChiAnnealer {
public:
    SQAChiAnnealer(const Hamiltonian &hamiltonian,
                   std::size_t trotter_slices,
                   std::size_t replicas = 1);
    SQAChiAnnealer(std::shared_ptr<Backend> backend,
                   std::size_t trotter_slices,
                   std::size_t replicas = 1);

    void set_seed(std::uint64_t seed);

    // Stage 1 (pilot): scan chi_B on `scan_points` values of s = A0/(A0+gamma),
    //   uniform between s(gamma_start) and s(gamma_end); scan_burn+scan_sweeps
    //   parity-parallel sweeps per point, on a FRESH random worldline each
    //   point (independent probes -- no state carried between grid points).
    // Stage 2 (schedule): weight w(s) = max(chi_B(s), floor), floor =
    //   max(chi_floor_fraction * max(chi_B), chi_floor_abs); allocate the
    //   num_steps time budget via tau(s) = int w / int w, inverted at uniform
    //   targets (inverse-CDF time allocation over the annealing parameter s).
    // Stage 3 (production): a phase-1 beta ramp (beta_ramp_start -> beta at
    //   fixed gamma_start, beta_ramp_fraction of the budget) followed by the
    //   phase-2 chi-weighted gamma(t) schedule, on a fresh worldline, run
    //   through the same parity-parallel kernel as the pilot.
    SQAChiResult run_chi(double beta,
                         double gamma_start,
                         double gamma_end,
                         std::size_t num_steps,
                         std::size_t sweeps_per_step,
                         std::size_t scan_points = 16,
                         std::size_t scan_sweeps = 30,
                         std::size_t scan_burn = 10,
                         double chi_floor_fraction = 1e-6,
                         double driver_A0 = 0.0,
                         double beta_ramp_fraction = 0.3,
                         double beta_ramp_start = 0.1,
                         const std::string &debug_csv_path = "");

private:
    std::shared_ptr<Backend> backend_;
    std::size_t slices_ = 0;
    std::size_t replicas_ = 0;
    std::mt19937_64 rng_;

    double trotter_coupling(double beta, double gamma, std::size_t slices) const;

    // Precomputed (replica, slice) cell indices for each Trotter parity class
    // (0 = even slice index, 1 = odd). Built once per run_chi() call and
    // reused across every sweep so the parallel-for target doesn't get
    // rebuilt (and reallocated) on every single sweep.
    using CellList = std::vector<std::pair<std::size_t, std::size_t>>;
    void build_parity_cells(std::size_t slices, CellList out[2]) const;

    // One full parity-parallel checkerboard sweep (both parities) over `state`,
    // updating `local_fields` and `cell_rng` in place. Returns the pooled
    // worldline magnetization m for each replica via `m_out` (resized to
    // replicas_; m_out[r] = mean spin over that replica's n*slices lattice
    // AFTER the sweep) so callers can accumulate <m^2>, <|m|> without a
    // second pass over the lattice.
    void checkerboard_sweep(SQAState &state,
                            std::vector<double> &local_fields,
                            std::vector<std::mt19937_64> &cell_rng,
                            const CellList parity_cells[2],
                            double beta_scale,
                            double j_perp,
                            std::size_t slices,
                            std::size_t n,
                            std::vector<double> &m_out) const;
};

}
