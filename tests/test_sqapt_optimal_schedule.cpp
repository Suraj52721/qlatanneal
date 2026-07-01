// Regression test for the budget-calibrated optimal J_perp schedule (run_optimal).
//
// History: the adaptive schedule used a caller-supplied constant eps_tilde with no tie to
// the step budget. When chi_B was large/flat the per-step delta_j was tiny and J_perp barely
// moved over the whole run ("frozen schedule") — invalidating opt-vs-std benchmarks. The fix
// calibrates eps_tilde from a pilot pre-pass so the schedule consumes its full num_steps
// budget and reaches j_perp_end by construction.
//
// This test asserts:
//   (1) calibration produces a positive eps_tilde and a non-decreasing trace,
//   (2) the calibrated schedule traverses (near) the full [start, end] range within budget,
//   (3) a legacy fixed tiny eps_tilde does NOT (demonstrating the bug the fix removes),
//   (4) chi_B is recorded and actually varies across the run (Task 1.2 diagnostic intent).

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "qanneal/dense_ising.hpp"
#include "qanneal/sqa_parallel_tempering.hpp"

namespace {

// A dense Ising instance with strong, mixed-sign couplings so the imaginary-time bond
// statistics (chi_B) are non-trivial across the J_perp range.
qanneal::DenseIsing make_dense_problem(std::size_t n) {
    std::vector<double> h(n, 0.0);
    std::vector<double> J(n * n, 0.0);
    // Deterministic pseudo-random couplings in roughly [-3, 3].
    std::uint64_t s = 0x9e3779b97f4a7c15ULL;
    auto nextd = [&]() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (static_cast<double>(s % 100000) / 100000.0 - 0.5) * 6.0;
    };
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double w = nextd();
            J[i * n + j] = w;
            J[j * n + i] = w;
        }
    }
    return qanneal::DenseIsing(h, J, n);
}

} // namespace

int main() {
    const std::size_t n = 10;
    qanneal::DenseIsing ham = make_dense_problem(n);

    std::vector<double> betas = {0.2, 0.5, 1.0, 2.0};
    std::vector<double> gammas = {2.0, 1.0, 0.5, 0.25};
    const std::size_t slices = 8;
    const std::size_t num_steps = 150;
    const double j_perp_end = 4.0;

    // --- (A) Calibrated schedule (eps_tilde <= 0 requests budget calibration) ---
    qanneal::SQAParallelTemperingAnnealer cal(ham, betas, gammas, slices);
    cal.set_seed(2024);
    auto rc = cal.run_optimal(num_steps, /*sweeps_per_step=*/8, /*worldline_sweeps=*/0,
                              /*eps_tilde=*/0.0, /*alpha=*/15.0 / 14.0,
                              /*j_perp_end=*/j_perp_end, /*cluster_sweeps=*/0,
                              /*swap_interval=*/1, /*continuous_time_slices=*/0,
                              /*calib_probes=*/12, /*calib_sweeps=*/8);

    assert(rc.calibrated_eps_tilde > 0.0);
    assert(rc.resolved_j_perp_end == j_perp_end);
    assert(!rc.j_perp_trace.empty());
    assert(rc.chi_B_trace.size() == rc.j_perp_trace.size());
    // The optimal run must consume its full step budget (no early break) so its total sweep
    // count matches the standard schedule — essential for a fair opt-vs-std benchmark.
    assert(rc.j_perp_trace.size() == num_steps);

    // Trace starts at the resolved start and is monotone non-decreasing.
    assert(std::abs(rc.j_perp_trace.front() - rc.j_perp_start) < 1e-9);
    for (std::size_t i = 1; i < rc.j_perp_trace.size(); ++i) {
        assert(rc.j_perp_trace[i] >= rc.j_perp_trace[i - 1] - 1e-12);
    }

    const double range = j_perp_end - rc.j_perp_start;
    assert(range > 0.5);  // sanity: the test ladder leaves real room to anneal
    // The j_perp_trace records the PRE-update value at each step; final_j_perp is the value
    // after the last update. Use final_j_perp to measure how far the schedule actually reached.
    const double reached = rc.final_j_perp;
    const double frac = (reached - rc.j_perp_start) / range;
    std::printf("[calibrated] eps_tilde=%.6g start=%.4f end=%.4f reached=%.4f frac=%.3f steps=%zu\n",
                rc.calibrated_eps_tilde, rc.j_perp_start, j_perp_end, reached, frac,
                rc.j_perp_trace.size());
    // Budget calibration must traverse (near) the whole range within budget.
    assert(frac > 0.9);

    // chi_B must actually vary across the run (a flat chi_B was the symptom of the bug).
    double chi_min = 1e300, chi_max = -1e300;
    for (double c : rc.chi_B_trace) { chi_min = std::min(chi_min, c); chi_max = std::max(chi_max, c); }
    assert(chi_max > chi_min);

    // --- (B) Legacy fixed tiny eps_tilde freezes (demonstrates the bug the fix removes) ---
    qanneal::SQAParallelTemperingAnnealer fix(ham, betas, gammas, slices);
    fix.set_seed(2024);
    auto rf = fix.run_optimal(num_steps, 8, 0, /*eps_tilde=*/1e-4, 15.0 / 14.0,
                              j_perp_end, 0, 1, 0);
    const double reached_fixed = rf.final_j_perp;
    const double frac_fixed = (reached_fixed - rf.j_perp_start) / range;
    std::printf("[legacy-fixed] eps_tilde=%.6g reached=%.4f frac=%.3f\n",
                rf.calibrated_eps_tilde, reached_fixed, frac_fixed);
    // The frozen schedule barely moves; calibration must traverse dramatically further.
    assert(frac_fixed < 0.5);
    assert(frac > frac_fixed + 0.3);

    std::printf("test_sqapt_optimal_schedule: OK\n");
    return 0;
}
