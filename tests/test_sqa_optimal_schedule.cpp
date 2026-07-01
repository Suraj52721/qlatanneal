// Regression test for the budget-calibrated optimal J_perp schedule in SQAAnnealer::run_optimal.
//
// This mirrors test_sqapt_optimal_schedule.cpp. The SQA path originally required eps_tilde > 0
// and used the online update delta_j = eps_tilde * chi_B^(-alpha) with a caller-supplied
// constant — the same "frozen schedule" bug that was fixed for SQAPT. The calibration was later
// ported to SQA; this test pins that port down:
//   (1) eps_tilde <= 0 requests calibration and yields a positive calibrated eps_tilde,
//   (2) the calibrated schedule traverses (near) the full [start, end] range within num_steps,
//   (3) a legacy fixed tiny eps_tilde does NOT (the bug), and
//   (4) chi_B is recorded and varies across the run.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "qanneal/dense_ising.hpp"
#include "qanneal/sqa_annealer.hpp"
#include "qanneal/sqa_schedule.hpp"

namespace {

qanneal::DenseIsing make_dense_problem(std::size_t n) {
    std::vector<double> h(n, 0.0);
    std::vector<double> J(n * n, 0.0);
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

    auto sched = qanneal::SQASchedule::from_vectors({1.0}, {1.0});
    const std::size_t slices = 8;
    const std::size_t replicas = 4;
    const std::size_t num_steps = 150;
    const double beta = 1.0;
    const double j_perp_start = 0.5;
    const double j_perp_end = 5.0;
    const double range = j_perp_end - j_perp_start;

    // --- (A) Calibrated schedule (eps_tilde <= 0 requests budget calibration) ---
    qanneal::SQAAnnealer cal(ham, sched, slices, replicas);
    cal.set_seed(2024);
    auto rc = cal.run_optimal(beta, j_perp_start, j_perp_end, /*eps_tilde=*/0.0,
                              /*alpha=*/15.0 / 14.0, num_steps, /*sweeps_per_step=*/8,
                              /*worldline_sweeps=*/0, /*cluster_sweeps=*/0,
                              /*calib_probes=*/12, /*calib_sweeps=*/8);

    assert(rc.calibrated_eps_tilde > 0.0);
    assert(rc.resolved_j_perp_end == j_perp_end);
    assert(std::abs(rc.j_perp_start - j_perp_start) < 1e-9);
    assert(!rc.j_perp_trace.empty());
    assert(rc.chi_B_trace.size() == rc.j_perp_trace.size());
    // No early break: the run consumes its full budget for a fair opt-vs-std comparison.
    assert(rc.j_perp_trace.size() == num_steps);

    assert(std::abs(rc.j_perp_trace.front() - rc.j_perp_start) < 1e-9);
    for (std::size_t i = 1; i < rc.j_perp_trace.size(); ++i) {
        assert(rc.j_perp_trace[i] >= rc.j_perp_trace[i - 1] - 1e-12);
    }

    const double reached = rc.final_j_perp;
    const double frac = (reached - rc.j_perp_start) / range;
    std::printf("[sqa calibrated] eps_tilde=%.6g start=%.4f end=%.4f reached=%.4f frac=%.3f steps=%zu\n",
                rc.calibrated_eps_tilde, rc.j_perp_start, j_perp_end, reached, frac,
                rc.j_perp_trace.size());
    assert(frac > 0.9);

    double chi_min = 1e300, chi_max = -1e300;
    for (double c : rc.chi_B_trace) { chi_min = std::min(chi_min, c); chi_max = std::max(chi_max, c); }
    assert(chi_max > chi_min);

    // --- (B) Legacy fixed tiny eps_tilde freezes (demonstrates the bug the fix removes) ---
    qanneal::SQAAnnealer fix(ham, sched, slices, replicas);
    fix.set_seed(2024);
    auto rf = fix.run_optimal(beta, j_perp_start, j_perp_end, /*eps_tilde=*/1e-4,
                              15.0 / 14.0, num_steps, 8, 0, 0);
    const double reached_fixed = rf.final_j_perp;
    const double frac_fixed = (reached_fixed - rf.j_perp_start) / range;
    std::printf("[sqa legacy-fixed] eps_tilde=%.6g reached=%.4f frac=%.3f\n",
                rf.calibrated_eps_tilde, reached_fixed, frac_fixed);
    assert(frac_fixed < 0.5);
    assert(frac > frac_fixed + 0.3);

    std::printf("test_sqa_optimal_schedule: OK\n");
    return 0;
}
