// Test for the bond-susceptibility SURROGATE schedule in SQAAnnealer::run_surrogate.
//
// The surrogate method (paper: "Bond Susceptibility as a Surrogate for Spectral Gaps in
// Quantum Annealing Schedule Design") is a two-stage protocol:
//   1. pilot scan of chi_B = Var(B) on a uniform s-grid (s = A0/(A0+gamma)),
//   2. time allocation with weight w(s) = chi_B(s) + chi_0 via the cumulative integral,
//      then a fresh standard SQA anneal along the resulting gamma(t).
// This test pins down:
//   (1) diagnostics have the right shapes and the chi_B profile is non-trivial,
//   (2) the QCP estimate (chi_B peak) lies inside the scanned s-range,
//   (3) the gamma schedule is monotone non-increasing and spans [gamma_end, gamma_start],
//   (4) the schedule dwells near the chi_B peak (denser steps than average),
//   (5) the anneal actually optimizes: exact ground state of a ferromagnetic chain.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "qanneal/dense_ising.hpp"
#include "qanneal/sqa_annealer.hpp"
#include "qanneal/sqa_schedule.hpp"

namespace {

// Ferromagnetic chain: J_{i,i+1} = -1, ground state all-up/all-down, E0 = -(n-1).
qanneal::DenseIsing make_ferro_chain(std::size_t n) {
    std::vector<double> h(n, 0.0);
    std::vector<double> J(n * n, 0.0);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        J[i * n + (i + 1)] = -1.0;
        J[(i + 1) * n + i] = -1.0;
    }
    return qanneal::DenseIsing(h, J, n);
}

qanneal::DenseIsing make_sk_problem(std::size_t n) {
    std::vector<double> h(n, 0.0);
    std::vector<double> J(n * n, 0.0);
    std::uint64_t s = 0x9e3779b97f4a7c15ULL;
    auto nextd = [&]() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (static_cast<double>(s % 100000) / 100000.0 - 0.5) * 2.0;
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
    const std::size_t slices = 8;
    const std::size_t replicas = 4;
    auto dummy_sched = qanneal::SQASchedule::from_vectors({1.0}, {1.0});

    // --- (A) Diagnostics + schedule structure on an SK instance ---
    {
        const std::size_t n = 10;
        qanneal::DenseIsing ham = make_sk_problem(n);
        qanneal::SQAAnnealer ann(ham, dummy_sched, slices, replicas);
        ann.set_seed(2026);

        const double beta = 4.0;
        const double gamma_start = 4.0;
        const double gamma_end = 0.05;
        const std::size_t num_steps = 120;
        const std::size_t scan_points = 12;

        auto res = ann.run_surrogate(beta, gamma_start, gamma_end, num_steps,
                                     /*sweeps_per_step=*/8, /*worldline_sweeps=*/2,
                                     /*cluster_sweeps=*/1, scan_points,
                                     /*scan_sweeps=*/24, /*scan_burn=*/8);

        // (1) shapes and profile sanity
        assert(res.scan_s.size() == scan_points);
        assert(res.scan_gamma.size() == scan_points);
        assert(res.scan_chi_B.size() == scan_points);
        assert(res.beta_schedule.size() == num_steps);
        assert(res.gamma_schedule.size() == num_steps);
        assert(res.s_schedule.size() == num_steps);
        assert(res.energy_trace.size() == num_steps);
        assert(res.chi0 > 0.0);
        assert(res.driver_A0 == gamma_start);

        double chi_min = 1e300, chi_max = -1e300;
        for (double c : res.scan_chi_B) {
            assert(c >= 0.0);
            chi_min = std::min(chi_min, c);
            chi_max = std::max(chi_max, c);
        }
        assert(chi_max > chi_min);  // profile is not flat

        // (2) QCP estimate inside the scanned range
        assert(res.s_star >= res.scan_s.front() && res.s_star <= res.scan_s.back());
        assert(res.gamma_star >= gamma_end - 1e-9 && res.gamma_star <= gamma_start + 1e-9);
        assert(res.j_perp_star > 0.0);

        // (3) monotone gamma schedule spanning the full range
        for (std::size_t i = 1; i < res.gamma_schedule.size(); ++i) {
            assert(res.gamma_schedule[i] <= res.gamma_schedule[i - 1] + 1e-12);
        }
        assert(std::abs(res.gamma_schedule.front() - gamma_start) < 1e-9);
        assert(std::abs(res.gamma_schedule.back() - gamma_end) < 1e-6);
        // s_schedule ascends over phase 2
        for (std::size_t i = 1; i < res.s_schedule.size(); ++i) {
            assert(res.s_schedule[i] >= res.s_schedule[i - 1] - 1e-12);
        }

        // (4) time allocation dwells near the chi_B peak: the average s-step
        // inside a window around s_star must not exceed the global average
        // (w(s) >= chi0 everywhere, and w(s_star) = max chi + chi0).
        const double s_lo = res.scan_s.front(), s_hi = res.scan_s.back();
        const double window = 0.15 * (s_hi - s_lo);
        double sum_all = 0.0, sum_peak = 0.0;
        std::size_t cnt_all = 0, cnt_peak = 0;
        for (std::size_t i = 1; i < res.s_schedule.size(); ++i) {
            const double ds = res.s_schedule[i] - res.s_schedule[i - 1];
            if (ds <= 0.0) continue;  // phase-1 beta ramp holds s constant
            sum_all += ds;
            ++cnt_all;
            const double mid = 0.5 * (res.s_schedule[i] + res.s_schedule[i - 1]);
            if (std::abs(mid - res.s_star) < window) {
                sum_peak += ds;
                ++cnt_peak;
            }
        }
        assert(cnt_all > 0 && cnt_peak > 0);
        const double mean_all = sum_all / static_cast<double>(cnt_all);
        const double mean_peak = sum_peak / static_cast<double>(cnt_peak);
        std::printf("[surrogate sk] s*=%.3f gamma*=%.3f chi0=%.4g mean_ds=%.5f mean_ds_peak=%.5f\n",
                    res.s_star, res.gamma_star, res.chi0, mean_all, mean_peak);
        assert(mean_peak <= mean_all * 1.05);
    }

    // --- (B) End-to-end optimization: exact ground state of a ferromagnetic chain ---
    {
        const std::size_t n = 12;
        qanneal::DenseIsing ham = make_ferro_chain(n);
        qanneal::SQAAnnealer ann(ham, dummy_sched, slices, replicas);
        ann.set_seed(7);

        auto res = ann.run_surrogate(/*beta=*/4.0, /*gamma_start=*/3.0, /*gamma_end=*/0.05,
                                     /*num_steps=*/100, /*sweeps_per_step=*/10,
                                     /*worldline_sweeps=*/2, /*cluster_sweeps=*/1,
                                     /*scan_points=*/10, /*scan_sweeps=*/20, /*scan_burn=*/6);

        const double e0 = -static_cast<double>(n - 1);
        std::printf("[surrogate ferro] best=%.4f exact=%.4f\n", res.best_energy, e0);
        assert(std::abs(res.best_energy - e0) < 1e-9);
    }

    std::printf("test_sqa_surrogate_schedule: OK\n");
    return 0;
}
