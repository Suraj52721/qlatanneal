// Test for the worldline-magnetization susceptibility method (SQAChiAnnealer::run_chi,
// method "sqa_chi"). Generalized from a TFIM validation script (dense J, checkerboard
// vectorised in NumPy) to qanneal's Backend abstraction (dense or sparse Ising) with an
// exact-detailed-balance parity-parallel Trotter-slice checkerboard kernel.
//
// Checks:
//   (1) diagnostics have the right shapes and the chi_B profile is non-trivial,
//   (2) the QCP estimate (chi_B peak) lies inside the scanned s-range,
//   (3) the gamma schedule is monotone non-increasing and spans [gamma_end, gamma_start],
//   (4) the schedule dwells near the chi_B peak (denser steps than average),
//   (5) end-to-end optimization on a dense ferromagnetic chain (exact ground state),
//   (6) end-to-end optimization on a SPARSE Hamiltonian (generality across backends),
//   (7) degenerate profile (gamma_start == gamma_end-adjacent tiny range) does not crash.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "qanneal/dense_ising.hpp"
#include "qanneal/sparse_ising.hpp"
#include "qanneal/sqa_chi_annealer.hpp"

namespace {

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

// Sparse ferromagnetic ring: J_{i,i+1} = -1 (cyclic), ground state all-up/down, E0 = -n.
qanneal::SparseIsing make_sparse_ferro_ring(std::size_t n) {
    std::vector<double> h(n, 0.0);
    std::vector<qanneal::SparseEdge> edges;
    edges.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        edges.push_back({i, (i + 1) % n, -1.0});
    }
    return qanneal::SparseIsing(h, edges, n);
}

} // namespace

int main() {
    const std::size_t slices = 8;
    const std::size_t replicas = 4;

    // --- (A) Diagnostics + schedule structure on an SK instance ---
    {
        const std::size_t n = 10;
        qanneal::DenseIsing ham = make_sk_problem(n);
        qanneal::SQAChiAnnealer ann(ham, slices, replicas);
        ann.set_seed(2026);

        const double beta = 4.0;
        const double gamma_start = 4.0;
        const double gamma_end = 0.05;
        const std::size_t num_steps = 120;
        const std::size_t scan_points = 12;

        auto res = ann.run_chi(beta, gamma_start, gamma_end, num_steps,
                               /*sweeps_per_step=*/8, scan_points,
                               /*scan_sweeps=*/24, /*scan_burn=*/8);

        assert(res.scan_s.size() == scan_points);
        assert(res.scan_gamma.size() == scan_points);
        assert(res.scan_chi_B.size() == scan_points);
        assert(res.beta_schedule.size() == num_steps);
        assert(res.gamma_schedule.size() == num_steps);
        assert(res.s_schedule.size() == num_steps);
        assert(res.energy_trace.size() == num_steps);
        assert(res.chi_floor > 0.0);
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
        for (std::size_t i = 1; i < res.s_schedule.size(); ++i) {
            assert(res.s_schedule[i] >= res.s_schedule[i - 1] - 1e-12);
        }

        // (4) time allocation dwells near the chi_B peak
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
        std::printf("[chi sk] s*=%.3f gamma*=%.3f floor=%.4g mean_ds=%.5f mean_ds_peak=%.5f\n",
                    res.s_star, res.gamma_star, res.chi_floor, mean_all, mean_peak);
        assert(mean_peak <= mean_all * 1.05);
    }

    // --- (B) End-to-end optimization: exact ground state of a dense ferromagnetic chain ---
    {
        const std::size_t n = 12;
        qanneal::DenseIsing ham = make_ferro_chain(n);
        qanneal::SQAChiAnnealer ann(ham, slices, replicas);
        ann.set_seed(7);

        auto res = ann.run_chi(/*beta=*/4.0, /*gamma_start=*/3.0, /*gamma_end=*/0.05,
                               /*num_steps=*/100, /*sweeps_per_step=*/10,
                               /*scan_points=*/10, /*scan_sweeps=*/20, /*scan_burn=*/6);

        const double e0 = -static_cast<double>(n - 1);
        std::printf("[chi ferro dense] best=%.4f exact=%.4f\n", res.best_energy, e0);
        assert(std::abs(res.best_energy - e0) < 1e-9);
    }

    // --- (C) End-to-end optimization on a SPARSE Hamiltonian ---
    {
        const std::size_t n = 12;
        qanneal::SparseIsing ham = make_sparse_ferro_ring(n);
        qanneal::SQAChiAnnealer ann(ham, slices, replicas);
        ann.set_seed(11);

        auto res = ann.run_chi(/*beta=*/4.0, /*gamma_start=*/3.0, /*gamma_end=*/0.05,
                               /*num_steps=*/100, /*sweeps_per_step=*/10,
                               /*scan_points=*/10, /*scan_sweeps=*/20, /*scan_burn=*/6);

        const double e0 = -static_cast<double>(n);  // ring: n bonds, all satisfied
        std::printf("[chi ferro sparse] best=%.4f exact=%.4f\n", res.best_energy, e0);
        assert(std::abs(res.best_energy - e0) < 1e-9);
    }

    // --- (D) Degenerate / tiny-range profile does not crash and still produces a valid schedule ---
    {
        const std::size_t n = 6;
        qanneal::DenseIsing ham = make_ferro_chain(n);
        qanneal::SQAChiAnnealer ann(ham, 4, 2);
        ann.set_seed(3);

        auto res = ann.run_chi(/*beta=*/0.5, /*gamma_start=*/0.2, /*gamma_end=*/0.15,
                               /*num_steps=*/20, /*sweeps_per_step=*/4,
                               /*scan_points=*/4, /*scan_sweeps=*/4, /*scan_burn=*/2);
        assert(res.gamma_schedule.size() == 20);
        for (double g : res.gamma_schedule) {
            assert(std::isfinite(g) && g > 0.0);
        }
        std::printf("[chi degenerate] best=%.4f (no crash)\n", res.best_energy);
    }

    std::printf("test_sqa_chi_annealer: OK\n");
    return 0;
}
