#include "qanneal/sqa_chi_annealer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace qanneal {

SQAChiAnnealer::SQAChiAnnealer(const Hamiltonian &hamiltonian,
                               std::size_t trotter_slices,
                               std::size_t replicas)
    : backend_(make_backend(BackendKind::CPU, hamiltonian)),
      slices_(trotter_slices),
      replicas_(replicas),
      rng_(std::random_device{}()) {
    if (slices_ == 0) {
        throw std::invalid_argument("trotter_slices must be > 0.");
    }
    if (replicas_ == 0) {
        throw std::invalid_argument("replicas must be > 0.");
    }
}

SQAChiAnnealer::SQAChiAnnealer(std::shared_ptr<Backend> backend,
                               std::size_t trotter_slices,
                               std::size_t replicas)
    : backend_(std::move(backend)),
      slices_(trotter_slices),
      replicas_(replicas),
      rng_(std::random_device{}()) {
    if (!backend_) {
        throw std::invalid_argument("SQAChiAnnealer requires a backend.");
    }
    if (slices_ == 0) {
        throw std::invalid_argument("trotter_slices must be > 0.");
    }
    if (replicas_ == 0) {
        throw std::invalid_argument("replicas must be > 0.");
    }
}

void SQAChiAnnealer::set_seed(std::uint64_t seed) {
    rng_.seed(seed);
}

double SQAChiAnnealer::trotter_coupling(double beta, double gamma, std::size_t slices) const {
    const double eps = 1e-12;
    const double x = std::max(beta * gamma / static_cast<double>(slices), eps);
    return 0.5 * std::log(1.0 / std::tanh(x));
}

void SQAChiAnnealer::build_parity_cells(std::size_t slices, CellList out[2]) const {
    out[0].clear();
    out[1].clear();
    out[0].reserve(replicas_ * ((slices + 1) / 2));
    out[1].reserve(replicas_ * (slices / 2));
    for (std::size_t r = 0; r < replicas_; ++r) {
        for (std::size_t t = 0; t < slices; ++t) {
            out[t % 2].emplace_back(r, t);
        }
    }
}

// ---------------------------------------------------------------------------
// Parity-parallel checkerboard sweep.
//
// Slices of one Trotter parity are never directly coupled to each other (the
// Trotter term only links slice t to t-1/t+1, which belong to the OTHER
// parity), so every (replica, slice) cell of a parity class can be updated
// fully independently -- and in parallel -- while the other parity's slices
// are held fixed as boundary values. Within a cell, all n spins of that one
// slice are swept sequentially (exact single-spin-flip Metropolis via the
// backend's cached local fields), so detailed balance is exact for ANY
// coupling graph (dense or sparse, bipartite or not) -- unlike a same-slice
// simultaneous spin update, which is only exact when the spatial couplings
// are themselves bipartite.
// ---------------------------------------------------------------------------
void SQAChiAnnealer::checkerboard_sweep(SQAState &state,
                                        std::vector<double> &local_fields,
                                        std::vector<std::mt19937_64> &cell_rng,
                                        const CellList parity_cells[2],
                                        double beta_scale,
                                        double j_perp,
                                        std::size_t slices,
                                        std::size_t n,
                                        std::vector<double> &m_out) const {
    for (int parity = 0; parity < 2; ++parity) {
        const CellList &cells = parity_cells[parity];
        const long long ncells = static_cast<long long>(cells.size());
#ifdef _OPENMP
#pragma omp parallel for if(ncells > 1) schedule(static)
#endif
        for (long long ci = 0; ci < ncells; ++ci) {
            const std::size_t r = cells[static_cast<std::size_t>(ci)].first;
            const std::size_t t = cells[static_cast<std::size_t>(ci)].second;
            auto &local_rng = cell_rng[r * slices + t];
            std::uniform_real_distribution<double> uniform(0.0, 1.0);

            int8_t *sptr = state.slice_ptr(r, t);
            double *fields = local_fields.data() + (r * slices + t) * n;
            const std::size_t prev_t = (t == 0) ? (slices - 1) : (t - 1);
            const std::size_t next_t = (t + 1) % slices;
            const int8_t *prev_ptr = state.slice_ptr(r, prev_t);
            const int8_t *next_ptr = state.slice_ptr(r, next_t);

            for (std::size_t spin = 0; spin < n; ++spin) {
                const int8_t s = sptr[spin];
                const int nn_sum = static_cast<int>(prev_ptr[spin]) + static_cast<int>(next_ptr[spin]);
                const double de_classical = -2.0 * static_cast<double>(s) * fields[spin];
                const double de_trotter =
                    2.0 * j_perp * static_cast<double>(s) * static_cast<double>(nn_sum);
                const double de = beta_scale * de_classical + de_trotter;
                if (de <= 0.0 || uniform(local_rng) < std::exp(-de)) {
                    sptr[spin] = static_cast<int8_t>(-s);
                    backend_->update_local_fields_after_flip(fields, sptr, n, spin, s);
                }
            }
        }
    }

    m_out.assign(replicas_, 0.0);
    const double inv_nM = 1.0 / static_cast<double>(n * slices);
#ifdef _OPENMP
#pragma omp parallel for if(replicas_ > 1) schedule(static)
#endif
    for (long long rr = 0; rr < static_cast<long long>(replicas_); ++rr) {
        const std::size_t r = static_cast<std::size_t>(rr);
        long long sum = 0;
        for (std::size_t t = 0; t < slices; ++t) {
            const int8_t *sptr = state.slice_ptr(r, t);
            for (std::size_t spin = 0; spin < n; ++spin) {
                sum += sptr[spin];
            }
        }
        m_out[r] = static_cast<double>(sum) * inv_nM;
    }
}

// ---------------------------------------------------------------------------
// run_chi: worldline-magnetization susceptibility schedule (method "sqa_chi").
// See sqa_chi_annealer.hpp for the algorithm description.
// ---------------------------------------------------------------------------
SQAChiResult SQAChiAnnealer::run_chi(double beta,
                                     double gamma_start,
                                     double gamma_end,
                                     std::size_t num_steps,
                                     std::size_t sweeps_per_step,
                                     std::size_t scan_points,
                                     std::size_t scan_sweeps,
                                     std::size_t scan_burn,
                                     double chi_floor_fraction,
                                     double driver_A0,
                                     double beta_ramp_fraction,
                                     double beta_ramp_start,
                                     const std::string &debug_csv_path) {
    if (num_steps < 2) {
        throw std::invalid_argument("num_steps must be >= 2.");
    }
    if (sweeps_per_step == 0) {
        throw std::invalid_argument("sweeps_per_step must be > 0.");
    }
    if (!(beta > 0.0)) {
        throw std::invalid_argument("beta must be > 0.");
    }
    if (!(gamma_end > 0.0) || !(gamma_start > gamma_end)) {
        throw std::invalid_argument("Require gamma_start > gamma_end > 0.");
    }
    if (scan_points < 4) {
        throw std::invalid_argument("scan_points must be >= 4.");
    }
    if (scan_sweeps == 0) {
        throw std::invalid_argument("scan_sweeps must be > 0.");
    }
    if (chi_floor_fraction < 0.0) {
        throw std::invalid_argument("chi_floor_fraction must be >= 0.");
    }

    // s = A0/(A0+gamma): exact annealing parameter for the symmetric path
    // A(s) = A0*(1-s), B(s) = s (the driver/problem strength ratio equals
    // gamma exactly when A0*(1-s)/s = gamma).
    const double A0 = (driver_A0 > 0.0) ? driver_A0 : gamma_start;
    auto gamma_to_s = [A0](double g) { return A0 / (A0 + g); };
    auto s_to_gamma = [A0](double s) {
        s = std::min(std::max(s, 1e-10), 1.0 - 1e-10);
        return A0 * (1.0 - s) / s;
    };

    const double s_lo = gamma_to_s(gamma_start);  // quantum end
    const double s_hi = gamma_to_s(gamma_end);    // classical end

    const std::size_t n = backend_->size();
    const std::size_t slices = slices_;
    const double beta_scale = beta / static_cast<double>(slices);

    SQAChiResult result;
    result.driver_A0 = A0;

    std::ofstream dbg;
    if (!debug_csv_path.empty()) {
        dbg.open(debug_csv_path);
        if (dbg) {
            dbg << "phase,index,s,gamma,j_perp,chi_B,beta\n";
            dbg.setf(std::ios::scientific);
            dbg.precision(8);
        } else {
            std::cerr << "[qanneal SQAChiAnnealer run_chi] WARNING: could not open debug CSV '"
                      << debug_csv_path << "'.\n";
        }
    }

    CellList parity_cells[2];
    build_parity_cells(slices, parity_cells);

    // ------------------------------------------------------------------
    // Stage 1: pilot chi_B(s) scan. Each grid point starts from an
    // independent, freshly randomized worldline (no state carried between
    // probes), matching the reference validation script: every chi_B(s)
    // estimate is a statistically independent measurement.
    // ------------------------------------------------------------------
    const std::size_t P = scan_points;
    result.scan_s.resize(P);
    result.scan_gamma.resize(P);
    result.scan_chi_B.resize(P);

    std::vector<double> local_fields(replicas_ * slices * n, 0.0);
    std::vector<std::mt19937_64> cell_rng(replicas_ * slices);
    std::vector<double> m_sweep;

    auto pilot_point = [&](double j_perp) -> double {
        SQAState scan_state = SQAState::random(replicas_, slices, n, rng_);
        for (std::size_t r = 0; r < replicas_; ++r) {
            for (std::size_t t = 0; t < slices; ++t) {
                backend_->compute_local_fields(
                    scan_state.slice_ptr(r, t), n,
                    &local_fields[(r * slices + t) * n]);
            }
        }
        for (auto &g : cell_rng) {
            g.seed(rng_());
        }

        double m2_sum = 0.0, mabs_sum = 0.0;
        std::size_t count = 0;
        const std::size_t total_sweeps = scan_burn + scan_sweeps;
        for (std::size_t sweep = 0; sweep < total_sweeps; ++sweep) {
            checkerboard_sweep(scan_state, local_fields, cell_rng, parity_cells,
                               beta_scale, j_perp, slices, n, m_sweep);
            if (sweep >= scan_burn) {
                for (double m : m_sweep) {
                    m2_sum += m * m;
                    mabs_sum += std::fabs(m);
                    ++count;
                }
            }
        }
        if (count < 2) {
            return 0.0;
        }
        const double m2_mean = m2_sum / static_cast<double>(count);
        const double mabs_mean = mabs_sum / static_cast<double>(count);
        const double nM = static_cast<double>(n * slices);
        return std::max(nM * (m2_mean - mabs_mean * mabs_mean), 0.0);
    };

    for (std::size_t m = 0; m < P; ++m) {
        const double s_m = s_lo + (s_hi - s_lo) * static_cast<double>(m) /
                                  static_cast<double>(P - 1);
        const double gamma_m = s_to_gamma(s_m);
        const double j_perp_m = trotter_coupling(beta, gamma_m, slices);
        const double chi = pilot_point(j_perp_m);
        result.scan_s[m] = s_m;
        result.scan_gamma[m] = gamma_m;
        result.scan_chi_B[m] = chi;
        if (dbg) {
            dbg << "scan," << m << ',' << s_m << ',' << gamma_m << ','
                << j_perp_m << ',' << chi << ',' << beta << '\n';
        }
    }

    // QCP estimate: the chi_B peak.
    const std::size_t peak =
        static_cast<std::size_t>(std::distance(
            result.scan_chi_B.begin(),
            std::max_element(result.scan_chi_B.begin(), result.scan_chi_B.end())));
    result.s_star = result.scan_s[peak];
    result.gamma_star = result.scan_gamma[peak];
    result.j_perp_star = trotter_coupling(beta, result.gamma_star, slices);

    // Floor chi_B before integrating: max(chi_floor_fraction * max(chi_B),
    // a tiny absolute floor). The absolute floor keeps the integrand
    // strictly positive when the scan is exactly flat/zero (degenerate
    // profile -> falls back to a linear-in-s schedule); the relative floor
    // generalizes the reference script's fixed 1e-8 clamp, which is not
    // scale-invariant across problem sizes (chi_B ~ O(n*M)).
    const double max_chi = result.scan_chi_B[peak];
    const bool degenerate = !(max_chi > 0.0) || !std::isfinite(max_chi);
    if (degenerate) {
        std::cerr << "[qanneal SQAChiAnnealer run_chi] WARNING: chi_B scan is flat/non-finite; "
                     "falling back to a linear-in-s schedule.\n";
        result.chi_floor = 1.0;
    } else {
        result.chi_floor = std::max(chi_floor_fraction * max_chi, 1e-12);
    }
    const double floor = result.chi_floor;

    // ------------------------------------------------------------------
    // Stage 2: build the schedule.
    //   w(s) = max(chi_B(s), floor)  (linear interpolation between scan points)
    //   tau(s) = int_{s_lo}^{s} w / int_{s_lo}^{s_hi} w,  s_k = tau^{-1}(k/(n2-1))
    // ------------------------------------------------------------------
    auto weight_at = [&](double s) -> double {
        const auto &xs = result.scan_s;
        const auto &ys = result.scan_chi_B;
        double chi;
        if (s <= xs.front()) {
            chi = ys.front();
        } else if (s >= xs.back()) {
            chi = ys.back();
        } else {
            std::size_t hi = 1;
            while (hi < P && xs[hi] < s) ++hi;
            const std::size_t lo = hi - 1;
            const double t = (s - xs[lo]) / (xs[hi] - xs[lo]);
            chi = ys[lo] + t * (ys[hi] - ys[lo]);
        }
        return std::max(chi, floor);
    };

    std::size_t n_phase1 = 0;
    if (beta_ramp_fraction > 0.0) {
        n_phase1 = static_cast<std::size_t>(beta_ramp_fraction * static_cast<double>(num_steps));
        if (n_phase1 > num_steps - 2) n_phase1 = num_steps - 2;
    }
    const std::size_t n_phase2 = num_steps - n_phase1;

    const std::size_t GRID = 4096;
    const double hs = (s_hi - s_lo) / static_cast<double>(GRID);
    std::vector<double> Wcum(GRID + 1, 0.0);
    double prev_w = weight_at(s_lo);
    for (std::size_t k = 1; k <= GRID; ++k) {
        const double sk = s_lo + static_cast<double>(k) * hs;
        const double cur_w = weight_at(sk);
        Wcum[k] = Wcum[k - 1] + 0.5 * (prev_w + cur_w) * hs;
        prev_w = cur_w;
    }
    const double W_total = Wcum[GRID];

    std::vector<double> betas(num_steps, beta);
    std::vector<double> gammas(num_steps, gamma_start);
    result.s_schedule.resize(num_steps, s_lo);

    for (std::size_t step = 0; step < n_phase1; ++step) {
        const double frac = (n_phase1 > 1)
            ? static_cast<double>(step) / static_cast<double>(n_phase1 - 1)
            : 1.0;
        betas[step] = beta_ramp_start + (beta - beta_ramp_start) * frac;
    }

    std::size_t k_idx = 0;
    for (std::size_t t = 0; t < n_phase2; ++t) {
        const double target = (W_total > 0.0)
            ? (static_cast<double>(t) / static_cast<double>(n_phase2 - 1)) * W_total
            : 0.0;
        while (k_idx < GRID && Wcum[k_idx + 1] < target) ++k_idx;
        const double w0 = Wcum[k_idx], w1 = Wcum[k_idx + 1];
        const double frac = (w1 > w0) ? (target - w0) / (w1 - w0) : 0.0;
        const double s_t = s_lo + (static_cast<double>(k_idx) + frac) * hs;
        const double g_t = std::min(std::max(s_to_gamma(s_t), gamma_end), gamma_start);
        const std::size_t step = n_phase1 + t;
        result.s_schedule[step] = s_t;
        gammas[step] = g_t;
    }

    result.beta_schedule = betas;
    result.gamma_schedule = gammas;

    if (dbg) {
        for (std::size_t step = 0; step < num_steps; ++step) {
            dbg << (step < n_phase1 ? "beta_ramp," : "run,") << step << ','
                << result.s_schedule[step] << ',' << gammas[step] << ','
                << trotter_coupling(betas[step], gammas[step], slices) << ",,"
                << betas[step] << '\n';
        }
    }

    // ------------------------------------------------------------------
    // Stage 3: main anneal. Fresh worldline (independent of the pilot scan),
    // run through the same parity-parallel checkerboard kernel, following
    // the resolved (beta_schedule, gamma_schedule).
    // ------------------------------------------------------------------
    SQAState state = SQAState::random(replicas_, slices, n, rng_);
    for (std::size_t r = 0; r < replicas_; ++r) {
        for (std::size_t t = 0; t < slices; ++t) {
            backend_->compute_local_fields(
                state.slice_ptr(r, t), n,
                &local_fields[(r * slices + t) * n]);
        }
    }
    for (auto &g : cell_rng) {
        g.seed(rng_());
    }

    result.best_energy = std::numeric_limits<double>::infinity();
    result.energy_trace.reserve(num_steps);

    for (std::size_t step = 0; step < num_steps; ++step) {
        const double step_beta = result.beta_schedule[step];
        const double step_gamma = result.gamma_schedule[step];
        const double step_beta_scale = step_beta / static_cast<double>(slices);
        const double j_perp = trotter_coupling(step_beta, step_gamma, slices);

        for (std::size_t sweep = 0; sweep < sweeps_per_step; ++sweep) {
            checkerboard_sweep(state, local_fields, cell_rng, parity_cells,
                               step_beta_scale, j_perp, slices, n, m_sweep);
        }

        double avg_energy = 0.0;
        const std::size_t total_states = replicas_ * slices;
        for (std::size_t r = 0; r < replicas_; ++r) {
            for (std::size_t t = 0; t < slices; ++t) {
                const int8_t *sptr = state.slice_ptr(r, t);
                const double e = backend_->energy(sptr, n);
                avg_energy += e;
                if (e < result.best_energy) {
                    result.best_energy = e;
                    result.best_state = state.slice_state(r, t);
                }
            }
        }
        avg_energy /= static_cast<double>(total_states);
        result.energy_trace.push_back(avg_energy);
    }

    return result;
}

}
