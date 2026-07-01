#include "qanneal/sqa_parallel_tempering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "qanneal/sqa_state.hpp"

namespace qanneal {

namespace {

double delta_trotter_single(const SQAState &state,
                            std::size_t slice,
                            std::size_t spin,
                            double j_perp,
                            std::size_t slices) {
    const std::size_t prev = (slice == 0) ? (slices - 1) : (slice - 1);
    const std::size_t next = (slice + 1) % slices;
    const int8_t s = state.at(0, slice, spin);
    const int8_t s_prev = state.at(0, prev, spin);
    const int8_t s_next = state.at(0, next, spin);
    return 2.0 * j_perp * static_cast<double>(s) * static_cast<double>(s_prev + s_next);
}

} // namespace

SQAParallelTemperingAnnealer::SQAParallelTemperingAnnealer(const Hamiltonian &hamiltonian,
                                                           std::vector<double> betas,
                                                           std::vector<double> gammas,
                                                           std::size_t trotter_slices)
    : backend_(make_backend(BackendKind::CPU, hamiltonian)),
      betas_(std::move(betas)),
      gammas_(std::move(gammas)),
      slices_(trotter_slices),
      rng_(std::random_device{}()) {
    if (betas_.size() < 2 || betas_.size() != gammas_.size()) {
        throw std::invalid_argument("SQA parallel tempering requires >=2 beta/gamma pairs.");
    }
    if (slices_ == 0) {
        throw std::invalid_argument("trotter_slices must be > 0.");
    }
}

SQAParallelTemperingAnnealer::SQAParallelTemperingAnnealer(std::shared_ptr<Backend> backend,
                                                           std::vector<double> betas,
                                                           std::vector<double> gammas,
                                                           std::size_t trotter_slices)
    : backend_(std::move(backend)),
      betas_(std::move(betas)),
      gammas_(std::move(gammas)),
      slices_(trotter_slices),
      rng_(std::random_device{}()) {
    if (!backend_) {
        throw std::invalid_argument("SQAParallelTemperingAnnealer requires a backend.");
    }
    if (betas_.size() < 2 || betas_.size() != gammas_.size()) {
        throw std::invalid_argument("SQA parallel tempering requires >=2 beta/gamma pairs.");
    }
    if (slices_ == 0) {
        throw std::invalid_argument("trotter_slices must be > 0.");
    }
}

void SQAParallelTemperingAnnealer::set_seed(std::uint64_t seed) {
    rng_.seed(seed);
}

double SQAParallelTemperingAnnealer::trotter_coupling(double beta, double gamma, std::size_t slices) const {
    const double eps = 1e-12;
    const double x = std::max(beta * gamma / static_cast<double>(slices), eps);
    return 0.5 * std::log(1.0 / std::tanh(x));
}

SQAParallelTemperingResult SQAParallelTemperingAnnealer::run(std::size_t sweeps_per_step,
                                                             std::size_t worldline_sweeps,
                                                             std::size_t steps,
                                                             std::size_t swap_interval,
                                                             std::size_t cluster_sweeps,
                                                             std::size_t continuous_time_slices) {
    if (steps == 0) {
        throw std::invalid_argument("steps must be > 0.");
    }
    if (swap_interval == 0) {
        throw std::invalid_argument("swap_interval must be > 0.");
    }
    if (sweeps_per_step == 0 && worldline_sweeps == 0 && cluster_sweeps == 0) {
        throw std::invalid_argument("At least one of sweeps_per_step/worldline_sweeps/cluster_sweeps must be > 0.");
    }

    const std::size_t n = backend_->size();
    const std::size_t replicas = betas_.size();
    const std::size_t slices = (continuous_time_slices > 0 && continuous_time_slices > slices_)
                                 ? continuous_time_slices
                                 : slices_;

    std::vector<SQAState> states;
    states.reserve(replicas);
    for (std::size_t r = 0; r < replicas; ++r) {
        states.emplace_back(SQAState::random(1, slices, n, rng_));
    }

    std::vector<std::mt19937_64> replica_rng(replicas);
    for (std::size_t r = 0; r < replicas; ++r) {
        replica_rng[r].seed(rng_());
    }

    // Pre-allocated cluster buffers (per replica for OpenMP thread safety)
    std::vector<std::vector<char>> cluster_buf(replicas, std::vector<char>(slices));
    std::vector<std::vector<std::size_t>> stack_buf(replicas);
    for (auto &s : stack_buf) {
        s.reserve(slices);
    }

    // Per-replica spin visit order — shuffled each sweep for better ergodicity
    std::vector<std::vector<std::size_t>> spin_orders(replicas);
    for (std::size_t r = 0; r < replicas; ++r) {
        spin_orders[r].resize(n);
        std::iota(spin_orders[r].begin(), spin_orders[r].end(), 0);
    }

    auto classical_sum = [&](const SQAState &state) {
        double sum = 0.0;
        for (std::size_t t = 0; t < slices; ++t) {
            sum += backend_->energy(state.slice_ptr(0, t), n);
        }
        return sum;
    };

    auto bond_sum = [&](const SQAState &state) {
        double sum = 0.0;
        for (std::size_t t = 0; t < slices; ++t) {
            const std::size_t tn = (t + 1) % slices;
            const int8_t *a = state.slice_ptr(0, t);
            const int8_t *b = state.slice_ptr(0, tn);
            for (std::size_t i = 0; i < n; ++i) {
                sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            }
        }
        return sum;
    };

    auto action = [&](const SQAState &state, double beta, double gamma) {
        const double csum = classical_sum(state);
        const double bsum = bond_sum(state);
        const double j_perp = trotter_coupling(beta, gamma, slices);
        return (beta / static_cast<double>(slices)) * csum - j_perp * bsum;
    };

    auto projected_energy = [&](const SQAState &state, std::size_t *best_slice_out = nullptr) {
        double best = std::numeric_limits<double>::infinity();
        std::size_t best_slice = 0;
        for (std::size_t t = 0; t < slices; ++t) {
            const double e = backend_->energy(state.slice_ptr(0, t), n);
            if (e < best) {
                best = e;
                best_slice = t;
            }
        }
        if (best_slice_out) {
            *best_slice_out = best_slice;
        }
        return best;
    };

    SQAParallelTemperingResult result;
    result.best_energy = std::numeric_limits<double>::infinity();
    result.average_energy_trace.reserve(steps);
    result.swap_acceptance_trace.reserve(steps);

    std::vector<double> energies(replicas, 0.0);
    for (std::size_t r = 0; r < replicas; ++r) {
        std::size_t best_slice = 0;
        energies[r] = projected_energy(states[r], &best_slice);
        if (energies[r] < result.best_energy) {
            result.best_energy = energies[r];
            result.best_state = states[r].slice_state(0, best_slice);
        }
    }

    for (std::size_t step = 0; step < steps; ++step) {
#ifdef _OPENMP
#pragma omp parallel for if(replicas > 1) schedule(static)
#endif
        for (std::size_t r = 0; r < replicas; ++r) {
            auto &state = states[r];
            auto &rr = replica_rng[r];
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            std::uniform_int_distribution<std::size_t> slice_pick(0, slices - 1);

            const double beta = betas_[r];
            const double gamma = gammas_[r];
            const double beta_scale = beta / static_cast<double>(slices);
            const double j_perp = trotter_coupling(beta, gamma, slices);
            const double join_prob = 1.0 - std::exp(-2.0 * j_perp);

            auto &my_spin_order = spin_orders[r];
            auto &my_cluster = cluster_buf[r];
            auto &my_stack = stack_buf[r];

            for (std::size_t sweep = 0; sweep < sweeps_per_step; ++sweep) {
                std::shuffle(my_spin_order.begin(), my_spin_order.end(), rr);
                for (std::size_t t = 0; t < slices; ++t) {
                    int8_t *slice_ptr = state.slice_ptr(0, t);
                    for (std::size_t idx = 0; idx < n; ++idx) {
                        const std::size_t i = my_spin_order[idx];
                        const double dclass = backend_->delta_energy(slice_ptr, n, i);
                        const double dtotal = beta_scale * dclass + delta_trotter_single(state, t, i, j_perp, slices);
                        if (dtotal <= 0.0 || uniform(rr) < std::exp(-dtotal)) {
                            slice_ptr[i] = static_cast<int8_t>(-slice_ptr[i]);
                        }
                    }
                }
            }

            for (std::size_t sweep = 0; sweep < cluster_sweeps; ++sweep) {
                std::shuffle(my_spin_order.begin(), my_spin_order.end(), rr);
                for (std::size_t idx = 0; idx < n; ++idx) {
                    const std::size_t i = my_spin_order[idx];
                    const std::size_t seed = slice_pick(rr);
                    std::fill(my_cluster.begin(), my_cluster.end(), 0);
                    my_stack.clear();
                    my_stack.push_back(seed);
                    my_cluster[seed] = 1;
                    const int8_t seed_spin = state.slice_ptr(0, seed)[i];

                    while (!my_stack.empty()) {
                        const std::size_t t = my_stack.back();
                        my_stack.pop_back();
                        const std::size_t prev = (t == 0) ? (slices - 1) : (t - 1);
                        const std::size_t next = (t + 1) % slices;
                        if (!my_cluster[prev] && state.slice_ptr(0, prev)[i] == seed_spin && uniform(rr) < join_prob) {
                            my_cluster[prev] = 1;
                            my_stack.push_back(prev);
                        }
                        if (!my_cluster[next] && state.slice_ptr(0, next)[i] == seed_spin && uniform(rr) < join_prob) {
                            my_cluster[next] = 1;
                            my_stack.push_back(next);
                        }
                    }

                    double dclass = 0.0;
                    double dtime = 0.0;
                    for (std::size_t t = 0; t < slices; ++t) {
                        if (!my_cluster[t]) {
                            continue;
                        }
                        int8_t *slice_ptr = state.slice_ptr(0, t);
                        dclass += backend_->delta_energy(slice_ptr, n, i);
                        const std::size_t prev = (t == 0) ? (slices - 1) : (t - 1);
                        const std::size_t next = (t + 1) % slices;
                        const int8_t s_t = slice_ptr[i];
                        if (!my_cluster[prev]) {
                            dtime += 2.0 * j_perp * static_cast<double>(s_t) * static_cast<double>(state.slice_ptr(0, prev)[i]);
                        }
                        if (!my_cluster[next]) {
                            dtime += 2.0 * j_perp * static_cast<double>(s_t) * static_cast<double>(state.slice_ptr(0, next)[i]);
                        }
                    }

                    const double dtotal = beta_scale * dclass + dtime;
                    if (dtotal <= 0.0 || uniform(rr) < std::exp(-dtotal)) {
                        for (std::size_t t = 0; t < slices; ++t) {
                            if (my_cluster[t]) {
                                int8_t *slice_ptr = state.slice_ptr(0, t);
                                slice_ptr[i] = static_cast<int8_t>(-slice_ptr[i]);
                            }
                        }
                    }
                }
            }

            for (std::size_t sweep = 0; sweep < worldline_sweeps; ++sweep) {
                std::shuffle(my_spin_order.begin(), my_spin_order.end(), rr);
                for (std::size_t idx = 0; idx < n; ++idx) {
                    const std::size_t i = my_spin_order[idx];
                    double dclass = 0.0;
                    for (std::size_t t = 0; t < slices; ++t) {
                        dclass += backend_->delta_energy(state.slice_ptr(0, t), n, i);
                    }
                    const double dtotal = beta_scale * dclass;
                    if (dtotal <= 0.0 || uniform(rr) < std::exp(-dtotal)) {
                        for (std::size_t t = 0; t < slices; ++t) {
                            int8_t *slice_ptr = state.slice_ptr(0, t);
                            slice_ptr[i] = static_cast<int8_t>(-slice_ptr[i]);
                        }
                    }
                }
            }
        }

        for (std::size_t r = 0; r < replicas; ++r) {
            std::size_t best_slice = 0;
            energies[r] = projected_energy(states[r], &best_slice);
            if (energies[r] < result.best_energy) {
                result.best_energy = energies[r];
                result.best_state = states[r].slice_state(0, best_slice);
            }
        }

        double accepted = 0.0;
        double attempted = 0.0;
        if ((step + 1) % swap_interval == 0) {
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            for (std::size_t r = 0; r + 1 < replicas; ++r) {
                const double e_aa = action(states[r], betas_[r], gammas_[r]);
                const double e_bb = action(states[r + 1], betas_[r + 1], gammas_[r + 1]);
                const double e_ab = action(states[r + 1], betas_[r], gammas_[r]);
                const double e_ba = action(states[r], betas_[r + 1], gammas_[r + 1]);
                const double delta = (e_ab + e_ba) - (e_aa + e_bb);
                ++attempted;
                if (delta <= 0.0 || uniform(rng_) < std::exp(-delta)) {
                    std::swap(states[r], states[r + 1]);
                    std::swap(energies[r], energies[r + 1]);
                    ++accepted;
                }
            }
        }

        double avg_energy = 0.0;
        for (double e : energies) {
            avg_energy += e;
        }
        avg_energy /= static_cast<double>(replicas);

        result.average_energy_trace.push_back(avg_energy);
        result.swap_acceptance_trace.push_back(attempted > 0.0 ? accepted / attempted : 0.0);
    }

    result.final_states.reserve(replicas);
    result.final_energies.assign(replicas, 0.0);
    for (std::size_t r = 0; r < replicas; ++r) {
        std::size_t best_slice = 0;
        result.final_energies[r] = projected_energy(states[r], &best_slice);
        result.final_states.push_back(states[r].slice_state(0, best_slice));
    }

    return result;
}

SQAParallelTemperingResult SQAParallelTemperingAnnealer::run_optimal(
        std::size_t num_steps,
        std::size_t sweeps_per_step,
        std::size_t worldline_sweeps,
        double eps_tilde,
        double alpha,
        double j_perp_end,
        std::size_t cluster_sweeps,
        std::size_t swap_interval,
        std::size_t continuous_time_slices,
        std::size_t calib_probes,
        std::size_t calib_sweeps,
        const std::string &debug_csv_path) {
    if (num_steps == 0) throw std::invalid_argument("num_steps must be > 0.");
    if (swap_interval == 0) throw std::invalid_argument("swap_interval must be > 0.");
    if (sweeps_per_step == 0 && worldline_sweeps == 0 && cluster_sweeps == 0) {
        throw std::invalid_argument("At least one of sweeps_per_step/worldline_sweeps/cluster_sweeps must be > 0.");
    }
    // eps_tilde <= 0 is no longer an error: it requests budget calibration (see below).
    // A positive eps_tilde is an explicit override that skips calibration.
    const bool calibrate = (eps_tilde <= 0.0);

    const std::size_t n = backend_->size();
    const std::size_t replicas = betas_.size();
    const std::size_t slices = (continuous_time_slices > 0 && continuous_time_slices > slices_)
                                 ? continuous_time_slices
                                 : slices_;
    const double nM = static_cast<double>(n * slices);

    // Start j_perp from the hottest replica's coupling (most quantum end of the ladder).
    const double j_perp_start = trotter_coupling(betas_[0], gammas_[0], slices);
    double j_perp = j_perp_start;
    // If j_perp_end not given, use the coldest replica's initial coupling as target.
    const bool j_perp_end_was_sentinel = (j_perp_end <= 0.0);
    if (j_perp_end_was_sentinel) {
        j_perp_end = trotter_coupling(betas_.back(), gammas_.back(), slices);
        std::cerr << "[qanneal run_optimal] WARNING: j_perp_end not specified; "
                     "defaulted to coldest-replica trotter_coupling = " << j_perp_end
                  << " (j_perp_start = " << j_perp_start << "). "
                     "For benchmark runs pass an explicit j_perp_end (e.g. 5*j_rms).\n";
    }
    if (j_perp_end <= j_perp) {
        // Degenerate target: the schedule has no room to anneal. Rather than silently
        // installing an effectively-unbounded ceiling (which hides the misconfiguration
        // and lets the run "freeze" undetected), warn loudly. We still install a finite
        // ceiling so the loop terminates by num_steps, but the calibration below cannot
        // produce a meaningful traversal and will fall back to the legacy heuristic.
        std::cerr << "[qanneal run_optimal] WARNING: resolved j_perp_end (" << j_perp_end
                  << ") <= j_perp_start (" << j_perp_start << "). The adaptive schedule has "
                     "no range to traverse and will not anneal. Installing finite ceiling "
                     "j_perp_start + 1e6 so the run terminates by num_steps. "
                     "Check your beta/gamma ladder or pass an explicit j_perp_end > j_perp_start.\n";
        j_perp_end = j_perp + 1e6;
    }

    std::vector<SQAState> states;
    states.reserve(replicas);
    for (std::size_t r = 0; r < replicas; ++r) {
        states.emplace_back(SQAState::random(1, slices, n, rng_));
    }

    std::vector<std::mt19937_64> replica_rng(replicas);
    for (std::size_t r = 0; r < replicas; ++r) replica_rng[r].seed(rng_());

    std::vector<std::vector<char>> cluster_buf(replicas, std::vector<char>(slices));
    std::vector<std::vector<std::size_t>> stack_buf(replicas);
    for (auto &s : stack_buf) s.reserve(slices);

    std::vector<std::vector<std::size_t>> spin_orders(replicas);
    for (std::size_t r = 0; r < replicas; ++r) {
        spin_orders[r].resize(n);
        std::iota(spin_orders[r].begin(), spin_orders[r].end(), 0);
    }

    // Per-replica B sample buffers: one sample per (Metropolis + cluster) sweep per replica.
    // replicas * (sweeps_per_step + cluster_sweeps) samples per step.
    const std::size_t samples_per_replica = sweeps_per_step + cluster_sweeps;
    std::vector<std::vector<double>> B_sweep_samples(replicas);
    for (auto &v : B_sweep_samples) {
        v.reserve(samples_per_replica > 0 ? samples_per_replica : 1);
    }

    auto classical_sum = [&](const SQAState &state) {
        double sum = 0.0;
        for (std::size_t t = 0; t < slices; ++t)
            sum += backend_->energy(state.slice_ptr(0, t), n);
        return sum;
    };

    auto projected_energy = [&](const SQAState &state, std::size_t *best_slice_out = nullptr) {
        double best = std::numeric_limits<double>::infinity();
        std::size_t best_slice = 0;
        for (std::size_t t = 0; t < slices; ++t) {
            const double e = backend_->energy(state.slice_ptr(0, t), n);
            if (e < best) { best = e; best_slice = t; }
        }
        if (best_slice_out) *best_slice_out = best_slice;
        return best;
    };

    // B = sum_{k,i} s^k_i s^{k+1}_i for one SQAPT state (single-replica SQAState).
    auto bond_sum_one = [&](const SQAState &state) {
        double sum = 0.0;
        for (std::size_t t = 0; t < slices; ++t) {
            const std::size_t tn = (t + 1) % slices;
            const int8_t *a = state.slice_ptr(0, t);
            const int8_t *b = state.slice_ptr(0, tn);
            for (std::size_t i = 0; i < n; ++i)
                sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }
        return sum;
    };

    // One step of per-replica sweeps at a fixed j_perp_local, filling B_sweep_samples.
    // Shared by the calibration pre-pass and the main schedule loop so the chi_B that
    // calibration measures is produced by exactly the same sampler the run uses.
    auto sweep_replicas = [&](std::vector<SQAState> &st,
                              std::vector<std::mt19937_64> &rngs,
                              double j_perp_local) {
        const double join_prob = 1.0 - std::exp(-2.0 * j_perp_local);
#ifdef _OPENMP
#pragma omp parallel for if(replicas > 1) schedule(static)
#endif
        for (std::size_t r = 0; r < replicas; ++r) {
            auto &state = st[r];
            auto &rr = rngs[r];
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            std::uniform_int_distribution<std::size_t> slice_pick(0, slices - 1);

            const double beta_scale = betas_[r] / static_cast<double>(slices);
            auto &my_spin_order = spin_orders[r];
            auto &my_cluster = cluster_buf[r];
            auto &my_stack = stack_buf[r];

            // Running B — exact incremental updates, no recomputation during sweeps.
            double b_r = bond_sum_one(state);
            auto &samples = B_sweep_samples[r];
            samples.clear();

            for (std::size_t sweep = 0; sweep < sweeps_per_step; ++sweep) {
                std::shuffle(my_spin_order.begin(), my_spin_order.end(), rr);
                for (std::size_t t = 0; t < slices; ++t) {
                    int8_t *slice_ptr = state.slice_ptr(0, t);
                    const std::size_t prev_t = (t == 0) ? (slices - 1) : (t - 1);
                    const std::size_t next_t = (t + 1) % slices;
                    const int8_t *prev_ptr = state.slice_ptr(0, prev_t);
                    const int8_t *next_ptr = state.slice_ptr(0, next_t);
                    for (std::size_t idx = 0; idx < n; ++idx) {
                        const std::size_t i = my_spin_order[idx];
                        const int8_t s = slice_ptr[i];
                        const int8_t s_prev = prev_ptr[i];
                        const int8_t s_next = next_ptr[i];
                        const int nn_sum = static_cast<int>(s_prev) + static_cast<int>(s_next);
                        const double dclass = backend_->delta_energy(slice_ptr, n, i);
                        const double dtime  = 2.0 * j_perp_local * static_cast<double>(s) *
                                              static_cast<double>(nn_sum);
                        const double dtotal = beta_scale * dclass + dtime;
                        if (dtotal <= 0.0 || uniform(rr) < std::exp(-dtotal)) {
                            slice_ptr[i] = static_cast<int8_t>(-s);
                            b_r += -2.0 * static_cast<double>(s) * static_cast<double>(nn_sum);
                        }
                    }
                }
                samples.push_back(b_r);
            }

            for (std::size_t sweep = 0; sweep < cluster_sweeps; ++sweep) {
                std::shuffle(my_spin_order.begin(), my_spin_order.end(), rr);
                for (std::size_t idx = 0; idx < n; ++idx) {
                    const std::size_t i = my_spin_order[idx];
                    const std::size_t seed = slice_pick(rr);
                    std::fill(my_cluster.begin(), my_cluster.end(), 0);
                    my_stack.clear();
                    my_stack.push_back(seed);
                    my_cluster[seed] = 1;
                    const int8_t seed_spin = state.slice_ptr(0, seed)[i];

                    while (!my_stack.empty()) {
                        const std::size_t t = my_stack.back(); my_stack.pop_back();
                        const std::size_t prev = (t == 0) ? (slices - 1) : (t - 1);
                        const std::size_t next = (t + 1) % slices;
                        if (!my_cluster[prev] && state.slice_ptr(0, prev)[i] == seed_spin &&
                            uniform(rr) < join_prob) {
                            my_cluster[prev] = 1; my_stack.push_back(prev);
                        }
                        if (!my_cluster[next] && state.slice_ptr(0, next)[i] == seed_spin &&
                            uniform(rr) < join_prob) {
                            my_cluster[next] = 1; my_stack.push_back(next);
                        }
                    }

                    double dclass = 0.0, dtime = 0.0, delta_B = 0.0;
                    for (std::size_t t = 0; t < slices; ++t) {
                        if (!my_cluster[t]) continue;
                        int8_t *sptr = state.slice_ptr(0, t);
                        dclass += backend_->delta_energy(sptr, n, i);
                        const std::size_t prev = (t == 0) ? (slices - 1) : (t - 1);
                        const std::size_t next = (t + 1) % slices;
                        const int8_t s_t = sptr[i];
                        if (!my_cluster[prev]) {
                            const double bond = static_cast<double>(s_t) *
                                               static_cast<double>(state.slice_ptr(0, prev)[i]);
                            dtime   += 2.0 * j_perp_local * bond;
                            delta_B += -2.0 * bond;
                        }
                        if (!my_cluster[next]) {
                            const double bond = static_cast<double>(s_t) *
                                               static_cast<double>(state.slice_ptr(0, next)[i]);
                            dtime   += 2.0 * j_perp_local * bond;
                            delta_B += -2.0 * bond;
                        }
                    }
                    const double dtotal = beta_scale * dclass + dtime;
                    if (dtotal <= 0.0 || uniform(rr) < std::exp(-dtotal)) {
                        for (std::size_t t = 0; t < slices; ++t) {
                            if (my_cluster[t]) {
                                int8_t *sptr = state.slice_ptr(0, t);
                                sptr[i] = static_cast<int8_t>(-sptr[i]);
                            }
                        }
                        b_r += delta_B;
                    }
                }
                samples.push_back(b_r);
            }

            for (std::size_t sweep = 0; sweep < worldline_sweeps; ++sweep) {
                std::shuffle(my_spin_order.begin(), my_spin_order.end(), rr);
                for (std::size_t idx = 0; idx < n; ++idx) {
                    const std::size_t i = my_spin_order[idx];
                    double dclass = 0.0;
                    for (std::size_t t = 0; t < slices; ++t)
                        dclass += backend_->delta_energy(state.slice_ptr(0, t), n, i);
                    const double dtotal = beta_scale * dclass;
                    if (dtotal <= 0.0 || uniform(rr) < std::exp(-dtotal)) {
                        for (std::size_t t = 0; t < slices; ++t) {
                            int8_t *sptr = state.slice_ptr(0, t);
                            sptr[i] = static_cast<int8_t>(-sptr[i]);
                        }
                        // Worldline flips preserve all imaginary-time bonds: ΔB = 0.
                    }
                }
                // No B sample: worldline sweeps cannot change B.
            }
        }
    };

    // Raw chi_B = Var(B) from the current B_sweep_samples (no floor applied).
    // Returns -1.0 if fewer than 2 samples were collected (caller handles fallback).
    auto measure_chi_B_raw = [&](const std::vector<SQAState> &st) {
        double B_sum = 0.0, B2_sum = 0.0;
        std::size_t total_samples = 0;
        for (std::size_t r = 0; r < replicas; ++r) {
            for (double b : B_sweep_samples[r]) {
                B_sum  += b;
                B2_sum += b * b;
                ++total_samples;
            }
        }
        if (total_samples >= 2) {
            const double B_mean  = B_sum  / static_cast<double>(total_samples);
            const double B2_mean = B2_sum / static_cast<double>(total_samples);
            return B2_mean - B_mean * B_mean;
        }
        // Single-replica fallback: per-bond variance approximation.
        const double b = bond_sum_one(st[0]);
        const double b_avg = b / nM;
        return nM * (1.0 - b_avg * b_avg);
    };

    SQAParallelTemperingResult result;
    result.best_energy = std::numeric_limits<double>::infinity();
    result.average_energy_trace.reserve(num_steps);
    result.swap_acceptance_trace.reserve(num_steps);

    std::vector<double> energies(replicas, 0.0);
    for (std::size_t r = 0; r < replicas; ++r) {
        std::size_t best_slice = 0;
        energies[r] = projected_energy(states[r], &best_slice);
        if (energies[r] < result.best_energy) {
            result.best_energy = energies[r];
            result.best_state = states[r].slice_state(0, best_slice);
        }
    }

    // ----------------------------------------------------------------------------------
    // Calibration pre-pass (Task 2): fix the overall scale eps_tilde from the step budget.
    //
    // The schedule's *shape* (slow near criticality, fast away) comes from chi_B(J)^(-alpha)
    // and is independent of eps_tilde; eps_tilde is a free normalization that must be set so
    // the T = num_steps updates traverse exactly [j_perp_start, j_perp_end].
    //
    // From the discrete update J_{t+1} = J_t + eps_tilde * chi_B(J_t)^(-alpha), the number of
    // steps to cross the range is  T = (1/eps_tilde) * integral_{J0}^{Jend} chi_B(J)^alpha dJ.
    // Hence the budget-correct value is
    //     eps_tilde = (1/T) * integral_{J0}^{Jend} chi_B(J)^alpha dJ
    //               ~= (dJ / (T*M)) * sum_m chi_B(J_m)^alpha            (midpoint rule, M probes)
    // NOTE: the +alpha exponent (and the /T) are essential. The naive form
    // eps_tilde = dJ / sum_m chi_B(J_m)^(-alpha) does NOT consume the budget when chi_B varies
    // (it under/overshoots and re-freezes); the integral form above does, by construction.
    double min_chi_B = 1e-4 * nM;  // legacy floor; rescaled below when calibrating

    // Calibration profile: chi_B measured at probe J_perp points. When non-empty (calibrate
    // path) the schedule is DRIVEN by this equilibrium-like profile chi_B(J), interpolated at
    // the current j_perp — NOT by the noisy, path-dependent online variance. This is what makes
    // the traversal budget-exact "by construction": the same chi_B(J) used to set eps_tilde
    // (via the integral) drives each step, so the discrete sum of delta_j reaches j_perp_end in
    // ~num_steps. (Driving from the online variance instead lets the run order faster than the
    // pilot, inflating chi_B^(-alpha) and making the schedule leap to the end in a few steps.)
    std::vector<double> probe_J;    // probe abscissae (ascending)
    std::vector<double> probe_chi;  // chi_B at each probe

    // Optional per-step diagnostic CSV (Task 1 / Task 5). Opened before calibration so the
    // pilot probe points are recorded too (phase=calib), then per-step rows (phase=run).
    std::ofstream dbg;
    if (!debug_csv_path.empty()) {
        dbg.open(debug_csv_path);
        if (dbg) {
            dbg << "phase,step_index,j_perp,chi_B,delta_j,eps_tilde,alpha,floor_hit,chi_B_online\n";
            dbg.setf(std::ios::scientific);
            dbg.precision(8);
        } else {
            std::cerr << "[qanneal run_optimal] WARNING: could not open debug CSV '"
                      << debug_csv_path << "'.\n";
        }
    }

    if (calibrate) {
        const std::size_t M = std::max<std::size_t>(calib_probes, 2);
        const std::size_t cs = std::max<std::size_t>(calib_sweeps, 1);

        // Scratch copies so the pilot does not bias the real run's initial states.
        std::vector<SQAState> scratch = states;
        std::vector<std::mt19937_64> scratch_rng(replicas);
        for (std::size_t r = 0; r < replicas; ++r) scratch_rng[r].seed(rng_());

        // Temporarily run the shared sampler with `cs` sweeps per probe.
        const std::size_t saved_sweeps = sweeps_per_step;
        const std::size_t saved_cluster = cluster_sweeps;
        const std::size_t saved_worldline = worldline_sweeps;
        sweeps_per_step = cs;
        // Cluster sweeps materially change chi_B, so mirror their presence during calibration.
        cluster_sweeps = saved_cluster > 0 ? std::max<std::size_t>(saved_cluster, 1) : 0;
        worldline_sweeps = 0;

        const double dJ = j_perp_end - j_perp_start;
        const double dJ_probe = dJ / static_cast<double>(M);
        probe_J.resize(M);
        probe_chi.resize(M, 0.0);
        double integral = 0.0;  // sum_m chi_B(J_m)^alpha * dJ_probe
        for (std::size_t m = 0; m < M; ++m) {
            // Midpoint of the m-th sub-interval.
            const double Jm = j_perp_start + (static_cast<double>(m) + 0.5) * dJ_probe;
            sweep_replicas(scratch, scratch_rng, Jm);
            double chi = measure_chi_B_raw(scratch);
            if (!(chi > 0.0)) chi = 0.0;
            probe_J[m] = Jm;
            probe_chi[m] = chi;
            integral += std::pow(std::max(chi, 1e-12), alpha) * dJ_probe;
            if (dbg) {
                // Negative step index marks calibration rows; eps_tilde/delta_j unknown yet.
                dbg << "calib," << (static_cast<long long>(m) - static_cast<long long>(M)) << ','
                    << Jm << ',' << chi << ",0,0," << alpha << ",0," << chi << '\n';
            }
        }

        // Budget-constrained scale (see derivation above).
        eps_tilde = integral / static_cast<double>(num_steps);
        if (!(eps_tilde > 0.0) || !std::isfinite(eps_tilde)) {
            // Degenerate (e.g. chi_B ~ 0 everywhere, or j_perp_end <= j_perp_start). Fall back
            // to a linear ramp that still consumes the budget.
            std::cerr << "[qanneal run_optimal] WARNING: calibration produced non-positive "
                         "eps_tilde (integral=" << integral << "); falling back to linear ramp.\n";
            eps_tilde = std::max(dJ, 0.0) / static_cast<double>(num_steps);
            if (!(eps_tilde > 0.0)) eps_tilde = 1e-9;
        }

        // Rescale the chi_B floor (Task 3). Empirically chi_B is LARGE near the quantum
        // transition and collapses toward 0 in the ordered phase (it does not diverge at
        // criticality for this model), so chi_B^(-alpha) — and hence the step size — is
        // largest deep in the ordered phase. Without a floor a single ordered-phase step can
        // leap across a wide swath of J_perp. Floor relative to the MAX probed chi_B so the
        // floor only bites where chi_B is genuinely tiny (ordered phase), bounding the step.
        std::vector<double> sorted = probe_chi;
        std::sort(sorted.begin(), sorted.end());
        const double max_chi = sorted.back();
        min_chi_B = std::max(1e-4 * max_chi, 1e-12);

        // Restore the run's sweep counts.
        sweeps_per_step = saved_sweeps;
        cluster_sweeps = saved_cluster;
        worldline_sweeps = saved_worldline;
    }
    result.calibrated_eps_tilde = eps_tilde;
    result.j_perp_start = j_perp_start;
    result.resolved_j_perp_end = j_perp_end;
    result.j_perp_trace.reserve(num_steps);
    result.chi_B_trace.reserve(num_steps);

    // Interpolate the calibration profile chi_B(J) at an arbitrary J (clamped to the probe
    // range, linear between probes). Returns -1 when no profile exists (eps_tilde override).
    auto chi_profile_at = [&](double J) -> double {
        const std::size_t M = probe_J.size();
        if (M == 0) return -1.0;
        if (J <= probe_J.front()) return probe_chi.front();
        if (J >= probe_J.back())  return probe_chi.back();
        // probe_J is ascending and evenly spaced; locate the bracketing interval.
        std::size_t hi = 1;
        while (hi < M && probe_J[hi] < J) ++hi;
        const std::size_t lo = hi - 1;
        const double t = (J - probe_J[lo]) / (probe_J[hi] - probe_J[lo]);
        return probe_chi[lo] + t * (probe_chi[hi] - probe_chi[lo]);
    };

    // Precompute the full J_perp(step) trajectory by inverting the cumulative "adiabatic time"
    // G(J) = integral_{J0}^{J} chi_B(J')^alpha dJ'. Step t is placed at J such that
    // G(J) = (t/num_steps) * G(j_perp_end). This guarantees, BY CONSTRUCTION and independent
    // of the chi_B profile shape, that the schedule (a) consumes its full num_steps budget,
    // (b) reaches j_perp_end, and (c) dwells where chi_B^alpha is large (near criticality) and
    // crosses quickly where it is small. We floor chi_B in the integrand so the ordered region
    // (chi_B -> 0) still receives a small, finite share of steps instead of a single leap.
    std::vector<double> j_traj;  // size num_steps when calibrating; empty for the legacy path
    if (calibrate && !probe_J.empty() && j_perp_end > j_perp_start) {
        const std::size_t GRID = 4096;
        const double h = (j_perp_end - j_perp_start) / static_cast<double>(GRID);
        std::vector<double> Gcum(GRID + 1, 0.0);
        double prev = std::pow(std::max(chi_profile_at(j_perp_start), min_chi_B), alpha);
        for (std::size_t k = 1; k <= GRID; ++k) {
            const double Jk = j_perp_start + static_cast<double>(k) * h;
            const double cur = std::pow(std::max(chi_profile_at(Jk), min_chi_B), alpha);
            Gcum[k] = Gcum[k - 1] + 0.5 * (prev + cur) * h;
            prev = cur;
        }
        const double G_total = Gcum[GRID];
        j_traj.resize(num_steps);
        std::size_t k = 0;
        const double denom = (num_steps > 1) ? static_cast<double>(num_steps - 1) : 1.0;
        for (std::size_t t = 0; t < num_steps; ++t) {
            // t=0 -> J_start, t=num_steps-1 -> J_end (endpoints inclusive).
            const double target = (G_total > 0.0)
                ? (static_cast<double>(t) / denom) * G_total
                : 0.0;
            while (k < GRID && Gcum[k + 1] < target) ++k;
            // Linear interpolation of J within grid cell [k, k+1] at cumulative `target`.
            const double g0 = Gcum[k], g1 = Gcum[k + 1];
            const double frac = (g1 > g0) ? (target - g0) / (g1 - g0) : 0.0;
            j_traj[t] = j_perp_start + (static_cast<double>(k) + frac) * h;
        }
    }

    for (std::size_t step = 0; step < num_steps; ++step) {
        // Profile-driven trajectory when calibrating; otherwise j_perp advances online below.
        if (!j_traj.empty()) j_perp = j_traj[step];
        sweep_replicas(states, replica_rng, j_perp);

        // --- Update best energies ---
        for (std::size_t r = 0; r < replicas; ++r) {
            std::size_t best_slice = 0;
            energies[r] = projected_energy(states[r], &best_slice);
            if (energies[r] < result.best_energy) {
                result.best_energy = energies[r];
                result.best_state = states[r].slice_state(0, best_slice);
            }
        }

        // --- Schedule bookkeeping ---
        // When calibrating, j_perp follows the precomputed inverse-CDF trajectory j_traj (set
        // at the top of the loop), so the online delta_j here is recorded purely as a
        // diagnostic. On the legacy eps_tilde-override path (no profile/trajectory) the online
        // variance drives the update as before.
        const double chi_online = measure_chi_B_raw(states);
        const double chi_prof = chi_profile_at(j_perp);
        const double chi_drive_raw = (chi_prof >= 0.0) ? chi_prof : chi_online;
        const bool floor_hit = (chi_drive_raw < min_chi_B);
        const double chi_drive = std::max(chi_drive_raw, min_chi_B);
        const double delta_j = eps_tilde * std::pow(chi_drive, -alpha);

        result.j_perp_trace.push_back(j_perp);
        result.chi_B_trace.push_back(chi_drive);
        if (dbg) {
            dbg << "run," << step << ',' << j_perp << ',' << chi_drive << ',' << delta_j
                << ',' << eps_tilde << ',' << alpha << ',' << (floor_hit ? 1 : 0)
                << ',' << std::max(chi_online, 0.0) << '\n';
        }

        if (j_traj.empty()) {
            // Legacy online schedule (eps_tilde override): advance by the local adiabaticity step.
            j_perp = std::min(j_perp + delta_j, j_perp_end);
        }
        // else: j_perp is set from j_traj at the start of the next iteration.

        // --- PT swaps: Trotter terms cancel with shared j_perp → purely classical criterion ---
        double accepted = 0.0, attempted = 0.0;
        if ((step + 1) % swap_interval == 0) {
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            for (std::size_t r = 0; r + 1 < replicas; ++r) {
                const double bs_r  = betas_[r]     / static_cast<double>(slices);
                const double bs_r1 = betas_[r + 1] / static_cast<double>(slices);
                const double csum_r  = classical_sum(states[r]);
                const double csum_r1 = classical_sum(states[r + 1]);
                const double delta = (bs_r - bs_r1) * (csum_r1 - csum_r);
                ++attempted;
                if (delta <= 0.0 || uniform(rng_) < std::exp(-delta)) {
                    std::swap(states[r], states[r + 1]);
                    std::swap(energies[r], energies[r + 1]);
                    ++accepted;
                }
            }
        }

        double avg_energy = 0.0;
        for (double e : energies) avg_energy += e;
        avg_energy /= static_cast<double>(replicas);
        result.average_energy_trace.push_back(avg_energy);
        result.swap_acceptance_trace.push_back(attempted > 0.0 ? accepted / attempted : 0.0);

        // NOTE: we deliberately do NOT break when j_perp reaches j_perp_end. The adaptive
        // schedule concentrates its steps near criticality and then crosses the trivial
        // ordered region quickly; once it saturates at j_perp_end it keeps sweeping there for
        // the remaining budget. Running the full num_steps keeps the total sweep budget equal
        // to the standard schedule, which is essential for a fair opt-vs-std benchmark.
    }

    result.final_j_perp = j_perp;
    result.final_states.reserve(replicas);
    result.final_energies.assign(replicas, 0.0);
    for (std::size_t r = 0; r < replicas; ++r) {
        std::size_t best_slice = 0;
        result.final_energies[r] = projected_energy(states[r], &best_slice);
        result.final_states.push_back(states[r].slice_state(0, best_slice));
    }

    return result;
}

} // namespace qanneal
