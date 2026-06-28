#include "qanneal/sqa_parallel_tempering.hpp"

#include <algorithm>
#include <cmath>
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
        std::size_t continuous_time_slices) {
    if (num_steps == 0) throw std::invalid_argument("num_steps must be > 0.");
    if (swap_interval == 0) throw std::invalid_argument("swap_interval must be > 0.");
    if (sweeps_per_step == 0 && worldline_sweeps == 0 && cluster_sweeps == 0) {
        throw std::invalid_argument("At least one of sweeps_per_step/worldline_sweeps/cluster_sweeps must be > 0.");
    }
    if (eps_tilde <= 0.0) throw std::invalid_argument("eps_tilde must be > 0.");

    const std::size_t n = backend_->size();
    const std::size_t replicas = betas_.size();
    const std::size_t slices = (continuous_time_slices > 0 && continuous_time_slices > slices_)
                                 ? continuous_time_slices
                                 : slices_;
    const double nM = static_cast<double>(n * slices);
    const double min_chi_B = 1e-4 * nM;

    // Start j_perp from the hottest replica's coupling (most quantum end of the ladder).
    double j_perp = trotter_coupling(betas_[0], gammas_[0], slices);
    // If j_perp_end not given, use the coldest replica's initial coupling as target.
    if (j_perp_end <= 0.0) {
        j_perp_end = trotter_coupling(betas_.back(), gammas_.back(), slices);
    }
    if (j_perp_end <= j_perp) {
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

    for (std::size_t step = 0; step < num_steps; ++step) {
        const double join_prob = 1.0 - std::exp(-2.0 * j_perp);

#ifdef _OPENMP
#pragma omp parallel for if(replicas > 1) schedule(static)
#endif
        for (std::size_t r = 0; r < replicas; ++r) {
            auto &state = states[r];
            auto &rr = replica_rng[r];
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
                        const double dtime  = 2.0 * j_perp * static_cast<double>(s) *
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
                            dtime   += 2.0 * j_perp * bond;
                            delta_B += -2.0 * bond;
                        }
                        if (!my_cluster[next]) {
                            const double bond = static_cast<double>(s_t) *
                                               static_cast<double>(state.slice_ptr(0, next)[i]);
                            dtime   += 2.0 * j_perp * bond;
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

        // --- Update best energies ---
        for (std::size_t r = 0; r < replicas; ++r) {
            std::size_t best_slice = 0;
            energies[r] = projected_energy(states[r], &best_slice);
            if (energies[r] < result.best_energy) {
                result.best_energy = energies[r];
                result.best_state = states[r].slice_state(0, best_slice);
            }
        }

        // --- Compute chi_B from all within-step B samples across all replicas ---
        // Total: replicas * (sweeps_per_step + cluster_sweeps) samples per step.
        {
            double B_sum = 0.0, B2_sum = 0.0;
            std::size_t total_samples = 0;
            for (std::size_t r = 0; r < replicas; ++r) {
                for (double b : B_sweep_samples[r]) {
                    B_sum  += b;
                    B2_sum += b * b;
                    ++total_samples;
                }
            }
            double chi_B;
            if (total_samples >= 2) {
                const double B_mean  = B_sum  / static_cast<double>(total_samples);
                const double B2_mean = B2_sum / static_cast<double>(total_samples);
                chi_B = std::max(B2_mean - B_mean * B_mean, min_chi_B);
            } else {
                const double b = bond_sum_one(states[0]);
                const double b_avg = b / nM;
                chi_B = std::max(nM * (1.0 - b_avg * b_avg), min_chi_B);
            }

            // --- Optimal schedule update ---
            const double delta_j = eps_tilde * std::pow(chi_B, -alpha);
            j_perp = std::min(j_perp + delta_j, j_perp_end);
        }

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

        if (j_perp >= j_perp_end) break;
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

} // namespace qanneal
