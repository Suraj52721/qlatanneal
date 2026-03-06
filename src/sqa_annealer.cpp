#include "qanneal/sqa_annealer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace qanneal {

SQAAnnealer::SQAAnnealer(const Hamiltonian &hamiltonian,
                         SQASchedule schedule,
                         std::size_t trotter_slices,
                         std::size_t replicas)
    : backend_(make_backend(BackendKind::CPU, hamiltonian)),
      schedule_(std::move(schedule)),
      slices_(trotter_slices),
      replicas_(replicas),
      rng_(std::random_device{}()) {
    if (schedule_.betas.empty()) {
        throw std::invalid_argument("SQA schedule must contain betas.");
    }
    if (schedule_.betas.size() != schedule_.gammas.size()) {
        throw std::invalid_argument("SQA schedule betas/gammas length mismatch.");
    }
    if (slices_ == 0) {
        throw std::invalid_argument("trotter_slices must be > 0.");
    }
    if (replicas_ == 0) {
        throw std::invalid_argument("replicas must be > 0.");
    }
}

SQAAnnealer::SQAAnnealer(std::shared_ptr<Backend> backend,
                         SQASchedule schedule,
                         std::size_t trotter_slices,
                         std::size_t replicas)
    : backend_(std::move(backend)),
      schedule_(std::move(schedule)),
      slices_(trotter_slices),
      replicas_(replicas),
      rng_(std::random_device{}()) {
    if (!backend_) {
        throw std::invalid_argument("SQAAnnealer requires a backend.");
    }
    if (schedule_.betas.empty()) {
        throw std::invalid_argument("SQA schedule must contain betas.");
    }
    if (schedule_.betas.size() != schedule_.gammas.size()) {
        throw std::invalid_argument("SQA schedule betas/gammas length mismatch.");
    }
    if (slices_ == 0) {
        throw std::invalid_argument("trotter_slices must be > 0.");
    }
    if (replicas_ == 0) {
        throw std::invalid_argument("replicas must be > 0.");
    }
}

void SQAAnnealer::set_seed(std::uint64_t seed) {
    rng_.seed(seed);
}

double SQAAnnealer::trotter_coupling(double beta, double gamma, std::size_t slices) const {
    const double eps = 1e-12;
    const double x = std::max(beta * gamma / static_cast<double>(slices), eps);
    return 0.5 * std::log(1.0 / std::tanh(x));
}

double SQAAnnealer::delta_trotter(const SQAState &state,
                                  std::size_t replica,
                                  std::size_t slice,
                                  std::size_t spin,
                                  double j_perp,
                                  std::size_t slices) const {
    const std::size_t prev = (slice == 0) ? (slices - 1) : (slice - 1);
    const std::size_t next = (slice + 1) % slices;
    const int8_t s = state.at(replica, slice, spin);
    const int8_t s_prev = state.at(replica, prev, spin);
    const int8_t s_next = state.at(replica, next, spin);
    return 2.0 * j_perp * static_cast<double>(s) *
           static_cast<double>(s_prev + s_next);
}

SQAResult SQAAnnealer::run(std::size_t sweeps_per_beta,
                           std::size_t worldline_sweeps,
                           std::size_t cluster_sweeps,
                           std::size_t continuous_time_slices,
                           SQAObserver *observer) {
    if (sweeps_per_beta == 0 && worldline_sweeps == 0 && cluster_sweeps == 0) {
        throw std::invalid_argument("At least one of sweeps_per_beta, worldline_sweeps, or cluster_sweeps must be > 0.");
    }

    const std::size_t n = backend_->size();
    const std::size_t slices = (continuous_time_slices > 0 && continuous_time_slices > slices_)
                               ? continuous_time_slices
                               : slices_;
    SQAState state = SQAState::random(replicas_, slices, n, rng_);

    SQAResult result;
    result.best_energy = std::numeric_limits<double>::infinity();
    result.energy_trace.reserve(schedule_.size());

    auto *sweep_observer = dynamic_cast<SQASweepObserver *>(observer);
    std::vector<double> replica_energies;
    std::vector<std::mt19937_64> replica_rng(replicas_);
    for (std::size_t r = 0; r < replicas_; ++r) {
        replica_rng[r].seed(rng_());
    }

    auto compute_avg_energy = [&](const SQAState &state) {
        replica_energies.assign(replicas_, 0.0);
        double total = 0.0;
        for (std::size_t r = 0; r < replicas_; ++r) {
            double sum = 0.0;
            for (std::size_t t = 0; t < slices; ++t) {
                const int8_t *slice_ptr = state.slice_ptr(r, t);
                sum += backend_->energy(slice_ptr, n);
            }
            replica_energies[r] = sum / static_cast<double>(slices);
            total += sum;
        }
        return total / static_cast<double>(replicas_ * slices);
    };

    // Local field cache: flat array indexed as [replica * slices + slice][spin]
    // fields[i] = h[i] + sum_{j!=i} J[i,j]*s[j]  ->  O(1) delta: -2*s[flip]*fields[flip]
    std::vector<double> local_fields(replicas_ * slices * n, 0.0);
    for (std::size_t r = 0; r < replicas_; ++r) {
        for (std::size_t t = 0; t < slices; ++t) {
            backend_->compute_local_fields(
                state.slice_ptr(r, t), n,
                &local_fields[(r * slices + t) * n]);
        }
    }

    // Pre-allocated cluster sweep buffers (one set per replica for thread safety)
    std::vector<std::vector<char>> cluster_buf(replicas_, std::vector<char>(slices));
    std::vector<std::vector<std::size_t>> stack_buf(replicas_);
    for (auto &s : stack_buf) {
        s.reserve(slices);
    }

    // Per-replica spin visit order — shuffled each sweep for better ergodicity
    std::vector<std::vector<std::size_t>> spin_orders(replicas_);
    for (std::size_t r = 0; r < replicas_; ++r) {
        spin_orders[r].resize(n);
        std::iota(spin_orders[r].begin(), spin_orders[r].end(), 0);
    }

    for (std::size_t step = 0; step < schedule_.size(); ++step) {
        const double beta = schedule_.betas[step];
        const double gamma = schedule_.gammas[step];
        const double j_perp = trotter_coupling(beta, gamma, slices);
        const double beta_scale = beta / static_cast<double>(slices);
        const double join_prob = 1.0 - std::exp(-2.0 * j_perp);

        if (!sweep_observer) {
#ifdef _OPENMP
#pragma omp parallel for if(replicas_ > 1) schedule(static)
#endif
            for (long long rr = 0; rr < static_cast<long long>(replicas_); ++rr) {
                const std::size_t replica = static_cast<std::size_t>(rr);
                auto &local_rng = replica_rng[replica];
                std::uniform_real_distribution<double> uniform(0.0, 1.0);
                std::uniform_int_distribution<std::size_t> slice_pick(0, slices - 1);

                // Each replica owns exclusive regions of these arrays
                double *my_lf = local_fields.data() + replica * slices * n;
                auto &my_spin_order = spin_orders[replica];
                auto &my_cluster = cluster_buf[replica];
                auto &my_stack = stack_buf[replica];

                // ----------------------------------------------------------------
                // Metropolis slice sweeps
                // Hot path: O(1) delta per spin via cached local field
                // ----------------------------------------------------------------
                for (std::size_t sweep = 0; sweep < sweeps_per_beta; ++sweep) {
                    std::shuffle(my_spin_order.begin(), my_spin_order.end(), local_rng);
                    for (std::size_t t = 0; t < slices; ++t) {
                        int8_t *sptr = state.slice_ptr(replica, t);
                        double *fields = my_lf + t * n;
                        const std::size_t prev_t = (t == 0) ? (slices - 1) : (t - 1);
                        const std::size_t next_t = (t + 1) % slices;
                        const int8_t *prev_ptr = state.slice_ptr(replica, prev_t);
                        const int8_t *next_ptr = state.slice_ptr(replica, next_t);

                        for (std::size_t idx = 0; idx < n; ++idx) {
                            const std::size_t spin = my_spin_order[idx];
                            const int8_t s = sptr[spin];
                            // O(1) classical delta from cached field
                            const double de_classical =
                                -2.0 * static_cast<double>(s) * fields[spin];
                            // O(1) Trotter delta via raw pointer (no bounds check)
                            const double de_trotter =
                                2.0 * j_perp * static_cast<double>(s) *
                                static_cast<double>(prev_ptr[spin] + next_ptr[spin]);
                            const double de = beta_scale * de_classical + de_trotter;
                            if (de <= 0.0 || uniform(local_rng) < std::exp(-de)) {
                                sptr[spin] = static_cast<int8_t>(-s);
                                // O(N) dense / O(degree) sparse field update
                                backend_->update_local_fields_after_flip(
                                    fields, sptr, n, spin, s);
                            }
                        }
                    }
                }

                // ----------------------------------------------------------------
                // Cluster sweeps (Swendsen-Wang along imaginary-time axis)
                // Pre-allocated buffers: no heap allocs inside the spin loop
                // ----------------------------------------------------------------
                for (std::size_t sweep = 0; sweep < cluster_sweeps; ++sweep) {
                    std::shuffle(my_spin_order.begin(), my_spin_order.end(), local_rng);
                    for (std::size_t idx = 0; idx < n; ++idx) {
                        const std::size_t spin = my_spin_order[idx];
                        const std::size_t seed = slice_pick(local_rng);

                        std::fill(my_cluster.begin(), my_cluster.end(), 0);
                        my_stack.clear();
                        my_stack.push_back(seed);
                        my_cluster[seed] = 1;
                        // All spins joining the cluster share this value
                        const int8_t seed_spin = state.slice_ptr(replica, seed)[spin];

                        while (!my_stack.empty()) {
                            const std::size_t t = my_stack.back();
                            my_stack.pop_back();
                            const std::size_t prev = (t == 0) ? (slices - 1) : (t - 1);
                            const std::size_t next = (t + 1) % slices;

                            if (!my_cluster[prev] &&
                                state.slice_ptr(replica, prev)[spin] == seed_spin &&
                                uniform(local_rng) < join_prob) {
                                my_cluster[prev] = 1;
                                my_stack.push_back(prev);
                            }
                            if (!my_cluster[next] &&
                                state.slice_ptr(replica, next)[spin] == seed_spin &&
                                uniform(local_rng) < join_prob) {
                                my_cluster[next] = 1;
                                my_stack.push_back(next);
                            }
                        }

                        // Compute delta using cached fields (O(1) per cluster slice)
                        double delta_classical = 0.0;
                        double delta_time = 0.0;
                        for (std::size_t t = 0; t < slices; ++t) {
                            if (!my_cluster[t]) {
                                continue;
                            }
                            delta_classical +=
                                -2.0 * static_cast<double>(seed_spin) *
                                (my_lf + t * n)[spin];

                            const std::size_t prev = (t == 0) ? (slices - 1) : (t - 1);
                            const std::size_t next = (t + 1) % slices;
                            if (!my_cluster[prev]) {
                                delta_time += 2.0 * j_perp *
                                              static_cast<double>(seed_spin) *
                                              static_cast<double>(
                                                  state.slice_ptr(replica, prev)[spin]);
                            }
                            if (!my_cluster[next]) {
                                delta_time += 2.0 * j_perp *
                                              static_cast<double>(seed_spin) *
                                              static_cast<double>(
                                                  state.slice_ptr(replica, next)[spin]);
                            }
                        }

                        const double delta = beta_scale * delta_classical + delta_time;
                        if (delta <= 0.0 || uniform(local_rng) < std::exp(-delta)) {
                            for (std::size_t t = 0; t < slices; ++t) {
                                if (!my_cluster[t]) {
                                    continue;
                                }
                                int8_t *sptr = state.slice_ptr(replica, t);
                                sptr[spin] = static_cast<int8_t>(-seed_spin);
                                backend_->update_local_fields_after_flip(
                                    my_lf + t * n, sptr, n, spin, seed_spin);
                            }
                        }
                    }
                }

                // ----------------------------------------------------------------
                // Worldline sweeps (flip spin across all slices simultaneously)
                // Trotter term cancels — only classical delta matters
                // ----------------------------------------------------------------
                for (std::size_t sweep = 0; sweep < worldline_sweeps; ++sweep) {
                    std::shuffle(my_spin_order.begin(), my_spin_order.end(), local_rng);
                    for (std::size_t idx = 0; idx < n; ++idx) {
                        const std::size_t spin = my_spin_order[idx];
                        double de_classical = 0.0;
                        for (std::size_t t = 0; t < slices; ++t) {
                            const int8_t s_t = state.slice_ptr(replica, t)[spin];
                            de_classical += -2.0 * static_cast<double>(s_t) *
                                            (my_lf + t * n)[spin];
                        }
                        const double de = beta_scale * de_classical;
                        if (de <= 0.0 || uniform(local_rng) < std::exp(-de)) {
                            for (std::size_t t = 0; t < slices; ++t) {
                                int8_t *sptr = state.slice_ptr(replica, t);
                                const int8_t old = sptr[spin];
                                sptr[spin] = static_cast<int8_t>(-old);
                                backend_->update_local_fields_after_flip(
                                    my_lf + t * n, sptr, n, spin, old);
                            }
                        }
                    }
                }
            }
        } else {
            // Observer path: sequential, used only for tracing/debugging.
            // Uses pre-allocated cluster buffers but skips local field cache
            // to keep this branch simple.
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            std::uniform_int_distribution<std::size_t> slice_pick(0, slices - 1);
            auto &my_cluster = cluster_buf[0];
            auto &my_stack = stack_buf[0];

            for (std::size_t replica = 0; replica < replicas_; ++replica) {
                for (std::size_t sweep = 0; sweep < sweeps_per_beta; ++sweep) {
                    for (std::size_t slice = 0; slice < slices; ++slice) {
                        int8_t *slice_ptr = state.slice_ptr(replica, slice);
                        for (std::size_t spin = 0; spin < n; ++spin) {
                            const double delta_classical =
                                backend_->delta_energy(slice_ptr, n, spin);
                            const double delta = beta_scale * delta_classical +
                                                 delta_trotter(state, replica, slice, spin, j_perp, slices);
                            if (delta <= 0.0 || uniform(rng_) < std::exp(-delta)) {
                                slice_ptr[spin] = static_cast<int8_t>(-slice_ptr[spin]);
                            }
                        }
                    }
                    const double avg_energy = compute_avg_energy(state);
                    sweep_observer->record_sweep(step,
                                                 replica,
                                                 sweep,
                                                 SQASweepPhase::Slice,
                                                 beta,
                                                 gamma,
                                                 avg_energy,
                                                 replica_energies,
                                                 state);
                }

                for (std::size_t sweep = 0; sweep < cluster_sweeps; ++sweep) {
                    for (std::size_t spin = 0; spin < n; ++spin) {
                        const std::size_t seed = slice_pick(rng_);
                        std::fill(my_cluster.begin(), my_cluster.end(), 0);
                        my_stack.clear();
                        my_stack.push_back(seed);
                        my_cluster[seed] = 1;
                        const int8_t seed_spin = state.at(replica, seed, spin);

                        while (!my_stack.empty()) {
                            const std::size_t t = my_stack.back();
                            my_stack.pop_back();
                            const std::size_t prev = (t == 0) ? (slices - 1) : (t - 1);
                            const std::size_t next = (t + 1) % slices;

                            if (!my_cluster[prev] &&
                                state.at(replica, prev, spin) == seed_spin &&
                                uniform(rng_) < join_prob) {
                                my_cluster[prev] = 1;
                                my_stack.push_back(prev);
                            }
                            if (!my_cluster[next] &&
                                state.at(replica, next, spin) == seed_spin &&
                                uniform(rng_) < join_prob) {
                                my_cluster[next] = 1;
                                my_stack.push_back(next);
                            }
                        }

                        double delta_classical = 0.0;
                        double delta_time = 0.0;
                        for (std::size_t t = 0; t < slices; ++t) {
                            if (!my_cluster[t]) {
                                continue;
                            }
                            int8_t *slice_ptr = state.slice_ptr(replica, t);
                            delta_classical += backend_->delta_energy(slice_ptr, n, spin);

                            const std::size_t prev = (t == 0) ? (slices - 1) : (t - 1);
                            const std::size_t next = (t + 1) % slices;
                            const int8_t s_t = slice_ptr[spin];
                            if (!my_cluster[prev]) {
                                const int8_t s_prev = state.at(replica, prev, spin);
                                delta_time += 2.0 * j_perp * static_cast<double>(s_t) * static_cast<double>(s_prev);
                            }
                            if (!my_cluster[next]) {
                                const int8_t s_next = state.at(replica, next, spin);
                                delta_time += 2.0 * j_perp * static_cast<double>(s_t) * static_cast<double>(s_next);
                            }
                        }

                        const double delta = beta_scale * delta_classical + delta_time;
                        if (delta <= 0.0 || uniform(rng_) < std::exp(-delta)) {
                            for (std::size_t t = 0; t < slices; ++t) {
                                if (!my_cluster[t]) {
                                    continue;
                                }
                                int8_t *slice_ptr = state.slice_ptr(replica, t);
                                slice_ptr[spin] = static_cast<int8_t>(-slice_ptr[spin]);
                            }
                        }
                    }
                    const double avg_energy = compute_avg_energy(state);
                    sweep_observer->record_sweep(step,
                                                 replica,
                                                 sweep,
                                                 SQASweepPhase::Cluster,
                                                 beta,
                                                 gamma,
                                                 avg_energy,
                                                 replica_energies,
                                                 state);
                }

                for (std::size_t sweep = 0; sweep < worldline_sweeps; ++sweep) {
                    for (std::size_t spin = 0; spin < n; ++spin) {
                        double delta_classical = 0.0;
                        for (std::size_t slice = 0; slice < slices; ++slice) {
                            const int8_t *slice_ptr = state.slice_ptr(replica, slice);
                            delta_classical += backend_->delta_energy(slice_ptr, n, spin);
                        }
                        const double delta = beta_scale * delta_classical;
                        if (delta <= 0.0 || uniform(rng_) < std::exp(-delta)) {
                            for (std::size_t slice = 0; slice < slices; ++slice) {
                                int8_t *slice_ptr = state.slice_ptr(replica, slice);
                                slice_ptr[spin] = static_cast<int8_t>(-slice_ptr[spin]);
                            }
                        }
                    }
                    const double avg_energy = compute_avg_energy(state);
                    sweep_observer->record_sweep(step,
                                                 replica,
                                                 sweep,
                                                 SQASweepPhase::Worldline,
                                                 beta,
                                                 gamma,
                                                 avg_energy,
                                                 replica_energies,
                                                 state);
                }
            }
        }

        double avg_energy = 0.0;
        std::size_t total_states = replicas_ * slices;
        for (std::size_t replica = 0; replica < replicas_; ++replica) {
            for (std::size_t slice = 0; slice < slices; ++slice) {
                const int8_t *slice_ptr = state.slice_ptr(replica, slice);
                const double e = backend_->energy(slice_ptr, n);
                avg_energy += e;
                if (e < result.best_energy) {
                    result.best_energy = e;
                    result.best_state = state.slice_state(replica, slice);
                }
            }
        }
        avg_energy /= static_cast<double>(total_states);
        result.energy_trace.push_back(avg_energy);

        if (observer) {
            observer->record(step, beta, gamma, avg_energy, state);
        }
    }

    return result;
}

}
