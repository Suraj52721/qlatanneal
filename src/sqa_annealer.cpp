#include "qanneal/sqa_annealer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
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

// ---------------------------------------------------------------------------
// Compute the imaginary-time bond sum B(sigma) = sum_{k,i} s^k_i s^{k+1}_i
// for a single replica inside an SQAState.
// ---------------------------------------------------------------------------
static double bond_sum_replica(const SQAState &state,
                                std::size_t replica,
                                std::size_t slices,
                                std::size_t n) {
    double sum = 0.0;
    for (std::size_t t = 0; t < slices; ++t) {
        const std::size_t tn = (t + 1) % slices;
        const int8_t *a = state.slice_ptr(replica, t);
        const int8_t *b = state.slice_ptr(replica, tn);
        for (std::size_t i = 0; i < n; ++i) {
            sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }
    }
    return sum;
}

SQAResult SQAAnnealer::run_optimal(double beta,
                                   double j_perp_start,
                                   double j_perp_end,
                                   double eps_tilde,
                                   double alpha,
                                   std::size_t num_steps,
                                   std::size_t sweeps_per_step,
                                   std::size_t worldline_sweeps,
                                   std::size_t cluster_sweeps,
                                   std::size_t calib_probes,
                                   std::size_t calib_sweeps,
                                   const std::string &debug_csv_path,
                                   double beta_ramp_fraction,
                                   double beta_ramp_start) {
    if (num_steps == 0) {
        throw std::invalid_argument("num_steps must be > 0.");
    }
    if (sweeps_per_step == 0 && worldline_sweeps == 0 && cluster_sweeps == 0) {
        throw std::invalid_argument("At least one of sweeps_per_step, worldline_sweeps, or cluster_sweeps must be > 0.");
    }
    if (j_perp_end <= j_perp_start) {
        throw std::invalid_argument("j_perp_end must be > j_perp_start.");
    }
    // eps_tilde <= 0 now requests budget calibration (see calibration block below); a positive
    // value keeps the legacy online update. This matches SQAParallelTemperingAnnealer::run_optimal.
    const bool calibrate = (eps_tilde <= 0.0);

    const std::size_t n = backend_->size();
    const std::size_t slices = slices_;
    const double beta_scale = beta / static_cast<double>(slices);
    const double nM = static_cast<double>(n * slices);
    double min_chi_B = 1e-4 * nM;  // rescaled below when calibrating

    SQAState state = SQAState::random(replicas_, slices, n, rng_);

    SQAResult result;
    result.best_energy = std::numeric_limits<double>::infinity();
    result.energy_trace.reserve(num_steps);
    result.j_perp_trace.reserve(num_steps);

    std::vector<std::mt19937_64> replica_rng(replicas_);
    for (std::size_t r = 0; r < replicas_; ++r) {
        replica_rng[r].seed(rng_());
    }

    std::vector<double> local_fields(replicas_ * slices * n, 0.0);
    for (std::size_t r = 0; r < replicas_; ++r) {
        for (std::size_t t = 0; t < slices; ++t) {
            backend_->compute_local_fields(
                state.slice_ptr(r, t), n,
                &local_fields[(r * slices + t) * n]);
        }
    }

    std::vector<std::vector<char>> cluster_buf(replicas_, std::vector<char>(slices));
    std::vector<std::vector<std::size_t>> stack_buf(replicas_);
    for (auto &s : stack_buf) s.reserve(slices);

    std::vector<std::vector<std::size_t>> spin_orders(replicas_);
    for (std::size_t r = 0; r < replicas_; ++r) {
        spin_orders[r].resize(n);
        std::iota(spin_orders[r].begin(), spin_orders[r].end(), 0);
    }

    // Per-replica B sample buffers: one sample per (Metropolis + cluster) sweep per replica.
    // These give replicas_ * (sweeps_per_step + cluster_sweeps) chi_B samples each step,
    // replacing the old single-snapshot (and broken single-replica) estimators.
    const std::size_t samples_per_replica = sweeps_per_step + cluster_sweeps;
    std::vector<std::vector<double>> B_sweep_samples(replicas_);
    for (auto &v : B_sweep_samples) {
        v.reserve(samples_per_replica > 0 ? samples_per_replica : 1);
    }

    // Per-step replica sweep, factored so the calibration pre-pass and the main loop drive
    // chi_B with the *same* sampler (mirrors SQAParallelTemperingAnnealer::run_optimal). Operates
    // on the captured `state`, `local_fields`, and `replica_rng`; j_perp and the sweep counts are
    // parameters so calibration can probe at a fixed j_perp with fewer sweeps. Fills B_sweep_samples.
    auto sweep_current = [&](double j_perp_local, double beta_scale_local, std::size_t n_metro,
                             std::size_t n_cluster, std::size_t n_world) {
        const double join_prob = 1.0 - std::exp(-2.0 * j_perp_local);
#ifdef _OPENMP
#pragma omp parallel for if(replicas_ > 1) schedule(static)
#endif
        for (long long rr = 0; rr < static_cast<long long>(replicas_); ++rr) {
            const std::size_t replica = static_cast<std::size_t>(rr);
            auto &local_rng = replica_rng[replica];
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            std::uniform_int_distribution<std::size_t> slice_pick(0, slices - 1);

            double *my_lf = local_fields.data() + replica * slices * n;
            auto &my_spin_order = spin_orders[replica];
            auto &my_cluster = cluster_buf[replica];
            auto &my_stack = stack_buf[replica];

            // Running B for this replica — exact incremental updates after each accepted flip.
            double b_r = bond_sum_replica(state, replica, slices, n);
            auto &samples = B_sweep_samples[replica];
            samples.clear();

            for (std::size_t sweep = 0; sweep < n_metro; ++sweep) {
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
                        const int8_t s_prev = prev_ptr[spin];
                        const int8_t s_next = next_ptr[spin];
                        const int nn_sum = static_cast<int>(s_prev) + static_cast<int>(s_next);
                        const double de_classical = -2.0 * static_cast<double>(s) * fields[spin];
                        const double de_trotter =
                            2.0 * j_perp_local * static_cast<double>(s) * static_cast<double>(nn_sum);
                        const double de = beta_scale_local * de_classical + de_trotter;
                        if (de <= 0.0 || uniform(local_rng) < std::exp(-de)) {
                            sptr[spin] = static_cast<int8_t>(-s);
                            backend_->update_local_fields_after_flip(fields, sptr, n, spin, s);
                            // delta_B = -2*s*(s_prev + s_next): exact, no division by j_perp.
                            b_r += -2.0 * static_cast<double>(s) * static_cast<double>(nn_sum);
                        }
                    }
                }
                // One B sample per full Metropolis sweep.
                samples.push_back(b_r);
            }

            for (std::size_t sweep = 0; sweep < n_cluster; ++sweep) {
                std::shuffle(my_spin_order.begin(), my_spin_order.end(), local_rng);
                for (std::size_t idx = 0; idx < n; ++idx) {
                    const std::size_t spin = my_spin_order[idx];
                    const std::size_t seed = slice_pick(local_rng);

                    std::fill(my_cluster.begin(), my_cluster.end(), 0);
                    my_stack.clear();
                    my_stack.push_back(seed);
                    my_cluster[seed] = 1;
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

                    double delta_classical = 0.0;
                    double delta_time = 0.0;
                    double delta_B = 0.0;
                    for (std::size_t t = 0; t < slices; ++t) {
                        if (!my_cluster[t]) continue;
                        delta_classical += -2.0 * static_cast<double>(seed_spin) *
                                           (my_lf + t * n)[spin];
                        const std::size_t prev = (t == 0) ? (slices - 1) : (t - 1);
                        const std::size_t next = (t + 1) % slices;
                        if (!my_cluster[prev]) {
                            const double bond =
                                static_cast<double>(seed_spin) *
                                static_cast<double>(state.slice_ptr(replica, prev)[spin]);
                            delta_time += 2.0 * j_perp_local * bond;
                            delta_B    += -2.0 * bond;
                        }
                        if (!my_cluster[next]) {
                            const double bond =
                                static_cast<double>(seed_spin) *
                                static_cast<double>(state.slice_ptr(replica, next)[spin]);
                            delta_time += 2.0 * j_perp_local * bond;
                            delta_B    += -2.0 * bond;
                        }
                    }

                    const double delta = beta_scale_local * delta_classical + delta_time;
                    if (delta <= 0.0 || uniform(local_rng) < std::exp(-delta)) {
                        for (std::size_t t = 0; t < slices; ++t) {
                            if (!my_cluster[t]) continue;
                            int8_t *sptr = state.slice_ptr(replica, t);
                            sptr[spin] = static_cast<int8_t>(-seed_spin);
                            backend_->update_local_fields_after_flip(
                                my_lf + t * n, sptr, n, spin, seed_spin);
                        }
                        b_r += delta_B;
                    }
                }
                // One B sample per full cluster sweep.
                samples.push_back(b_r);
            }

            for (std::size_t sweep = 0; sweep < n_world; ++sweep) {
                std::shuffle(my_spin_order.begin(), my_spin_order.end(), local_rng);
                for (std::size_t idx = 0; idx < n; ++idx) {
                    const std::size_t spin = my_spin_order[idx];
                    double de_classical = 0.0;
                    for (std::size_t t = 0; t < slices; ++t) {
                        const int8_t s_t = state.slice_ptr(replica, t)[spin];
                        de_classical += -2.0 * static_cast<double>(s_t) *
                                        (my_lf + t * n)[spin];
                    }
                    const double de = beta_scale_local * de_classical;
                    if (de <= 0.0 || uniform(local_rng) < std::exp(-de)) {
                        for (std::size_t t = 0; t < slices; ++t) {
                            int8_t *sptr = state.slice_ptr(replica, t);
                            const int8_t old = sptr[spin];
                            sptr[spin] = static_cast<int8_t>(-old);
                            backend_->update_local_fields_after_flip(
                                my_lf + t * n, sptr, n, spin, old);
                        }
                        // Worldline flips preserve all imaginary-time bonds: ΔB = 0.
                    }
                }
                // No B sample: worldline sweeps cannot change B.
            }
        }
    };

    // Raw chi_B = Var(B) from the current B_sweep_samples (NO min floor; caller floors). Uses the
    // captured `state` for the single-replica fallback/guard. Matches the SQAPT estimator.
    auto measure_chi_B_raw = [&]() -> double {
        double B_sum = 0.0, B2_sum = 0.0;
        std::size_t total_samples = 0;
        for (std::size_t r = 0; r < replicas_; ++r) {
            for (double b : B_sweep_samples[r]) {
                B_sum  += b;
                B2_sum += b * b;
                ++total_samples;
            }
        }
        double var;
        if (total_samples >= 2) {
            const double B_mean  = B_sum  / static_cast<double>(total_samples);
            const double B2_mean = B2_sum / static_cast<double>(total_samples);
            var = B2_mean - B_mean * B_mean;
        } else {
            const double B = bond_sum_replica(state, 0, slices, n);
            const double b_avg = B / nM;
            var = nM * (1.0 - b_avg * b_avg);
        }
        // Single-replica guard: within-step variance underestimates near criticality.
        if (replicas_ == 1 && !B_sweep_samples[0].empty()) {
            const double b_avg = B_sweep_samples[0].back() / nM;
            var = std::max(var, nM * (1.0 - b_avg * b_avg));
        }
        return var;
    };

    // Scan all worldline slices: update best energy/state and push (avg_energy, j_perp, chi) traces.
    // Shared by the phase-1 thermal ramp and the phase-2 adaptive J_perp loop.
    auto record_step = [&](double j_perp_val, double chi_val) {
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
        result.energy_trace.push_back(avg_energy / static_cast<double>(total_states));
        result.j_perp_trace.push_back(j_perp_val);
        result.chi_B_trace.push_back(chi_val);
    };

    // --- Two-phase split (Day-1 protocol) ---
    // Phase 1 (beta_ramp_fraction of the budget): ramp beta from beta_ramp_start up to the target
    //   `beta` at FIXED j_perp = j_perp_start (strong transverse field) — this is the THERMAL anneal
    //   that fixed-beta quantum-only SQA was missing, and without which SQA-opt read out garbage.
    // Phase 2 (the rest): fix beta and run the calibrated adaptive J_perp schedule.
    // beta_ramp_fraction <= 0 recovers the single-phase behaviour.
    const double bscale_cold = beta / static_cast<double>(slices);
    std::size_t n_phase1 = 0;
    if (beta_ramp_fraction > 0.0) {
        n_phase1 = static_cast<std::size_t>(beta_ramp_fraction * static_cast<double>(num_steps));
        if (n_phase1 >= num_steps) n_phase1 = num_steps - 1;  // always leave >=1 step for phase 2
    }
    const std::size_t n_phase2 = num_steps - n_phase1;

    // Optional per-step diagnostic CSV (opened before calibration so probe rows are captured).
    std::vector<double> probe_J, probe_chi;
    std::ofstream dbg;
    if (!debug_csv_path.empty()) {
        dbg.open(debug_csv_path);
        if (dbg) {
            dbg << "phase,step_index,j_perp,chi_B,delta_j,eps_tilde,alpha,floor_hit,chi_B_online\n";
            dbg.setf(std::ios::scientific);
            dbg.precision(8);
        } else {
            std::cerr << "[qanneal SQA run_optimal] WARNING: could not open debug CSV '"
                      << debug_csv_path << "'.\n";
        }
    }

    // --- PHASE 1: thermal anneal (ramp beta at fixed j_perp_start) ---
    for (std::size_t step = 0; step < n_phase1; ++step) {
        const double frac = (n_phase1 > 1) ? static_cast<double>(step) / static_cast<double>(n_phase1 - 1) : 1.0;
        const double beta_t = beta_ramp_start + (beta - beta_ramp_start) * frac;
        sweep_current(j_perp_start, beta_t / static_cast<double>(slices),
                      sweeps_per_step, cluster_sweeps, worldline_sweeps);
        const double chi = std::max(measure_chi_B_raw(), 0.0);
        record_step(j_perp_start, chi);
        if (dbg) {
            dbg << "beta_ramp," << step << ',' << j_perp_start << ',' << chi << ",0,0,"
                << alpha << ",0," << beta_t << '\n';
        }
    }

    // --- Calibration pre-pass: set eps_tilde = (1/n_phase2) * integral chi_B(J)^alpha dJ, and
    //     build the profile chi_B(J) that drives the inverse-CDF trajectory (see SQAPT for the
    //     full derivation). The +alpha exponent and the /n_phase2 are what make the traversal
    //     budget-exact. Runs at the cold beta on the post-phase-1 state; the pilot mutates
    //     `state`/`local_fields`, so save and restore them.
    if (calibrate) {
        const std::size_t M = std::max<std::size_t>(calib_probes, 2);
        const std::size_t cs = std::max<std::size_t>(calib_sweeps, 1);
        const std::size_t pilot_cluster = cluster_sweeps > 0 ? std::max<std::size_t>(cluster_sweeps, 1) : 0;

        SQAState saved_state = state;
        std::vector<double> saved_fields = local_fields;

        const double dJ = j_perp_end - j_perp_start;
        const double dJ_probe = dJ / static_cast<double>(M);
        probe_J.resize(M);
        probe_chi.resize(M, 0.0);
        double integral = 0.0;
        for (std::size_t m = 0; m < M; ++m) {
            const double Jm = j_perp_start + (static_cast<double>(m) + 0.5) * dJ_probe;
            sweep_current(Jm, bscale_cold, cs, pilot_cluster, 0);
            double chi = measure_chi_B_raw();
            if (!(chi > 0.0)) chi = 0.0;
            probe_J[m] = Jm;
            probe_chi[m] = chi;
            integral += std::pow(std::max(chi, 1e-12), alpha) * dJ_probe;
            if (dbg) {
                dbg << "calib," << (static_cast<long long>(m) - static_cast<long long>(M)) << ','
                    << Jm << ',' << chi << ",0,0," << alpha << ",0," << chi << '\n';
            }
        }

        // Restore the run's post-phase-1 state; RNG streams are allowed to advance (pilot noise).
        state = saved_state;
        local_fields = saved_fields;

        eps_tilde = integral / static_cast<double>(n_phase2);
        if (!(eps_tilde > 0.0) || !std::isfinite(eps_tilde)) {
            std::cerr << "[qanneal SQA run_optimal] WARNING: calibration produced non-positive "
                         "eps_tilde (integral=" << integral << "); falling back to linear ramp.\n";
            eps_tilde = std::max(dJ, 0.0) / static_cast<double>(num_steps);
            if (!(eps_tilde > 0.0)) eps_tilde = 1e-9;
        }

        std::vector<double> sorted = probe_chi;
        std::sort(sorted.begin(), sorted.end());
        const double max_chi = sorted.back();
        min_chi_B = std::max(1e-4 * max_chi, 1e-12);
    }
    result.calibrated_eps_tilde = eps_tilde;
    result.j_perp_start = j_perp_start;
    result.resolved_j_perp_end = j_perp_end;
    result.chi_B_trace.reserve(num_steps);

    // Interpolate the calibration profile chi_B(J) (linear between probes; clamped). -1 when no
    // profile exists (eps_tilde override path).
    auto chi_profile_at = [&](double J) -> double {
        const std::size_t M = probe_J.size();
        if (M == 0) return -1.0;
        if (J <= probe_J.front()) return probe_chi.front();
        if (J >= probe_J.back())  return probe_chi.back();
        std::size_t hi = 1;
        while (hi < M && probe_J[hi] < J) ++hi;
        const std::size_t lo = hi - 1;
        const double t = (J - probe_J[lo]) / (probe_J[hi] - probe_J[lo]);
        return probe_chi[lo] + t * (probe_chi[hi] - probe_chi[lo]);
    };

    // Precompute the J_perp(step) trajectory by inverting G(J) = integral chi_B(J')^alpha dJ', so
    // the schedule consumes all num_steps, reaches j_perp_end, and dwells where chi_B is large.
    std::vector<double> j_traj;
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
        j_traj.resize(n_phase2);
        std::size_t k = 0;
        const double denom = (n_phase2 > 1) ? static_cast<double>(n_phase2 - 1) : 1.0;
        for (std::size_t t = 0; t < n_phase2; ++t) {
            const double target = (G_total > 0.0)
                ? (static_cast<double>(t) / denom) * G_total
                : 0.0;
            while (k < GRID && Gcum[k + 1] < target) ++k;
            const double g0 = Gcum[k], g1 = Gcum[k + 1];
            const double frac = (g1 > g0) ? (target - g0) / (g1 - g0) : 0.0;
            j_traj[t] = j_perp_start + (static_cast<double>(k) + frac) * h;
        }
    }

    // --- PHASE 2: adaptive J_perp schedule at fixed (cold) beta ---
    double j_perp = j_perp_start;

    for (std::size_t step = 0; step < n_phase2; ++step) {
        // Profile-driven trajectory when calibrating; otherwise j_perp advances online below.
        if (!j_traj.empty()) j_perp = j_traj[step];
        sweep_current(j_perp, bscale_cold, sweeps_per_step, cluster_sweeps, worldline_sweeps);

        // --- chi_B: profile-driven when calibrating, online otherwise (online kept as diagnostic) ---
        const double chi_online = measure_chi_B_raw();
        const double chi_prof = chi_profile_at(j_perp);
        const double chi_drive_raw = (chi_prof >= 0.0) ? chi_prof : chi_online;
        const bool floor_hit = (chi_drive_raw < min_chi_B);
        const double chi_drive = std::max(chi_drive_raw, min_chi_B);
        const double delta_j = eps_tilde * std::pow(chi_drive, -alpha);

        record_step(j_perp, chi_drive);
        if (dbg) {
            dbg << "run," << (n_phase1 + step) << ',' << j_perp << ',' << chi_drive << ',' << delta_j
                << ',' << eps_tilde << ',' << alpha << ',' << (floor_hit ? 1 : 0)
                << ',' << std::max(chi_online, 0.0) << '\n';
        }

        // NOTE: we deliberately do NOT break when j_perp reaches j_perp_end — the run always
        // consumes all num_steps so opt and std use an identical sweep budget (fair benchmark).
        if (j_traj.empty()) {
            j_perp = std::min(j_perp + delta_j, j_perp_end);  // legacy online schedule
        }
    }

    result.final_j_perp = j_perp;
    return result;
}


}
