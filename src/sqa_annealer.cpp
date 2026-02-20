#include "qanneal/sqa_annealer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

double SQAAnnealer::trotter_coupling(double beta, double gamma) const {
    const double eps = 1e-12;
    const double x = std::max(beta * gamma / static_cast<double>(slices_), eps);
    return 0.5 * std::log(1.0 / std::tanh(x));
}

double SQAAnnealer::delta_trotter(const SQAState &state,
                                  std::size_t replica,
                                  std::size_t slice,
                                  std::size_t spin,
                                  double j_perp) const {
    const std::size_t prev = (slice == 0) ? (slices_ - 1) : (slice - 1);
    const std::size_t next = (slice + 1) % slices_;
    const int8_t s = state.at(replica, slice, spin);
    const int8_t s_prev = state.at(replica, prev, spin);
    const int8_t s_next = state.at(replica, next, spin);
    return 2.0 * j_perp * static_cast<double>(s) *
           static_cast<double>(s_prev + s_next);
}

SQAResult SQAAnnealer::run(std::size_t sweeps_per_beta,
                           std::size_t worldline_sweeps,
                           SQAObserver *observer) {
    if (sweeps_per_beta == 0) {
        throw std::invalid_argument("sweeps_per_beta must be > 0.");
    }

    const std::size_t n = backend_->size();
    SQAState state = SQAState::random(replicas_, slices_, n, rng_);

    SQAResult result;
    result.best_energy = std::numeric_limits<double>::infinity();
    result.energy_trace.reserve(schedule_.size());

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    auto *sweep_observer = dynamic_cast<SQASweepObserver *>(observer);
    std::vector<double> replica_energies;

    auto compute_avg_energy = [&](const SQAState &state) {
        replica_energies.assign(replicas_, 0.0);
        double total = 0.0;
        for (std::size_t r = 0; r < replicas_; ++r) {
            double sum = 0.0;
            for (std::size_t t = 0; t < slices_; ++t) {
                const int8_t *slice_ptr = state.slice_ptr(r, t);
                sum += backend_->energy(slice_ptr, n);
            }
            replica_energies[r] = sum / static_cast<double>(slices_);
            total += sum;
        }
        return total / static_cast<double>(replicas_ * slices_);
    };

    for (std::size_t step = 0; step < schedule_.size(); ++step) {
        const double beta = schedule_.betas[step];
        const double gamma = schedule_.gammas[step];
        const double j_perp = trotter_coupling(beta, gamma);

        const double beta_scale = beta / static_cast<double>(slices_);

        for (std::size_t replica = 0; replica < replicas_; ++replica) {
            for (std::size_t sweep = 0; sweep < sweeps_per_beta; ++sweep) {
                for (std::size_t slice = 0; slice < slices_; ++slice) {
                    int8_t *slice_ptr = state.slice_ptr(replica, slice);
                    for (std::size_t spin = 0; spin < n; ++spin) {
                        const double delta_classical =
                            backend_->delta_energy(slice_ptr, n, spin);
                        const double delta = beta_scale * delta_classical +
                                             delta_trotter(state, replica, slice, spin, j_perp);
                        if (delta <= 0.0 || uniform(rng_) < std::exp(-delta)) {
                            slice_ptr[spin] = static_cast<int8_t>(-slice_ptr[spin]);
                        }
                    }
                }
                if (sweep_observer) {
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
            }

            for (std::size_t sweep = 0; sweep < worldline_sweeps; ++sweep) {
                for (std::size_t spin = 0; spin < n; ++spin) {
                    double delta_classical = 0.0;
                    for (std::size_t slice = 0; slice < slices_; ++slice) {
                        const int8_t *slice_ptr = state.slice_ptr(replica, slice);
                        delta_classical += backend_->delta_energy(slice_ptr, n, spin);
                    }
                    const double delta = beta_scale * delta_classical;
                    if (delta <= 0.0 || uniform(rng_) < std::exp(-delta)) {
                        for (std::size_t slice = 0; slice < slices_; ++slice) {
                            int8_t *slice_ptr = state.slice_ptr(replica, slice);
                            slice_ptr[spin] = static_cast<int8_t>(-slice_ptr[spin]);
                        }
                    }
                }
                if (sweep_observer) {
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
        std::size_t total_states = replicas_ * slices_;
        for (std::size_t replica = 0; replica < replicas_; ++replica) {
            for (std::size_t slice = 0; slice < slices_; ++slice) {
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
