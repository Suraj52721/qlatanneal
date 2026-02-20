#include "qanneal/replica_annealer.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace qanneal {

ReplicaAnnealer::ReplicaAnnealer(const Hamiltonian &hamiltonian,
                                 AnnealSchedule schedule,
                                 std::size_t replicas)
    : backend_(make_backend(BackendKind::CPU, hamiltonian)),
      schedule_(std::move(schedule)),
      replicas_(replicas),
      rng_(std::random_device{}()) {
    if (schedule_.betas.empty()) {
        throw std::invalid_argument("Schedule must contain at least one beta.");
    }
    if (replicas_ == 0) {
        throw std::invalid_argument("replicas must be > 0.");
    }
}

ReplicaAnnealer::ReplicaAnnealer(std::shared_ptr<Backend> backend,
                                 AnnealSchedule schedule,
                                 std::size_t replicas)
    : backend_(std::move(backend)),
      schedule_(std::move(schedule)),
      replicas_(replicas),
      rng_(std::random_device{}()) {
    if (!backend_) {
        throw std::invalid_argument("ReplicaAnnealer requires a backend.");
    }
    if (schedule_.betas.empty()) {
        throw std::invalid_argument("Schedule must contain at least one beta.");
    }
    if (replicas_ == 0) {
        throw std::invalid_argument("replicas must be > 0.");
    }
}

void ReplicaAnnealer::set_seed(std::uint64_t seed) {
    rng_.seed(seed);
}

MultiAnnealResult ReplicaAnnealer::run(std::size_t sweeps_per_beta) {
    if (sweeps_per_beta == 0) {
        throw std::invalid_argument("sweeps_per_beta must be > 0.");
    }

    const std::size_t n = backend_->size();

    std::vector<State> states;
    states.reserve(replicas_);
    std::vector<double> energies(replicas_, 0.0);

    for (std::size_t r = 0; r < replicas_; ++r) {
        states.push_back(State::random(n, rng_));
        energies[r] = backend_->energy(states[r].spins.data(), states[r].size());
    }

    MultiAnnealResult result;
    result.replicas.resize(replicas_);
    result.global_best_energy = std::numeric_limits<double>::infinity();
    result.average_energy_trace.reserve(schedule_.size());
    result.average_magnetization_trace.reserve(schedule_.size());

    for (std::size_t r = 0; r < replicas_; ++r) {
        result.replicas[r].best_state = states[r];
        result.replicas[r].best_energy = energies[r];
        result.replicas[r].energy_trace.reserve(schedule_.size());
        result.replicas[r].magnetization_trace.reserve(schedule_.size());
        if (energies[r] < result.global_best_energy) {
            result.global_best_energy = energies[r];
            result.global_best_state = states[r];
        }
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (std::size_t step = 0; step < schedule_.betas.size(); ++step) {
        const double beta = schedule_.betas[step];

        for (std::size_t r = 0; r < replicas_; ++r) {
            auto &state = states[r];
            double energy = energies[r];
            for (std::size_t sweep = 0; sweep < sweeps_per_beta; ++sweep) {
                for (std::size_t i = 0; i < n; ++i) {
                    const double delta = backend_->delta_energy(state.spins.data(), state.size(), i);
                    if (delta <= 0.0 || uniform(rng_) < std::exp(-beta * delta)) {
                        state[i] = static_cast<int8_t>(-state[i]);
                        energy += delta;
                        if (energy < result.replicas[r].best_energy) {
                            result.replicas[r].best_energy = energy;
                            result.replicas[r].best_state = state;
                        }
                        if (energy < result.global_best_energy) {
                            result.global_best_energy = energy;
                            result.global_best_state = state;
                        }
                    }
                }
            }
            energies[r] = energy;
        }

        double avg_energy = 0.0;
        double avg_mag = 0.0;
        for (std::size_t r = 0; r < replicas_; ++r) {
            const double energy = energies[r];
            const double mag = magnetization(states[r]);
            result.replicas[r].energy_trace.push_back(energy);
            result.replicas[r].magnetization_trace.push_back(mag);
            avg_energy += energy;
            avg_mag += mag;
        }
        avg_energy /= static_cast<double>(replicas_);
        avg_mag /= static_cast<double>(replicas_);
        result.average_energy_trace.push_back(avg_energy);
        result.average_magnetization_trace.push_back(avg_mag);
    }

    return result;
}

}
