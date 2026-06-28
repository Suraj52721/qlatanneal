#include "qanneal/parallel_tempering.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace qanneal {

ParallelTemperingAnnealer::ParallelTemperingAnnealer(const Hamiltonian &hamiltonian,
                                                     std::vector<double> betas)
    : backend_(make_backend(BackendKind::CPU, hamiltonian)),
      betas_(std::move(betas)),
      rng_(std::random_device{}()) {
    if (betas_.size() < 2) {
        throw std::invalid_argument("Parallel tempering requires at least two betas.");
    }
}

ParallelTemperingAnnealer::ParallelTemperingAnnealer(std::shared_ptr<Backend> backend,
                                                     std::vector<double> betas)
    : backend_(std::move(backend)),
      betas_(std::move(betas)),
      rng_(std::random_device{}()) {
    if (!backend_) {
        throw std::invalid_argument("ParallelTemperingAnnealer requires a backend.");
    }
    if (betas_.size() < 2) {
        throw std::invalid_argument("Parallel tempering requires at least two betas.");
    }
}

void ParallelTemperingAnnealer::set_seed(std::uint64_t seed) {
    rng_.seed(seed);
}

ParallelTemperingResult ParallelTemperingAnnealer::run(std::size_t sweeps_per_step,
                                                       std::size_t steps,
                                                       std::size_t swap_interval) {
    if (sweeps_per_step == 0) {
        throw std::invalid_argument("sweeps_per_step must be > 0.");
    }
    if (steps == 0) {
        throw std::invalid_argument("steps must be > 0.");
    }
    if (swap_interval == 0) {
        throw std::invalid_argument("swap_interval must be > 0.");
    }

    const std::size_t n = backend_->size();
    const std::size_t replicas = betas_.size();

    std::vector<State> states;
    states.reserve(replicas);
    std::vector<double> energies(replicas, 0.0);

    for (std::size_t r = 0; r < replicas; ++r) {
        states.push_back(State::random(n, rng_));
        energies[r] = backend_->energy(states[r].spins.data(), states[r].size());
    }

    ParallelTemperingResult result;
    result.final_states = states;
    result.final_energies = energies;
    result.best_energy = std::numeric_limits<double>::infinity();
    result.average_energy_trace.reserve(steps);
    result.swap_acceptance_trace.reserve(steps);

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t r = 0; r < replicas; ++r) {
            const double beta = betas_[r];
            auto &state = states[r];
            double energy = energies[r];
            for (std::size_t sweep = 0; sweep < sweeps_per_step; ++sweep) {
                for (std::size_t i = 0; i < n; ++i) {
                    const double delta = backend_->delta_energy(state.spins.data(), state.size(), i);
                    if (delta <= 0.0 || uniform(rng_) < std::exp(-beta * delta)) {
                        state[i] = static_cast<int8_t>(-state[i]);
                        energy += delta;
                    }
                }
            }
            energies[r] = energy;
            if (energy < result.best_energy) {
                result.best_energy = energy;
                result.best_state = state;
            }
        }

        double accepted = 0.0;
        double attempted = 0.0;
        if ((step + 1) % swap_interval == 0) {
            for (std::size_t r = 0; r + 1 < replicas; ++r) {
                const double beta_i = betas_[r];
                const double beta_j = betas_[r + 1];
                const double e_i = energies[r];
                const double e_j = energies[r + 1];
                const double delta = (beta_i - beta_j) * (e_j - e_i);
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
        result.swap_acceptance_trace.push_back(attempted > 0.0 ? (accepted / attempted) : 0.0);
    }

    result.final_states = states;
    result.final_energies = energies;

    return result;
}

} // namespace qanneal
