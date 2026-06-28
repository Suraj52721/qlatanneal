#include "qanneal/annealer.hpp"

#include <cmath>
#include <stdexcept>

namespace qanneal {

Annealer::Annealer(const Hamiltonian &hamiltonian, AnnealSchedule schedule)
    : backend_(make_backend(BackendKind::CPU, hamiltonian)),
      schedule_(std::move(schedule)),
      rng_(std::random_device{}()) {
    if (schedule_.betas.empty()) {
        throw std::invalid_argument("Schedule must contain at least one beta.");
    }
}

Annealer::Annealer(std::shared_ptr<Backend> backend, AnnealSchedule schedule)
    : backend_(std::move(backend)),
      schedule_(std::move(schedule)),
      rng_(std::random_device{}()) {
    if (!backend_) {
        throw std::invalid_argument("Annealer requires a backend.");
    }
    if (schedule_.betas.empty()) {
        throw std::invalid_argument("Schedule must contain at least one beta.");
    }
}

void Annealer::set_seed(std::uint64_t seed) {
    rng_.seed(seed);
}

AnnealResult Annealer::run(std::size_t sweeps_per_beta, Observer *observer) {
    if (sweeps_per_beta == 0) {
        throw std::invalid_argument("sweeps_per_beta must be > 0.");
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    State state = State::random(backend_->size(), rng_);
    double energy = backend_->energy(state.spins.data(), state.size());

    AnnealResult result;
    result.best_state = state;
    result.best_energy = energy;
    result.energy_trace.reserve(schedule_.size());

    auto *sweep_observer = dynamic_cast<SweepObserver *>(observer);

    for (std::size_t step = 0; step < schedule_.betas.size(); ++step) {
        const double beta = schedule_.betas[step];
        for (std::size_t sweep = 0; sweep < sweeps_per_beta; ++sweep) {
            for (std::size_t i = 0; i < backend_->size(); ++i) {
                const double delta = backend_->delta_energy(state.spins.data(), state.size(), i);
                if (delta <= 0.0 || uniform(rng_) < std::exp(-beta * delta)) {
                    state[i] = static_cast<int8_t>(-state[i]);
                    energy += delta;
                    if (energy < result.best_energy) {
                        result.best_energy = energy;
                        result.best_state = state;
                    }
                }
            }
            if (sweep_observer) {
                sweep_observer->record_sweep(step, sweep, beta, energy, state);
            }
        }
        result.energy_trace.push_back(energy);
        if (observer) {
            observer->record(step, beta, energy, state);
        }
    }

    return result;
}

}
