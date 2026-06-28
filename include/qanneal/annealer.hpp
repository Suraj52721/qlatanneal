#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "qanneal/backend.hpp"
#include "qanneal/observer.hpp"
#include "qanneal/schedule.hpp"
#include "qanneal/state.hpp"

namespace qanneal {

struct AnnealResult {
    State best_state;
    double best_energy = 0.0;
    std::vector<double> energy_trace;
};

class Annealer {
public:
    explicit Annealer(const Hamiltonian &hamiltonian,
                      AnnealSchedule schedule);
    explicit Annealer(std::shared_ptr<Backend> backend,
                      AnnealSchedule schedule);

    void set_seed(std::uint64_t seed);

    AnnealResult run(std::size_t sweeps_per_beta,
                     Observer *observer = nullptr);

private:
    std::shared_ptr<Backend> backend_;
    AnnealSchedule schedule_;
    std::mt19937_64 rng_;
};

}
