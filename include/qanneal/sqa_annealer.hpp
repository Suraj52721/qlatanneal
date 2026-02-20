#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "qanneal/backend.hpp"
#include "qanneal/sqa_observer.hpp"
#include "qanneal/sqa_schedule.hpp"
#include "qanneal/sqa_state.hpp"
#include "qanneal/state.hpp"

namespace qanneal {

struct SQAResult {
    State best_state;
    double best_energy = 0.0;
    std::vector<double> energy_trace;
};

class SQAAnnealer {
public:
    SQAAnnealer(const Hamiltonian &hamiltonian,
                SQASchedule schedule,
                std::size_t trotter_slices,
                std::size_t replicas = 1);
    SQAAnnealer(std::shared_ptr<Backend> backend,
                SQASchedule schedule,
                std::size_t trotter_slices,
                std::size_t replicas = 1);

    void set_seed(std::uint64_t seed);

    SQAResult run(std::size_t sweeps_per_beta,
                  std::size_t worldline_sweeps,
                  SQAObserver *observer = nullptr);

private:
    std::shared_ptr<Backend> backend_;
    SQASchedule schedule_;
    std::size_t slices_ = 0;
    std::size_t replicas_ = 0;
    std::mt19937_64 rng_;

    double trotter_coupling(double beta, double gamma) const;
    double delta_trotter(const SQAState &state,
                         std::size_t replica,
                         std::size_t slice,
                         std::size_t spin,
                         double j_perp) const;
};

}
