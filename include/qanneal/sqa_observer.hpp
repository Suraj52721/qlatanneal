#pragma once

#include <cstddef>
#include <vector>

#include "qanneal/sqa_state.hpp"

namespace qanneal {

class SQAObserver {
public:
    virtual ~SQAObserver() = default;
    virtual void record(std::size_t step,
                        double beta,
                        double gamma,
                        double avg_energy,
                        const SQAState &state) = 0;
};

enum class SQASweepPhase : int {
    Slice = 0,
    Worldline = 1,
    Cluster = 2
};

class SQASweepObserver {
public:
    virtual ~SQASweepObserver() = default;
    virtual void record_sweep(std::size_t step,
                              std::size_t replica,
                              std::size_t sweep,
                              SQASweepPhase phase,
                              double beta,
                              double gamma,
                              double avg_energy,
                              const std::vector<double> &replica_energies,
                              const SQAState &state) = 0;
};

}
