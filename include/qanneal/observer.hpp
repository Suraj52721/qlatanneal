#pragma once

#include <cstddef>

#include "qanneal/state.hpp"

namespace qanneal {

class Observer {
public:
    virtual ~Observer() = default;
    virtual void record(std::size_t step,
                        double beta,
                        double energy,
                        const State &state) = 0;
};

class SweepObserver {
public:
    virtual ~SweepObserver() = default;
    virtual void record_sweep(std::size_t step,
                              std::size_t sweep,
                              double beta,
                              double energy,
                              const State &state) = 0;
};

}
