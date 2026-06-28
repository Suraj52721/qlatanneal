#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "qanneal/state.hpp"

namespace qanneal {

class SQAState {
public:
    SQAState() = default;

    SQAState(std::size_t replicas, std::size_t slices, std::size_t spins)
        : replicas_(replicas), slices_(slices), spins_(spins),
          data_(replicas * slices * spins, 1) {
        if (replicas_ == 0 || slices_ == 0 || spins_ == 0) {
            throw std::invalid_argument("SQAState dimensions must be > 0.");
        }
    }

    std::size_t replicas() const { return replicas_; }
    std::size_t slices() const { return slices_; }
    std::size_t spins() const { return spins_; }

    int8_t &at(std::size_t replica, std::size_t slice, std::size_t spin) {
        return data_[index(replica, slice, spin)];
    }

    const int8_t &at(std::size_t replica, std::size_t slice, std::size_t spin) const {
        return data_[index(replica, slice, spin)];
    }

    int8_t *slice_ptr(std::size_t replica, std::size_t slice) {
        return &data_[index(replica, slice, 0)];
    }

    const int8_t *slice_ptr(std::size_t replica, std::size_t slice) const {
        return &data_[index(replica, slice, 0)];
    }

    State slice_state(std::size_t replica, std::size_t slice) const {
        State s(spins_);
        const int8_t *ptr = slice_ptr(replica, slice);
        for (std::size_t i = 0; i < spins_; ++i) {
            s[i] = ptr[i];
        }
        return s;
    }

    void set_slice_state(std::size_t replica, std::size_t slice, const State &state) {
        if (state.size() != spins_) {
            throw std::invalid_argument("State size mismatch.");
        }
        int8_t *ptr = slice_ptr(replica, slice);
        for (std::size_t i = 0; i < spins_; ++i) {
            ptr[i] = state[i];
        }
    }

    template <class URNG>
    static SQAState random(std::size_t replicas, std::size_t slices, std::size_t spins, URNG &rng) {
        SQAState state(replicas, slices, spins);
        std::uniform_int_distribution<int> dist(0, 1);
        for (auto &v : state.data_) {
            v = dist(rng) ? 1 : -1;
        }
        return state;
    }

private:
    std::size_t replicas_ = 0;
    std::size_t slices_ = 0;
    std::size_t spins_ = 0;
    std::vector<int8_t> data_;

    std::size_t index(std::size_t replica, std::size_t slice, std::size_t spin) const {
        if (replica >= replicas_ || slice >= slices_ || spin >= spins_) {
            throw std::invalid_argument("SQAState index out of range.");
        }
        return (replica * slices_ + slice) * spins_ + spin;
    }
};

}
