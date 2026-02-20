#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "qanneal/metrics.hpp"
#include "qanneal/observer.hpp"
#include "qanneal/sqa_observer.hpp"

namespace qanneal {

class MetricsObserver : public Observer {
public:
    std::vector<double> energy_trace;
    std::vector<double> magnetization_trace;

    void record(std::size_t step,
                double beta,
                double energy,
                const State &state) override {
        (void)step;
        (void)beta;
        energy_trace.push_back(energy);
        magnetization_trace.push_back(magnetization(state));
    }

    void clear() {
        energy_trace.clear();
        magnetization_trace.clear();
    }
};

class SQAMetricsObserver : public SQAObserver {
public:
    std::vector<double> energy_trace;
    std::vector<double> magnetization_trace;

    void record(std::size_t step,
                double beta,
                double gamma,
                double avg_energy,
                const SQAState &state) override {
        (void)step;
        (void)beta;
        (void)gamma;
        energy_trace.push_back(avg_energy);

        const std::size_t replicas = state.replicas();
        const std::size_t slices = state.slices();
        const std::size_t spins = state.spins();
        double sum = 0.0;
        for (std::size_t r = 0; r < replicas; ++r) {
            for (std::size_t t = 0; t < slices; ++t) {
                const int8_t *ptr = state.slice_ptr(r, t);
                for (std::size_t i = 0; i < spins; ++i) {
                    sum += static_cast<double>(ptr[i]);
                }
            }
        }
        const double denom = static_cast<double>(replicas * slices * spins);
        magnetization_trace.push_back(denom > 0.0 ? sum / denom : 0.0);
    }

    void clear() {
        energy_trace.clear();
        magnetization_trace.clear();
    }
};

class StateTraceObserver : public Observer, public SweepObserver {
public:
    std::size_t stride = 1;
    std::vector<std::size_t> step_trace;
    std::vector<std::size_t> sweep_trace;
    std::vector<double> beta_trace;
    std::vector<double> energy_trace;
    std::vector<std::vector<int8_t>> state_trace;

    void record(std::size_t step,
                double beta,
                double energy,
                const State &state) override {
        (void)step;
        (void)beta;
        (void)energy;
        (void)state;
    }

    void record_sweep(std::size_t step,
                      std::size_t sweep,
                      double beta,
                      double energy,
                      const State &state) override {
        if (stride > 1 && (sweep % stride) != 0) {
            return;
        }
        step_trace.push_back(step);
        sweep_trace.push_back(sweep);
        beta_trace.push_back(beta);
        energy_trace.push_back(energy);
        state_trace.push_back(state.spins);
    }

    void clear() {
        step_trace.clear();
        sweep_trace.clear();
        beta_trace.clear();
        energy_trace.clear();
        state_trace.clear();
    }
};

class SQAStateTraceObserver : public SQAObserver, public SQASweepObserver {
public:
    std::size_t stride = 1;
    std::size_t replicas = 0;
    std::size_t slices = 0;
    std::size_t spins = 0;
    std::vector<std::size_t> step_trace;
    std::vector<std::size_t> replica_trace;
    std::vector<std::size_t> sweep_trace;
    std::vector<int> phase_trace;
    std::vector<double> beta_trace;
    std::vector<double> gamma_trace;
    std::vector<double> avg_energy_trace;
    std::vector<std::vector<double>> replica_energy_trace;
    std::vector<std::vector<int8_t>> state_trace;

    void record(std::size_t step,
                double beta,
                double gamma,
                double avg_energy,
                const SQAState &state) override {
        (void)step;
        (void)beta;
        (void)gamma;
        (void)avg_energy;
        (void)state;
    }

    void record_sweep(std::size_t step,
                      std::size_t replica,
                      std::size_t sweep,
                      SQASweepPhase phase,
                      double beta,
                      double gamma,
                      double avg_energy,
                      const std::vector<double> &replica_energies,
                      const SQAState &state) override {
        if (stride > 1 && (sweep % stride) != 0) {
            return;
        }
        if (replicas == 0) {
            replicas = state.replicas();
            slices = state.slices();
            spins = state.spins();
        }
        step_trace.push_back(step);
        replica_trace.push_back(replica);
        sweep_trace.push_back(sweep);
        phase_trace.push_back(static_cast<int>(phase));
        beta_trace.push_back(beta);
        gamma_trace.push_back(gamma);
        avg_energy_trace.push_back(avg_energy);
        replica_energy_trace.push_back(replica_energies);

        std::vector<int8_t> flat;
        flat.reserve(replicas * slices * spins);
        for (std::size_t r = 0; r < replicas; ++r) {
            for (std::size_t t = 0; t < slices; ++t) {
                const int8_t *ptr = state.slice_ptr(r, t);
                flat.insert(flat.end(), ptr, ptr + spins);
            }
        }
        state_trace.push_back(std::move(flat));
    }

    void clear() {
        replicas = 0;
        slices = 0;
        spins = 0;
        step_trace.clear();
        replica_trace.clear();
        sweep_trace.clear();
        phase_trace.clear();
        beta_trace.clear();
        gamma_trace.clear();
        avg_energy_trace.clear();
        replica_energy_trace.clear();
        state_trace.clear();
    }
};

}
