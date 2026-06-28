#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "qanneal/dense_ising.hpp"
#include "qanneal/sparse_ising.hpp"
#include "qanneal/sqa_schedule.hpp"
#include "qanneal/state.hpp"

namespace qanneal {

struct CTPIMCResult {
    State best_state;
    double best_energy = 0.0;
    std::vector<double> energy_trace;
};

class CTPIMCAnnealer {
public:
    CTPIMCAnnealer(const DenseIsing &ising,
                   SQASchedule schedule,
                   std::size_t qubits_per_update = 1,
                   std::size_t qubits_per_chain = 1);

    CTPIMCAnnealer(const SparseIsing &ising,
                   SQASchedule schedule,
                   std::size_t qubits_per_update = 1,
                   std::size_t qubits_per_chain = 1);

    CTPIMCAnnealer(int Lperiodic,
                   double inv_temp_over_J,
                   double gamma_over_J,
                   int initial_condition,
                   std::size_t qubits_per_update = 1,
                   std::size_t qubits_per_chain = 1);

    void set_seed(std::uint64_t seed);
    void set_initial_state(const State &state);

    CTPIMCResult run(std::size_t sweeps_per_beta,
                     std::size_t reads = 1);

private:
    struct Edge {
        std::size_t i;
        std::size_t j;
        double value;
    };

    bool lattice_mode_ = false;
    int lattice_L_ = 0;
    double lattice_inv_temp_over_J_ = 0.0;
    double lattice_gamma_over_J_ = 0.0;
    int lattice_initial_condition_ = 0;

    std::size_t qubits_per_update_ = 1;
    std::size_t qubits_per_chain_ = 1;

    std::size_t n_ = 0;
    std::vector<double> h_;
    std::vector<Edge> edges_;
    double c_ = 0.0;

    SQASchedule schedule_;
    std::mt19937_64 rng_;
    std::optional<std::vector<int8_t>> initial_spins_;

    void init_from_dense(const DenseIsing &ising);
    void init_from_sparse(const SparseIsing &ising);

    CTPIMCResult run_general(std::size_t sweeps_per_beta,
                             std::size_t reads);
    CTPIMCResult run_lattice(std::size_t sweeps,
                             std::size_t reads);

    double energy_of(const std::vector<int> &spins) const;
};

}
