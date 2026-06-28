#include "qanneal/ctpimc_annealer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "localPIMC.hpp"

namespace qanneal {

namespace {
constexpr double kGammaEps = 1e-9;
}

CTPIMCAnnealer::CTPIMCAnnealer(const DenseIsing &ising,
                               SQASchedule schedule,
                               std::size_t qubits_per_update,
                               std::size_t qubits_per_chain)
    : qubits_per_update_(qubits_per_update),
      qubits_per_chain_(qubits_per_chain),
      schedule_(std::move(schedule)),
      rng_(std::random_device{}()) {
    init_from_dense(ising);
    if (schedule_.betas.empty()) {
        throw std::invalid_argument("CTPIMC schedule must contain betas.");
    }
    if (schedule_.betas.size() != schedule_.gammas.size()) {
        throw std::invalid_argument("CTPIMC schedule betas/gammas length mismatch.");
    }
    if (qubits_per_update_ == 0 || qubits_per_chain_ == 0) {
        throw std::invalid_argument("qubits_per_update and qubits_per_chain must be > 0.");
    }
    if (qubits_per_chain_ < qubits_per_update_) {
        throw std::invalid_argument("qubits_per_chain must be >= qubits_per_update.");
    }
}

CTPIMCAnnealer::CTPIMCAnnealer(const SparseIsing &ising,
                               SQASchedule schedule,
                               std::size_t qubits_per_update,
                               std::size_t qubits_per_chain)
    : qubits_per_update_(qubits_per_update),
      qubits_per_chain_(qubits_per_chain),
      schedule_(std::move(schedule)),
      rng_(std::random_device{}()) {
    init_from_sparse(ising);
    if (schedule_.betas.empty()) {
        throw std::invalid_argument("CTPIMC schedule must contain betas.");
    }
    if (schedule_.betas.size() != schedule_.gammas.size()) {
        throw std::invalid_argument("CTPIMC schedule betas/gammas length mismatch.");
    }
    if (qubits_per_update_ == 0 || qubits_per_chain_ == 0) {
        throw std::invalid_argument("qubits_per_update and qubits_per_chain must be > 0.");
    }
    if (qubits_per_chain_ < qubits_per_update_) {
        throw std::invalid_argument("qubits_per_chain must be >= qubits_per_update.");
    }
}

CTPIMCAnnealer::CTPIMCAnnealer(int Lperiodic,
                               double inv_temp_over_J,
                               double gamma_over_J,
                               int initial_condition,
                               std::size_t qubits_per_update,
                               std::size_t qubits_per_chain)
    : lattice_mode_(true),
      lattice_L_(Lperiodic),
      lattice_inv_temp_over_J_(inv_temp_over_J),
      lattice_gamma_over_J_(gamma_over_J),
      lattice_initial_condition_(initial_condition),
      qubits_per_update_(qubits_per_update),
      qubits_per_chain_(qubits_per_chain),
      rng_(std::random_device{}()) {
    if (lattice_L_ <= 0) {
        throw std::invalid_argument("Lperiodic must be > 0.");
    }
    if (lattice_inv_temp_over_J_ <= 0.0) {
        throw std::invalid_argument("inv_temp_over_J must be > 0.");
    }
    if (lattice_gamma_over_J_ <= 0.0) {
        throw std::invalid_argument("gamma_over_J must be > 0.");
    }
    if (qubits_per_update_ == 0 || qubits_per_chain_ == 0) {
        throw std::invalid_argument("qubits_per_update and qubits_per_chain must be > 0.");
    }
    if (qubits_per_chain_ < qubits_per_update_) {
        throw std::invalid_argument("qubits_per_chain must be >= qubits_per_update.");
    }
}

void CTPIMCAnnealer::set_seed(std::uint64_t seed) {
    rng_.seed(seed);
}

void CTPIMCAnnealer::set_initial_state(const State &state) {
    if (state.spins.size() != n_) {
        throw std::invalid_argument("Initial state size mismatch.");
    }
    initial_spins_ = state.spins;
}

CTPIMCResult CTPIMCAnnealer::run(std::size_t sweeps_per_beta,
                                std::size_t reads) {
    if (reads == 0) {
        throw std::invalid_argument("reads must be > 0.");
    }
    if (sweeps_per_beta == 0) {
        throw std::invalid_argument("sweeps_per_beta must be > 0.");
    }
    if (lattice_mode_) {
        return run_lattice(sweeps_per_beta, reads);
    }
    return run_general(sweeps_per_beta, reads);
}

void CTPIMCAnnealer::init_from_dense(const DenseIsing &ising) {
    n_ = ising.size();
    h_ = ising.h();
    c_ = ising.constant();
    edges_.clear();
    edges_.reserve(n_ * 2);
    const auto &J = ising.J();
    for (std::size_t i = 0; i < n_; ++i) {
        for (std::size_t j = i + 1; j < n_; ++j) {
            const double v = J[i * n_ + j];
            if (std::abs(v) > 0.0) {
                edges_.push_back({i, j, v});
            }
        }
    }
}

void CTPIMCAnnealer::init_from_sparse(const SparseIsing &ising) {
    n_ = ising.size();
    h_ = ising.h();
    c_ = ising.constant();
    edges_.clear();
    edges_.reserve(ising.edges().size());
    for (const auto &edge : ising.edges()) {
        if (edge.i == edge.j) {
            if (edge.i < h_.size()) {
                h_[edge.i] += edge.value;
            }
            continue;
        }
        const std::size_t i = std::min(edge.i, edge.j);
        const std::size_t j = std::max(edge.i, edge.j);
        edges_.push_back({i, j, edge.value});
    }
}

CTPIMCResult CTPIMCAnnealer::run_general(std::size_t sweeps_per_beta,
                                        std::size_t reads) {
    const std::size_t steps = schedule_.size();
    if (steps == 0) {
        throw std::invalid_argument("CTPIMC schedule is empty.");
    }

    std::vector<std::vector<int> > adj(n_);
    std::vector<std::vector<double> > baseJ(n_);
    for (const auto &edge : edges_) {
        if (edge.i >= n_ || edge.j >= n_) {
            throw std::invalid_argument("Edge index out of range.");
        }
        adj[edge.i].push_back(static_cast<int>(edge.j));
        adj[edge.j].push_back(static_cast<int>(edge.i));
        baseJ[edge.i].push_back(edge.value);
        baseJ[edge.j].push_back(edge.value);
    }

    // Pre-generate seeds for each read so the parallel loop is thread-safe
    std::vector<unsigned int> read_seeds(reads);
    std::vector<std::vector<int>> read_initials(reads, std::vector<int>(n_));
    {
        std::uniform_int_distribution<int> spin_pick(0, 1);
        for (std::size_t read = 0; read < reads; ++read) {
            read_seeds[read] = static_cast<unsigned int>(rng_());
            if (initial_spins_) {
                for (std::size_t i = 0; i < n_; ++i) {
                    read_initials[read][i] = static_cast<int>((*initial_spins_)[i]);
                }
            } else {
                for (std::size_t i = 0; i < n_; ++i) {
                    read_initials[read][i] = spin_pick(rng_) == 0 ? -1 : 1;
                }
            }
        }
    }

    // Per-read results (written independently, merged after parallel section)
    std::vector<double> read_best_energies(reads, std::numeric_limits<double>::infinity());
    std::vector<std::vector<int>> read_best_spins(reads);
    std::vector<std::vector<double>> read_traces(reads, std::vector<double>(steps, 0.0));

#ifdef _OPENMP
#pragma omp parallel for if(reads > 1) schedule(dynamic)
#endif
    for (long long read_ll = 0; read_ll < static_cast<long long>(reads); ++read_ll) {
        const std::size_t read = static_cast<std::size_t>(read_ll);

        const double beta0 = schedule_.betas.front();
        const double gamma0 = std::max(schedule_.gammas.front(), kGammaEps);

        std::vector<std::vector<double> > scaledJ = baseJ;
        std::vector<double> scaledH(h_.size(), 0.0);
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t k = 0; k < scaledJ[i].size(); ++k) {
                scaledJ[i][k] *= beta0;
            }
            scaledH[i] = h_[i] * beta0;
        }

        localPIMC pimc(beta0 * gamma0,
                       0.0,
                       static_cast<int>(qubits_per_update_),
                       static_cast<int>(qubits_per_chain_),
                       adj,
                       scaledJ,
                       scaledH,
                       read_initials[read],
                       read_seeds[read]);

        double local_best_energy = std::numeric_limits<double>::infinity();
        std::vector<int> local_best_spins;

        for (std::size_t step = 0; step < steps; ++step) {
            const double beta = schedule_.betas[step];
            const double gamma = std::max(schedule_.gammas[step], kGammaEps);

            for (std::size_t i = 0; i < n_; ++i) {
                for (std::size_t k = 0; k < baseJ[i].size(); ++k) {
                    scaledJ[i][k] = baseJ[i][k] * beta;
                }
                scaledH[i] = h_[i] * beta;
            }

            if (step > 0) {
                pimc.set_scaled_params(scaledJ, scaledH, beta * gamma);
            }

            pimc.run(static_cast<int>(sweeps_per_beta));

            const std::vector<int> &spins = pimc.current_spins();
            const double energy = energy_of(spins);
            read_traces[read][step] = energy;
            if (energy < local_best_energy) {
                local_best_energy = energy;
                local_best_spins = spins;
            }
        }

        read_best_energies[read] = local_best_energy;
        read_best_spins[read] = std::move(local_best_spins);
    }

    // Merge per-read results
    CTPIMCResult result;
    result.best_energy = std::numeric_limits<double>::infinity();
    result.energy_trace.assign(steps, 0.0);

    for (std::size_t read = 0; read < reads; ++read) {
        for (std::size_t step = 0; step < steps; ++step) {
            result.energy_trace[step] += read_traces[read][step];
        }
        if (read_best_energies[read] < result.best_energy) {
            result.best_energy = read_best_energies[read];
            result.best_state.spins.assign(
                read_best_spins[read].begin(), read_best_spins[read].end());
        }
    }

    for (double &e : result.energy_trace) {
        e /= static_cast<double>(reads);
    }

    return result;
}

CTPIMCResult CTPIMCAnnealer::run_lattice(std::size_t sweeps,
                                        std::size_t reads) {
    CTPIMCResult result;
    result.best_energy = std::numeric_limits<double>::infinity();
    result.energy_trace.assign(1, 0.0);

    for (std::size_t read = 0; read < reads; ++read) {
        const unsigned int seed = static_cast<unsigned int>(rng_());
        localPIMC pimc(lattice_L_,
                       lattice_inv_temp_over_J_,
                       lattice_gamma_over_J_,
                       lattice_initial_condition_,
                       static_cast<int>(qubits_per_update_),
                       static_cast<int>(qubits_per_chain_),
                       seed);

        pimc.run(static_cast<int>(sweeps));

        const std::vector<int> &spins = pimc.current_spins();
        double energy_scaled = pimc.scaled_energy(spins);
        double energy = energy_scaled / lattice_inv_temp_over_J_;

        result.energy_trace[0] += energy;
        if (energy < result.best_energy) {
            result.best_energy = energy;
            result.best_state.spins.assign(spins.begin(), spins.end());
        }
    }

    result.energy_trace[0] /= static_cast<double>(reads);
    return result;
}

double CTPIMCAnnealer::energy_of(const std::vector<int> &spins) const {
    if (spins.size() != n_) {
        throw std::invalid_argument("Spin vector size mismatch.");
    }
    double energy = c_;
    for (std::size_t i = 0; i < n_; ++i) {
        energy += h_[i] * static_cast<double>(spins[i]);
    }
    for (const auto &edge : edges_) {
        energy += edge.value * static_cast<double>(spins[edge.i]) * static_cast<double>(spins[edge.j]);
    }
    return energy;
}

}
