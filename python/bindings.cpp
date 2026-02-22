#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "qanneal/annealer.hpp"
#include "qanneal/backend.hpp"
#include "qanneal/dense_ising.hpp"
#include "qanneal/hamiltonian.hpp"
#include "qanneal/metrics.hpp"
#include "qanneal/metrics_observer.hpp"
#include "qanneal/parallel_tempering.hpp"
#include "qanneal/qubo.hpp"
#include "qanneal/replica_annealer.hpp"
#include "qanneal/schedule.hpp"
#include "qanneal/sparse_ising.hpp"
#include "qanneal/sqa_annealer.hpp"
#include "qanneal/sqa_observer.hpp"
#include "qanneal/sqa_schedule.hpp"
#include "qanneal/sqa_state.hpp"
#include "qanneal/state.hpp"
#include "qanneal/version.hpp"

namespace py = pybind11;

namespace {

std::vector<double> array_to_vector_1d(const py::array_t<double, py::array::c_style | py::array::forcecast> &arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Expected 1D array.");
    }
    const auto *ptr = static_cast<const double *>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.shape[0]);
}

std::vector<double> array_to_vector_2d(const py::array_t<double, py::array::c_style | py::array::forcecast> &arr,
                                       std::size_t &n) {
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument("Expected 2D array.");
    }
    if (buf.shape[0] != buf.shape[1]) {
        throw std::invalid_argument("Expected square matrix.");
    }
    n = static_cast<std::size_t>(buf.shape[0]);
    const auto *ptr = static_cast<const double *>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.shape[0] * buf.shape[1]);
}

std::vector<int8_t> seq_to_spins(const py::sequence &seq) {
    std::vector<int8_t> spins;
    spins.reserve(seq.size());
    for (auto item : seq) {
        spins.push_back(static_cast<int8_t>(py::cast<int>(item)));
    }
    return spins;
}

} // namespace

PYBIND11_MODULE(_qanneal, m) {
    m.doc() = "qanneal core bindings";

    m.def("version_string", &qanneal::version_string);
    m.attr("version_major") = qanneal::version_major;
    m.attr("version_minor") = qanneal::version_minor;
    m.attr("version_patch") = qanneal::version_patch;

    py::class_<qanneal::State>(m, "State")
        .def(py::init<std::size_t>())
        .def_property("spins",
                      [](const qanneal::State &s) { return s.spins; },
                      [](qanneal::State &s, const std::vector<int8_t> &spins) { s.spins = spins; })
        .def("size", &qanneal::State::size);

    py::class_<qanneal::Hamiltonian, std::shared_ptr<qanneal::Hamiltonian>>(m, "Hamiltonian");

    py::class_<qanneal::DenseIsing, qanneal::Hamiltonian, std::shared_ptr<qanneal::DenseIsing>>(m, "DenseIsing")
        .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> h,
                         py::array_t<double, py::array::c_style | py::array::forcecast> J,
                         double c) {
            std::vector<double> hv = array_to_vector_1d(h);
            std::size_t n = 0;
            std::vector<double> Jv = array_to_vector_2d(J, n);
            if (hv.size() != n) {
                throw std::invalid_argument("h vector length mismatch.");
            }
            return qanneal::DenseIsing(std::move(hv), std::move(Jv), n, c);
        }), py::arg("h"), py::arg("J"), py::arg("c") = 0.0)
        .def("size", &qanneal::DenseIsing::size)
        .def("energy", [](const qanneal::DenseIsing &ham, const py::sequence &spins) {
            auto data = seq_to_spins(spins);
            return ham.energy(data.data(), data.size());
        })
        .def("delta_energy", [](const qanneal::DenseIsing &ham, const py::sequence &spins, std::size_t flip) {
            auto data = seq_to_spins(spins);
            return ham.delta_energy(data.data(), data.size(), flip);
        });

    py::class_<qanneal::SparseEdge>(m, "SparseEdge")
        .def(py::init<std::size_t, std::size_t, double>())
        .def_readwrite("i", &qanneal::SparseEdge::i)
        .def_readwrite("j", &qanneal::SparseEdge::j)
        .def_readwrite("value", &qanneal::SparseEdge::value);

    py::class_<qanneal::SparseIsing, qanneal::Hamiltonian, std::shared_ptr<qanneal::SparseIsing>>(m, "SparseIsing")
        .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> h,
                         const std::vector<qanneal::SparseEdge> &edges,
                         std::size_t n,
                         double c) {
            std::vector<double> hv = array_to_vector_1d(h);
            if (hv.size() != n) {
                throw std::invalid_argument("h vector length mismatch.");
            }
            return qanneal::SparseIsing(std::move(hv), edges, n, c);
        }), py::arg("h"), py::arg("edges"), py::arg("n"), py::arg("c") = 0.0)
        .def("size", &qanneal::SparseIsing::size)
        .def("energy", [](const qanneal::SparseIsing &ham, const py::sequence &spins) {
            auto data = seq_to_spins(spins);
            return ham.energy(data.data(), data.size());
        })
        .def("delta_energy", [](const qanneal::SparseIsing &ham, const py::sequence &spins, std::size_t flip) {
            auto data = seq_to_spins(spins);
            return ham.delta_energy(data.data(), data.size(), flip);
        });

    py::class_<qanneal::QUBO>(m, "QUBO")
        .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> Q) {
            std::size_t n = 0;
            std::vector<double> qv = array_to_vector_2d(Q, n);
            return qanneal::QUBO(std::move(qv), n);
        }))
        .def(py::init([](const std::vector<std::tuple<std::size_t, std::size_t, double>> &entries,
                         std::size_t n) {
            std::vector<std::pair<std::pair<std::size_t, std::size_t>, double>> native;
            native.reserve(entries.size());
            for (const auto &e : entries) {
                native.push_back({{std::get<0>(e), std::get<1>(e)}, std::get<2>(e)});
            }
            return qanneal::QUBO(native, n);
        }), py::arg("entries"), py::arg("n"))
        .def("size", &qanneal::QUBO::size)
        .def("to_ising", &qanneal::QUBO::to_ising);

    py::class_<qanneal::AnnealSchedule>(m, "AnnealSchedule")
        .def(py::init<>())
        .def_readwrite("betas", &qanneal::AnnealSchedule::betas)
        .def_static("linear", &qanneal::AnnealSchedule::linear,
                    py::arg("beta_start"), py::arg("beta_end"), py::arg("steps"))
        .def_static("from_betas", [](const std::vector<double> &betas) {
            qanneal::AnnealSchedule sched;
            sched.betas = betas;
            if (sched.betas.empty()) {
                throw std::invalid_argument("Schedule must contain betas.");
            }
            return sched;
        });

    py::class_<qanneal::Observer, std::shared_ptr<qanneal::Observer>>(m, "Observer");
    py::class_<qanneal::MetricsObserver, qanneal::Observer, std::shared_ptr<qanneal::MetricsObserver>>(m, "MetricsObserver")
        .def(py::init<>())
        .def_readonly("energy_trace", &qanneal::MetricsObserver::energy_trace)
        .def_readonly("magnetization_trace", &qanneal::MetricsObserver::magnetization_trace)
        .def("clear", &qanneal::MetricsObserver::clear);
    py::class_<qanneal::StateTraceObserver, qanneal::Observer, std::shared_ptr<qanneal::StateTraceObserver>>(m, "StateTraceObserver")
        .def(py::init<>())
        .def_readwrite("stride", &qanneal::StateTraceObserver::stride)
        .def_readonly("step_trace", &qanneal::StateTraceObserver::step_trace)
        .def_readonly("sweep_trace", &qanneal::StateTraceObserver::sweep_trace)
        .def_readonly("beta_trace", &qanneal::StateTraceObserver::beta_trace)
        .def_readonly("energy_trace", &qanneal::StateTraceObserver::energy_trace)
        .def_readonly("state_trace", &qanneal::StateTraceObserver::state_trace)
        .def("clear", &qanneal::StateTraceObserver::clear);

    py::class_<qanneal::AnnealResult>(m, "AnnealResult")
        .def_readonly("best_state", &qanneal::AnnealResult::best_state)
        .def_readonly("best_energy", &qanneal::AnnealResult::best_energy)
        .def_readonly("energy_trace", &qanneal::AnnealResult::energy_trace);

    py::class_<qanneal::Annealer>(m, "Annealer")
        .def(py::init([](std::shared_ptr<qanneal::Hamiltonian> ham,
                         qanneal::AnnealSchedule schedule,
                         const std::string &backend) {
            auto kind = qanneal::backend_from_string(backend);
            auto be = qanneal::make_backend(kind, std::move(ham));
            return qanneal::Annealer(std::move(be), std::move(schedule));
        }),
        py::arg("hamiltonian"),
        py::arg("schedule"),
        py::arg("backend") = "cpu",
        py::keep_alive<1, 2>())
        .def("set_seed", &qanneal::Annealer::set_seed)
        .def("run", [](qanneal::Annealer &self,
                       std::size_t sweeps_per_beta,
                       std::shared_ptr<qanneal::Observer> obs) {
            return self.run(sweeps_per_beta, obs.get());
        }, py::arg("sweeps_per_beta"), py::arg("observer") = nullptr);

    py::class_<qanneal::ReplicaResult>(m, "ReplicaResult")
        .def_readonly("best_state", &qanneal::ReplicaResult::best_state)
        .def_readonly("best_energy", &qanneal::ReplicaResult::best_energy)
        .def_readonly("energy_trace", &qanneal::ReplicaResult::energy_trace)
        .def_readonly("magnetization_trace", &qanneal::ReplicaResult::magnetization_trace);

    py::class_<qanneal::MultiAnnealResult>(m, "MultiAnnealResult")
        .def_readonly("replicas", &qanneal::MultiAnnealResult::replicas)
        .def_readonly("global_best_state", &qanneal::MultiAnnealResult::global_best_state)
        .def_readonly("global_best_energy", &qanneal::MultiAnnealResult::global_best_energy)
        .def_readonly("average_energy_trace", &qanneal::MultiAnnealResult::average_energy_trace)
        .def_readonly("average_magnetization_trace", &qanneal::MultiAnnealResult::average_magnetization_trace);

    py::class_<qanneal::ReplicaAnnealer>(m, "ReplicaAnnealer")
        .def(py::init([](std::shared_ptr<qanneal::Hamiltonian> ham,
                         qanneal::AnnealSchedule schedule,
                         std::size_t replicas,
                         const std::string &backend) {
            auto kind = qanneal::backend_from_string(backend);
            auto be = qanneal::make_backend(kind, std::move(ham));
            return qanneal::ReplicaAnnealer(std::move(be), std::move(schedule), replicas);
        }),
        py::arg("hamiltonian"),
        py::arg("schedule"),
        py::arg("replicas"),
        py::arg("backend") = "cpu",
        py::keep_alive<1, 2>())
        .def("set_seed", &qanneal::ReplicaAnnealer::set_seed)
        .def("run", &qanneal::ReplicaAnnealer::run, py::arg("sweeps_per_beta"));

    py::class_<qanneal::ParallelTemperingResult>(m, "ParallelTemperingResult")
        .def_readonly("final_states", &qanneal::ParallelTemperingResult::final_states)
        .def_readonly("final_energies", &qanneal::ParallelTemperingResult::final_energies)
        .def_readonly("best_state", &qanneal::ParallelTemperingResult::best_state)
        .def_readonly("best_energy", &qanneal::ParallelTemperingResult::best_energy)
        .def_readonly("average_energy_trace", &qanneal::ParallelTemperingResult::average_energy_trace)
        .def_readonly("swap_acceptance_trace", &qanneal::ParallelTemperingResult::swap_acceptance_trace);

    py::class_<qanneal::ParallelTemperingAnnealer>(m, "ParallelTemperingAnnealer")
        .def(py::init([](std::shared_ptr<qanneal::Hamiltonian> ham,
                         const std::vector<double> &betas,
                         const std::string &backend) {
            auto kind = qanneal::backend_from_string(backend);
            auto be = qanneal::make_backend(kind, std::move(ham));
            return qanneal::ParallelTemperingAnnealer(std::move(be), betas);
        }),
        py::arg("hamiltonian"),
        py::arg("betas"),
        py::arg("backend") = "cpu",
        py::keep_alive<1, 2>())
        .def("set_seed", &qanneal::ParallelTemperingAnnealer::set_seed)
        .def("run", &qanneal::ParallelTemperingAnnealer::run,
             py::arg("sweeps_per_step"),
             py::arg("steps"),
             py::arg("swap_interval") = 1);

    py::class_<qanneal::SQASchedule>(m, "SQASchedule")
        .def(py::init<>())
        .def_readwrite("betas", &qanneal::SQASchedule::betas)
        .def_readwrite("gammas", &qanneal::SQASchedule::gammas)
        .def_static("from_vectors", &qanneal::SQASchedule::from_vectors,
                    py::arg("betas"), py::arg("gammas"));

    py::class_<qanneal::SQAObserver, std::shared_ptr<qanneal::SQAObserver>>(m, "SQAObserver");
    py::enum_<qanneal::SQASweepPhase>(m, "SQASweepPhase")
        .value("SLICE", qanneal::SQASweepPhase::Slice)
        .value("WORLDLINE", qanneal::SQASweepPhase::Worldline);
    py::class_<qanneal::SQAMetricsObserver, qanneal::SQAObserver, std::shared_ptr<qanneal::SQAMetricsObserver>>(m, "SQAMetricsObserver")
        .def(py::init<>())
        .def_readonly("energy_trace", &qanneal::SQAMetricsObserver::energy_trace)
        .def_readonly("magnetization_trace", &qanneal::SQAMetricsObserver::magnetization_trace)
        .def("clear", &qanneal::SQAMetricsObserver::clear);
    py::class_<qanneal::SQAStateTraceObserver, qanneal::SQAObserver, std::shared_ptr<qanneal::SQAStateTraceObserver>>(m, "SQAStateTraceObserver")
        .def(py::init<>())
        .def_readwrite("stride", &qanneal::SQAStateTraceObserver::stride)
        .def_readonly("replicas", &qanneal::SQAStateTraceObserver::replicas)
        .def_readonly("slices", &qanneal::SQAStateTraceObserver::slices)
        .def_readonly("spins", &qanneal::SQAStateTraceObserver::spins)
        .def_readonly("step_trace", &qanneal::SQAStateTraceObserver::step_trace)
        .def_readonly("replica_trace", &qanneal::SQAStateTraceObserver::replica_trace)
        .def_readonly("sweep_trace", &qanneal::SQAStateTraceObserver::sweep_trace)
        .def_readonly("phase_trace", &qanneal::SQAStateTraceObserver::phase_trace)
        .def_readonly("beta_trace", &qanneal::SQAStateTraceObserver::beta_trace)
        .def_readonly("gamma_trace", &qanneal::SQAStateTraceObserver::gamma_trace)
        .def_readonly("avg_energy_trace", &qanneal::SQAStateTraceObserver::avg_energy_trace)
        .def_readonly("replica_energy_trace", &qanneal::SQAStateTraceObserver::replica_energy_trace)
        .def_readonly("state_trace", &qanneal::SQAStateTraceObserver::state_trace)
        .def("clear", &qanneal::SQAStateTraceObserver::clear);

    py::class_<qanneal::SQAResult>(m, "SQAResult")
        .def_readonly("best_state", &qanneal::SQAResult::best_state)
        .def_readonly("best_energy", &qanneal::SQAResult::best_energy)
        .def_readonly("energy_trace", &qanneal::SQAResult::energy_trace);

    py::class_<qanneal::SQAAnnealer>(m, "SQAAnnealer")
        .def(py::init([](std::shared_ptr<qanneal::Hamiltonian> ham,
                         qanneal::SQASchedule schedule,
                         std::size_t trotter_slices,
                         std::size_t replicas,
                         const std::string &backend) {
            auto kind = qanneal::backend_from_string(backend);
            auto be = qanneal::make_backend(kind, std::move(ham));
            return qanneal::SQAAnnealer(std::move(be), std::move(schedule), trotter_slices, replicas);
        }),
        py::arg("hamiltonian"),
        py::arg("schedule"),
        py::arg("trotter_slices"),
        py::arg("replicas") = 1,
        py::arg("backend") = "cpu",
        py::keep_alive<1, 2>())
        .def("set_seed", &qanneal::SQAAnnealer::set_seed)
        .def("run", [](qanneal::SQAAnnealer &self,
                       std::size_t sweeps_per_beta,
                       std::size_t worldline_sweeps,
                       std::shared_ptr<qanneal::SQAObserver> obs) {
            return self.run(sweeps_per_beta, worldline_sweeps, obs.get());
        }, py::arg("sweeps_per_beta"), py::arg("worldline_sweeps"), py::arg("observer") = nullptr);

    m.def("magnetization", [](const py::sequence &spins) {
        auto data = seq_to_spins(spins);
        return qanneal::magnetization(data.data(), data.size());
    });

    m.def("overlap", [](const py::sequence &a, const py::sequence &b) {
        auto da = seq_to_spins(a);
        auto db = seq_to_spins(b);
        if (da.size() != db.size()) {
            throw std::invalid_argument("Spin vectors must be same length.");
        }
        return qanneal::overlap(da.data(), db.data(), da.size());
    });
}
