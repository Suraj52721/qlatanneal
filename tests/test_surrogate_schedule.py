"""Bond-susceptibility surrogate schedule — SQAAnnealer.run_surrogate + solve() wiring.

The surrogate method (paper: "Bond Susceptibility as a Surrogate for Spectral Gaps in
Quantum Annealing Schedule Design") replaces the spectral gap in Roland-Cerf scheduling
with chi_B = Var(B) measured in a pilot SQA scan, allocating annealing time with weight
w(s) = chi_B(s) + chi_0 through the cumulative integral tau(s).

These tests pin down:
  * run_surrogate returns well-formed diagnostics (scan profile, QCP estimate, schedule),
  * the gamma schedule is monotone and spans the requested range,
  * it finds the exact ground state of a ferromagnetic chain,
  * solve(..., schedule_type="surrogate") round-trips through the high-level API,
  * surrogate is rejected for non-SQA methods.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Use the in-tree build (python/qanneal/_qanneal*.so) without requiring an install.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "python"))

solver = pytest.importorskip("qanneal.solver")
_q = pytest.importorskip("qanneal._qanneal")
DenseIsing = _q.DenseIsing
SQAAnnealer = _q.SQAAnnealer
SQASchedule = _q.SQASchedule

if not hasattr(SQAAnnealer, "run_surrogate"):
    pytest.skip("qanneal build predates run_surrogate; rebuild _qanneal", allow_module_level=True)


def _ferro_chain(n: int) -> DenseIsing:
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = J[i + 1, i] = -1.0
    return DenseIsing(np.zeros(n), J)


def _sk(n: int, seed: int = 0) -> DenseIsing:
    rng = np.random.default_rng(seed)
    J = rng.normal(0.0, 1.0 / np.sqrt(n), size=(n, n))
    J = np.triu(J, 1)
    J = J + J.T
    return DenseIsing(np.zeros(n), J)


def _annealer(ham: DenseIsing, slices: int = 8, replicas: int = 4) -> SQAAnnealer:
    sched = SQASchedule.from_vectors([1.0], [1.0])
    return SQAAnnealer(ham, sched, slices, replicas)


def test_run_surrogate_diagnostics_and_schedule():
    ann = _annealer(_sk(10, seed=3))
    ann.set_seed(2026)
    num_steps, scan_points = 120, 12
    res = ann.run_surrogate(
        beta=4.0, gamma_start=4.0, gamma_end=0.05,
        num_steps=num_steps, sweeps_per_step=8,
        worldline_sweeps=2, cluster_sweeps=1,
        scan_points=scan_points, scan_sweeps=24, scan_burn=8,
    )

    scan_s = np.array(res.scan_s)
    scan_chi = np.array(res.scan_chi_B)
    gammas = np.array(res.gamma_schedule)
    s_sched = np.array(res.s_schedule)

    assert scan_s.shape == (scan_points,) and scan_chi.shape == (scan_points,)
    assert len(res.beta_schedule) == num_steps
    assert gammas.shape == (num_steps,) and s_sched.shape == (num_steps,)
    assert len(res.energy_trace) == num_steps

    # chi_B profile: non-negative, not flat
    assert (scan_chi >= 0.0).all()
    assert scan_chi.max() > scan_chi.min()
    assert res.chi0 > 0.0
    assert res.driver_A0 == pytest.approx(4.0)

    # QCP estimate at the chi_B peak, inside the scanned range
    assert scan_s[0] <= res.s_star <= scan_s[-1]
    assert res.s_star == pytest.approx(scan_s[np.argmax(scan_chi)])
    assert res.j_perp_star > 0.0

    # gamma schedule: monotone non-increasing, spans the full range
    assert (np.diff(gammas) <= 1e-12).all()
    assert gammas[0] == pytest.approx(4.0)
    assert gammas[-1] == pytest.approx(0.05, abs=1e-6)
    # s ascends over the anneal
    assert (np.diff(s_sched) >= -1e-12).all()


def test_run_surrogate_finds_ferro_ground_state():
    n = 12
    ann = _annealer(_ferro_chain(n))
    ann.set_seed(7)
    res = ann.run_surrogate(
        beta=4.0, gamma_start=3.0, gamma_end=0.05,
        num_steps=100, sweeps_per_step=10,
        worldline_sweeps=2, cluster_sweeps=1,
        scan_points=10, scan_sweeps=20, scan_burn=6,
    )
    assert res.best_energy == pytest.approx(-(n - 1))
    assert abs(sum(res.best_state.spins)) == n  # fully aligned


def test_solve_surrogate_roundtrip():
    n = 10
    res = solver.solve(
        _sk(n, seed=1),
        method="sqa",
        schedule_type="surrogate",
        reads=1,
        sweeps_per_beta=8,
        worldline_sweeps=2,
        trotter_slices=8,
        replicas=2,
        seed=11,
        optimal_num_steps=80,
        surrogate_scan_points=8,
        surrogate_scan_sweeps=12,
        surrogate_scan_burn=4,
        progress=False,
    )
    assert res.method == "sqa"
    assert len(res.best_sample) == n
    assert set(np.unique(res.best_sample)).issubset({-1, 1})
    assert np.isfinite(res.best_energy)
    assert res.trace is not None and len(res.trace) == 80


def test_solve_surrogate_rejects_non_sqa():
    with pytest.raises(ValueError, match="surrogate"):
        solver.solve(_sk(6), method="sqapt", schedule_type="surrogate", progress=False)
    with pytest.raises(ValueError, match="surrogate"):
        solver.solve(_sk(6), method="sa", schedule_type="surrogate", progress=False)
