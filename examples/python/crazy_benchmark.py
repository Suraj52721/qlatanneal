#!/usr/bin/env python3
"""
crazy_benchmark.py
==================
A comprehensive head-to-head benchmark across FIVE problem families:

  1. SK spin glass         – fully-connected random ±J Ising (canonical NP-hard)
  2. Frustrated magnet     – antiferromagnetic triangular-grid Ising
  3. MaxCut                – random 3-regular graph, QUBO formulation
  4. Dense random QUBO     – no structure, brute-force verified for n≤20
  5. Number partitioning   – balanced-partition hard instances

Solvers:
  • qanneal-SA     (C++, tuned schedule)
  • qanneal-SQA    (Trotter path-integral, tuned schedule)
  • qanneal-SQAPT  (SQA + parallel tempering, tuned ladder)
  • dwave-NEAL     (reference pure-Python SA)   [optional]
  • dwave-PIMC     (D-Wave path-integral)        [optional]
  • random-search  (baseline)

Metrics per instance:
  • best energy found
  • gap to oracle  (brute-force ≤20, else inter-solver oracle)
  • normalised residual  = (E - E_opt) / |E_opt|
  • wall-clock time
  • ground-state hit rate (for exact problems)
  • approximate TTS at 99 % success (extrapolated from per-read hit probability)

Scaling sweep: n ∈ {8, 12, 16, 20, 30, 50} (configurable)

Outputs:
  crazy_benchmark_dashboard.png   – 6-panel summary dashboard
  crazy_benchmark_scaling.png     – scaling plots per problem family
  crazy_benchmark_results.json    – full per-instance results
  crazy_benchmark_summary.csv     – aggregated summary table
  crazy_benchmark_traces.png      – energy-convergence curves (one per family)

Usage examples:
  python crazy_benchmark.py                              # defaults
  python crazy_benchmark.py --instances 20 --reads 50   # more statistics
  python crazy_benchmark.py --no-sqapt --no-neal        # quick run
  python crazy_benchmark.py --accurate                  # slow but thorough
  python crazy_benchmark.py --fair                      # scientifically equal compute budget
  python crazy_benchmark.py --fair --budget-updates 1e7 # explicit budget
  python crazy_benchmark.py --fair --accurate           # both: thorough + fair

--fair mode:
  Each solver's sweep count is adjusted so every method performs approximately
  the same total number of spin-update operations for a given problem size n:

    budget  = reads × steps × sweeps × n   (SA baseline)
    SA      sweeps = sweeps                          (unchanged)
    SQA     sweeps = budget / (reads × steps × n × slices × replicas)
    SQAPT   sweeps = budget / (reads × pt_steps × n × slices × replicas)
    NEAL/PIMC num_sweeps = budget / (reads × n)

  This removes the "more compute = better result" confound and reveals true
  algorithmic quality differences between classical and quantum-inspired methods.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Optional third-party imports ─────────────────────────────────────────────

try:
    import qanneal
    from qanneal import (
        DenseIsing,
        solve,
        auto_schedule_sa_tuned,
        auto_schedule_sqa_tuned,
        auto_ladder_sqa_tuned,
    )
    HAS_QANNEAL = True
except ImportError:
    HAS_QANNEAL = False
    print("WARNING: qanneal not installed – run `pip install -e .` from repo root", file=sys.stderr)

try:
    import dimod  # type: ignore
    HAS_DIMOD = True
except ImportError:
    dimod = None  # type: ignore
    HAS_DIMOD = False

try:
    import neal  # type: ignore
    HAS_NEAL = True
except ImportError:
    neal = None  # type: ignore
    HAS_NEAL = False

try:
    from dwave.samplers import PathIntegralAnnealingSampler  # type: ignore
    HAS_DWAVE_PIMC = True
except ImportError:
    PathIntegralAnnealingSampler = None  # type: ignore
    HAS_DWAVE_PIMC = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
    HAS_MPL = True
except ImportError:
    plt = None  # type: ignore
    HAS_MPL = False
    print("WARNING: matplotlib not installed – plots disabled", file=sys.stderr)

try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except ImportError:
    def tqdm(x, **kw):  # type: ignore
        return x
    HAS_TQDM = False

# ── Colour palette (warm parchment theme matching qanneal GUI) ────────────────
COLOURS = {
    "qanneal-SA":    "#2a9d8f",
    "qanneal-SQA":   "#e9c46a",
    "qanneal-SQAPT": "#e76f51",
    "dwave-NEAL":    "#264653",
    "dwave-PIMC":    "#a8dadc",
    "random":        "#bdbdbd",
}
DEFAULT_COLOUR = "#888888"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PROBLEM GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def gen_sk_spin_glass(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sherrington-Kirkpatrick model: fully-connected, J_ij ~ N(0, 1/√n)."""
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n), dtype=float)
    scale = 1.0 / max(np.sqrt(n), 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            v = rng.normal(0.0, scale)
            J[i, j] = v
            J[j, i] = v
    h = rng.normal(0.0, 0.05, size=n)
    return h, J


def gen_frustrated_magnet(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangular-strip antiferromagnet: spins on a 1D chain with next-nearest
    couplings, all J > 0, heavily frustrated.  J_nn = 1.0, J_nnn = 0.5.
    """
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        J[i, i + 1] = 1.0 + rng.uniform(-0.1, 0.1)
        J[i + 1, i] = J[i, i + 1]
    for i in range(n - 2):
        J[i, i + 2] = 0.5 + rng.uniform(-0.05, 0.05)
        J[i + 2, i] = J[i, i + 2]
    h = rng.normal(0.0, 0.02, size=n)
    return h, J


def gen_maxcut_qubo(n: int, seed: int, degree: int = 3) -> np.ndarray:
    """
    MaxCut on a random d-regular graph → upper-triangular QUBO Q.
    Maximise cut = Σ w_ij (x_i + x_j - 2 x_i x_j)
    Minimise QUBO: Q_ii -= w, Q_ij += w  (so solvers minimise → maximise cut).
    """
    rng = np.random.default_rng(seed)
    # Build random graph: each node has ~degree edges
    adj: List[set] = [set() for _ in range(n)]
    for i in range(n):
        while len(adj[i]) < degree and len(adj[i]) < n - 1:
            j = int(rng.integers(0, n))
            if j != i and len(adj[j]) < degree:
                adj[i].add(j)
                adj[j].add(i)

    Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in adj[i]:
            if i < j:
                w = float(rng.uniform(0.5, 1.5))
                Q[i, i] -= w
                Q[j, j] -= w
                Q[i, j] += w
                Q[j, i] += w
    return Q


def gen_random_qubo(n: int, seed: int, density: float = 0.6) -> np.ndarray:
    """Dense random QUBO with entries ~ U(-1, 1), upper triangular."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        Q[i, i] = rng.uniform(-1.0, 1.0)
        for j in range(i + 1, n):
            if rng.random() < density:
                v = rng.uniform(-1.0, 1.0)
                Q[i, j] = v
                Q[j, i] = v
    return Q


def gen_number_partition_ising(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Number partitioning as Ising: minimise (Σ a_i s_i)^2.
    Expands to h=0, J_ij = a_i a_j, c = Σ a_i^2.
    Hard instances have numbers drawn from U(1, 2^20).
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2 ** 20, size=n).astype(float)
    J = np.outer(a, a)
    np.fill_diagonal(J, 0.0)
    h = np.zeros(n, dtype=float)
    c = float(np.dot(a, a))
    return h, J, c


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ORACLE / BRUTE-FORCE
# ═══════════════════════════════════════════════════════════════════════════════

def brute_force_ising(h: np.ndarray, J: np.ndarray, c: float = 0.0) -> Tuple[float, np.ndarray]:
    n = len(h)
    assert n <= 22, f"brute force only safe for n≤22, got n={n}"
    best_e = float("inf")
    best_s = np.ones(n, dtype=int)
    for mask in range(1 << n):
        s = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)], dtype=float)
        e = float(np.dot(h, s) + 0.5 * s @ J @ s) + c
        if e < best_e:
            best_e = e
            best_s = s.astype(int)
    return best_e, best_s


def brute_force_qubo(Q: np.ndarray) -> Tuple[float, np.ndarray]:
    n = Q.shape[0]
    assert n <= 22
    best_e = float("inf")
    best_x = np.zeros(n, dtype=int)
    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        e = float(x @ Q @ x)
        if e < best_e:
            best_e = e
            best_x = x.astype(int)
    return best_e, best_x


def partition_diff(a: np.ndarray, spins: np.ndarray) -> float:
    return float(abs(np.dot(a, spins)))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SOLVER WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolverResult:
    method: str
    energies: List[float]           # one per read
    best_energy: float
    wall_time_s: float
    trace: Optional[List[float]] = None   # last-read energy trace
    ok: bool = True
    error: Optional[str] = None


def _safe_run(fn: Callable, method: str) -> SolverResult:
    t0 = time.perf_counter()
    try:
        result: SolverResult = fn()
        result.wall_time_s = time.perf_counter() - t0
        return result
    except Exception as exc:
        dt = time.perf_counter() - t0
        return SolverResult(method=method, energies=[], best_energy=float("inf"),
                            wall_time_s=dt, ok=False, error=str(exc))


def make_ising(h: np.ndarray, J: np.ndarray, c: float = 0.0) -> "DenseIsing":
    try:
        return DenseIsing(h, J, c)
    except TypeError:
        return DenseIsing(h.tolist(), J.flatten().tolist(), len(h), c)


# ── qanneal SA ────────────────────────────────────────────────────────────────

def run_qanneal_sa(ising: "DenseIsing", reads: int, cfg) -> SolverResult:
    def _inner():
        sched = auto_schedule_sa_tuned(ising, mode=cfg.mode, steps=cfg.steps, seed=cfg.seed)
        res = solve(ising, method="sa", reads=reads,
                    sweeps_per_beta=cfg.sa_sweeps, schedule=sched,
                    seed=cfg.seed, progress=False)
        return SolverResult(
            method="qanneal-SA",
            energies=list(res.energies),
            best_energy=float(res.best_energy),
            wall_time_s=0.0,
            trace=list(res.trace or []),
        )
    return _safe_run(_inner, "qanneal-SA")


# ── qanneal SQA ───────────────────────────────────────────────────────────────

def run_qanneal_sqa(ising: "DenseIsing", reads: int, cfg) -> SolverResult:
    def _inner():
        sched = auto_schedule_sqa_tuned(ising, mode=cfg.mode, steps=cfg.steps, seed=cfg.seed)
        res = solve(ising, method="sqa", reads=reads,
                    sweeps_per_beta=cfg.sqa_sweeps,
                    worldline_sweeps=cfg.worldline,
                    cluster_sweeps=cfg.cluster,
                    trotter_slices=cfg.slices,
                    replicas=cfg.replicas,
                    schedule=sched,
                    seed=cfg.seed, progress=False)
        return SolverResult(
            method="qanneal-SQA",
            energies=list(res.energies),
            best_energy=float(res.best_energy),
            wall_time_s=0.0,
            trace=list(res.trace or []),
        )
    return _safe_run(_inner, "qanneal-SQA")


# ── qanneal SQAPT ─────────────────────────────────────────────────────────────

def run_qanneal_sqapt(ising: "DenseIsing", reads: int, cfg) -> SolverResult:
    def _inner():
        ladder = auto_ladder_sqa_tuned(ising, replicas=cfg.replicas, mode=cfg.mode, seed=cfg.seed)
        res = solve(ising, method="sqapt", reads=reads,
                    sweeps_per_beta=cfg.sqapt_sweeps,
                    worldline_sweeps=cfg.worldline,
                    cluster_sweeps=cfg.cluster,
                    trotter_slices=cfg.slices,
                    replicas=cfg.replicas,
                    pt_steps=cfg.pt_steps,
                    swap_interval=cfg.swap_interval,
                    schedule=ladder,
                    seed=cfg.seed, progress=False)
        return SolverResult(
            method="qanneal-SQAPT",
            energies=list(res.energies),
            best_energy=float(res.best_energy),
            wall_time_s=0.0,
            trace=list(res.trace or []),
        )
    return _safe_run(_inner, "qanneal-SQAPT")


# ── dwave-NEAL SA ─────────────────────────────────────────────────────────────

def _build_spin_bqm(h: np.ndarray, J: np.ndarray):
    if not HAS_DIMOD:
        return None
    linear = {i: float(h[i]) for i in range(len(h))}
    quad = {(i, j): float(J[i, j])
            for i in range(len(h))
            for j in range(i + 1, len(h))
            if J[i, j] != 0.0}
    return dimod.BinaryQuadraticModel(linear, quad, 0.0, vartype="SPIN")


def _build_binary_bqm(Q: np.ndarray):
    if not HAS_DIMOD:
        return None
    n = Q.shape[0]
    linear = {i: float(Q[i, i]) for i in range(n)}
    quad = {(i, j): float(Q[i, j] + Q[j, i])
            for i in range(n) for j in range(i + 1, n)
            if Q[i, j] != 0.0 or Q[j, i] != 0.0}
    return dimod.BinaryQuadraticModel(linear, quad, 0.0, vartype="BINARY")


def run_neal_sa(bqm, reads: int, cfg) -> SolverResult:
    if not HAS_NEAL or bqm is None:
        return SolverResult("dwave-NEAL", [], float("inf"), 0.0, ok=False,
                            error="dwave-neal not installed")
    def _inner():
        sampler = neal.SimulatedAnnealingSampler()
        num_sweeps = max(1, cfg.neal_sweeps)
        ss = sampler.sample(bqm, num_reads=reads, num_sweeps=num_sweeps,
                            beta_range=(0.1, 4.0), beta_schedule_type="geometric",
                            seed=cfg.seed)
        energies = sorted(float(s.energy) for s in ss.samples())
        return SolverResult("dwave-NEAL", energies, energies[0], 0.0)
    return _safe_run(_inner, "dwave-NEAL")


# ── dwave-PIMC ────────────────────────────────────────────────────────────────

def run_dwave_pimc(bqm, reads: int, cfg) -> SolverResult:
    if not HAS_DWAVE_PIMC or bqm is None:
        return SolverResult("dwave-PIMC", [], float("inf"), 0.0, ok=False,
                            error="dwave.samplers not installed")
    def _inner():
        sampler = PathIntegralAnnealingSampler()
        num_sweeps = max(1, cfg.pimc_sweeps)
        ss = sampler.sample(bqm, num_reads=reads, num_sweeps=num_sweeps)
        energies = sorted(float(s.energy) for s in ss.samples())
        return SolverResult("dwave-PIMC", energies, energies[0], 0.0)
    return _safe_run(_inner, "dwave-PIMC")


# ── Random search baseline ────────────────────────────────────────────────────

def run_random_ising(h: np.ndarray, J: np.ndarray, c: float, reads: int, cfg) -> SolverResult:
    rng = np.random.default_rng(cfg.seed)
    n = len(h)
    trials = reads * max(100, cfg.steps * cfg.sweeps // max(n, 1))
    energies_list = []
    best_e = float("inf")
    for _ in range(trials):
        s = rng.choice([-1, 1], size=n).astype(float)
        e = float(np.dot(h, s) + 0.5 * s @ J @ s) + c
        energies_list.append(e)
        if e < best_e:
            best_e = e
    # subsample down to `reads` representative values
    idx = np.round(np.linspace(0, len(energies_list) - 1, reads)).astype(int)
    return SolverResult("random", [energies_list[i] for i in idx], best_e, 0.0)


def run_random_qubo(Q: np.ndarray, reads: int, cfg) -> SolverResult:
    rng = np.random.default_rng(cfg.seed)
    n = Q.shape[0]
    trials = reads * max(100, cfg.steps * cfg.sweeps // max(n, 1))
    energies_list = []
    best_e = float("inf")
    for _ in range(trials):
        x = rng.integers(0, 2, size=n).astype(float)
        e = float(x @ Q @ x)
        energies_list.append(e)
        if e < best_e:
            best_e = e
    idx = np.round(np.linspace(0, len(energies_list) - 1, reads)).astype(int)
    return SolverResult("random", [energies_list[i] for i in idx], best_e, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INSTANCE RUNNER  (Ising or QUBO)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InstanceResult:
    problem: str
    n: int
    instance_id: int
    oracle: float
    methods: List[SolverResult] = field(default_factory=list)


def gap(e: float, oracle: float) -> float:
    if not np.isfinite(oracle) or oracle == 0.0:
        return max(0.0, e - oracle) if np.isfinite(oracle) else float("inf")
    return max(0.0, (e - oracle) / max(abs(oracle), 1e-12))


def hit_rate(energies: List[float], oracle: float, tol: float = 1e-6) -> float:
    if not energies:
        return 0.0
    hits = sum(1 for e in energies if abs(e - oracle) <= tol * max(abs(oracle), 1.0))
    return hits / len(energies)


def tts_99(hit_prob: float) -> float:
    """TTS for 99% success probability given per-read hit probability."""
    if hit_prob <= 0.0:
        return float("inf")
    if hit_prob >= 1.0:
        return 1.0
    return np.log(1.0 - 0.99) / np.log(1.0 - hit_prob)


def run_ising_instance(problem: str, n: int, inst: int,
                       h: np.ndarray, J: np.ndarray, c: float,
                       reads: int, cfg, exact_max_n: int = 20,
                       with_neal: bool = True,
                       with_pimc: bool = True,
                       with_sqapt: bool = True) -> InstanceResult:
    ising = make_ising(h, J, c)
    bqm = _build_spin_bqm(h, J)

    methods: List[SolverResult] = []
    if HAS_QANNEAL:
        methods.append(run_qanneal_sa(ising, reads, cfg))
        methods.append(run_qanneal_sqa(ising, reads, cfg))
        if with_sqapt:
            methods.append(run_qanneal_sqapt(ising, reads, cfg))
    if with_neal:
        methods.append(run_neal_sa(bqm, reads, cfg))
    if with_pimc:
        methods.append(run_dwave_pimc(bqm, reads, cfg))
    methods.append(run_random_ising(h, J, c, reads, cfg))

    # oracle
    if n <= exact_max_n:
        oracle_e, _ = brute_force_ising(h, J, c)
    else:
        ok_energies = [r.best_energy for r in methods if r.ok and np.isfinite(r.best_energy)]
        oracle_e = min(ok_energies) if ok_energies else float("inf")

    return InstanceResult(problem=problem, n=n, instance_id=inst,
                          oracle=oracle_e, methods=methods)


def run_qubo_instance(problem: str, n: int, inst: int,
                      Q: np.ndarray,
                      reads: int, cfg, exact_max_n: int = 20,
                      with_neal: bool = True,
                      with_pimc: bool = True,
                      with_sqapt: bool = True) -> InstanceResult:
    if not HAS_QANNEAL:
        return InstanceResult(problem=problem, n=n, instance_id=inst,
                              oracle=float("inf"), methods=[])
    ising = make_ising(*_qubo_to_ising_numpy(Q))
    bqm = _build_binary_bqm(Q)

    methods: List[SolverResult] = []
    if HAS_QANNEAL:
        # Solve as QUBO via the Python solver (handles QUBO→Ising internally)
        methods.append(_run_qanneal_sa_qubo(Q, reads, cfg))
        methods.append(_run_qanneal_sqa_qubo(Q, reads, cfg))
        if with_sqapt:
            methods.append(_run_qanneal_sqapt_qubo(Q, reads, cfg))
    if with_neal:
        methods.append(run_neal_sa(bqm, reads, cfg))
    if with_pimc:
        methods.append(run_dwave_pimc(bqm, reads, cfg))
    methods.append(run_random_qubo(Q, reads, cfg))

    if n <= exact_max_n:
        oracle_e, _ = brute_force_qubo(Q)
    else:
        ok_e = [r.best_energy for r in methods if r.ok and np.isfinite(r.best_energy)]
        oracle_e = min(ok_e) if ok_e else float("inf")

    return InstanceResult(problem=problem, n=n, instance_id=inst,
                          oracle=oracle_e, methods=methods)


def _qubo_to_ising_numpy(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Manual QUBO→Ising conversion matching the fixed qubo.cpp formula."""
    n = Q.shape[0]
    W = 0.5 * (Q + Q.T)
    row_sums = W.sum(axis=1)
    h = 0.5 * row_sums
    J = 0.5 * W
    np.fill_diagonal(J, 0.0)
    diag_sum = np.diag(W).sum()
    c = 0.25 * diag_sum + 0.25 * W.sum()
    return h, J, float(c)


def _run_qanneal_sa_qubo(Q: np.ndarray, reads: int, cfg) -> SolverResult:
    def _inner():
        sched = auto_schedule_sa_tuned(Q, mode=cfg.mode, steps=cfg.steps, seed=cfg.seed)
        res = solve(Q, method="sa", reads=reads,
                    sweeps_per_beta=cfg.sa_sweeps, schedule=sched,
                    seed=cfg.seed, progress=False, return_bits=True)
        return SolverResult("qanneal-SA", list(res.energies),
                            float(res.best_energy), 0.0, list(res.trace or []))
    return _safe_run(_inner, "qanneal-SA")


def _run_qanneal_sqa_qubo(Q: np.ndarray, reads: int, cfg) -> SolverResult:
    def _inner():
        sched = auto_schedule_sqa_tuned(Q, mode=cfg.mode, steps=cfg.steps, seed=cfg.seed)
        res = solve(Q, method="sqa", reads=reads,
                    sweeps_per_beta=cfg.sqa_sweeps,
                    worldline_sweeps=cfg.worldline,
                    cluster_sweeps=cfg.cluster,
                    trotter_slices=cfg.slices,
                    replicas=cfg.replicas,
                    schedule=sched,
                    seed=cfg.seed, progress=False, return_bits=True)
        return SolverResult("qanneal-SQA", list(res.energies),
                            float(res.best_energy), 0.0, list(res.trace or []))
    return _safe_run(_inner, "qanneal-SQA")


def _run_qanneal_sqapt_qubo(Q: np.ndarray, reads: int, cfg) -> SolverResult:
    def _inner():
        ladder = auto_ladder_sqa_tuned(Q, replicas=cfg.replicas, mode=cfg.mode, seed=cfg.seed)
        res = solve(Q, method="sqapt", reads=reads,
                    sweeps_per_beta=cfg.sqapt_sweeps,
                    worldline_sweeps=cfg.worldline,
                    cluster_sweeps=cfg.cluster,
                    trotter_slices=cfg.slices,
                    replicas=cfg.replicas,
                    pt_steps=cfg.pt_steps,
                    swap_interval=cfg.swap_interval,
                    schedule=ladder,
                    seed=cfg.seed, progress=False, return_bits=True)
        return SolverResult("qanneal-SQAPT", list(res.energies),
                            float(res.best_energy), 0.0, list(res.trace or []))
    return _safe_run(_inner, "qanneal-SQAPT")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AggRow:
    problem: str
    n: int
    method: str
    n_inst: int
    med_gap: float       # median normalised residual gap
    mean_gap: float
    std_gap: float
    med_time: float
    mean_time: float
    hit_rate: float      # fraction of instances with gap < 1e-6
    med_tts: float       # median TTS(99%) in reads
    best_gap: float      # best gap found across all instances


def aggregate(instances: List[InstanceResult]) -> List[AggRow]:
    # Group by (problem, n, method)
    from collections import defaultdict
    store: Dict[Tuple[str, int, str], Dict[str, List]] = defaultdict(
        lambda: {"gaps": [], "times": [], "hr": [], "tts": []}
    )

    for ir in instances:
        for sr in ir.methods:
            key = (ir.problem, ir.n, sr.method)
            if not sr.ok or not np.isfinite(sr.best_energy):
                continue
            g = gap(sr.best_energy, ir.oracle)
            store[key]["gaps"].append(g)
            store[key]["times"].append(sr.wall_time_s)
            hr = hit_rate(sr.energies, ir.oracle)
            store[key]["hr"].append(hr)
            store[key]["tts"].append(tts_99(hr))

    rows: List[AggRow] = []
    for (prob, n, meth), d in store.items():
        gaps = np.array(d["gaps"], dtype=float)
        times = np.array(d["times"], dtype=float)
        hr_vals = np.array(d["hr"], dtype=float)
        tts_vals = np.array([v for v in d["tts"] if np.isfinite(v)], dtype=float)
        rows.append(AggRow(
            problem=prob, n=n, method=meth,
            n_inst=len(gaps),
            med_gap=float(np.median(gaps)),
            mean_gap=float(np.mean(gaps)),
            std_gap=float(np.std(gaps)),
            med_time=float(np.median(times)),
            mean_time=float(np.mean(times)),
            hit_rate=float(np.mean(hr_vals)),
            med_tts=float(np.median(tts_vals)) if len(tts_vals) else float("inf"),
            best_gap=float(np.min(gaps)),
        ))
    rows.sort(key=lambda r: (r.problem, r.n, r.med_gap))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _method_colour(method: str) -> str:
    return COLOURS.get(method, DEFAULT_COLOUR)


def plot_dashboard(rows: List[AggRow], instances: List[InstanceResult],
                   out_path: Path) -> None:
    if not HAS_MPL:
        return

    problems = sorted(set(r.problem for r in rows))
    methods = [m for m in ["qanneal-SA", "qanneal-SQA", "qanneal-SQAPT",
                           "dwave-NEAL", "dwave-PIMC", "random"]
               if any(r.method == m for r in rows)]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#fdf8ef")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Median gap per problem family (bar chart) ────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#fdf8ef")
    n_problems = len(problems)
    n_methods = len(methods)
    bar_w = 0.8 / max(n_methods, 1)
    x_pos = np.arange(n_problems)
    for mi, meth in enumerate(methods):
        vals = []
        for prob in problems:
            matched = [r for r in rows if r.problem == prob and r.method == meth]
            if matched:
                # take median across all n sizes for this panel
                v = float(np.median([r.med_gap for r in matched]))
            else:
                v = float("nan")
            vals.append(v)
        offset = (mi - n_methods / 2 + 0.5) * bar_w
        bars = ax1.bar(x_pos + offset, vals, bar_w * 0.9,
                       label=meth, color=_method_colour(meth), alpha=0.85)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(problems, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Median norm. gap (↓ better)")
    ax1.set_title("Solution quality across problem families")
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_yscale("symlog", linthresh=1e-4)
    ax1.axhline(0, color="#888", linewidth=0.5)
    ax1.set_facecolor("#fdf8ef")

    # ── Panel 2: Wall-clock time per problem family ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#fdf8ef")
    for mi, meth in enumerate(methods):
        vals = []
        for prob in problems:
            matched = [r for r in rows if r.problem == prob and r.method == meth]
            v = float(np.median([r.med_time for r in matched])) if matched else float("nan")
            vals.append(v)
        offset = (mi - n_methods / 2 + 0.5) * bar_w
        ax2.bar(x_pos + offset, vals, bar_w * 0.9,
                label=meth, color=_method_colour(meth), alpha=0.85)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(problems, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Median wall time (s)")
    ax2.set_title("Runtime comparison")
    ax2.set_yscale("log")
    ax2.set_facecolor("#fdf8ef")

    # ── Panel 3–7: Scaling plots  (one per problem) ───────────────────────────
    subplot_idx = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    for pi, prob in enumerate(problems[:5]):
        r_idx, c_idx = subplot_idx[pi]
        ax = fig.add_subplot(gs[r_idx, c_idx])
        ax.set_facecolor("#fdf8ef")
        prob_rows = [r for r in rows if r.problem == prob]
        ns = sorted(set(r.n for r in prob_rows))
        for meth in methods:
            ys = []
            xs_used = []
            for nv in ns:
                matched = [r for r in prob_rows if r.method == meth and r.n == nv]
                if matched:
                    ys.append(matched[0].med_gap)
                    xs_used.append(nv)
            if xs_used and any(np.isfinite(y) for y in ys):
                ax.plot(xs_used, ys, "o-", color=_method_colour(meth),
                        label=meth, linewidth=1.8, markersize=5, alpha=0.9)
        ax.set_xlabel("n")
        ax.set_ylabel("Median gap")
        ax.set_title(f"Scaling: {prob}")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.legend(fontsize=7)
        ax.set_facecolor("#fdf8ef")

    # ── Panel 8: hit-rate bar ─────────────────────────────────────────────────
    ax_hr = fig.add_subplot(gs[2, 2])
    ax_hr.set_facecolor("#fdf8ef")
    hr_vals_per = {m: [] for m in methods}
    for r in rows:
        if r.method in hr_vals_per:
            hr_vals_per[r.method].append(r.hit_rate)
    meth_labels = [m for m in methods if hr_vals_per[m]]
    hr_means = [float(np.mean(hr_vals_per[m])) for m in meth_labels]
    ax_hr.bar(meth_labels, hr_means, color=[_method_colour(m) for m in meth_labels], alpha=0.85)
    ax_hr.set_ylabel("Mean ground-state hit rate")
    ax_hr.set_title("Oracle hit rate (all problems, n≤20)")
    ax_hr.set_ylim(0, 1)
    ax_hr.tick_params(axis="x", rotation=25)
    ax_hr.set_facecolor("#fdf8ef")

    fig.suptitle("qanneal vs dwave-neal benchmark dashboard", fontsize=14, y=0.99)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fdf8ef")
    plt.close(fig)
    print(f"  [plot] {out_path}")


def plot_traces(instances: List[InstanceResult], out_path: Path) -> None:
    """Energy-convergence traces, one panel per problem family."""
    if not HAS_MPL:
        return

    problems = sorted(set(ir.problem for ir in instances))
    fig, axes = plt.subplots(1, len(problems), figsize=(5 * len(problems), 4.5))
    if len(problems) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#fdf8ef")

    for ax, prob in zip(axes, problems):
        ax.set_facecolor("#fdf8ef")
        prob_insts = [ir for ir in instances if ir.problem == prob]
        # pick a representative instance (smallest n, first instance)
        if not prob_insts:
            continue
        rep = min(prob_insts, key=lambda ir: (ir.n, ir.instance_id))
        plotted = set()
        for sr in rep.methods:
            if sr.trace and len(sr.trace) > 1 and sr.method not in plotted:
                trace = np.array(sr.trace, dtype=float)
                steps = np.arange(len(trace))
                ax.plot(steps, trace, color=_method_colour(sr.method),
                        label=sr.method, linewidth=1.8, alpha=0.85)
                plotted.add(sr.method)
        if np.isfinite(rep.oracle):
            ax.axhline(rep.oracle, color="#cc3d2b", linestyle="--",
                       linewidth=1.2, label="oracle")
        ax.set_xlabel("Annealing step")
        ax.set_ylabel("Energy")
        ax.set_title(f"{prob} (n={rep.n})")
        ax.legend(fontsize=7)

    fig.suptitle("Energy convergence traces", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fdf8ef")
    plt.close(fig)
    print(f"  [plot] {out_path}")


def plot_pareto(rows: List[AggRow], out_path: Path) -> None:
    """Quality vs runtime Pareto scatter per problem family."""
    if not HAS_MPL:
        return

    problems = sorted(set(r.problem for r in rows))
    fig, axes = plt.subplots(1, len(problems), figsize=(5 * len(problems), 4.5))
    if len(problems) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#fdf8ef")

    for ax, prob in zip(axes, problems):
        ax.set_facecolor("#fdf8ef")
        prob_rows = [r for r in rows if r.problem == prob]
        plotted_methods = set()
        for r in prob_rows:
            if r.med_time <= 0 or not np.isfinite(r.med_gap):
                continue
            color = _method_colour(r.method)
            label = r.method if r.method not in plotted_methods else "_"
            ax.scatter(r.med_time, r.med_gap, s=80 + 20 * np.log1p(r.n),
                       color=color, label=label, alpha=0.7, edgecolors="white", linewidths=0.5)
            plotted_methods.add(r.method)
        ax.set_xlabel("Median time (s)")
        ax.set_ylabel("Median gap (lower = better)")
        ax.set_title(f"Pareto: {prob}")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.legend(fontsize=7)
        ax.set_facecolor("#fdf8ef")

    fig.suptitle("Quality–runtime Pareto (bubble size ∝ log n)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fdf8ef")
    plt.close(fig)
    print(f"  [plot] {out_path}")


def plot_boxplots(instances: List[InstanceResult], out_path: Path) -> None:
    """Box plots of normalised gap distribution per method, per problem."""
    if not HAS_MPL:
        return

    from collections import defaultdict
    problems = sorted(set(ir.problem for ir in instances))
    methods_seen = sorted(set(sr.method for ir in instances for sr in ir.methods if sr.ok))
    methods_order = [m for m in ["qanneal-SA", "qanneal-SQA", "qanneal-SQAPT",
                                  "dwave-NEAL", "dwave-PIMC", "random"]
                     if m in methods_seen]

    fig, axes = plt.subplots(1, len(problems), figsize=(5 * len(problems), 5))
    if len(problems) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#fdf8ef")

    for ax, prob in zip(axes, problems):
        ax.set_facecolor("#fdf8ef")
        prob_insts = [ir for ir in instances if ir.problem == prob]
        box_data = []
        labels = []
        for meth in methods_order:
            gaps_all = []
            for ir in prob_insts:
                for sr in ir.methods:
                    if sr.method == meth and sr.ok and np.isfinite(sr.best_energy):
                        g = gap(sr.best_energy, ir.oracle)
                        if np.isfinite(g):
                            gaps_all.append(g)
            if gaps_all:
                box_data.append(gaps_all)
                labels.append(meth)

        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                            medianprops=dict(color="#cc3d2b", linewidth=2))
            for patch, meth in zip(bp["boxes"], labels):
                patch.set_facecolor(_method_colour(meth))
                patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Norm. gap (↓ better)")
        ax.set_title(f"Distribution: {prob}")
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.set_facecolor("#fdf8ef")

    fig.suptitle("Gap distribution (box = IQR, red line = median)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fdf8ef")
    plt.close(fig)
    print(f"  [plot] {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CSV / JSON EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def save_csv(rows: List[AggRow], path: Path) -> None:
    fields = ["problem", "n", "method", "n_inst", "med_gap", "mean_gap",
              "std_gap", "med_time", "mean_time", "hit_rate", "med_tts", "best_gap"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"  [csv] {path}")


def save_json(instances: List[InstanceResult], rows: List[AggRow],
              cfg: Any, path: Path) -> None:
    payload = {
        "config": vars(cfg),
        "summary": [asdict(r) for r in rows],
        "instances": [
            {
                "problem": ir.problem,
                "n": ir.n,
                "instance_id": ir.instance_id,
                "oracle": ir.oracle,
                "methods": [
                    {
                        "method": sr.method,
                        "ok": sr.ok,
                        "best_energy": sr.best_energy,
                        "energies": sr.energies[:20],   # cap to avoid huge JSON
                        "wall_time_s": sr.wall_time_s,
                        "error": sr.error,
                    }
                    for sr in ir.methods
                ],
            }
            for ir in instances
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  [json] {path}")


def print_summary_table(rows: List[AggRow]) -> None:
    problems = sorted(set(r.problem for r in rows))
    for prob in problems:
        print(f"\n{'─'*70}")
        print(f"  {prob}")
        print(f"{'─'*70}")
        prob_rows = [r for r in rows if r.problem == prob]
        ns = sorted(set(r.n for r in prob_rows))
        print(f"  {'method':16s} {'n':>4s} {'med_gap':>10s} {'mean_gap':>10s} "
              f"{'hit_rate':>9s} {'med_tts':>8s} {'time(s)':>8s}")
        print(f"  {'─'*16} {'─'*4} {'─'*10} {'─'*10} {'─'*9} {'─'*8} {'─'*8}")
        for nv in ns:
            n_rows = sorted([r for r in prob_rows if r.n == nv], key=lambda r: r.med_gap)
            for r in n_rows:
                tts_str = f"{r.med_tts:.1f}" if np.isfinite(r.med_tts) else "  ∞"
                print(f"  {r.method:16s} {r.n:4d} {r.med_gap:10.4f} {r.mean_gap:10.4f} "
                      f"{r.hit_rate:9.3f} {tts_str:>8s} {r.med_time:8.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

class Cfg:
    """Flat config container (avoids argparse namespace pollution)."""
    def __init__(self, args):
        self.mode = args.mode
        self.steps = args.steps
        self.sweeps = args.sweeps
        self.worldline = args.worldline
        self.slices = args.slices
        self.replicas = args.replicas
        self.cluster = args.cluster
        self.pt_steps = args.pt_steps
        self.swap_interval = args.swap_interval
        self.seed = args.seed
        self.fair = getattr(args, "fair", False)
        self.budget_updates = getattr(args, "budget_updates", 0.0)
        # Per-solver sweep counts.  In normal mode all are equal to ``sweeps``.
        # In --fair mode, set_fair_sweeps() recomputes them per problem size so
        # every solver receives the same total number of spin-update operations.
        self.sa_sweeps    = args.sweeps
        self.sqa_sweeps   = args.sweeps
        self.sqapt_sweeps = args.sweeps
        self.neal_sweeps  = args.steps * args.sweeps   # NEAL counts full n-spin sweeps
        self.pimc_sweeps  = args.steps * args.sweeps

    def set_fair_sweeps(self, n: int, reads: int) -> None:
        """Recompute per-solver sweep counts so each method consumes ≈ the same
        total number of spin-update operations.

        Budget baseline = SA cost = reads × steps × sweeps × n.

        SA updates per run  = reads × steps × sa_sweeps × n
        SQA updates per run = reads × steps × sqa_sweeps × n × slices × replicas
        SQAPT updates per run = reads × pt_steps × sqapt_sweeps × n × slices × replicas
        NEAL/PIMC updates per run = reads × neal_sweeps × n  (1 full sweep = n flips)
        """
        base = (self.budget_updates if self.budget_updates > 0
                else reads * self.steps * self.sweeps * n)

        # SA is the reference – its sweep count is left unchanged.
        self.sa_sweeps = self.sweeps

        # SQA pays an extra factor of slices × replicas per step.
        slices   = max(1, self.slices)
        replicas = max(1, self.replicas)
        denom_sqa = reads * self.steps * n * slices * replicas
        self.sqa_sweeps = max(1,
            round(base / denom_sqa) - self.worldline - self.cluster)

        # SQAPT: same cost model but uses pt_steps instead of steps.
        denom_sqapt = reads * max(1, self.pt_steps) * n * slices * replicas
        self.sqapt_sweeps = max(1,
            round(base / denom_sqapt) - self.worldline - self.cluster)

        # NEAL / PIMC: each "sweep" = n single-spin-flip attempts → divides by n.
        self.neal_sweeps = max(1, round(base / (reads * n)))
        self.pimc_sweeps = self.neal_sweeps


def main():
    parser = argparse.ArgumentParser(
        description="Crazy full benchmark: qanneal vs dwave-neal across 5 problem families",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Problem scope
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 20, 30],
                        metavar="N", help="Problem sizes to sweep (default: 8 12 16 20 30)")
    parser.add_argument("--instances", type=int, default=10,
                        help="Random instances per (problem, n) cell (default: 10)")
    parser.add_argument("--reads", type=int, default=20,
                        help="Annealing reads per solver call (default: 20)")
    parser.add_argument("--exact-max-n", type=int, default=20,
                        help="Brute-force ground truth up to this n (default: 20)")
    parser.add_argument("--seed", type=int, default=42)

    # Solver parameters
    parser.add_argument("--mode", choices=["fast", "balanced", "accurate"],
                        default="balanced", help="Schedule quality mode (default: balanced)")
    parser.add_argument("--accurate", action="store_true",
                        help="Shorthand for --mode accurate --steps 120 --sweeps 100 --reads 40")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--sweeps", type=int, default=60)
    parser.add_argument("--worldline", type=int, default=4)
    parser.add_argument("--slices", type=int, default=24)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--cluster", type=int, default=1)
    parser.add_argument("--pt-steps", type=int, default=60)
    parser.add_argument("--swap-interval", type=int, default=1)

    # Solver inclusion flags
    parser.add_argument("--no-sqapt", action="store_true", help="Skip SQAPT solver")
    parser.add_argument("--no-neal", action="store_true", help="Skip dwave-NEAL")
    parser.add_argument("--no-pimc", action="store_true", help="Skip dwave-PIMC")

    # Problem family flags
    parser.add_argument("--no-sk", action="store_true", help="Skip SK spin glass")
    parser.add_argument("--no-frustrated", action="store_true", help="Skip frustrated magnet")
    parser.add_argument("--no-maxcut", action="store_true", help="Skip MaxCut")
    parser.add_argument("--no-qubo", action="store_true", help="Skip random QUBO")
    parser.add_argument("--no-numpart", action="store_true", help="Skip number partitioning")

    # Fair compute-budget mode
    parser.add_argument(
        "--fair", action="store_true",
        help=(
            "Equalize total spin-update operations across all solvers for each n. "
            "SA is the baseline; SQA/SQAPT sweeps are reduced to compensate for "
            "their slices×replicas overhead; NEAL/PIMC num_sweeps are set to match. "
            "Makes solution-quality comparisons scientifically rigorous."
        ),
    )
    parser.add_argument(
        "--budget-updates", type=float, default=0.0, metavar="B",
        help=(
            "Override the spin-update budget used by --fair mode. "
            "Default (0) auto-computes it from reads×steps×sweeps×n (SA cost)."
        ),
    )

    # Output
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Directory for output files (default: .)")

    args = parser.parse_args()

    # --accurate shorthand
    if args.accurate:
        args.mode = "accurate"
        args.steps = 120
        args.sweeps = 100
        args.reads = 40

    if not HAS_QANNEAL:
        print("ERROR: qanneal is required. Build and install it first.", file=sys.stderr)
        sys.exit(1)

    cfg = Cfg(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with_sqapt = not args.no_sqapt
    with_neal  = not args.no_neal  and HAS_NEAL
    with_pimc  = not args.no_pimc  and HAS_DWAVE_PIMC
    exact_max_n = args.exact_max_n

    print("=" * 70)
    print("  crazy_benchmark.py")
    print("=" * 70)
    print(f"  sizes={args.sizes}  instances={args.instances}  reads={args.reads}")
    print(f"  mode={args.mode}  steps={args.steps}  sweeps={args.sweeps}")
    print(f"  sqapt={with_sqapt}  neal={with_neal}  pimc={with_pimc}")
    print(f"  exact brute-force for n ≤ {exact_max_n}")
    if args.fair:
        print(f"  [FAIR] compute-budget mode ON  "
              f"(budget_updates={'auto' if args.budget_updates == 0 else args.budget_updates})")
        print("         Each solver receives ≈ the same total spin-update operations per size.")
    if not HAS_NEAL:
        print("  [!] dwave-neal not found – install with: pip install dwave-neal dimod")
    if not HAS_DWAVE_PIMC:
        print("  [!] dwave.samplers not found – install with: pip install dwave-samplers")
    print()

    all_instances: List[InstanceResult] = []

    # ── Build problem list ────────────────────────────────────────────────────
    problem_families = []
    if not args.no_sk:
        problem_families.append("SK-spin-glass")
    if not args.no_frustrated:
        problem_families.append("frustrated-magnet")
    if not args.no_maxcut:
        problem_families.append("MaxCut")
    if not args.no_qubo:
        problem_families.append("random-QUBO")
    if not args.no_numpart:
        problem_families.append("num-partition")

    total_cells = len(problem_families) * len(args.sizes) * args.instances
    done = 0

    for prob in problem_families:
        for n in args.sizes:
            # ── Fair-mode: recompute per-solver sweeps to match SA budget ────
            if args.fair:
                cfg.set_fair_sweeps(n, args.reads)
                base_budget = (args.budget_updates if args.budget_updates > 0
                               else args.reads * cfg.steps * cfg.sweeps * n)
                print(f"  [FAIR n={n:3d}] budget≈{base_budget:.0f} spin-updates  │  "
                      f"SA={cfg.sa_sweeps}sw  SQA={cfg.sqa_sweeps}sw  "
                      f"SQAPT={cfg.sqapt_sweeps}sw  NEAL/PIMC={cfg.neal_sweeps}sw")

            for inst_id in range(args.instances):
                inst_seed = args.seed + 10000 * problem_families.index(prob) + 1000 * n + inst_id
                cfg.seed = inst_seed

                if prob == "SK-spin-glass":
                    h, J = gen_sk_spin_glass(n, inst_seed)
                    ir = run_ising_instance(prob, n, inst_id, h, J, 0.0,
                                            args.reads, cfg, exact_max_n,
                                            with_neal, with_pimc, with_sqapt)

                elif prob == "frustrated-magnet":
                    h, J = gen_frustrated_magnet(n, inst_seed)
                    ir = run_ising_instance(prob, n, inst_id, h, J, 0.0,
                                            args.reads, cfg, exact_max_n,
                                            with_neal, with_pimc, with_sqapt)

                elif prob == "MaxCut":
                    Q = gen_maxcut_qubo(n, inst_seed)
                    ir = run_qubo_instance(prob, n, inst_id, Q,
                                           args.reads, cfg, exact_max_n,
                                           with_neal, with_pimc, with_sqapt)

                elif prob == "random-QUBO":
                    Q = gen_random_qubo(n, inst_seed)
                    ir = run_qubo_instance(prob, n, inst_id, Q,
                                           args.reads, cfg, exact_max_n,
                                           with_neal, with_pimc, with_sqapt)

                elif prob == "num-partition":
                    h, J, c = gen_number_partition_ising(n, inst_seed)
                    ir = run_ising_instance(prob, n, inst_id, h, J, c,
                                            args.reads, cfg, exact_max_n,
                                            with_neal, with_pimc, with_sqapt)

                else:
                    continue

                all_instances.append(ir)
                done += 1

                # Quick progress line
                method_strs = []
                for sr in ir.methods:
                    if sr.ok:
                        g = gap(sr.best_energy, ir.oracle)
                        method_strs.append(f"{sr.method}={g:.4f}")
                    else:
                        method_strs.append(f"{sr.method}=ERR")
                print(f"  [{done:4d}/{total_cells}] {prob:18s} n={n:3d} "
                      f"inst={inst_id:2d}  oracle={ir.oracle:.3f}  "
                      + "  ".join(method_strs))

    if not all_instances:
        print("No instances ran. Check that qanneal is installed and at least one problem family is selected.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\nAggregating...")
    agg_rows = aggregate(all_instances)

    # ── Print table ───────────────────────────────────────────────────────────
    print_summary_table(agg_rows)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving outputs...")
    save_csv(agg_rows, out_dir / "crazy_benchmark_summary.csv")
    save_json(all_instances, agg_rows, args, out_dir / "crazy_benchmark_results.json")

    if HAS_MPL:
        plot_dashboard(agg_rows, all_instances, out_dir / "crazy_benchmark_dashboard.png")
        plot_traces(all_instances, out_dir / "crazy_benchmark_traces.png")
        plot_pareto(agg_rows, out_dir / "crazy_benchmark_pareto.png")
        plot_boxplots(all_instances, out_dir / "crazy_benchmark_boxplots.png")

    # ── Final winner summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WINNER SUMMARY  (method with lowest median gap per problem family)")
    print("=" * 70)
    for prob in sorted(set(r.problem for r in agg_rows)):
        prob_rows = [r for r in agg_rows if r.problem == prob]
        if not prob_rows:
            continue
        winner = min(prob_rows, key=lambda r: r.med_gap)
        runner_up = sorted(prob_rows, key=lambda r: r.med_gap)
        print(f"\n  {prob}")
        for rank, r in enumerate(runner_up[:4], 1):
            marker = "★" if rank == 1 else f"  {rank}."
            hr_str = f"hit={r.hit_rate:.3f}" if r.hit_rate > 0 else ""
            print(f"   {marker} {r.method:16s} med_gap={r.med_gap:.5f}  "
                  f"time={r.med_time:.3f}s  {hr_str}")
    print()


if __name__ == "__main__":
    main()
