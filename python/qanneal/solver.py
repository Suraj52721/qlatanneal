from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._qanneal import (
    AnnealSchedule,
    SQASchedule,
    Annealer,
    SQAAnnealer,
    DenseIsing,
    SparseIsing,
    QUBO,
)

try:
    import dimod  # type: ignore
except Exception:  # pragma: no cover
    dimod = None

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class SolveResult:
    method: str
    samples: List[np.ndarray]
    energies: List[float]
    best_sample: np.ndarray
    best_energy: float
    trace: Optional[List[float]]
    var_order: List[Any]
    return_bits: bool


def auto_schedule_sa(steps: int = 50,
                     beta_start: float = 0.1,
                     beta_end: float = 4.0) -> AnnealSchedule:
    return AnnealSchedule.linear(beta_start, beta_end, steps)


def auto_schedule_sqa(steps: int = 50,
                      beta_start: float = 0.1,
                      beta_end: float = 4.0,
                      gamma_start: float = 5.0,
                      gamma_end: float = 0.01) -> SQASchedule:
    betas = np.linspace(beta_start, beta_end, steps).tolist()
    gammas = np.linspace(gamma_start, gamma_end, steps).tolist()
    return SQASchedule.from_vectors(betas, gammas)


def _guess_n_from_entries(entries: Iterable[Tuple[int, int, float]]) -> int:
    max_idx = -1
    for i, j, _ in entries:
        max_idx = max(max_idx, i, j)
    return max_idx + 1


def _qubo_from_graph(graph: Any) -> Tuple[QUBO, List[Any]]:
    if nx is None:
        raise ValueError("networkx not available; install `networkx` to use graph inputs.")

    nodes = list(graph.nodes())
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    Q = np.zeros((n, n), dtype=float)

    for node, data in graph.nodes(data=True):
        bias = float(data.get("bias", 0.0))
        Q[index[node], index[node]] += bias

    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 0.0))
        i = index[u]
        j = index[v]
        Q[i, j] += w
        Q[j, i] += w

    return QUBO(Q), nodes


def _normalize_problem(problem: Any,
                       n: Optional[int] = None) -> Tuple[Any, List[Any], bool]:
    # Returns (ising, var_order, is_qubo)
    if isinstance(problem, (DenseIsing, SparseIsing)):
        return problem, list(range(problem.size())), False

    if isinstance(problem, QUBO):
        return problem.to_ising(), list(range(problem.size())), True

    if dimod is not None and isinstance(problem, dimod.BinaryQuadraticModel):
        var_order = list(problem.variables)
        qubo = QUBO(problem)
        return qubo.to_ising(), var_order, True

    if nx is not None and isinstance(problem, nx.Graph):
        qubo, var_order = _qubo_from_graph(problem)
        return qubo.to_ising(), var_order, True

    if isinstance(problem, np.ndarray):
        qubo = QUBO(problem)
        return qubo.to_ising(), list(range(problem.shape[0])), True

    if isinstance(problem, dict):
        if n is None:
            n = _guess_n_from_entries([(i, j, float(v)) for (i, j), v in problem.items()])
        qubo = QUBO(problem, n)
        return qubo.to_ising(), list(range(n)), True

    if isinstance(problem, (list, tuple)) and len(problem) > 0 and isinstance(problem[0], (list, tuple)):
        entries = [(int(e[0]), int(e[1]), float(e[2])) for e in problem]
        if n is None:
            n = _guess_n_from_entries(entries)
        qubo = QUBO(entries, n)
        return qubo.to_ising(), list(range(n)), True

    raise ValueError("Unsupported problem type. Provide QUBO, DenseIsing, SparseIsing, dimod BQM, "
                     "networkx graph, ndarray, dict, or entries list.")


def _iter_reads(total: int, enabled: bool, logger: logging.Logger) -> Iterable[int]:
    if total <= 1:
        return range(total)
    if enabled and tqdm is not None:
        return tqdm(range(total), total=total, desc="qanneal")
    if enabled:
        logger.info("Running %d reads...", total)
    return range(total)


def solve(problem: Any,
          method: str = "sqa",
          reads: int = 1,
          sweeps_per_beta: int = 20,
          worldline_sweeps: int = 5,
          trotter_slices: int = 32,
          replicas: int = 1,
          schedule: Optional[Any] = None,
          seed: Optional[int] = None,
          backend: str = "cpu",
          progress: bool = True,
          logger: Optional[logging.Logger] = None,
          n: Optional[int] = None,
          return_bits: bool = False) -> SolveResult:
    """
    Solve a problem using SA or SQA.

    problem: QUBO / DenseIsing / SparseIsing / dimod BQM / networkx Graph /
             ndarray / dict / entries list.
    method: "sa" or "sqa"
    return_bits: if True, convert spins to bits {0,1}.
    """
    method = method.lower().strip()
    if method not in ("sa", "sqa"):
        raise ValueError("method must be 'sa' or 'sqa'")

    if logger is None:
        logger = logging.getLogger("qanneal.solve")

    ising, var_order, is_qubo = _normalize_problem(problem, n=n)

    if schedule is None:
        schedule = auto_schedule_sqa() if method == "sqa" else auto_schedule_sa()

    if method == "sqa":
        annealer = SQAAnnealer(ising, schedule, trotter_slices, replicas, backend=backend)
    else:
        annealer = Annealer(ising, schedule, backend=backend)

    samples: List[np.ndarray] = []
    energies: List[float] = []
    trace: Optional[List[float]] = None

    for r in _iter_reads(reads, progress, logger):
        if seed is not None:
            annealer.set_seed(seed + int(r))
        if method == "sqa":
            res = annealer.run(sweeps_per_beta=sweeps_per_beta, worldline_sweeps=worldline_sweeps)
        else:
            res = annealer.run(sweeps_per_beta=sweeps_per_beta)

        spins = np.array(res.best_state.spins, dtype=int)
        if return_bits and is_qubo:
            sample = ((spins + 1) // 2).astype(int)
        else:
            sample = spins

        samples.append(sample)
        energies.append(float(res.best_energy))
        if trace is None:
            trace = list(res.energy_trace)

    best_idx = int(np.argmin(energies))
    return SolveResult(
        method=method,
        samples=samples,
        energies=energies,
        best_sample=samples[best_idx],
        best_energy=energies[best_idx],
        trace=trace,
        var_order=var_order,
        return_bits=return_bits,
    )
