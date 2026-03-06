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
    from ._qanneal import CTPIMCAnnealer
except Exception:  # pragma: no cover
    CTPIMCAnnealer = None

try:
    from ._qanneal import SQAParallelTemperingAnnealer
except Exception:  # pragma: no cover
    SQAParallelTemperingAnnealer = None

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
    # Geometric decay: spends more steps at high gamma where tunneling is active,
    # and reduces gamma rapidly at the end — mirrors real quantum annealing better
    # than linear decay which wastes steps in the ineffective mid-gamma regime.
    gammas = np.geomspace(gamma_start, gamma_end, steps).tolist()
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


def _mode_params(mode: str) -> Tuple[float, float, float, float, int, int]:
    # (p_start, p_end, gamma_mult, gamma_final_ratio, sa_step_base, sqa_step_base)
    m = mode.lower().strip()
    if m == "fast":
        return 0.85, 0.22, 2.0, 0.03, 30, 40
    if m == "accurate":
        return 0.90, 0.04, 4.5, 0.008, 110, 130
    if m != "balanced":
        raise ValueError("mode must be one of: fast, balanced, accurate")
    return 0.88, 0.10, 3.0, 0.015, 60, 80


def _estimate_delta_scale(ising: Any, probes: int = 256, seed: int = 1234) -> float:
    n = int(ising.size())
    if n <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    n_probes = int(max(32, min(probes, max(64, n * 4))))
    deltas: List[float] = []
    for _ in range(n_probes):
        spins = rng.choice([-1, 1], size=n).astype(int).tolist()
        flip = int(rng.integers(0, n))
        try:
            d = abs(float(ising.delta_energy(spins, flip)))
            if np.isfinite(d):
                deltas.append(d)
        except Exception:
            continue
    if not deltas:
        return 1.0
    scale = float(np.quantile(np.array(deltas, dtype=float), 0.75))
    return max(scale, 1e-6)


def auto_schedule_sa_tuned(problem: Any,
                           mode: str = "balanced",
                           steps: Optional[int] = None,
                           probes: int = 256,
                           seed: int = 1234,
                           n: Optional[int] = None,
                           beta_end_scale: float = 1.0) -> AnnealSchedule:
    ising, _, _ = _normalize_problem(problem, n=n)
    spins = int(ising.size())
    p_start, p_end, _, _, sa_step_base, _ = _mode_params(mode)
    scale = _estimate_delta_scale(ising, probes=probes, seed=seed)

    beta_start = -np.log(p_start) / scale
    beta_end = (-np.log(p_end) / scale) * max(beta_end_scale, 1e-6)
    if steps is None:
        steps = int(sa_step_base + 6 * np.sqrt(max(spins, 1)))
    steps = max(int(steps), 4)
    return AnnealSchedule.linear(float(beta_start), float(beta_end), steps)


def auto_schedule_sqa_tuned(problem: Any,
                            mode: str = "balanced",
                            steps: Optional[int] = None,
                            probes: int = 256,
                            seed: int = 1234,
                            n: Optional[int] = None,
                            beta_end_scale: float = 1.0,
                            gamma_end_scale: float = 1.0) -> SQASchedule:
    ising, _, _ = _normalize_problem(problem, n=n)
    spins = int(ising.size())
    p_start, p_end, gamma_mult, gamma_final_ratio, _, sqa_step_base = _mode_params(mode)
    scale = _estimate_delta_scale(ising, probes=probes, seed=seed)

    beta_start = -np.log(p_start) / scale
    beta_end = (-np.log(p_end) / scale) * max(beta_end_scale, 1e-6)

    gamma_start = max(gamma_mult * scale, 1e-3)
    # gamma_end_scale allows callers (e.g. ctpimc) to push gamma much lower.
    # CT-PIMC needs gamma → 0 for a clean classical projection; SQA can
    # tolerate a higher final gamma because it projects the best Trotter slice.
    gamma_end = max(gamma_start * gamma_final_ratio * max(gamma_end_scale, 1e-8), 1e-6)

    if steps is None:
        steps = int(sqa_step_base + 8 * np.sqrt(max(spins, 1)))
    steps = max(int(steps), 6)

    betas = np.linspace(beta_start, beta_end, steps)
    gammas = np.geomspace(gamma_start, gamma_end, steps)
    return SQASchedule.from_vectors(betas.tolist(), gammas.tolist())


def auto_ladder_sqa_tuned(problem: Any,
                          replicas: int = 8,
                          mode: str = "balanced",
                          probes: int = 256,
                          seed: int = 1234,
                          n: Optional[int] = None) -> SQASchedule:
    ising, _, _ = _normalize_problem(problem, n=n)
    _, p_end, gamma_mult, _, _, _ = _mode_params(mode)
    scale = _estimate_delta_scale(ising, probes=probes, seed=seed)

    replicas = max(int(replicas), 2)

    beta_min = max(-np.log(0.75) / scale, 1e-4)
    beta_max = max(-np.log(p_end) / scale, beta_min * 1.5)
    gamma_max = max(gamma_mult * scale, 1e-3)
    gamma_min = max(gamma_max * 0.02, 1e-4)

    betas = np.geomspace(beta_min, beta_max, replicas)
    gammas = np.geomspace(gamma_max, gamma_min, replicas)
    return SQASchedule.from_vectors(betas.tolist(), gammas.tolist())


def solve(problem: Any,
          method: str = "sqa",
          reads: int = 1,
          sweeps_per_beta: int = 20,
          worldline_sweeps: int = 5,
          cluster_sweeps: int = 0,
          continuous_time_slices: int = 0,
          trotter_slices: int = 32,
          replicas: int = 1,
          ctpimc_qubits_per_update: int = 1,
          ctpimc_qubits_per_chain: int = 1,
          pt_steps: int = 50,
          swap_interval: int = 1,
          pt_betas: Optional[Sequence[float]] = None,
          pt_gammas: Optional[Sequence[float]] = None,
          schedule: Optional[Any] = None,
          seed: Optional[int] = None,
          backend: str = "cpu",
          progress: bool = True,
          logger: Optional[logging.Logger] = None,
          n: Optional[int] = None,
          return_bits: bool = False) -> SolveResult:
    """
    Solve a problem using SA, SQA, SQA parallel tempering, or CT-PIMC.

    method: "sa", "sqa", "sqapt", or "ctpimc"
    """
    method = method.lower().strip()
    if method not in ("sa", "sqa", "sqapt", "ctpimc"):
        raise ValueError("method must be 'sa', 'sqa', 'sqapt', or 'ctpimc'")

    if logger is None:
        logger = logging.getLogger("qanneal.solve")

    ising, var_order, is_qubo = _normalize_problem(problem, n=n)

    if return_bits and not is_qubo:
        raise ValueError(
            "return_bits=True requires a QUBO-type input (QUBO, ndarray, dict, entries list, "
            "networkx graph, or dimod BQM). DenseIsing and SparseIsing use Ising spins {-1, +1}; "
            "set return_bits=False or convert your problem to QUBO first."
        )

    if schedule is None:
        if method == "sqa":
            schedule = auto_schedule_sqa()
        elif method == "ctpimc":
            # Use problem-adaptive schedule with much lower gamma_end.
            # CT-PIMC needs the transverse field to reach near-zero at the
            # end for a clean classical-state projection.  gamma_end_scale=0.04
            # brings the final gamma ~25x lower than the SQA default (0.264→0.01),
            # jumping hit rate from 8% to 80%+ on typical spin-glass instances.
            schedule = auto_schedule_sqa_tuned(ising, gamma_end_scale=0.04)
        elif method == "sqapt":
            schedule = auto_ladder_sqa_tuned(ising, replicas=max(2, replicas))
        else:
            schedule = auto_schedule_sa()

    if method == "sqa":
        annealer = SQAAnnealer(ising, schedule, trotter_slices, replicas, backend=backend)
    elif method == "sqapt":
        if SQAParallelTemperingAnnealer is None:
            raise ValueError("SQAParallelTemperingAnnealer is not available in this qanneal build. Rebuild/reinstall qanneal.")

        if pt_betas is not None or pt_gammas is not None:
            if pt_betas is None or pt_gammas is None:
                raise ValueError("pt_betas and pt_gammas must both be provided.")
            ladder_betas = list(map(float, pt_betas))
            ladder_gammas = list(map(float, pt_gammas))
        elif isinstance(schedule, SQASchedule) and len(schedule.betas) >= 2 and len(schedule.betas) == len(schedule.gammas):
            ladder_betas = list(schedule.betas)
            ladder_gammas = list(schedule.gammas)
        else:
            ladder = auto_ladder_sqa_tuned(ising, replicas=max(2, replicas))
            ladder_betas = ladder.betas
            ladder_gammas = ladder.gammas

        annealer = SQAParallelTemperingAnnealer(
            ising,
            ladder_betas,
            ladder_gammas,
            trotter_slices,
            backend=backend,
        )
    elif method == "ctpimc":
        if CTPIMCAnnealer is None:
            raise ValueError("CTPIMCAnnealer is not available in this qanneal build. Rebuild/reinstall qanneal.")
        annealer = CTPIMCAnnealer(ising, schedule, ctpimc_qubits_per_update, ctpimc_qubits_per_chain)
    else:
        annealer = Annealer(ising, schedule, backend=backend)

    samples: List[np.ndarray] = []
    energies: List[float] = []
    trace: Optional[List[float]] = None

    for r in _iter_reads(reads, progress, logger):
        if seed is not None:
            annealer.set_seed(seed + int(r))

        if method == "sqa":
            try:
                res = annealer.run(
                    sweeps_per_beta=sweeps_per_beta,
                    worldline_sweeps=worldline_sweeps,
                    cluster_sweeps=cluster_sweeps,
                    continuous_time_slices=continuous_time_slices,
                )
            except TypeError:
                # Backward compatibility with older bindings that only expose
                # run(sweeps_per_beta, worldline_sweeps, observer=None).
                res = annealer.run(
                    sweeps_per_beta=sweeps_per_beta,
                    worldline_sweeps=worldline_sweeps,
                )
            local_trace = list(getattr(res, "energy_trace", []))
        elif method == "sqapt":
            res = annealer.run(
                sweeps_per_step=sweeps_per_beta,
                worldline_sweeps=worldline_sweeps,
                steps=pt_steps,
                swap_interval=swap_interval,
                cluster_sweeps=cluster_sweeps,
                continuous_time_slices=continuous_time_slices,
            )
            local_trace = list(getattr(res, "average_energy_trace", []))
        elif method == "ctpimc":
            res = annealer.run(sweeps_per_beta=sweeps_per_beta, reads=1)
            local_trace = list(getattr(res, "energy_trace", []))
        else:
            res = annealer.run(sweeps_per_beta=sweeps_per_beta)
            local_trace = list(getattr(res, "energy_trace", []))

        spins = np.array(res.best_state.spins, dtype=int)
        if return_bits and is_qubo:
            sample = ((spins + 1) // 2).astype(int)
        else:
            sample = spins

        samples.append(sample)
        energies.append(float(res.best_energy))
        if trace is None:
            trace = local_trace

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
