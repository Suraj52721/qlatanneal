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
    HigherOrderIsing,
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
    from ._qanneal import SQAChiAnnealer
except Exception:  # pragma: no cover
    SQAChiAnnealer = None

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


def j_perp_from_beta_gamma(beta: float, gamma: float, trotter_slices: int = 32) -> float:
    """Convert (beta, gamma) to the Trotter coupling J_perp = 0.5*ln(1/tanh(beta*gamma/M))."""
    import math
    eps = 1e-12
    x = max(beta * gamma / trotter_slices, eps)
    return 0.5 * math.log(1.0 / math.tanh(x))


def j_rms_from_problem(problem: Any, n: Optional[int] = None) -> Optional[float]:
    """
    Best-effort RMS of the off-diagonal couplings J_ij over the *present* (non-zero)
    edges:  j_rms = sqrt(mean_{i<j, J_ij != 0} J_ij^2).

    Used to pick a physically-motivated j_perp_end = 5 * j_rms for the optimal schedule
    (Task 4.2). Returns None when the couplings are not directly inspectable from Python
    (e.g. an opaque DenseIsing/SparseIsing handle) — callers then fall back to the
    beta/gamma-derived j_perp_end from optimal_j_perp_params.
    """
    import math

    def _rms_from_matrix(M: np.ndarray) -> Optional[float]:
        M = np.asarray(M, dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            return None
        iu = np.triu_indices(M.shape[0], k=1)
        off = M[iu]
        # Symmetric QUBO/Ising matrices store J_ij in both triangles; the upper
        # triangle alone is the set of distinct couplings.
        nz = off[off != 0.0]
        if nz.size == 0:
            return None
        return float(math.sqrt(float(np.mean(nz * nz))))

    try:
        # qanneal DenseIsing (and anything exposing a couplings() matrix accessor).
        if hasattr(problem, "couplings") and callable(getattr(problem, "couplings")):
            return _rms_from_matrix(problem.couplings())
        if isinstance(problem, np.ndarray):
            return _rms_from_matrix(problem)
        if nx is not None and isinstance(problem, nx.Graph):
            w = [float(d.get("weight", 0.0)) for _, _, d in problem.edges(data=True)]
            w = [x for x in w if x != 0.0]
            if not w:
                return None
            return float(math.sqrt(sum(x * x for x in w) / len(w)))
        if isinstance(problem, dict):
            vals = [float(v) for (i, j), v in problem.items() if i != j and float(v) != 0.0]
            if not vals:
                return None
            return float(math.sqrt(sum(x * x for x in vals) / len(vals)))
        if isinstance(problem, (list, tuple)) and len(problem) > 0 and isinstance(problem[0], (list, tuple)):
            vals = [float(e[2]) for e in problem if int(e[0]) != int(e[1]) and float(e[2]) != 0.0]
            if not vals:
                return None
            return float(math.sqrt(sum(x * x for x in vals) / len(vals)))
        if dimod is not None and isinstance(problem, dimod.BinaryQuadraticModel):
            vals = [float(b) for b in problem.quadratic.values() if float(b) != 0.0]
            if not vals:
                return None
            return float(math.sqrt(sum(x * x for x in vals) / len(vals)))
    except Exception:
        return None
    return None


def optimal_j_perp_params(problem: Any,
                          mode: str = "balanced",
                          trotter_slices: int = 32,
                          probes: int = 256,
                          seed: int = 1234,
                          n: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Return (beta, j_perp_start, j_perp_end) for the optimal adaptive schedule.

    beta       — fixed inverse temperature for the (phase-2) adaptive J_perp ramp. It must be
                 cold enough that the classical energy term dominates thermal noise, i.e.
                 beta * j_rms >= C. We therefore set

                     beta_end = max(C / j_rms, beta_min)      (C=4, beta_min=1.0)

                 The previous formula beta_end = -log(p_end)/scale scaled INVERSELY with the
                 coupling magnitude, so for strong couplings (large j_rms) it collapsed to
                 beta ~ 0.02 (near-infinite temperature): the classical spins never committed
                 and SQA-opt read out garbage. Tying beta to j_rms fixes that. Falls back to
                 the delta-scale formula only when the couplings can't be inspected.
    j_perp_start — J_perp corresponding to gamma_start at that beta (quantum end).
    j_perp_end   — J_perp corresponding to gamma_end at that beta (classical end).
    """
    ising, _, _ = _normalize_problem(problem, n=n)
    p_start, p_end, gamma_mult, gamma_final_ratio, _, sqa_step_base = _mode_params(mode)
    scale = _estimate_delta_scale(ising, probes=probes, seed=seed)

    # beta must satisfy beta * j_rms >= C so the classical energy term dominates thermal noise.
    _C, _beta_min = 4.0, 1.0
    _jrms = j_rms_from_problem(problem, n=n)
    if _jrms is not None and _jrms > 0.0:
        beta_end = max(_C / _jrms, _beta_min)
    else:
        beta_end = -np.log(p_end) / scale  # fallback when couplings aren't inspectable
    gamma_start = max(gamma_mult * scale, 1e-3)
    gamma_end = max(gamma_start * gamma_final_ratio, 1e-6)

    j_perp_start = j_perp_from_beta_gamma(float(beta_end), float(gamma_start), trotter_slices)
    j_perp_end   = j_perp_from_beta_gamma(float(beta_end), float(gamma_end),   trotter_slices)
    return float(beta_end), float(j_perp_start), float(j_perp_end)


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


def _is_higher_order_keys(keys: Iterable[Any]) -> bool:
    # A mapping is higher-order if any key is a collection whose length != 2
    # (a 3+-body or 1-body interaction). Plain (i, j) pairs stay on the QUBO path.
    for k in keys:
        if isinstance(k, (tuple, list, frozenset, set)):
            if len(k) != 2:
                return True
        else:
            # A bare int key is a 1-body (linear) term -> higher-order dict form.
            return True
    return False


def _hubo_from_dict(problem: dict, n: Optional[int]) -> Tuple[HigherOrderIsing, List[Any]]:
    # Build a native spin HUBO from a mapping {vars: coeff}, where vars is a
    # tuple/list/set of spin indices (or a bare int for a linear term).
    max_idx = -1
    for k in problem:
        idxs = (k,) if isinstance(k, int) else tuple(k)
        for i in idxs:
            max_idx = max(max_idx, int(i))
    size = n if n is not None else max_idx + 1
    ham = HigherOrderIsing(problem, size)
    return ham, list(range(size))


def _hubo_from_binary_polynomial(poly: Any) -> Tuple[HigherOrderIsing, List[Any], bool]:
    # Convert a dimod BinaryPolynomial to a native spin HUBO.
    #   - SPIN vartype: coefficients map straight onto spin terms.
    #   - BINARY vartype: expand x_i = (1 - s_i)/2 into spin terms, so the
    #     reported energy matches the binary polynomial and bits are returned.
    # Returns (ham, var_order, is_binary).
    variables = list(poly.variables)
    index = {v: i for i, v in enumerate(variables)}
    n = len(variables)
    vartype = str(getattr(poly, "vartype", "SPIN"))
    is_binary = "BINARY" in vartype

    spin_terms: dict = {}

    def add(vars_tuple: Tuple[int, ...], coeff: float) -> None:
        key = tuple(sorted(vars_tuple))
        spin_terms[key] = spin_terms.get(key, 0.0) + coeff

    for term, bias in poly.items():
        bias = float(bias)
        idxs = [index[v] for v in term]
        if not is_binary:
            add(tuple(idxs), bias)
            continue
        # x = (1 + s)/2  (library convention: spin +1 -> bit 1, matching the
        # downstream ((spins+1)//2) bit readout and QUBO.to_ising).
        #   prod_k x_{i_k} = (1/2^d) * prod_k (1 + s_{i_k})
        #                  = (1/2^d) * sum_{S subset} prod_{i in S} s_i
        # Every subset contributes with a +1 sign.
        d = len(idxs)
        scale = bias / (2 ** d) if d > 0 else bias
        for mask in range(1 << d):
            subset = [idxs[b] for b in range(d) if (mask >> b) & 1]
            add(tuple(subset), scale)

    ham = HigherOrderIsing(spin_terms, n)
    return ham, variables, is_binary


def _normalize_problem(problem: Any,
                       n: Optional[int] = None) -> Tuple[Any, List[Any], bool]:
    # Returns (ising, var_order, is_qubo)
    if isinstance(problem, (DenseIsing, SparseIsing)):
        return problem, list(range(problem.size())), False

    if isinstance(problem, HigherOrderIsing):
        return problem, list(range(problem.size())), False

    if isinstance(problem, QUBO):
        return problem.to_ising(), list(range(problem.size())), True

    if dimod is not None and isinstance(problem, dimod.BinaryQuadraticModel):
        var_order = list(problem.variables)
        qubo = QUBO(problem)
        return qubo.to_ising(), var_order, True

    if dimod is not None and isinstance(problem, dimod.BinaryPolynomial):
        # Native higher-order path. BINARY polynomials return bits; SPIN ones
        # return spins (is_qubo mirrors "returns bits").
        ham, var_order, is_binary = _hubo_from_binary_polynomial(problem)
        return ham, var_order, is_binary

    if nx is not None and isinstance(problem, nx.Graph):
        qubo, var_order = _qubo_from_graph(problem)
        return qubo.to_ising(), var_order, True

    if isinstance(problem, np.ndarray):
        qubo = QUBO(problem)
        return qubo.to_ising(), list(range(problem.shape[0])), True

    if isinstance(problem, dict):
        # A mapping with any non-pair key is a native spin HUBO; plain (i, j)
        # pair keys stay on the quadratic QUBO path (unchanged behavior).
        if _is_higher_order_keys(problem.keys()):
            ham, var_order = _hubo_from_dict(problem, n)
            return ham, var_order, False
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

    raise ValueError("Unsupported problem type. Provide QUBO, DenseIsing, SparseIsing, "
                     "HigherOrderIsing, dimod BQM/BinaryPolynomial, networkx graph, ndarray, "
                     "dict, or entries list.")


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
          return_bits: bool = False,
          schedule_type: str = "standard",
          optimal_eps_tilde: float = 0.0,
          optimal_alpha: float = 15.0 / 14.0,
          optimal_num_steps: Optional[int] = None,
          optimal_j_perp_start: Optional[float] = None,
          optimal_j_perp_end: Optional[float] = None,
          optimal_beta: Optional[float] = None,
          optimal_calib_probes: int = 12,
          optimal_calib_sweeps: int = 10,
          optimal_beta_ramp_fraction: float = 0.3,
          optimal_debug_csv: Optional[str] = None,
          chi_gamma_start: Optional[float] = None,
          chi_gamma_end: Optional[float] = None,
          chi_scan_points: int = 16,
          chi_scan_sweeps: int = 30,
          chi_scan_burn: int = 10,
          chi_floor_fraction: float = 1e-6,
          chi_driver_A0: float = 0.0) -> SolveResult:
    """
    Solve a problem using SA, SQA, SQA parallel tempering, CT-PIMC, or SQA-chi.

    method: "sa", "sqa", "sqapt", "ctpimc", or "sqa_chi"
        "sqa_chi" is a standalone worldline-QMC method (sibling to "sqa", not a
        schedule_type of it): it measures the susceptibility chi_B of the worldline
        magnetization order parameter m = (1/(n*M)) sum sigma_{i,k} on a pilot scan
        over s = A0/(A0+gamma), then allocates the annealing time budget with weight
        w(s) = max(chi_B(s), floor) via tau(s) = int w / int w (chi_B plays the role
        of 1/Delta(s)^2 in the Roland-Cerf condition), and runs the main anneal along
        the resulting gamma(t) at fixed beta. Update kernel: an exact-detailed-balance
        parity-parallel checkerboard sweep over Trotter slices (slices of the same
        parity never interact directly, so they update fully in parallel), generalized
        to any Backend (dense or sparse Ising). Ignores `schedule` and `schedule_type`
        (must be "standard" if given). See chi_* parameters below.
    schedule_type: "standard" (default) or "optimal"
        "optimal" uses the local adiabaticity ODE from the Roland-Cerf SQA derivation:
        dJ_perp/dt = eps_tilde * chi_B^(-alpha), where chi_B = Var(B) across replicas.
        Supported for methods "sqa" and "sqapt" (including SQAPT+SW via cluster_sweeps > 0).
    optimal_eps_tilde: adiabaticity parameter epsilon-tilde. Default 0.0 requests budget
        calibration (a short pilot pre-pass picks eps_tilde so the schedule traverses the
        full [j_perp_start, j_perp_end] range within optimal_num_steps). Pass a positive
        value to override calibration with a fixed scale (legacy behavior).
    optimal_alpha: exponent alpha = z/(2-eta)+1/2 (default 15/14, 1-D quantum Ising class).
    optimal_num_steps: max adaptive steps (default: same as standard schedule length).
    optimal_j_perp_start / optimal_j_perp_end / optimal_beta: auto-computed if None.
        For sqapt, optimal_j_perp_end auto-defaults to max(j_perp_start, 5*j_rms) when the
        couplings are inspectable, else to the beta/gamma-derived value.
    optimal_calib_probes / optimal_calib_sweeps: pilot grid size and sweeps-per-probe for
        budget calibration (sqapt only).
    optimal_debug_csv: if set (sqapt only), write per-step schedule diagnostics
        (phase,step_index,j_perp,chi_B,delta_j,eps_tilde,alpha,floor_hit) to this path,
        consumable by scripts/plot_schedule_trajectory.py. Also used by "sqa_chi" (see
        chi_* below), which writes (phase,index,s,gamma,j_perp,chi_B,beta) rows instead.
    chi_gamma_start / chi_gamma_end: transverse-field range for the sqa_chi pilot scan
        and schedule (default: max/min gamma of the resolved auto SQA schedule).
    chi_scan_points / chi_scan_sweeps / chi_scan_burn: pilot chi_B scan grid size,
        measurement sweeps, and burn-in sweeps per grid point (fresh worldline per point).
    chi_floor_fraction: floor = max(chi_floor_fraction * max(chi_B), 1e-12) applied to
        chi_B before integrating, so a near-zero-chi_B region still gets nonzero time.
    chi_driver_A0: driver scale A0 in s = A0/(A0+gamma); <= 0 uses chi_gamma_start.
    """
    method = method.lower().strip()
    schedule_type = schedule_type.lower().strip()
    if method not in ("sa", "sqa", "sqapt", "ctpimc", "sqa_chi"):
        raise ValueError("method must be 'sa', 'sqa', 'sqapt', 'ctpimc', or 'sqa_chi'")
    if schedule_type not in ("standard", "optimal"):
        raise ValueError("schedule_type must be 'standard' or 'optimal'")
    if method == "sqa_chi" and schedule_type != "standard":
        raise ValueError("method='sqa_chi' does not support schedule_type overrides; "
                         "leave schedule_type='standard' and use the chi_* parameters.")

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
        if method in ("sqa", "sqa_chi"):
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
    elif method == "sqa_chi":
        if SQAChiAnnealer is None:
            raise ValueError("SQAChiAnnealer is not available in this qanneal build. Rebuild/reinstall qanneal.")
        annealer = SQAChiAnnealer(ising, trotter_slices, replicas, backend=backend)
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

    # --- Pre-compute optimal schedule parameters (done once, outside the reads loop) ---
    _opt_beta: float = 0.0
    _opt_j_perp_start: float = 0.0
    _opt_j_perp_end: float = 0.0
    _opt_num_steps: int = 100
    if schedule_type == "optimal" and method in ("sqa", "sqapt"):
        if method not in ("sqa", "sqapt"):
            raise ValueError("schedule_type='optimal' is only supported for methods 'sqa' and 'sqapt'.")
        _auto_beta, _auto_jps, _auto_jpe = optimal_j_perp_params(
            ising, trotter_slices=trotter_slices)
        _opt_beta = float(optimal_beta) if optimal_beta is not None else _auto_beta
        _opt_j_perp_start = float(optimal_j_perp_start) if optimal_j_perp_start is not None else _auto_jps

        if optimal_j_perp_end is not None:
            _opt_j_perp_end = float(optimal_j_perp_end)
        else:
            # Prefer a physically-motivated target tied to the coupling scale: the quantum
            # critical J_perp ~ j_rms, so 5*j_rms sits well past the transition (Task 4.2).
            # Fall back to the beta/gamma-derived value when couplings aren't inspectable.
            _jrms = j_rms_from_problem(problem, n=n)
            if _jrms is not None and _jrms > 0.0:
                _opt_j_perp_end = max(_opt_j_perp_start, 5.0 * _jrms)
                if _opt_j_perp_end <= _opt_j_perp_start:
                    logger.warning(
                        "optimal schedule: 5*j_rms=%.4g <= j_perp_start=%.4g; the system "
                        "starts past criticality and the adaptive schedule has little range "
                        "to traverse. Consider a problem with larger j_rms.",
                        5.0 * _jrms, _opt_j_perp_start)
            else:
                _opt_j_perp_end = _auto_jpe
        _opt_num_steps = int(optimal_num_steps) if optimal_num_steps is not None else max(
            100, int(getattr(schedule, 'betas', [None] * 100).__len__()) if schedule is not None else 100
        )

    # --- Pre-compute sqa_chi parameters (done once, outside the reads loop) ---
    _chi_beta: float = 0.0
    _chi_gamma_start: float = 0.0
    _chi_gamma_end: float = 0.0
    _chi_num_steps: int = 100
    if method == "sqa_chi":
        _auto_beta, _, _ = optimal_j_perp_params(ising, trotter_slices=trotter_slices)
        _chi_beta = float(optimal_beta) if optimal_beta is not None else _auto_beta
        _sched_gammas = list(getattr(schedule, "gammas", []) or [])
        if chi_gamma_start is not None:
            _chi_gamma_start = float(chi_gamma_start)
        elif _sched_gammas:
            _chi_gamma_start = float(max(_sched_gammas))
        else:
            _chi_gamma_start = 5.0
        if chi_gamma_end is not None:
            _chi_gamma_end = float(chi_gamma_end)
        elif _sched_gammas:
            _chi_gamma_end = float(min(_sched_gammas))
        else:
            _chi_gamma_end = 0.01
        if not (_chi_gamma_start > _chi_gamma_end > 0.0):
            raise ValueError(
                "sqa_chi requires gamma_start > gamma_end > 0 "
                f"(got {_chi_gamma_start} and {_chi_gamma_end})."
            )
        _chi_num_steps = int(optimal_num_steps) if optimal_num_steps is not None else max(
            100, len(_sched_gammas) if _sched_gammas else 100
        )

    samples: List[np.ndarray] = []
    energies: List[float] = []
    trace: Optional[List[float]] = None

    for r in _iter_reads(reads, progress, logger):
        if seed is not None:
            annealer.set_seed(seed + int(r))

        if schedule_type == "optimal" and method == "sqa":
            # eps_tilde <= 0 (the default) triggers budget calibration inside SQA run_optimal
            # (ported from SQAPT); a positive value keeps the legacy online update.
            res = annealer.run_optimal(
                beta=_opt_beta,
                j_perp_start=_opt_j_perp_start,
                j_perp_end=_opt_j_perp_end,
                eps_tilde=optimal_eps_tilde,
                alpha=optimal_alpha,
                num_steps=_opt_num_steps,
                sweeps_per_step=sweeps_per_beta,
                worldline_sweeps=worldline_sweeps,
                cluster_sweeps=cluster_sweeps,
                calib_probes=int(optimal_calib_probes),
                calib_sweeps=int(optimal_calib_sweeps),
                debug_csv_path=(optimal_debug_csv or ""),
                beta_ramp_fraction=float(optimal_beta_ramp_fraction),
            )
            local_trace = list(getattr(res, "energy_trace", []))
        elif method == "sqa_chi":
            res = annealer.run_chi(
                beta=_chi_beta,
                gamma_start=_chi_gamma_start,
                gamma_end=_chi_gamma_end,
                num_steps=_chi_num_steps,
                sweeps_per_step=sweeps_per_beta,
                scan_points=int(chi_scan_points),
                scan_sweeps=int(chi_scan_sweeps),
                scan_burn=int(chi_scan_burn),
                chi_floor_fraction=float(chi_floor_fraction),
                driver_A0=float(chi_driver_A0),
                beta_ramp_fraction=float(optimal_beta_ramp_fraction),
                debug_csv_path=(optimal_debug_csv or ""),
            )
            local_trace = list(getattr(res, "energy_trace", []))
        elif schedule_type == "optimal" and method == "sqapt":
            # eps_tilde <= 0 (the default) triggers budget calibration inside run_optimal.
            res = annealer.run_optimal(
                num_steps=_opt_num_steps,
                sweeps_per_step=sweeps_per_beta,
                worldline_sweeps=worldline_sweeps,
                eps_tilde=optimal_eps_tilde,
                alpha=optimal_alpha,
                j_perp_end=_opt_j_perp_end,
                cluster_sweeps=cluster_sweeps,
                swap_interval=swap_interval,
                continuous_time_slices=continuous_time_slices,
                calib_probes=int(optimal_calib_probes),
                calib_sweeps=int(optimal_calib_sweeps),
                debug_csv_path=(optimal_debug_csv or ""),
            )
            local_trace = list(getattr(res, "average_energy_trace", []))
        elif method == "sqa":
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
