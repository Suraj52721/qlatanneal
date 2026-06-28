#!/usr/bin/env python3
"""
dwave_comparative_benchmark.py  —  qlatannealv4
═══════════════════════════════════════════════════════════════════════════════
Full parallel benchmark: qlatannealv4 (SA, SQA, SQAPT, CT-PIMC) vs
D-Wave SDK (dwave-NEAL, dwave-PIMC) on random Ising spin-glass instances.

qlatannealv4 features used
--------------------------
  • Problem-adaptive auto-tuned schedules (auto_schedule_sa_tuned,
    auto_schedule_sqa_tuned, auto_ladder_sqa_tuned)
  • Geometric gamma decay in SQA (vs linear in v1)
  • SQAParallelTemperingAnnealer (SQAPT) — SQA + parallel-tempering swap
  • CTPIMCAnnealer — continuous-time path integral Monte Carlo

Metrics
-------
  • Accuracy       – energy gap to oracle  (brute-force ≤ 20 spins,
                     else best-known across all methods)
  • Hit rate        – fraction of runs reaching oracle within ε = 1e-4
  • Wall time       – total seconds for all reads
  • Power           – hit_rate / mean_time  (higher = better)
  • Success rate    – gap ≤ SUCCESS_TOL × |oracle|
  • TTS @ 99%       – time-to-solution for 99 % probability

Parallelism
-----------
All (method × size × seed) tasks dispatched to ProcessPoolExecutor.

Usage
-----
    python dwave_comparative_benchmark.py [--workers N] [--no-pimc] [--quick]

Outputs (saved to this directory)
----------------------------------
    dwave_comparative_results_v4.json
    benchmark_v4_accuracy.png
    benchmark_v4_walltime.png
    benchmark_v4_power.png
    benchmark_v4_success_rate.png
    benchmark_v4_tts.png
    benchmark_v4_pareto.png
    benchmark_v4_boxplot.png
    benchmark_v4_radar.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  v4 IMPORT PATH  (workers inherit this via task dict)
# ─────────────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_V4_ROOT     = os.path.normpath(os.path.join(_HERE, "..", ".."))
V4_PY_PATH   = os.path.join(_V4_ROOT, "python")   # …/qlatannealv4/python

# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PROBLEM_SIZES = [20, 30, 40, 50, 60, 75, 90]
N_SEEDS       = 15         # independent random instances per size
SEED_BASE     = 42
DENSITY       = 0.45       # edge density for spin-glass instances
BRUTE_MAX_N   = 20         # use brute force oracle for n ≤ this

# qlatanneal SA (adaptive schedule, no trotter slices)
SA_READS      = 24
SA_STEPS      = 70         # passed to auto_schedule_sa_tuned
SA_SWEEPS     = 60

# qlatanneal SQA (adaptive SQASchedule, Trotter path integral)
SQA_READS     = 24
SQA_STEPS     = 70
SQA_SWEEPS    = 40
SQA_SLICES    = 24
SQA_REPLICAS  = 2
SQA_WORLDLINE = 4
SQA_CLUSTER   = 1

# qlatanneal SQAPT (SQA + Parallel Tempering, adaptive ladder)
SQAPT_READS     = 24
SQAPT_STEPS     = 60       # pt_steps
SQAPT_SWEEPS    = 40
SQAPT_SLICES    = 24
SQAPT_REPLICAS  = 4        # number of PT replicas / ladder rungs
SQAPT_WORLDLINE = 4
SQAPT_CLUSTER   = 1
SQAPT_SWAP      = 1

# qlatanneal CT-PIMC (continuous-time path integral MC)
CTPIMC_READS    = 16
CTPIMC_STEPS    = 70
CTPIMC_SWEEPS   = 60

# D-Wave NEAL (pure-Python SA reference)
NEAL_READS    = 24
NEAL_SWEEPS   = 4200       # ≈ SA_STEPS × SA_SWEEPS (fair budget)

# D-Wave PIMC (path-integral MC)
PIMC_READS    = 8
PIMC_SWEEPS   = 600

SUCCESS_TOL   = 0.02       # gap ≤ 2% of |oracle| counts as success
HIT_EPS       = 1e-4       # exact-hit threshold

OUT_DIR = _HERE

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR / MARKER SCHEME
# ─────────────────────────────────────────────────────────────────────────────
METHOD_COLOR = {
    "qanneal-SA":     "#2563EB",
    "qanneal-SQA":    "#16A34A",
    "qanneal-SQAPT":  "#9333EA",
    "qanneal-CTPIMC": "#EA580C",
    "dwave-NEAL":     "#DC2626",
    "dwave-PIMC":     "#0891B2",
}
METHOD_MARKER = {
    "qanneal-SA":     "o",
    "qanneal-SQA":    "s",
    "qanneal-SQAPT":  "^",
    "qanneal-CTPIMC": "D",
    "dwave-NEAL":     "v",
    "dwave-PIMC":     "P",
}

# ─────────────────────────────────────────────────────────────────────────────
#  PROBLEM HELPERS  (module-level — picklable)
# ─────────────────────────────────────────────────────────────────────────────

def _make_spin_glass(n: int, seed: int, density: float = DENSITY
                    ) -> Tuple[List[float], List[float]]:
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, 0.3, n)
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                v = float(rng.choice([-1., 1.])) * float(rng.uniform(0.5, 1.5))
                J[i, j] = J[j, i] = v
    return h.tolist(), J.flatten().tolist()


def _brute_force_energy(h_list: List[float], J_flat: List[float]) -> Optional[float]:
    n = len(h_list)
    if n > BRUTE_MAX_N:
        return None
    h = np.array(h_list, dtype=float)
    J = np.array(J_flat, dtype=float).reshape(n, n)
    best = float("inf")
    for mask in range(1 << n):
        s = np.array([(1 if (mask >> i) & 1 else -1) for i in range(n)], dtype=float)
        e = float(np.dot(h, s) + 0.5 * (s @ J @ s))
        if e < best:
            best = e
    return best


def _make_bqm(h_list: List[float], J_flat: List[float]):
    import dimod  # type: ignore
    n = len(h_list)
    J = np.array(J_flat, dtype=float).reshape(n, n)
    bqm = dimod.BinaryQuadraticModel(vartype="SPIN")
    for i in range(n):
        bqm.set_linear(i, float(h_list[i]))
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                bqm.set_quadratic(i, j, float(J[i, j]))
    return bqm


def _inject_v4(v4_py_path: str) -> None:
    if v4_py_path not in sys.path:
        sys.path.insert(0, v4_py_path)


# ─────────────────────────────────────────────────────────────────────────────
#  INDIVIDUAL WORKER FUNCTIONS  (all module-level for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _run_qanneal_sa(h_list, J_flat, seed, p, v4_py_path):
    _inject_v4(v4_py_path)
    from qanneal import DenseIsing, solve, auto_schedule_sa_tuned  # type: ignore
    import numpy as _np
    n = len(h_list)
    h = _np.array(h_list, dtype=float)
    J = _np.array(J_flat, dtype=float).reshape(n, n)
    ising = DenseIsing(h, J, 0.0)
    sched = auto_schedule_sa_tuned(ising, mode="balanced", steps=p["steps"], seed=seed)
    t0 = time.perf_counter()
    r = solve(ising, method="sa", reads=p["reads"], sweeps_per_beta=p["sweeps"],
              schedule=sched, seed=seed, progress=False)
    return {"energy": float(r.best_energy), "wall_time": time.perf_counter() - t0}


def _run_qanneal_sqa(h_list, J_flat, seed, p, v4_py_path):
    _inject_v4(v4_py_path)
    from qanneal import DenseIsing, solve, auto_schedule_sqa_tuned  # type: ignore
    import numpy as _np
    n = len(h_list)
    h = _np.array(h_list, dtype=float)
    J = _np.array(J_flat, dtype=float).reshape(n, n)
    ising = DenseIsing(h, J, 0.0)
    sched = auto_schedule_sqa_tuned(ising, mode="balanced", steps=p["steps"], seed=seed)
    t0 = time.perf_counter()
    r = solve(ising, method="sqa", reads=p["reads"], sweeps_per_beta=p["sweeps"],
              worldline_sweeps=p["worldline"], cluster_sweeps=p["cluster"],
              trotter_slices=p["slices"], replicas=p["replicas"],
              schedule=sched, seed=seed, progress=False)
    return {"energy": float(r.best_energy), "wall_time": time.perf_counter() - t0}


def _run_qanneal_sqapt(h_list, J_flat, seed, p, v4_py_path):
    _inject_v4(v4_py_path)
    from qanneal import DenseIsing, solve, auto_ladder_sqa_tuned  # type: ignore
    import numpy as _np
    n = len(h_list)
    h = _np.array(h_list, dtype=float)
    J = _np.array(J_flat, dtype=float).reshape(n, n)
    ising = DenseIsing(h, J, 0.0)
    ladder = auto_ladder_sqa_tuned(ising, replicas=p["replicas"], mode="balanced", seed=seed)
    t0 = time.perf_counter()
    r = solve(ising, method="sqapt", reads=p["reads"], sweeps_per_beta=p["sweeps"],
              worldline_sweeps=p["worldline"], cluster_sweeps=p["cluster"],
              trotter_slices=p["slices"], replicas=p["replicas"],
              pt_steps=p["pt_steps"], swap_interval=p["swap_interval"],
              schedule=ladder, seed=seed, progress=False)
    return {"energy": float(r.best_energy), "wall_time": time.perf_counter() - t0}


def _run_qanneal_ctpimc(h_list, J_flat, seed, p, v4_py_path):
    _inject_v4(v4_py_path)
    from qanneal import DenseIsing, solve, auto_schedule_sqa_tuned  # type: ignore
    import numpy as _np
    n = len(h_list)
    h = _np.array(h_list, dtype=float)
    J = _np.array(J_flat, dtype=float).reshape(n, n)
    ising = DenseIsing(h, J, 0.0)
    sched = auto_schedule_sqa_tuned(ising, mode="balanced", steps=p["steps"],
                                    seed=seed, gamma_end_scale=0.04)
    t0 = time.perf_counter()
    r = solve(ising, method="ctpimc", reads=p["reads"], sweeps_per_beta=p["sweeps"],
              schedule=sched, seed=seed, progress=False)
    return {"energy": float(r.best_energy), "wall_time": time.perf_counter() - t0}


def _run_dwave_neal(h_list, J_flat, seed, p):
    import neal  # type: ignore
    bqm = _make_bqm(h_list, J_flat)
    sampler = neal.SimulatedAnnealingSampler()
    t0 = time.perf_counter()
    try:
        ss = sampler.sample(bqm, num_reads=p["reads"], num_sweeps=p["sweeps"], seed=seed)
    except TypeError:
        ss = sampler.sample(bqm, num_reads=p["reads"], num_sweeps=p["sweeps"])
    return {"energy": float(ss.first.energy), "wall_time": time.perf_counter() - t0}


def _run_dwave_pimc(h_list, J_flat, p):
    from dwave.samplers import PathIntegralAnnealingSampler  # type: ignore
    bqm = _make_bqm(h_list, J_flat)
    sampler = PathIntegralAnnealingSampler()
    t0 = time.perf_counter()
    ss = sampler.sample(bqm, num_reads=p["reads"], num_sweeps=p["sweeps"])
    return {"energy": float(ss.first.energy), "wall_time": time.perf_counter() - t0}


# ─────────────────────────────────────────────────────────────────────────────
#  TOP-LEVEL PICKLABLE WORKER DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def _worker(task: Dict[str, Any]) -> Dict[str, Any]:
    method      = task["method"]
    h_list      = task["h"]
    J_flat      = task["J"]
    seed        = task["seed"]
    p           = task["params"]
    n           = task["n"]
    oracle      = task["oracle"]          # may be None
    v4_py_path  = task["v4_py_path"]

    try:
        dispatch = {
            "qanneal-SA":     lambda: _run_qanneal_sa(h_list, J_flat, seed, p, v4_py_path),
            "qanneal-SQA":    lambda: _run_qanneal_sqa(h_list, J_flat, seed, p, v4_py_path),
            "qanneal-SQAPT":  lambda: _run_qanneal_sqapt(h_list, J_flat, seed, p, v4_py_path),
            "qanneal-CTPIMC": lambda: _run_qanneal_ctpimc(h_list, J_flat, seed, p, v4_py_path),
            "dwave-NEAL":     lambda: _run_dwave_neal(h_list, J_flat, seed, p),
            "dwave-PIMC":     lambda: _run_dwave_pimc(h_list, J_flat, p),
        }
        result = dispatch[method]()
    except Exception as exc:
        result = {"energy": float("inf"), "wall_time": 0.0, "error": str(exc)}

    result["method"] = method
    result["n"]      = n
    result["seed"]   = seed
    result["oracle"] = oracle
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  TASK BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_tasks(sizes: List[int], n_seeds: int,
                 include_pimc: bool) -> List[Dict[str, Any]]:
    tasks: List[Dict] = []
    for n in sizes:
        for s_idx in range(n_seeds):
            seed     = SEED_BASE + s_idx * 1009 + n * 17
            h_list, J_flat = _make_spin_glass(n, seed)
            oracle   = _brute_force_energy(h_list, J_flat)   # None if n > BRUTE_MAX_N

            method_params: Dict[str, Dict] = {
                "qanneal-SA": dict(
                    reads=SA_READS, steps=SA_STEPS, sweeps=SA_SWEEPS,
                ),
                "qanneal-SQA": dict(
                    reads=SQA_READS, steps=SQA_STEPS, sweeps=SQA_SWEEPS,
                    slices=SQA_SLICES, replicas=SQA_REPLICAS,
                    worldline=SQA_WORLDLINE, cluster=SQA_CLUSTER,
                ),
                "qanneal-SQAPT": dict(
                    reads=SQAPT_READS, steps=SQAPT_STEPS, sweeps=SQAPT_SWEEPS,
                    slices=SQAPT_SLICES, replicas=SQAPT_REPLICAS,
                    worldline=SQAPT_WORLDLINE, cluster=SQAPT_CLUSTER,
                    pt_steps=SQAPT_STEPS, swap_interval=SQAPT_SWAP,
                ),
                "qanneal-CTPIMC": dict(
                    reads=CTPIMC_READS, steps=CTPIMC_STEPS, sweeps=CTPIMC_SWEEPS,
                ),
                "dwave-NEAL": dict(reads=NEAL_READS, sweeps=NEAL_SWEEPS),
                "dwave-PIMC": dict(reads=PIMC_READS, sweeps=PIMC_SWEEPS),
            }

            for method, params in method_params.items():
                if method == "dwave-PIMC" and not include_pimc:
                    continue
                tasks.append({
                    "method":     method,
                    "h":          h_list,
                    "J":          J_flat,
                    "seed":       seed,
                    "n":          n,
                    "oracle":     oracle,
                    "params":     params,
                    "v4_py_path": V4_PY_PATH,
                })
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
#  PARALLEL RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _run_parallel(tasks: List[Dict], max_workers: Optional[int]) -> List[Dict]:
    total   = len(tasks)
    results = []
    workers = max_workers or os.cpu_count() or 4
    print(f"  Dispatching {total} tasks → {workers} workers …", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done  += 1
            res    = fut.result()
            results.append(res)
            energy = res.get("energy", float("inf"))
            err    = f"  ERROR: {res.get('error','')}" if "error" in res else ""
            print(
                f"  [{done:4d}/{total}] {res['method']:18s} "
                f"n={res['n']:3d}  E={energy:10.4f}{err}",
                flush=True,
            )
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stats(raw: List[Dict]) -> Dict[Tuple, Dict]:
    from collections import defaultdict

    # Best energy found by ANY method for each (n, seed)
    best_known: Dict[Tuple, float] = defaultdict(lambda: float("inf"))
    for r in raw:
        if "error" not in r:
            key = (r["n"], r["seed"])
            best_known[key] = min(best_known[key], r["energy"])

    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in raw:
        if "error" not in r:
            groups[(r["method"], r["n"])].append(r)

    stats: Dict[Tuple, Dict] = {}
    for (method, n), runs in sorted(groups.items()):
        energies = [r["energy"] for r in runs]
        times    = [r["wall_time"] for r in runs]

        # Oracle for each run: brute-force if available, else best-known
        gaps, hits, successes = [], [], []
        for r in runs:
            oracle = r.get("oracle")
            if oracle is None:
                oracle = best_known[(r["n"], r["seed"])]
            gap = float(r["energy"] - oracle)
            gaps.append(gap)
            hits.append(1.0 if gap <= HIT_EPS else 0.0)
            tol_val = SUCCESS_TOL * max(abs(oracle), 1.0)
            successes.append(1 if gap <= tol_val else 0)

        mean_gap   = float(np.mean(gaps))
        std_gap    = float(np.std(gaps))
        med_gap    = float(np.median(gaps))
        mean_time  = float(np.mean(times))
        std_time   = float(np.std(times))
        hit_rate   = float(np.mean(hits))
        sr         = float(np.mean(successes))

        # Power: hit_rate / mean_time
        power = hit_rate / max(mean_time, 1e-9)

        # TTS @ 99%
        if 0.0 < sr < 1.0:
            tts = mean_time * math.log(1.0 - 0.99) / math.log(max(1.0 - sr, 1e-12))
        elif sr >= 1.0:
            tts = mean_time
        else:
            tts = float("inf")

        stats[(method, n)] = {
            "method":       method,
            "n":            n,
            "mean_gap":     mean_gap,
            "std_gap":      std_gap,
            "median_gap":   med_gap,
            "mean_time":    mean_time,
            "std_time":     std_time,
            "hit_rate":     hit_rate,
            "success_rate": sr,
            "power":        power,
            "tts_99":       tts,
            "n_runs":       len(runs),
        }

    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bar_chart(stats, methods, sizes, key_mean, key_std,
               ylabel, title, fname, out_dir, log_y=False):
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(sizes))
    w = 0.80 / max(len(methods), 1)
    for i, m in enumerate(methods):
        ys   = [stats.get((m, s), {}).get(key_mean, np.nan) for s in sizes]
        errs = [stats.get((m, s), {}).get(key_std,  np.nan) for s in sizes]
        off  = (i - len(methods) / 2 + 0.5) * w
        ax.bar(x + off, ys, w * 0.92, label=m,
               color=METHOD_COLOR.get(m, "gray"), alpha=0.85)
        ax.errorbar(x + off, ys, yerr=errs, fmt="none",
                    ecolor="black", capsize=3, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={s}" for s in sizes])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if log_y:
        valid = [v for m in methods for s in sizes
                 for v in [stats.get((m,s),{}).get(key_mean, np.nan)]
                 if np.isfinite(v) and v > 0]
        if valid:
            ax.set_yscale("log")
    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _line_chart(stats, methods, sizes, key, ylabel, title, fname, out_dir,
                log_y=False, scale=1.0):
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=(10, 5))
    for m in methods:
        ys = [stats.get((m, s), {}).get(key, np.nan) * scale for s in sizes]
        ax.plot(sizes, ys,
                color=METHOD_COLOR.get(m, "gray"),
                marker=METHOD_MARKER.get(m, "o"),
                label=m, linewidth=2, markersize=7)

    ax.set_xlabel("Problem size  n  (spins)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_pareto(raw, methods, sizes, out_dir):
    import matplotlib.pyplot as plt  # type: ignore

    ncols = len(sizes)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, sz in zip(axes, sizes):
        for m in methods:
            pts = []
            for r in raw:
                if r.get("method") == m and r.get("n") == sz and "error" not in r:
                    oracle = r.get("oracle")
                    bk = oracle if oracle is not None else float("inf")
                    gap = r["energy"] - bk if math.isfinite(bk) else 0.0
                    pts.append((r["wall_time"], max(gap, 0.0)))
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, alpha=0.55, s=20,
                           color=METHOD_COLOR.get(m, "gray"),
                           marker=METHOD_MARKER.get(m, "o"), label=m)
        ax.set_title(f"n = {sz}", fontsize=10)
        ax.set_xlabel("Wall time (s)", fontsize=9)
        ax.set_ylabel("Energy gap", fontsize=9)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=min(len(methods), 3), fontsize=8,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Pareto: Energy Gap vs Wall Time  (all seeds)", y=1.07, fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "benchmark_v4_pareto.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_boxplot(raw, methods, sizes, out_dir):
    import matplotlib.pyplot as plt  # type: ignore

    ncols = len(sizes)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 5), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, sz in zip(axes, sizes):
        data, labels_short, colors = [], [], []
        for m in methods:
            gaps = []
            for r in raw:
                if r.get("method") == m and r.get("n") == sz and "error" not in r:
                    oracle = r.get("oracle")
                    bk = oracle if oracle is not None else float("inf")
                    gap = r["energy"] - bk if math.isfinite(bk) else 0.0
                    gaps.append(max(gap, 0.0))
            if gaps:
                data.append(gaps)
                labels_short.append(
                    m.replace("qanneal-", "qa-").replace("dwave-", "dw-")
                )
                colors.append(METHOD_COLOR.get(m, "gray"))

        if data:
            bp = ax.boxplot(data, patch_artist=True, tick_labels=labels_short,
                            medianprops=dict(color="black", linewidth=1.5))
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.72)

        ax.set_title(f"n = {sz}", fontsize=10)
        ax.set_ylabel("Energy gap to oracle", fontsize=9)
        ax.tick_params(axis="x", rotation=38, labelsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Distribution of Energy Gap  (all seeds)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "benchmark_v4_boxplot.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_radar(stats, methods, sizes, out_dir):
    import matplotlib.pyplot as plt  # type: ignore

    mid_n    = sizes[len(sizes) // 2]
    axes_lbl = ["Accuracy\n(low gap)", "Speed\n(fast)", "Hit Rate", "Power"]
    N_ax     = len(axes_lbl)
    angles   = [k / N_ax * 2 * math.pi for k in range(N_ax)] + [0.0]

    vals_all = {m: stats.get((m, mid_n)) for m in methods}
    vals_all = {m: v for m, v in vals_all.items() if v is not None}
    if not vals_all:
        return None

    gaps  = [v["mean_gap"]  for v in vals_all.values() if np.isfinite(v["mean_gap"])]
    times = [v["mean_time"] for v in vals_all.values() if np.isfinite(v["mean_time"])]
    pwrs  = [v["power"]     for v in vals_all.values() if np.isfinite(v["power"])]

    def _ni(val, lst):   # normalise, invert (lower raw → higher score)
        lo, hi = min(lst), max(lst)
        return 1.0 - (val - lo) / (hi - lo) if hi > lo else 1.0

    def _n(val, lst):
        lo, hi = min(lst), max(lst)
        return (val - lo) / (hi - lo) if hi > lo else 1.0

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for m, s in vals_all.items():
        scores = [
            _ni(s["mean_gap"],  gaps)  if gaps  else 0.0,
            _ni(s["mean_time"], times) if times else 0.0,
            s["hit_rate"],
            _n(s["power"],      pwrs)  if pwrs  else 0.0,
        ]
        scores += [scores[0]]
        ax.plot(angles, scores,
                color=METHOD_COLOR.get(m, "gray"), linewidth=2, label=m)
        ax.fill(angles, scores,
                color=METHOD_COLOR.get(m, "gray"), alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_lbl, size=9)
    ax.set_ylim(0, 1)
    ax.set_title(f"Summary Radar  (n = {mid_n})", pad=20, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15), fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "benchmark_v4_radar.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(stats: Dict[Tuple, Dict]) -> None:
    hdr = (f"{'Method':18s} {'n':>4}  {'MeanGap':>10} {'StdGap':>9} "
           f"{'MedGap':>9} {'Time(s)':>8} {'HitRate':>8} {'SR%':>6} "
           f"{'Power':>9}  {'TTS99':>10}")
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    prev_m = None
    for (m, n), s in sorted(stats.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if prev_m and m != prev_m:
            print()
        prev_m = m
        tts_str = f"{s['tts_99']:10.3f}" if math.isfinite(s["tts_99"]) else "       inf"
        print(
            f"{m:18s} {n:4d}  "
            f"{s['mean_gap']:10.5f} {s['std_gap']:9.5f} "
            f"{s['median_gap']:9.5f} {s['mean_time']:8.4f} "
            f"{s['hit_rate']:8.3f} {s['success_rate']*100:6.1f} "
            f"{s['power']:9.4f}  {tts_str}"
        )
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="qlatannealv4 vs D-Wave SDK comparative benchmark"
    )
    parser.add_argument("--workers",  type=int,  default=None,
                        help="Parallel workers (default: cpu_count)")
    parser.add_argument("--no-pimc",  action="store_true",
                        help="Skip dwave-PIMC")
    parser.add_argument("--quick",    action="store_true",
                        help="Smoke-test: 3 sizes × 5 seeds")
    args = parser.parse_args()

    sizes   = PROBLEM_SIZES
    n_seeds = N_SEEDS
    if args.quick:
        sizes   = [20, 30, 40]
        n_seeds = 5

    include_pimc = not args.no_pimc
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    banner = "  qlatannealv4 vs D-Wave SDK — Comparative Benchmark  "
    print("═" * len(banner))
    print(banner)
    print("═" * len(banner))
    print(f"  Problem sizes  : {sizes}  (random Ising spin-glass, density={DENSITY})")
    print(f"  Seeds / size   : {n_seeds}")
    print(f"  Brute-force    : n ≤ {BRUTE_MAX_N}")
    print(f"  Include PIMC   : {include_pimc}")
    print(f"  v4 Python path : {V4_PY_PATH}")
    print(f"  Output dir     : {out_dir}")
    print()

    tasks     = _build_tasks(sizes, n_seeds, include_pimc=include_pimc)
    t_wall    = time.perf_counter()
    raw       = _run_parallel(tasks, max_workers=args.workers)
    t_elapsed = time.perf_counter() - t_wall
    print(f"\n  Finished {len(tasks)} tasks in {t_elapsed:.1f} s\n")

    # Resolve deferred oracles: replace None oracle with best-known
    from collections import defaultdict
    best_known: Dict[Tuple, float] = defaultdict(lambda: float("inf"))
    for r in raw:
        if "error" not in r:
            best_known[(r["n"], r["seed"])] = min(
                best_known[(r["n"], r["seed"])], r["energy"]
            )
    for r in raw:
        if r.get("oracle") is None:
            r["oracle"] = best_known.get((r["n"], r["seed"]), None)

    # Ordered method list
    seen: set = set()
    methods: List[str] = []
    for order in ["qanneal-SA", "qanneal-SQA", "qanneal-SQAPT",
                  "qanneal-CTPIMC", "dwave-NEAL", "dwave-PIMC"]:
        if any(r.get("method") == order and "error" not in r for r in raw):
            if order not in seen:
                methods.append(order)
                seen.add(order)

    stats = _compute_stats(raw)
    _print_table(stats)

    errors = [r for r in raw if "error" in r]
    if errors:
        print(f"\n  {len(errors)} task(s) failed:")
        for e in errors[:10]:
            print(f"    {e['method']} n={e['n']} seed={e['seed']}: {e['error']}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "dwave_comparative_results_v4.json")
    payload = {
        "config": {
            "problem_sizes":    sizes,
            "n_seeds":          n_seeds,
            "density":          DENSITY,
            "brute_max_n":      BRUTE_MAX_N,
            "success_tol":      SUCCESS_TOL,
            "hit_eps":          HIT_EPS,
            "sa_reads":         SA_READS, "sa_steps": SA_STEPS, "sa_sweeps": SA_SWEEPS,
            "sqa_reads":        SQA_READS, "sqa_slices": SQA_SLICES,
            "sqapt_reads":      SQAPT_READS, "sqapt_replicas": SQAPT_REPLICAS,
            "ctpimc_reads":     CTPIMC_READS,
            "neal_reads":       NEAL_READS, "neal_sweeps": NEAL_SWEEPS,
            "pimc_reads":       PIMC_READS, "pimc_sweeps": PIMC_SWEEPS,
            "total_wall_s":     round(t_elapsed, 3),
        },
        "statistics":   list(stats.values()),
        "raw_results":  [r for r in raw if "error" not in r],
        "errors":       errors,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results  → {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating figures …")
    saved: List[str] = []

    p = _bar_chart(stats, methods, sizes,
                   key_mean="mean_gap", key_std="std_gap",
                   ylabel="Mean energy gap to oracle  (lower = better)",
                   title="Accuracy — qlatannealv4 vs D-Wave SDK",
                   fname="benchmark_v4_accuracy.png", out_dir=out_dir,
                   log_y=True)
    saved.append(p)

    p = _bar_chart(stats, methods, sizes,
                   key_mean="mean_time", key_std="std_time",
                   ylabel="Mean wall time (s)",
                   title="Wall Time — qlatannealv4 vs D-Wave SDK",
                   fname="benchmark_v4_walltime.png", out_dir=out_dir)
    saved.append(p)

    p = _line_chart(stats, methods, sizes, key="power",
                    ylabel="Power  = hit_rate / mean_time  (higher = better)",
                    title="Computational Power — qlatannealv4 vs D-Wave SDK",
                    fname="benchmark_v4_power.png", out_dir=out_dir)
    saved.append(p)

    p = _line_chart(stats, methods, sizes, key="success_rate",
                    ylabel=f"Success rate  (gap ≤ {int(SUCCESS_TOL*100)}% |oracle|, %)",
                    title="Success Rate — qlatannealv4 vs D-Wave SDK",
                    fname="benchmark_v4_success_rate.png", out_dir=out_dir,
                    scale=100.0)
    saved.append(p)

    p = _line_chart(stats, methods, sizes, key="tts_99",
                    ylabel="TTS @ 99%  (s, lower = better)",
                    title="Time-to-Solution (99%) — qlatannealv4 vs D-Wave SDK",
                    fname="benchmark_v4_tts.png", out_dir=out_dir,
                    log_y=True)
    saved.append(p)

    p = _plot_pareto(raw, methods, sizes, out_dir)
    saved.append(p)

    p = _plot_boxplot(raw, methods, sizes, out_dir)
    saved.append(p)

    p = _plot_radar(stats, methods, sizes, out_dir)
    if p:
        saved.append(p)

    print("\n  Figures saved:")
    for f in saved:
        print(f"    {f}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
