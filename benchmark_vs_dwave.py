#!/usr/bin/env python3
"""
Benchmark: qlatannealv4 optimized SQA/SQAPT/CTPIMC vs D-Wave PIMC and NEAL.

Tests:
  1. Timing across problem sizes (wall-clock seconds per read)
  2. Solution quality on spin-glass instances (gap to oracle)
  3. Aggregate summary table + PNG plots
"""
from __future__ import annotations

import time
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── qanneal ──────────────────────────────────────────────────────────────────
from qanneal import (
    DenseIsing,
    solve,
    auto_schedule_sa_tuned,
    auto_schedule_sqa_tuned,
    auto_ladder_sqa_tuned,
)

# ── optional third-party ─────────────────────────────────────────────────────
try:
    import dimod
    from dwave.samplers import PathIntegralAnnealingSampler
    DWAVE_PIMC = True
except Exception:
    DWAVE_PIMC = False

try:
    import neal
    DWAVE_NEAL = True
except Exception:
    DWAVE_NEAL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# ─────────────────────────────────────────────────────────────────────────────
# Problem generation
# ─────────────────────────────────────────────────────────────────────────────

def random_spin_glass(n: int, seed: int, density: float = 0.45) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 0.3, n)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                v = rng.choice([-1., 1.]) * rng.uniform(0.5, 1.5)
                J[i, j] = J[j, i] = v
    return h, J


def make_ising(h, J):
    return DenseIsing(h, J, 0.0)


def make_bqm(h, J):
    bqm = dimod.BinaryQuadraticModel(vartype="SPIN")
    for i in range(len(h)):
        bqm.set_linear(i, float(h[i]))
    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            if J[i, j] != 0:
                bqm.set_quadratic(i, j, float(J[i, j]))
    return bqm


def brute_force(h, J, max_n=20):
    n = len(h)
    if n > max_n:
        return None
    best = np.inf
    for mask in range(1 << n):
        s = np.array([(1 if (mask >> i) & 1 else -1) for i in range(n)], dtype=float)
        e = np.dot(h, s) + 0.5 * (s @ J @ s)
        best = min(best, e)
    return float(best)


# ─────────────────────────────────────────────────────────────────────────────
# Runner helpers
# ─────────────────────────────────────────────────────────────────────────────

def timed(fn):
    t0 = time.perf_counter()
    try:
        val = fn()
        return True, val, time.perf_counter() - t0, None
    except Exception as exc:
        return False, None, time.perf_counter() - t0, str(exc)


def run_sa(ising, reads, steps, sweeps, seed):
    sched = auto_schedule_sa_tuned(ising, mode="balanced", steps=steps, seed=seed)
    r = solve(ising, method="sa", reads=reads, sweeps_per_beta=sweeps,
              schedule=sched, seed=seed, progress=False)
    return float(r.best_energy)


def run_sqa(ising, reads, steps, sweeps, slices, replicas, seed):
    sched = auto_schedule_sqa_tuned(ising, mode="balanced", steps=steps, seed=seed)
    r = solve(ising, method="sqa", reads=reads, sweeps_per_beta=sweeps,
              worldline_sweeps=4, cluster_sweeps=1,
              trotter_slices=slices, replicas=replicas,
              schedule=sched, seed=seed, progress=False)
    return float(r.best_energy)


def run_sqapt(ising, reads, steps, sweeps, slices, replicas, seed):
    ladder = auto_ladder_sqa_tuned(ising, replicas=replicas, mode="balanced", seed=seed)
    r = solve(ising, method="sqapt", reads=reads, sweeps_per_beta=sweeps,
              worldline_sweeps=4, cluster_sweeps=1,
              trotter_slices=slices, replicas=replicas,
              pt_steps=steps, swap_interval=1,
              schedule=ladder, seed=seed, progress=False)
    return float(r.best_energy)


def run_ctpimc(ising, reads, steps, sweeps, seed):
    sched = auto_schedule_sqa_tuned(ising, mode="balanced", steps=steps, seed=seed)
    r = solve(ising, method="ctpimc", reads=reads, sweeps_per_beta=sweeps,
              schedule=sched, seed=seed, progress=False)
    return float(r.best_energy)


def run_dwave_pimc(bqm, reads, sweeps):
    s = PathIntegralAnnealingSampler()
    ss = s.sample(bqm, num_reads=reads, num_sweeps=sweeps)
    return float(ss.first.energy)


def run_dwave_neal(bqm, reads, sweeps):
    s = neal.SimulatedAnnealingSampler()
    ss = s.sample(bqm, num_reads=reads, num_sweeps=sweeps)
    return float(ss.first.energy)


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 1: Timing vs problem size
# ─────────────────────────────────────────────────────────────────────────────

def bench_timing(sizes, reads, steps, sweeps, slices, replicas, seed):
    print("\n" + "="*70)
    print("BENCHMARK 1: Wall-clock time vs problem size (seconds total)")
    print(f"  reads={reads}  steps={steps}  sweeps={sweeps}  slices={slices}  replicas={replicas}")
    print("="*70)

    METHODS = ["qanneal-SA", "qanneal-SQA", "qanneal-SQAPT", "qanneal-CTPIMC"]
    if DWAVE_PIMC:
        METHODS.append("dwave-PIMC")
    if DWAVE_NEAL:
        METHODS.append("dwave-NEAL")

    header = f"{'N':>6} | " + " | ".join(f"{m:>14}" for m in METHODS)
    print(header)
    print("-" * len(header))

    timing_data = {m: [] for m in METHODS}
    ns = []

    for n in sizes:
        h, J = random_spin_glass(n, seed=seed + n)
        ising = make_ising(h, J)
        bqm = make_bqm(h, J) if (DWAVE_PIMC or DWAVE_NEAL) else None

        times = {}

        _, _, dt, _ = timed(lambda: run_sa(ising, reads, steps, sweeps, seed))
        times["qanneal-SA"] = dt

        _, _, dt, _ = timed(lambda: run_sqa(ising, reads, steps, sweeps, slices, replicas, seed))
        times["qanneal-SQA"] = dt

        _, _, dt, _ = timed(lambda: run_sqapt(ising, reads, steps, sweeps, slices, replicas, seed))
        times["qanneal-SQAPT"] = dt

        _, _, dt, _ = timed(lambda: run_ctpimc(ising, reads, steps, sweeps, seed))
        times["qanneal-CTPIMC"] = dt

        if DWAVE_PIMC and bqm is not None:
            _, _, dt, _ = timed(lambda: run_dwave_pimc(bqm, reads, sweeps))
            times["dwave-PIMC"] = dt

        if DWAVE_NEAL and bqm is not None:
            _, _, dt, _ = timed(lambda: run_dwave_neal(bqm, reads, sweeps))
            times["dwave-NEAL"] = dt

        row = f"{n:>6} | " + " | ".join(f"{times.get(m, float('nan')):>14.3f}" for m in METHODS)
        print(row)

        for m in METHODS:
            timing_data[m].append(times.get(m, float('nan')))
        ns.append(n)

    return ns, timing_data, METHODS


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 2: Solution quality on spin-glass instances
# ─────────────────────────────────────────────────────────────────────────────

def bench_quality(n, instances, reads, steps, sweeps, slices, replicas, seed):
    print("\n" + "="*70)
    print(f"BENCHMARK 2: Solution quality on {instances} spin-glass instances (N={n})")
    print(f"  reads={reads}  steps={steps}  sweeps={sweeps}  slices={slices}  replicas={replicas}")
    print("="*70)

    METHODS = ["qanneal-SA", "qanneal-SQA", "qanneal-SQAPT", "qanneal-CTPIMC"]
    if DWAVE_PIMC:
        METHODS.append("dwave-PIMC")
    if DWAVE_NEAL:
        METHODS.append("dwave-NEAL")

    all_gaps: Dict[str, List[float]] = {m: [] for m in METHODS}
    all_times: Dict[str, List[float]] = {m: [] for m in METHODS}
    hit_rates: Dict[str, List[float]] = {m: [] for m in METHODS}

    for inst in range(instances):
        inst_seed = seed + inst * 997
        h, J = random_spin_glass(n, seed=inst_seed)
        ising = make_ising(h, J)
        bqm = make_bqm(h, J) if (DWAVE_PIMC or DWAVE_NEAL) else None

        oracle = brute_force(h, J, max_n=20)

        results = {}
        raw_energies = {}

        ok, e, dt, err = timed(lambda: run_sa(ising, reads, steps, sweeps, inst_seed))
        results["qanneal-SA"] = (ok, e, dt)
        raw_energies["qanneal-SA"] = e if ok else np.inf

        ok, e, dt, err = timed(lambda: run_sqa(ising, reads, steps, sweeps, slices, replicas, inst_seed))
        results["qanneal-SQA"] = (ok, e, dt)
        raw_energies["qanneal-SQA"] = e if ok else np.inf

        ok, e, dt, err = timed(lambda: run_sqapt(ising, reads, steps, sweeps, slices, replicas, inst_seed))
        results["qanneal-SQAPT"] = (ok, e, dt)
        raw_energies["qanneal-SQAPT"] = e if ok else np.inf

        ok, e, dt, err = timed(lambda: run_ctpimc(ising, reads, steps, sweeps, inst_seed))
        results["qanneal-CTPIMC"] = (ok, e, dt)
        raw_energies["qanneal-CTPIMC"] = e if ok else np.inf

        if DWAVE_PIMC and bqm is not None:
            ok, e, dt, err = timed(lambda: run_dwave_pimc(bqm, reads, sweeps))
            results["dwave-PIMC"] = (ok, e, dt)
            raw_energies["dwave-PIMC"] = e if ok else np.inf

        if DWAVE_NEAL and bqm is not None:
            ok, e, dt, err = timed(lambda: run_dwave_neal(bqm, reads, sweeps))
            results["dwave-NEAL"] = (ok, e, dt)
            raw_energies["dwave-NEAL"] = e if ok else np.inf

        # Use best known energy as oracle if brute force not available
        if oracle is None:
            valid_e = [v for v in raw_energies.values() if np.isfinite(v)]
            oracle = min(valid_e) if valid_e else np.inf

        for m in METHODS:
            if m in results:
                ok, e, dt = results[m]
                gap = float(e - oracle) if ok and np.isfinite(oracle) else np.inf
                all_gaps[m].append(gap)
                all_times[m].append(dt)
                hit_rates[m].append(1.0 if gap <= 1e-6 else 0.0)

        sys.stdout.write(f"\r  instance {inst+1}/{instances} done")
        sys.stdout.flush()

    print()
    print(f"\n{'Method':>16} | {'Median gap':>12} | {'Mean gap':>12} | {'Time (s)':>10} | {'Hit rate':>10}")
    print("-" * 72)

    summary = []
    for m in METHODS:
        gaps = all_gaps[m]
        times_m = all_times[m]
        hits = hit_rates[m]
        median_gap = float(np.median(gaps)) if gaps else np.inf
        mean_gap = float(np.mean(gaps)) if gaps else np.inf
        median_t = float(np.median(times_m)) if times_m else np.inf
        hit = float(np.mean(hits)) if hits else 0.0
        print(f"{m:>16} | {median_gap:>12.6f} | {mean_gap:>12.6f} | {median_t:>10.3f} | {hit:>10.3f}")
        summary.append({
            "method": m,
            "median_gap": median_gap,
            "mean_gap": mean_gap,
            "median_time_s": median_t,
            "hit_rate": hit,
            "n": n,
        })

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 3: Speedup ratio table (qanneal-SQA vs dwave-PIMC)
# ─────────────────────────────────────────────────────────────────────────────

def bench_speedup(ns, timing_data):
    if "dwave-PIMC" not in timing_data:
        return
    print("\n" + "="*70)
    print("BENCHMARK 3: Speedup of qanneal-SQA vs dwave-PIMC")
    print("  (values > 1 mean qanneal-SQA is faster)")
    print("="*70)
    print(f"{'N':>6} | {'qanneal-SQA (s)':>16} | {'dwave-PIMC (s)':>16} | {'Speedup':>10}")
    print("-" * 58)
    for i, n in enumerate(ns):
        t_sqa = timing_data["qanneal-SQA"][i]
        t_pimc = timing_data["dwave-PIMC"][i]
        if np.isfinite(t_sqa) and np.isfinite(t_pimc) and t_sqa > 0:
            speedup = t_pimc / t_sqa
            print(f"{n:>6} | {t_sqa:>16.3f} | {t_pimc:>16.3f} | {speedup:>10.2f}×")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_timing(ns, timing_data, methods, out="bench_timing.png"):
    if not HAS_PLT:
        return
    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ef4444", "#64748b"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(methods):
        vals = timing_data[m]
        ax.plot(ns, vals, marker="o", label=m, color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel("Problem size N (spins)", fontsize=12)
    ax.set_ylabel("Wall-clock time (s)", fontsize=12)
    ax.set_title("Timing: qanneal optimized vs D-Wave samplers", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"  Saved {out}")


def plot_quality(summary, out="bench_quality.png"):
    if not HAS_PLT:
        return
    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ef4444", "#64748b"]
    methods = [r["method"] for r in summary]
    gaps = [max(r["median_gap"], 1e-12) for r in summary]
    hits = [r["hit_rate"] for r in summary]
    times = [max(r["median_time_s"], 1e-4) for r in summary]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Gap
    bars = axes[0].bar(methods, gaps, color=colors[:len(methods)])
    axes[0].set_yscale("log")
    axes[0].set_title("Median energy gap to oracle\n(lower = better)", fontsize=11)
    axes[0].set_ylabel("Gap (energy units)")
    axes[0].tick_params(axis="x", rotation=30)

    # Hit rate
    axes[1].bar(methods, hits, color=colors[:len(methods)])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Oracle hit rate\n(higher = better)", fontsize=11)
    axes[1].set_ylabel("Fraction of instances solved exactly")
    axes[1].tick_params(axis="x", rotation=30)

    # Time
    axes[2].bar(methods, times, color=colors[:len(methods)])
    axes[2].set_yscale("log")
    axes[2].set_title("Median wall-clock time\n(lower = faster)", fontsize=11)
    axes[2].set_ylabel("Seconds")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle("Solution quality & runtime vs D-Wave PIMC/NEAL", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

import sys

def main():
    SEED = 42
    READS = 16
    STEPS = 70
    SWEEPS = 60
    SLICES = 24
    REPLICAS = 4

    # ── Benchmark 1: timing vs size ──────────────────────────────────────────
    SIZES = [20, 30, 40, 50, 60, 75, 90]
    ns, timing_data, methods = bench_timing(
        sizes=SIZES, reads=READS, steps=STEPS, sweeps=SWEEPS,
        slices=SLICES, replicas=REPLICAS, seed=SEED,
    )

    # ── Benchmark 2: solution quality ────────────────────────────────────────
    N_QUALITY = 30   # small enough for brute-force oracle on ≤20 spins
    INSTANCES = 12
    quality_summary = bench_quality(
        n=N_QUALITY, instances=INSTANCES,
        reads=READS, steps=STEPS, sweeps=SWEEPS,
        slices=SLICES, replicas=REPLICAS, seed=SEED + 7,
    )

    # ── Benchmark 3: speedup table ───────────────────────────────────────────
    bench_speedup(ns, timing_data)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if HAS_PLT:
        print("\nGenerating plots...")
        plot_timing(ns, timing_data, methods, out="bench_timing.png")
        plot_quality(quality_summary, out="bench_quality.png")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    output = {
        "config": {
            "reads": READS, "steps": STEPS, "sweeps": SWEEPS,
            "slices": SLICES, "replicas": REPLICAS, "seed": SEED,
        },
        "timing": {m: list(timing_data[m]) for m in methods},
        "sizes": ns,
        "quality": quality_summary,
    }
    with open("bench_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Saved bench_results.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
