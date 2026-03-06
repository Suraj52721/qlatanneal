#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from qanneal import (
    DenseIsing,
    solve,
    auto_schedule_sa_tuned,
    auto_schedule_sqa_tuned,
    auto_ladder_sqa_tuned,
)

try:
    import dimod  # type: ignore
except Exception:
    dimod = None

try:
    import neal  # type: ignore
except Exception:
    neal = None

try:
    from dwave.samplers import PathIntegralAnnealingSampler  # type: ignore
except Exception:
    PathIntegralAnnealingSampler = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


@dataclass
class RunResult:
    instance: int
    method: str
    energy: float
    gap_to_oracle: float
    wall_time_s: float
    ok: bool
    error: Optional[str] = None


def make_dense_ising(h: np.ndarray, J: np.ndarray, c: float = 0.0) -> DenseIsing:
    try:
        return DenseIsing(h, J, c)
    except TypeError:
        return DenseIsing(h.tolist(), J.flatten().tolist(), len(h), c)


def random_spin_glass(n: int, density: float, field_scale: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, field_scale, size=n)
    J = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                Jij = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5)
                J[i, j] = Jij
                J[j, i] = Jij

    if np.count_nonzero(np.triu(J, 1)) == 0:
        i, j = 0, min(1, n - 1)
        J[i, j] = 1.0
        J[j, i] = 1.0

    return h, J


def ising_energy(h: np.ndarray, J: np.ndarray, spins: np.ndarray) -> float:
    return float(np.dot(h, spins) + 0.5 * spins @ J @ spins)


def brute_force_optimum(h: np.ndarray, J: np.ndarray, max_n: int) -> Optional[float]:
    n = len(h)
    if n > max_n:
        return None
    best_e = float("inf")
    for mask in range(1 << n):
        bits = np.array([(mask >> i) & 1 for i in range(n)], dtype=int)
        spins = 2 * bits - 1
        e = ising_energy(h, J, spins)
        if e < best_e:
            best_e = e
    return best_e


def run_qanneal_sa(ising: DenseIsing, args, seed: int) -> float:
    schedule = auto_schedule_sa_tuned(ising, mode=args.schedule_mode, steps=args.steps, seed=seed)
    res = solve(
        ising,
        method="sa",
        reads=args.reads,
        sweeps_per_beta=args.sweeps,
        schedule=schedule,
        seed=seed,
        progress=False,
    )
    return float(res.best_energy)


def run_qanneal_sqa(ising: DenseIsing, args, seed: int) -> float:
    schedule = auto_schedule_sqa_tuned(ising, mode=args.schedule_mode, steps=args.steps, seed=seed)
    res = solve(
        ising,
        method="sqa",
        reads=args.reads,
        sweeps_per_beta=args.sweeps,
        worldline_sweeps=args.worldline,
        cluster_sweeps=args.cluster,
        trotter_slices=args.slices,
        replicas=args.replicas,
        schedule=schedule,
        seed=seed,
        progress=False,
    )
    return float(res.best_energy)


def run_qanneal_sqapt(ising: DenseIsing, args, seed: int) -> float:
    ladder = auto_ladder_sqa_tuned(ising, replicas=args.replicas, mode=args.schedule_mode, seed=seed)
    res = solve(
        ising,
        method="sqapt",
        reads=args.reads,
        sweeps_per_beta=args.sweeps,
        worldline_sweeps=args.worldline,
        cluster_sweeps=args.cluster,
        trotter_slices=args.slices,
        replicas=args.replicas,
        pt_steps=args.pt_steps,
        swap_interval=args.swap_interval,
        schedule=ladder,
        seed=seed,
        progress=False,
    )
    return float(res.best_energy)


def run_qanneal_ctpimc(ising: DenseIsing, args, seed: int) -> float:
    schedule = auto_schedule_sqa_tuned(ising, mode=args.schedule_mode, steps=args.steps, seed=seed)
    res = solve(
        ising,
        method="ctpimc",
        reads=args.reads,
        sweeps_per_beta=args.sweeps,
        schedule=schedule,
        seed=seed,
        progress=False,
    )
    return float(res.best_energy)


def build_dimod_bqm(h: np.ndarray, J: np.ndarray):
    if dimod is None:
        return None
    linear = {i: float(h[i]) for i in range(len(h))}
    quadratic = {}
    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            if J[i, j] != 0.0:
                quadratic[(i, j)] = float(J[i, j])
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype="SPIN")


def run_neal(bqm, args) -> float:
    sampler = neal.SimulatedAnnealingSampler()
    ss = sampler.sample(bqm, num_reads=args.reads, num_sweeps=max(1, args.steps * args.sweeps))
    return float(ss.first.energy)


def run_pimc(bqm, args) -> float:
    sampler = PathIntegralAnnealingSampler()
    ss = sampler.sample(bqm, num_reads=args.reads, num_sweeps=max(1, args.steps * args.sweeps))
    return float(ss.first.energy)


def run_random_search(h: np.ndarray, J: np.ndarray, args, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(h)
    best_e = float("inf")
    trials = max(100, args.reads * 20)
    for _ in range(trials):
        s = rng.choice([-1, 1], size=n)
        e = ising_energy(h, J, s)
        if e < best_e:
            best_e = e
    return best_e


def timed_call(fn) -> Tuple[bool, float, float, Optional[str]]:
    t0 = time.perf_counter()
    try:
        e = float(fn())
        return True, e, time.perf_counter() - t0, None
    except Exception as exc:
        return False, float("inf"), time.perf_counter() - t0, str(exc)


def summarize(results: List[RunResult]) -> List[Dict[str, float]]:
    methods = sorted(set(r.method for r in results))
    rows = []
    for m in methods:
        mr = [r for r in results if r.method == m and r.ok]
        if not mr:
            rows.append(
                {
                    "method": m,
                    "runs": 0,
                    "median_energy": float("inf"),
                    "median_gap": float("inf"),
                    "median_time_s": float("inf"),
                    "hit_rate": 0.0,
                }
            )
            continue
        energies = np.array([r.energy for r in mr], dtype=float)
        gaps = np.array([r.gap_to_oracle for r in mr], dtype=float)
        times = np.array([r.wall_time_s for r in mr], dtype=float)
        rows.append(
            {
                "method": m,
                "runs": float(len(mr)),
                "median_energy": float(np.median(energies)),
                "median_gap": float(np.median(gaps)),
                "median_time_s": float(np.median(times)),
                "hit_rate": float(np.mean(gaps <= 1e-9)),
            }
        )
    rows.sort(key=lambda r: (r["median_gap"], r["median_time_s"]))
    return rows


def plot_summary(summary_rows: List[Dict[str, float]], out_png: Path) -> None:
    if plt is None:
        return

    names = [r["method"] for r in summary_rows]
    gaps = [max(r["median_gap"], 1e-12) for r in summary_rows]
    times = [max(r["median_time_s"], 1e-6) for r in summary_rows]
    hits = [r["hit_rate"] for r in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].bar(names, gaps, color="#2a9d8f")
    axes[0].set_title("Median Gap to Oracle")
    axes[0].set_ylabel("energy gap")
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, times, color="#e9c46a")
    axes[1].set_title("Median Runtime")
    axes[1].set_ylabel("seconds")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(names, hits, color="#e76f51")
    axes[2].set_title("Oracle Hit Rate")
    axes[2].set_ylabel("fraction")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spin-glass benchmark across qanneal and D-Wave samplers")
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--instances", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--density", type=float, default=0.35)
    parser.add_argument("--field-scale", type=float, default=0.2)
    parser.add_argument("--exact-max-n", type=int, default=22)

    parser.add_argument("--reads", type=int, default=32)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--sweeps", type=int, default=80)
    parser.add_argument("--schedule-mode", type=str, default="balanced", choices=["fast", "balanced", "accurate"])

    parser.add_argument("--worldline", type=int, default=4)
    parser.add_argument("--slices", type=int, default=24)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--cluster", type=int, default=1)
    parser.add_argument("--pt-steps", type=int, default=70)
    parser.add_argument("--swap-interval", type=int, default=1)

    parser.add_argument("--with-ctpimc", action="store_true")
    parser.add_argument("--out", type=str, default="spin_glass_results")
    args = parser.parse_args()

    all_rows: List[RunResult] = []

    for inst in range(args.instances):
        inst_seed = args.seed + 1000 * inst
        h, J = random_spin_glass(args.n, args.density, args.field_scale, inst_seed)
        ising = make_dense_ising(h, J, 0.0)
        bqm = build_dimod_bqm(h, J)

        method_calls = [
            ("qanneal-SA", lambda: run_qanneal_sa(ising, args, inst_seed + 1)),
            ("qanneal-SQA", lambda: run_qanneal_sqa(ising, args, inst_seed + 2)),
            ("qanneal-SQAPT", lambda: run_qanneal_sqapt(ising, args, inst_seed + 3)),
            ("random-search", lambda: run_random_search(h, J, args, inst_seed + 4)),
        ]
        if args.with_ctpimc:
            method_calls.append(("qanneal-CTPIMC", lambda: run_qanneal_ctpimc(ising, args, inst_seed + 5)))
        if bqm is not None and neal is not None:
            method_calls.append(("dwave-NEAL", lambda: run_neal(bqm, args)))
        if bqm is not None and PathIntegralAnnealingSampler is not None:
            method_calls.append(("dwave-PIMC", lambda: run_pimc(bqm, args)))

        raw_instance = []
        for name, fn in method_calls:
            ok, energy, dt, err = timed_call(fn)
            raw_instance.append((name, ok, energy, dt, err))

        exact = brute_force_optimum(h, J, args.exact_max_n)
        if exact is None:
            ok_energies = [x[2] for x in raw_instance if x[1]]
            oracle = min(ok_energies) if ok_energies else float("inf")
        else:
            oracle = exact

        for name, ok, energy, dt, err in raw_instance:
            gap = float(energy - oracle) if ok and np.isfinite(oracle) else float("inf")
            all_rows.append(
                RunResult(
                    instance=inst,
                    method=name,
                    energy=float(energy),
                    gap_to_oracle=gap,
                    wall_time_s=float(dt),
                    ok=ok,
                    error=err,
                )
            )

    summary_rows = summarize(all_rows)

    out_base = Path(args.out)
    out_json = out_base.with_suffix(".json")
    out_csv = out_base.with_suffix(".csv")
    out_png = out_base.with_suffix(".png")

    payload = {"config": vars(args), "runs": [asdict(r) for r in all_rows], "summary": summary_rows}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "runs", "median_energy", "median_gap", "median_time_s", "hit_rate"]
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    plot_summary(summary_rows, out_png)

    print("Saved:")
    print(" ", out_json)
    print(" ", out_csv)
    if plt is not None:
        print(" ", out_png)

    for row in summary_rows:
        print(
            f"{row['method']:16s} "
            f"median_gap={row['median_gap']:.6f} "
            f"median_time={row['median_time_s']:.3f}s "
            f"hit_rate={row['hit_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
