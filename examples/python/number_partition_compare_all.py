#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


@dataclass
class RunResult:
    instance_id: int
    method: str
    diff: float
    wall_time_s: float
    ok: bool
    error: Optional[str] = None


def diff_from_spins(nums: np.ndarray, spins: np.ndarray) -> float:
    return float(abs(np.dot(nums, spins)))


def build_ising_from_numbers(nums: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    n = len(nums)
    h = np.zeros(n, dtype=float)
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = 2.0 * nums[i] * nums[j]
            J[j, i] = J[i, j]
    c = float(np.dot(nums, nums))
    return h, J, c


def brute_force_opt(nums: np.ndarray, max_n: int) -> Optional[float]:
    n = len(nums)
    if n > max_n:
        return None
    best_diff = float("inf")
    for mask in range(1 << n):
        spins = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)], dtype=int)
        d = diff_from_spins(nums, spins)
        if d < best_diff:
            best_diff = d
    return best_diff


def greedy_heuristic(nums: np.ndarray) -> float:
    idx = np.argsort(-nums)
    a_sum = 0.0
    b_sum = 0.0
    for i in idx:
        if a_sum <= b_sum:
            a_sum += nums[i]
        else:
            b_sum += nums[i]
    return abs(a_sum - b_sum)


def karmarkar_karp(nums: np.ndarray) -> float:
    arr = list(nums.astype(float))
    arr.sort(reverse=True)
    while len(arr) > 1:
        a = arr.pop(0)
        b = arr.pop(0)
        arr.append(abs(a - b))
        arr.sort(reverse=True)
    return arr[0] if arr else 0.0


def random_baseline(nums: np.ndarray, reads: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(nums)
    best = float("inf")
    for _ in range(reads):
        spins = rng.choice([-1, 1], size=n)
        best = min(best, diff_from_spins(nums, spins))
    return best


def _run_method_task(payload: Dict) -> RunResult:
    instance_id = int(payload["instance_id"])
    method = str(payload["method"])
    nums = np.array(payload["nums"], dtype=float)
    reads = int(payload["reads"])
    steps = int(payload["steps"])
    sweeps = int(payload["sweeps"])
    worldline = int(payload["worldline"])
    slices = int(payload["slices"])
    replicas = int(payload["replicas"])
    cluster = int(payload["cluster"])
    ct_slices = int(payload["ct_slices"])
    schedule_mode = str(payload.get("schedule_mode", "balanced"))
    pt_steps = int(payload.get("pt_steps", steps))
    swap_interval = int(payload.get("swap_interval", 1))
    seed = int(payload["seed"])

    t0 = time.perf_counter()

    try:
        if method == "qanneal-SA":
            from qanneal import DenseIsing, solve, auto_schedule_sa, auto_schedule_sa_tuned

            h, J, c = build_ising_from_numbers(nums)
            try:
                ising = DenseIsing(h, J, c)
            except TypeError:
                ising = DenseIsing(h.tolist(), J.flatten().tolist(), len(nums), c)
            if schedule_mode == "legacy":
                schedule = auto_schedule_sa(steps=steps)
            else:
                schedule = auto_schedule_sa_tuned(ising, mode=schedule_mode, steps=steps)
            res = solve(ising, method="sa", reads=reads, sweeps_per_beta=sweeps, schedule=schedule, seed=seed, progress=False)
            d = diff_from_spins(nums, np.array(res.best_sample, dtype=int))

        elif method == "qanneal-SQA":
            from qanneal import DenseIsing, solve, auto_schedule_sqa, auto_schedule_sqa_tuned

            h, J, c = build_ising_from_numbers(nums)
            try:
                ising = DenseIsing(h, J, c)
            except TypeError:
                ising = DenseIsing(h.tolist(), J.flatten().tolist(), len(nums), c)
            if schedule_mode == "legacy":
                schedule = auto_schedule_sqa(steps=steps)
            else:
                schedule = auto_schedule_sqa_tuned(ising, mode=schedule_mode, steps=steps)
            res = solve(
                ising,
                method="sqa",
                reads=reads,
                sweeps_per_beta=sweeps,
                worldline_sweeps=worldline,
                cluster_sweeps=cluster,
                continuous_time_slices=ct_slices,
                trotter_slices=slices,
                replicas=replicas,
                schedule=schedule,
                seed=seed,
                progress=False,
            )
            d = diff_from_spins(nums, np.array(res.best_sample, dtype=int))

        elif method == "qanneal-CTPIMC":
            from qanneal import DenseIsing, solve, auto_schedule_sqa, auto_schedule_sqa_tuned

            h, J, c = build_ising_from_numbers(nums)
            try:
                ising = DenseIsing(h, J, c)
            except TypeError:
                ising = DenseIsing(h.tolist(), J.flatten().tolist(), len(nums), c)
            if schedule_mode == "legacy":
                schedule = auto_schedule_sqa(steps=steps)
            else:
                schedule = auto_schedule_sqa_tuned(ising, mode=schedule_mode, steps=steps)
            res = solve(ising, method="ctpimc", reads=reads, sweeps_per_beta=sweeps, schedule=schedule, seed=seed, progress=False)
            d = diff_from_spins(nums, np.array(res.best_sample, dtype=int))

        elif method == "qanneal-SQAPT":
            from qanneal import DenseIsing, solve, auto_schedule_sqa, auto_ladder_sqa_tuned

            h, J, c = build_ising_from_numbers(nums)
            try:
                ising = DenseIsing(h, J, c)
            except TypeError:
                ising = DenseIsing(h.tolist(), J.flatten().tolist(), len(nums), c)

            if schedule_mode == "legacy":
                ladder = auto_schedule_sqa(steps=max(2, replicas))
            else:
                ladder = auto_ladder_sqa_tuned(ising, replicas=max(2, replicas), mode=schedule_mode)

            res = solve(
                ising,
                method="sqapt",
                reads=reads,
                sweeps_per_beta=sweeps,
                worldline_sweeps=worldline,
                cluster_sweeps=cluster,
                continuous_time_slices=ct_slices,
                trotter_slices=slices,
                replicas=replicas,
                pt_steps=pt_steps,
                swap_interval=swap_interval,
                schedule=ladder,
                seed=seed,
                progress=False,
            )
            d = diff_from_spins(nums, np.array(res.best_sample, dtype=int))

        elif method == "dwave-NEAL":
            import dimod  # type: ignore
            import neal  # type: ignore

            bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, vartype="SPIN")
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    bqm.add_interaction(i, j, 2.0 * float(nums[i] * nums[j]))
            sampler = neal.SimulatedAnnealingSampler()
            ss = sampler.sample(bqm, num_reads=reads, num_sweeps=sweeps)
            best = ss.first.sample
            spins = np.array([best[i] for i in range(len(nums))], dtype=int)
            d = diff_from_spins(nums, spins)

        elif method == "dwave-PIMC":
            import dimod  # type: ignore
            from dwave.samplers import PathIntegralAnnealingSampler  # type: ignore

            bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, vartype="SPIN")
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    bqm.add_interaction(i, j, 2.0 * float(nums[i] * nums[j]))
            sampler = PathIntegralAnnealingSampler()
            ss = sampler.sample(bqm, num_reads=reads, num_sweeps=sweeps)
            best = ss.first.sample
            spins = np.array([best[i] for i in range(len(nums))], dtype=int)
            d = diff_from_spins(nums, spins)

        elif method == "greedy":
            d = greedy_heuristic(nums)

        elif method == "karmarkar-karp":
            d = karmarkar_karp(nums)

        elif method == "random":
            d = random_baseline(nums, reads=reads, seed=seed)

        else:
            raise ValueError(f"Unknown method: {method}")

        dt = time.perf_counter() - t0
        return RunResult(instance_id, method, float(d), dt, True, None)

    except Exception as exc:
        dt = time.perf_counter() - t0
        return RunResult(instance_id, method, float("inf"), dt, False, str(exc))


def summarize(results: List[RunResult], oracle: Dict[int, float]) -> List[Dict[str, object]]:
    methods = sorted({r.method for r in results})
    rows: List[Dict[str, object]] = []

    for method in methods:
        method_rows = [r for r in results if r.method == method and r.ok]
        if not method_rows:
            rows.append(
                {
                    "method": method,
                    "runs": 0,
                    "median_diff": None,
                    "mean_diff": None,
                    "median_time_s": None,
                    "mean_time_s": None,
                    "oracle_hit_rate": None,
                    "median_gap_to_oracle": None,
                }
            )
            continue

        diffs = [r.diff for r in method_rows]
        times = [r.wall_time_s for r in method_rows]
        gaps = [r.diff - oracle[r.instance_id] for r in method_rows if r.instance_id in oracle]
        hits = [1.0 if abs(r.diff - oracle[r.instance_id]) <= 1e-9 else 0.0 for r in method_rows if r.instance_id in oracle]

        rows.append(
            {
                "method": method,
                "runs": len(method_rows),
                "median_diff": statistics.median(diffs),
                "mean_diff": statistics.fmean(diffs),
                "median_time_s": statistics.median(times),
                "mean_time_s": statistics.fmean(times),
                "oracle_hit_rate": statistics.fmean(hits) if hits else None,
                "median_gap_to_oracle": statistics.median(gaps) if gaps else None,
            }
        )

    rows.sort(key=lambda x: (float("inf") if x["median_diff"] is None else x["median_diff"]))
    return rows


def plot_summary(rows: List[Dict[str, object]], out_prefix: str) -> None:
    if plt is None:
        return

    valid = [r for r in rows if r["median_diff"] is not None]
    if not valid:
        return

    names = [str(r["method"]) for r in valid]
    diffs = [float(r["median_diff"]) for r in valid]
    times = [max(float(r["median_time_s"]), 1e-6) for r in valid]
    hits = [float(r["oracle_hit_rate"]) if r["oracle_hit_rate"] is not None else 0.0 for r in valid]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].bar(names, diffs, color="#3b82f6")
    axes[0].set_title("Median diff")
    axes[0].set_ylabel("|sum a_i s_i|")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(names, times, color="#f59e0b")
    axes[1].set_title("Median runtime")
    axes[1].set_ylabel("seconds")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(names, hits, color="#10b981")
    axes[2].set_title("Oracle hit-rate")
    axes[2].set_ylabel("fraction")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_summary.png", dpi=180)


def execute_tasks(tasks: List[Dict], jobs: int) -> List[RunResult]:
    if jobs <= 1 or len(tasks) <= 1:
        return [_run_method_task(t) for t in tasks]

    results: List[RunResult] = []
    try:
        with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
            futs = [ex.submit(_run_method_task, t) for t in tasks]
            for fut in as_completed(futs):
                results.append(fut.result())
        return results
    except (PermissionError, OSError):
        results.clear()
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            futs = [ex.submit(_run_method_task, t) for t in tasks]
            for fut in as_completed(futs):
                results.append(fut.result())
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed all-method number partition benchmark with process parallelism.")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--instances", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--reads", type=int, default=32)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--sweeps", type=int, default=80)
    parser.add_argument("--schedule-mode", type=str, default="balanced",
                        choices=["legacy", "fast", "balanced", "accurate"])
    parser.add_argument("--worldline", type=int, default=5)
    parser.add_argument("--slices", type=int, default=16)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--cluster", type=int, default=0)
    parser.add_argument("--ct-slices", type=int, default=0)
    parser.add_argument("--with-sqapt", action="store_true")
    parser.add_argument("--pt-steps", type=int, default=60)
    parser.add_argument("--swap-interval", type=int, default=1)
    parser.add_argument("--bf-max-n", type=int, default=20)
    parser.add_argument("--out", type=str, default="number_partition_compare_all")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    problems = [rng.integers(1, 100, size=args.n).astype(float).tolist() for _ in range(args.instances)]

    methods = [
        "qanneal-SA",
        "qanneal-SQA",
        "qanneal-CTPIMC",
        "dwave-NEAL",
        "dwave-PIMC",
        "greedy",
        "karmarkar-karp",
        "random",
    ]
    if args.with_sqapt:
        methods.insert(2, "qanneal-SQAPT")

    tasks: List[Dict] = []
    for i, nums in enumerate(problems):
        for method in methods:
            tasks.append(
                {
                    "instance_id": i,
                    "method": method,
                    "nums": nums,
                    "reads": args.reads,
                    "steps": args.steps,
                    "sweeps": args.sweeps,
                    "worldline": args.worldline,
                    "slices": args.slices,
                    "replicas": args.replicas,
                    "cluster": args.cluster,
                    "ct_slices": args.ct_slices,
                    "schedule_mode": args.schedule_mode,
                    "pt_steps": args.pt_steps,
                    "swap_interval": args.swap_interval,
                    "seed": args.seed + 1000 * i,
                }
            )

    t0 = time.perf_counter()
    results = execute_tasks(tasks, args.jobs)
    total_time = time.perf_counter() - t0

    oracle: Dict[int, float] = {}
    for i, nums_list in enumerate(problems):
        nums = np.array(nums_list, dtype=float)
        exact = brute_force_opt(nums, args.bf_max_n)
        if exact is not None:
            oracle[i] = exact
            continue
        good_rows = [r for r in results if r.instance_id == i and r.ok]
        if good_rows:
            oracle[i] = min(r.diff for r in good_rows)

    summary_rows = summarize(results, oracle)

    out_json = f"{args.out}.json"
    out_csv = f"{args.out}.csv"

    payload = {
        "config": {
            "n": args.n,
            "instances": args.instances,
            "seed": args.seed,
            "jobs": args.jobs,
            "reads": args.reads,
            "steps": args.steps,
            "sweeps": args.sweeps,
            "schedule_mode": args.schedule_mode,
            "worldline": args.worldline,
            "slices": args.slices,
            "replicas": args.replicas,
            "cluster": args.cluster,
            "ct_slices": args.ct_slices,
            "with_sqapt": args.with_sqapt,
            "pt_steps": args.pt_steps,
            "swap_interval": args.swap_interval,
            "bf_max_n": args.bf_max_n,
        },
        "total_wall_time_s": total_time,
        "oracle": {str(k): v for k, v in oracle.items()},
        "summary": summary_rows,
        "runs": [r.__dict__ for r in results],
        "problems": problems,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "runs",
                "median_diff",
                "mean_diff",
                "median_time_s",
                "mean_time_s",
                "oracle_hit_rate",
                "median_gap_to_oracle",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    plot_summary(summary_rows, args.out)

    print("Detailed comparison complete")
    print("Total wall time (s):", round(total_time, 3))
    print("Saved:")
    print("  ", out_json)
    print("  ", out_csv)
    if plt is not None:
        print("  ", f"{args.out}_summary.png")

    for row in summary_rows:
        print(
            f"{row['method']:16s} "
            f"median_diff={row['median_diff']} "
            f"median_time={row['median_time_s']} "
            f"hit_rate={row['oracle_hit_rate']}"
        )


if __name__ == "__main__":
    main()
