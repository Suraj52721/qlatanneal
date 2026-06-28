#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from qanneal import (
    DenseIsing,
    solve,
    auto_schedule_sa,
    auto_schedule_sqa,
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
class MethodResult:
    name: str
    best_diff: float
    best_energy: float
    wall_time_s: float
    sample: Optional[List[int]] = None
    params: Optional[Dict[str, float]] = None
    error: Optional[str] = None


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


def make_dense_ising(h: np.ndarray, J: np.ndarray, c: float) -> DenseIsing:
    try:
        return DenseIsing(h, J, c)
    except TypeError:
        return DenseIsing(h.tolist(), J.flatten().tolist(), len(h), c)


def build_sa_schedule(ising: DenseIsing, steps: int, schedule_mode: str):
    if schedule_mode == "legacy":
        return auto_schedule_sa(steps=steps)
    return auto_schedule_sa_tuned(ising, mode=schedule_mode, steps=steps)


def build_sqa_schedule(ising: DenseIsing, steps: int, schedule_mode: str):
    if schedule_mode == "legacy":
        return auto_schedule_sqa(steps=steps)
    return auto_schedule_sqa_tuned(ising, mode=schedule_mode, steps=steps)


def build_sqapt_ladder(ising: DenseIsing, replicas: int, schedule_mode: str):
    if schedule_mode == "legacy":
        return auto_schedule_sqa(steps=max(2, replicas))
    return auto_ladder_sqa_tuned(ising, replicas=max(2, replicas), mode=schedule_mode)


def diff_from_spins(nums: np.ndarray, spins: np.ndarray) -> float:
    return float(abs(np.dot(nums, spins)))


def brute_force_opt(nums: np.ndarray, max_n: int) -> Optional[Tuple[float, np.ndarray]]:
    n = len(nums)
    if n > max_n:
        return None
    best_diff = float("inf")
    best_spin = None
    for mask in range(1 << n):
        spins = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)], dtype=int)
        d = diff_from_spins(nums, spins)
        if d < best_diff:
            best_diff = d
            best_spin = spins
    return best_diff, best_spin


def greedy_heuristic(nums: np.ndarray) -> Tuple[float, List[int]]:
    idx = np.argsort(-nums)
    a_sum = 0.0
    b_sum = 0.0
    spins = np.ones(len(nums), dtype=int)
    for i in idx:
        if a_sum <= b_sum:
            a_sum += nums[i]
            spins[i] = 1
        else:
            b_sum += nums[i]
            spins[i] = -1
    return abs(a_sum - b_sum), spins.tolist()


def karmarkar_karp(nums: np.ndarray) -> float:
    arr = list(nums.astype(float))
    arr.sort(reverse=True)
    while len(arr) > 1:
        a = arr.pop(0)
        b = arr.pop(0)
        arr.append(abs(a - b))
        arr.sort(reverse=True)
    return arr[0] if arr else 0.0


def _parallel_or_sequential(callable_fn, tasks: List[Tuple], jobs: int) -> List[Tuple[float, List[int]]]:
    if jobs <= 1 or len(tasks) <= 1:
        return [callable_fn(*t) for t in tasks]

    results: List[Tuple[float, List[int]]] = []
    try:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(callable_fn, *t) for t in tasks]
            for fut in as_completed(futs):
                results.append(fut.result())
        return results
    except (PermissionError, OSError):
        results.clear()
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(callable_fn, *t) for t in tasks]
            for fut in as_completed(futs):
                results.append(fut.result())
    return results


def _single_read_qanneal_sa(
    nums_list: List[float],
    steps: int,
    sweeps: int,
    seed: int,
    schedule_mode: str,
) -> Tuple[float, List[int]]:
    nums = np.array(nums_list, dtype=float)
    h, J, c = build_ising_from_numbers(nums)
    ising = make_dense_ising(h, J, c)
    schedule = build_sa_schedule(ising, steps, schedule_mode)
    res = solve(
        ising,
        method="sa",
        reads=1,
        sweeps_per_beta=sweeps,
        schedule=schedule,
        seed=seed,
        progress=False,
    )
    spins = np.array(res.best_sample, dtype=int)
    return diff_from_spins(nums, spins), spins.tolist()


def _single_read_qanneal_sqa(
    nums_list: List[float],
    steps: int,
    sweeps: int,
    worldline: int,
    slices: int,
    replicas: int,
    cluster_sweeps: int,
    continuous_time_slices: int,
    seed: int,
    schedule_mode: str,
) -> Tuple[float, List[int]]:
    nums = np.array(nums_list, dtype=float)
    h, J, c = build_ising_from_numbers(nums)
    ising = make_dense_ising(h, J, c)
    schedule = build_sqa_schedule(ising, steps, schedule_mode)
    res = solve(
        ising,
        method="sqa",
        reads=1,
        sweeps_per_beta=sweeps,
        worldline_sweeps=worldline,
        cluster_sweeps=cluster_sweeps,
        continuous_time_slices=continuous_time_slices,
        trotter_slices=slices,
        replicas=replicas,
        schedule=schedule,
        seed=seed,
        progress=False,
    )
    spins = np.array(res.best_sample, dtype=int)
    return diff_from_spins(nums, spins), spins.tolist()


def _single_read_qanneal_ctpimc(
    nums_list: List[float],
    steps: int,
    sweeps: int,
    seed: int,
    schedule_mode: str,
) -> Tuple[float, List[int]]:
    nums = np.array(nums_list, dtype=float)
    h, J, c = build_ising_from_numbers(nums)
    ising = make_dense_ising(h, J, c)
    schedule = build_sqa_schedule(ising, steps, schedule_mode)
    res = solve(
        ising,
        method="ctpimc",
        reads=1,
        sweeps_per_beta=sweeps,
        schedule=schedule,
        seed=seed,
        progress=False,
    )
    spins = np.array(res.best_sample, dtype=int)
    return diff_from_spins(nums, spins), spins.tolist()


def _single_read_qanneal_sqapt(
    nums_list: List[float],
    sweeps: int,
    worldline: int,
    slices: int,
    replicas: int,
    cluster_sweeps: int,
    continuous_time_slices: int,
    pt_steps: int,
    swap_interval: int,
    seed: int,
    schedule_mode: str,
) -> Tuple[float, List[int]]:
    nums = np.array(nums_list, dtype=float)
    h, J, c = build_ising_from_numbers(nums)
    ising = make_dense_ising(h, J, c)
    ladder = build_sqapt_ladder(ising, replicas, schedule_mode)
    res = solve(
        ising,
        method="sqapt",
        reads=1,
        sweeps_per_beta=sweeps,
        worldline_sweeps=worldline,
        cluster_sweeps=cluster_sweeps,
        continuous_time_slices=continuous_time_slices,
        trotter_slices=slices,
        replicas=replicas,
        pt_steps=pt_steps,
        swap_interval=swap_interval,
        schedule=ladder,
        seed=seed,
        progress=False,
    )
    spins = np.array(res.best_sample, dtype=int)
    return diff_from_spins(nums, spins), spins.tolist()


def _select_best(results: List[Tuple[float, List[int]]]) -> Tuple[float, List[int]]:
    best = min(results, key=lambda x: x[0])
    return best[0], best[1]


def run_qanneal_sa(
    nums: np.ndarray,
    reads: int,
    steps: int,
    sweeps: int,
    jobs: int,
    seed: int,
    schedule_mode: str,
) -> MethodResult:
    tasks = [(nums.tolist(), steps, sweeps, seed + i, schedule_mode) for i in range(reads)]
    t0 = time.perf_counter()
    out = _parallel_or_sequential(_single_read_qanneal_sa, tasks, jobs)
    dt = time.perf_counter() - t0
    d, spins = _select_best(out)
    return MethodResult(
        "qanneal-SA",
        d,
        d * d,
        dt,
        spins,
        params={
            "reads": reads,
            "steps": steps,
            "sweeps": sweeps,
            "jobs": jobs,
            "schedule_mode": schedule_mode,
        },
    )


def run_qanneal_sqa(
    nums: np.ndarray,
    reads: int,
    steps: int,
    sweeps: int,
    worldline: int,
    slices: int,
    replicas: int,
    cluster_sweeps: int,
    continuous_time_slices: int,
    jobs: int,
    seed: int,
    schedule_mode: str,
) -> MethodResult:
    tasks = [
        (
            nums.tolist(),
            steps,
            sweeps,
            worldline,
            slices,
            replicas,
            cluster_sweeps,
            continuous_time_slices,
            seed + i,
            schedule_mode,
        )
        for i in range(reads)
    ]
    t0 = time.perf_counter()
    out = _parallel_or_sequential(_single_read_qanneal_sqa, tasks, jobs)
    dt = time.perf_counter() - t0
    d, spins = _select_best(out)
    return MethodResult(
        "qanneal-SQA",
        d,
        d * d,
        dt,
        spins,
        params={
            "reads": reads,
            "steps": steps,
            "sweeps": sweeps,
            "worldline": worldline,
            "slices": slices,
            "replicas": replicas,
            "cluster_sweeps": cluster_sweeps,
            "continuous_time_slices": continuous_time_slices,
            "jobs": jobs,
            "schedule_mode": schedule_mode,
        },
    )


def run_qanneal_ctpimc(
    nums: np.ndarray,
    reads: int,
    steps: int,
    sweeps: int,
    jobs: int,
    seed: int,
    schedule_mode: str,
) -> MethodResult:
    tasks = [(nums.tolist(), steps, sweeps, seed + i, schedule_mode) for i in range(reads)]
    t0 = time.perf_counter()
    try:
        out = _parallel_or_sequential(_single_read_qanneal_ctpimc, tasks, jobs)
        dt = time.perf_counter() - t0
        d, spins = _select_best(out)
        return MethodResult(
            "qanneal-CTPIMC",
            d,
            d * d,
            dt,
            spins,
            params={
                "reads": reads,
                "steps": steps,
                "sweeps": sweeps,
                "jobs": jobs,
                "schedule_mode": schedule_mode,
            },
        )
    except Exception as exc:
        dt = time.perf_counter() - t0
        return MethodResult(
            "qanneal-CTPIMC",
            float("inf"),
            float("inf"),
            dt,
            None,
            params={
                "reads": reads,
                "steps": steps,
                "sweeps": sweeps,
                "jobs": jobs,
                "schedule_mode": schedule_mode,
            },
            error=str(exc),
        )


def run_qanneal_sqapt(
    nums: np.ndarray,
    reads: int,
    sweeps: int,
    worldline: int,
    slices: int,
    replicas: int,
    cluster_sweeps: int,
    continuous_time_slices: int,
    pt_steps: int,
    swap_interval: int,
    jobs: int,
    seed: int,
    schedule_mode: str,
) -> MethodResult:
    tasks = [
        (
            nums.tolist(),
            sweeps,
            worldline,
            slices,
            replicas,
            cluster_sweeps,
            continuous_time_slices,
            pt_steps,
            swap_interval,
            seed + i,
            schedule_mode,
        )
        for i in range(reads)
    ]
    t0 = time.perf_counter()
    try:
        out = _parallel_or_sequential(_single_read_qanneal_sqapt, tasks, jobs)
        dt = time.perf_counter() - t0
        d, spins = _select_best(out)
        return MethodResult(
            "qanneal-SQAPT",
            d,
            d * d,
            dt,
            spins,
            params={
                "reads": reads,
                "sweeps": sweeps,
                "worldline": worldline,
                "slices": slices,
                "replicas": replicas,
                "cluster_sweeps": cluster_sweeps,
                "continuous_time_slices": continuous_time_slices,
                "pt_steps": pt_steps,
                "swap_interval": swap_interval,
                "jobs": jobs,
                "schedule_mode": schedule_mode,
            },
        )
    except Exception as exc:
        dt = time.perf_counter() - t0
        return MethodResult(
            "qanneal-SQAPT",
            float("inf"),
            float("inf"),
            dt,
            None,
            params={
                "reads": reads,
                "sweeps": sweeps,
                "worldline": worldline,
                "slices": slices,
                "replicas": replicas,
                "cluster_sweeps": cluster_sweeps,
                "continuous_time_slices": continuous_time_slices,
                "pt_steps": pt_steps,
                "swap_interval": swap_interval,
                "jobs": jobs,
                "schedule_mode": schedule_mode,
            },
            error=str(exc),
        )


def _build_spin_bqm(nums: np.ndarray):
    if dimod is None:
        return None
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, vartype="SPIN")
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            bqm.add_interaction(i, j, 2.0 * float(nums[i] * nums[j]))
    return bqm


def run_neal_sa(nums: np.ndarray, reads: int, sweeps: int) -> Optional[MethodResult]:
    if neal is None or dimod is None:
        return None
    bqm = _build_spin_bqm(nums)
    sampler = neal.SimulatedAnnealingSampler()
    t0 = time.perf_counter()
    sampleset = sampler.sample(bqm, num_reads=reads, num_sweeps=sweeps)
    dt = time.perf_counter() - t0
    best = sampleset.first.sample
    spins = np.array([best[i] for i in range(len(nums))], dtype=int)
    d = diff_from_spins(nums, spins)
    return MethodResult("dwave-NEAL", d, d * d, dt, spins.tolist(), params={"reads": reads, "sweeps": sweeps})


def run_pimc(nums: np.ndarray, reads: int, sweeps: int) -> Optional[MethodResult]:
    if PathIntegralAnnealingSampler is None or dimod is None:
        return None
    bqm = _build_spin_bqm(nums)
    sampler = PathIntegralAnnealingSampler()
    t0 = time.perf_counter()
    sampleset = sampler.sample(bqm, num_reads=reads, num_sweeps=sweeps)
    dt = time.perf_counter() - t0
    best = sampleset.first.sample
    spins = np.array([best[i] for i in range(len(nums))], dtype=int)
    d = diff_from_spins(nums, spins)
    return MethodResult("dwave-PIMC", d, d * d, dt, spins.tolist(), params={"reads": reads, "sweeps": sweeps})


def plot_results(results: List[MethodResult], opt_diff: Optional[float], out_png: str) -> None:
    if plt is None:
        return

    valid = [r for r in results if np.isfinite(r.best_diff) and np.isfinite(r.wall_time_s)]
    if not valid:
        return

    names = [r.name for r in valid]
    diffs = [r.best_diff for r in valid]
    times = [max(r.wall_time_s, 1e-6) for r in valid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.bar(names, diffs, color="#4c8bf5")
    ax.set_ylabel("Partition difference |sum a_i s_i|")
    ax.set_title("Accuracy (lower is better)")
    ax.tick_params(axis="x", rotation=25)
    if opt_diff is not None:
        ax.axhline(opt_diff, color="#cc3d2b", linestyle="--", label="Optimal")
        ax.legend()

    ax2 = axes[1]
    ax2.bar(names, times, color="#c49b63")
    ax2.set_ylabel("Wall time (s)")
    ax2.set_title("Runtime")
    ax2.set_yscale("log")
    ax2.tick_params(axis="x", rotation=25)

    ax3 = axes[2]
    ax3.scatter(times, diffs, s=70, color="#2f7d4a")
    for i, name in enumerate(names):
        ax3.annotate(name, (times[i], diffs[i]), xytext=(4, 4), textcoords="offset points")
    ax3.set_xlabel("Wall time (s)")
    ax3.set_ylabel("Partition difference")
    ax3.set_title("Pareto view")
    ax3.set_xscale("log")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Number partitioning benchmark across solvers.")
    parser.add_argument("--n", type=int, default=30, help="Problem size")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reads", type=int, default=32)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--sweeps", type=int, default=80)
    parser.add_argument("--schedule-mode", type=str, default="balanced",
                        choices=["legacy", "fast", "balanced", "accurate"],
                        help="Schedule builder mode for qanneal SA/SQA/SQAPT")
    parser.add_argument("--worldline", type=int, default=5)
    parser.add_argument("--slices", type=int, default=16)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--cluster", type=int, default=0, help="SQA cluster sweeps per beta step")
    parser.add_argument("--ct-slices", type=int, default=0, help="SQA continuous-time slice override")
    parser.add_argument("--with-sqapt", action="store_true", help="Include qanneal SQA parallel tempering lane")
    parser.add_argument("--pt-steps", type=int, default=60, help="SQAPT Monte Carlo steps (ladder swaps occur inside)")
    parser.add_argument("--swap-interval", type=int, default=1, help="SQAPT swap interval in steps")
    parser.add_argument("--bf-max-n", type=int, default=22, help="Max N for brute-force optimum")
    parser.add_argument("--fair", action="store_true", help="Match compute budget across methods")
    parser.add_argument("--budget-updates", type=float, default=0.0,
                        help="Override update budget (spin updates). If 0, use SA budget.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel worker processes for qanneal read-level parallelism")
    parser.add_argument("--with-ctpimc", action="store_true", help="Include qanneal CT-PIMC lane")
    parser.add_argument("--out", type=str, default="number_partition_results")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    nums = rng.integers(1, 100, size=args.n).astype(float)

    results: List[MethodResult] = []
    n = len(nums)
    sa_sweeps = args.sweeps
    sqa_sweeps = args.sweeps
    sqapt_sweeps = args.sweeps
    neal_sweeps = args.sweeps
    pimc_sweeps = args.sweeps
    ctpimc_sweeps = args.sweeps

    if args.fair:
        base_updates = args.reads * args.steps * args.sweeps * n
        if args.budget_updates > 0:
            base_updates = args.budget_updates

        sa_sweeps = max(1, int(round(base_updates / (args.reads * args.steps * n))))

        denom_sqa = args.reads * args.steps * n * max(1, args.slices) * max(1, args.replicas)
        sqa_sweeps = max(1, int(round(base_updates / denom_sqa)) - args.worldline - args.cluster)
        denom_sqapt = args.reads * args.pt_steps * n * max(1, args.slices) * max(1, args.replicas)
        sqapt_sweeps = max(1, int(round(base_updates / denom_sqapt)) - args.worldline - args.cluster)

        neal_sweeps = max(1, int(round(base_updates / (args.reads * n))))
        pimc_sweeps = neal_sweeps
        ctpimc_sweeps = max(1, int(round(base_updates / (args.reads * args.steps * n))))

        print(f"[FAIR] update budget ~ {base_updates:.0f} spin-updates")
        print(f"[FAIR] SA sweeps={sa_sweeps}")
        print(
            f"[FAIR] SQA sweeps={sqa_sweeps} "
            f"(worldline={args.worldline}, cluster={args.cluster}, slices={args.slices}, replicas={args.replicas})"
        )
        if args.with_sqapt:
            print(
                f"[FAIR] SQAPT sweeps={sqapt_sweeps} "
                f"(pt_steps={args.pt_steps}, swap_interval={args.swap_interval}, "
                f"worldline={args.worldline}, cluster={args.cluster}, "
                f"slices={args.slices}, replicas={args.replicas})"
            )
        print(f"[FAIR] CT-PIMC sweeps={ctpimc_sweeps}")
        print(f"[FAIR] NEAL sweeps={neal_sweeps}")
        print(f"[FAIR] PIMC sweeps={pimc_sweeps}")

    results.append(
        run_qanneal_sa(
            nums,
            args.reads,
            args.steps,
            sa_sweeps,
            args.jobs,
            args.seed,
            args.schedule_mode,
        )
    )
    results.append(
        run_qanneal_sqa(
            nums,
            args.reads,
            args.steps,
            sqa_sweeps,
            args.worldline,
            args.slices,
            args.replicas,
            args.cluster,
            args.ct_slices,
            args.jobs,
            args.seed,
            args.schedule_mode,
        )
    )

    if args.with_sqapt:
        results.append(
            run_qanneal_sqapt(
                nums,
                args.reads,
                sqapt_sweeps,
                args.worldline,
                args.slices,
                args.replicas,
                args.cluster,
                args.ct_slices,
                args.pt_steps,
                args.swap_interval,
                args.jobs,
                args.seed,
                args.schedule_mode,
            )
        )

    if args.with_ctpimc:
        results.append(
            run_qanneal_ctpimc(
                nums,
                args.reads,
                args.steps,
                ctpimc_sweeps,
                args.jobs,
                args.seed,
                args.schedule_mode,
            )
        )

    neal_res = run_neal_sa(nums, args.reads, neal_sweeps)
    if neal_res is not None:
        results.append(neal_res)

    pimc_res = run_pimc(nums, args.reads, pimc_sweeps)
    if pimc_res is not None:
        results.append(pimc_res)

    d_greedy, s_greedy = greedy_heuristic(nums)
    results.append(MethodResult("greedy", d_greedy, d_greedy * d_greedy, 0.0, s_greedy))

    d_kk = karmarkar_karp(nums)
    results.append(MethodResult("karmarkar-karp", d_kk, d_kk * d_kk, 0.0, None))

    opt = brute_force_opt(nums, args.bf_max_n)
    opt_diff = opt[0] if opt is not None else None

    out_json = f"{args.out}.json"
    out_png = f"{args.out}.png"
    payload: Dict[str, object] = {
        "n": args.n,
        "seed": args.seed,
        "schedule_mode": args.schedule_mode,
        "numbers": nums.tolist(),
        "optimal_diff": opt_diff,
        "results": [asdict(r) for r in results],
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_results(results, opt_diff, out_png)

    print("Results saved:")
    print("  ", out_json)
    if plt is not None:
        print("  ", out_png)
    if opt_diff is not None:
        print("Optimal diff:", opt_diff)

    for r in sorted(results, key=lambda x: x.best_diff):
        if r.error:
            print(f"{r.name:16s} FAILED time={r.wall_time_s:.3f}s error={r.error}")
        else:
            print(f"{r.name:16s} diff={r.best_diff:.3f} time={r.wall_time_s:.3f}s")


if __name__ == "__main__":
    main()
