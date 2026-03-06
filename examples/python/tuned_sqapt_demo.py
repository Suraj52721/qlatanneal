#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Optional, Tuple

import numpy as np

from qanneal import (
    QUBO,
    solve,
    auto_schedule_sa_tuned,
    auto_schedule_sqa_tuned,
    auto_ladder_sqa_tuned,
)


def random_qubo(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0, size=(n, n))
    Q = 0.5 * (A + A.T)
    np.fill_diagonal(Q, np.diag(Q) + rng.uniform(-1.0, 1.0, size=n))
    return Q


def brute_force_qubo(Q: np.ndarray) -> Optional[Tuple[float, np.ndarray]]:
    n = Q.shape[0]
    if n > 22:
        return None
    best_e = float("inf")
    best_x = None
    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        e = float(x @ Q @ x)
        if e < best_e:
            best_e = e
            best_x = x.astype(int)
    return best_e, best_x


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate tuned schedules and SQA parallel tempering.")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--mode", type=str, default="balanced", choices=["fast", "balanced", "accurate"])
    parser.add_argument("--reads", type=int, default=16)
    parser.add_argument("--sa-steps", type=int, default=80)
    parser.add_argument("--sqa-steps", type=int, default=90)
    parser.add_argument("--pt-steps", type=int, default=70)
    parser.add_argument("--sweeps", type=int, default=60)
    parser.add_argument("--worldline", type=int, default=4)
    parser.add_argument("--slices", type=int, default=24)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--cluster", type=int, default=1)
    parser.add_argument("--swap-interval", type=int, default=1)
    args = parser.parse_args()

    Q = random_qubo(args.n, args.seed)
    qubo = QUBO(Q)
    ising = qubo.to_ising()

    sa_schedule = auto_schedule_sa_tuned(ising, mode=args.mode, steps=args.sa_steps, seed=args.seed)
    sqa_schedule = auto_schedule_sqa_tuned(ising, mode=args.mode, steps=args.sqa_steps, seed=args.seed)
    sqapt_ladder = auto_ladder_sqa_tuned(ising, replicas=args.replicas, mode=args.mode, seed=args.seed)

    sa = solve(
        ising,
        method="sa",
        reads=args.reads,
        sweeps_per_beta=args.sweeps,
        schedule=sa_schedule,
        seed=args.seed,
        progress=False,
    )
    sqa = solve(
        ising,
        method="sqa",
        reads=args.reads,
        sweeps_per_beta=args.sweeps,
        worldline_sweeps=args.worldline,
        cluster_sweeps=args.cluster,
        trotter_slices=args.slices,
        replicas=max(1, args.replicas // 2),
        schedule=sqa_schedule,
        seed=args.seed,
        progress=False,
    )
    sqapt = solve(
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
        schedule=sqapt_ladder,
        seed=args.seed,
        progress=False,
    )

    def to_bits(spins: np.ndarray) -> np.ndarray:
        return ((spins + 1) // 2).astype(int)

    sa_bits = to_bits(sa.best_sample)
    sqa_bits = to_bits(sqa.best_sample)
    sqapt_bits = to_bits(sqapt.best_sample)

    print("Problem size:", args.n)
    print("Mode:", args.mode)
    print("SA best Ising energy:", sa.best_energy, "QUBO energy:", float(sa_bits @ Q @ sa_bits))
    print("SQA best Ising energy:", sqa.best_energy, "QUBO energy:", float(sqa_bits @ Q @ sqa_bits))
    print("SQAPT best Ising energy:", sqapt.best_energy, "QUBO energy:", float(sqapt_bits @ Q @ sqapt_bits))
    print("SQAPT best sample bits:", sqapt_bits.tolist())

    exact = brute_force_qubo(Q)
    if exact is not None:
        print("Brute force best QUBO energy:", exact[0])
        print("Brute force sample:", exact[1].tolist())
    else:
        print("Brute force skipped (n > 22).")


if __name__ == "__main__":
    main()
