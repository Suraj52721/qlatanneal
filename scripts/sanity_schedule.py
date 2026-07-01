#!/usr/bin/env python3
"""Pre-flight sanity check for the optimal SQAPT schedule (Tasks 5 & 7).

Generates small instances of three problem classes (mwis_3reg, wmaxcut, sk), runs the
budget-calibrated optimal schedule, writes a debug CSV per class, and verifies the realized
J_perp(t) trajectory:

  (a) consumes the full step budget (runs all num_steps, no early stop),
  (b) reaches (near) j_perp_end,
  (c) dwells where chi_B is large (more steps in the lower part of the J_perp range) rather
      than ramping linearly — a linear/flat trajectory is the signature of the frozen-schedule
      bug this audit fixed.

MANDATORY before submitting any large-n HPC benchmark job: run this, inspect the printed
summary (and the plots with --plot), and only proceed if every class PASSES. See
docs/optimal_schedule.md, "Pre-flight check".

Usage:
    python scripts/sanity_schedule.py [--n 40] [--steps 150] [--plot] [--outdir /tmp/sched]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "python"))

from qanneal._qanneal import DenseIsing, SQAParallelTemperingAnnealer  # noqa: E402
from qanneal import solver  # noqa: E402


def make_mwis_3reg(n, seed):
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    for i in range(n):
        for _ in range(3):
            j = int(rng.integers(0, n))
            if j != i:
                J[i, j] = J[j, i] = 0.5
    return J


def make_wmaxcut(n, seed):
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                w = rng.uniform(1.0, 10.0)
                J[i, j] = J[j, i] = w
    return J


def make_sk(n, seed):
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    s = 1.0 / math.sqrt(n)
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = float(rng.normal(0.0, s))
    return J


CLASSES = {"mwis_3reg": make_mwis_3reg, "wmaxcut": make_wmaxcut, "sk": make_sk}


def run_one(name, J, steps, outdir, seed):
    n = J.shape[0]
    jrms = solver.j_rms_from_problem(J) or 0.0
    ham = DenseIsing(np.zeros(n), J)
    # A hot->cold ladder; the hottest replica sets j_perp_start.
    betas = [0.1, 0.3, 0.7, 1.5]
    gammas = [3.0, 1.5, 0.7, 0.3]
    ann = SQAParallelTemperingAnnealer(ham, betas, gammas, 32)
    ann.set_seed(seed)
    # Explicit, physically-motivated target (Task 4.2). For tiny-j_rms classes this may be
    # <= j_perp_start; pass 0.0 there to exercise (and surface) the sentinel fallback.
    csv_path = outdir / f"{name}_n{n}.csv"
    res = ann.run_optimal(num_steps=steps, sweeps_per_step=10, worldline_sweeps=0,
                          eps_tilde=0.0, j_perp_end=max(5.0 * jrms, 0.0),
                          calib_probes=12, calib_sweeps=10,
                          debug_csv_path=str(csv_path))
    traj = list(res.j_perp_trace)
    start = res.j_perp_start
    end = res.resolved_j_perp_end
    reached = max(traj) if traj else start
    rng_span = end - start
    # "Dwell" metric: fraction of steps spent in the lower 40% of the realized range. A frozen
    # or linear schedule spreads steps ~uniformly (dwell ~0.4); an adaptive one front-loads.
    lo_cut = start + 0.4 * max(reached - start, 1e-12)
    dwell = sum(1 for j in traj if j <= lo_cut) / max(len(traj), 1)
    # Degenerate class: end ~ start (or sentinel fallback to +1e6). Flag, don't hard-fail it.
    degenerate = (rng_span <= 0.2) or (end > start + 1e5)
    frac_reached = (reached - start) / rng_span if rng_span > 1e-9 else float("nan")

    if degenerate:
        status = "DEGENERATE"
        ok = True  # not a failure of the schedule; the problem has no quantum range to anneal
    else:
        # PASS requires consuming budget, reaching the end, and front-loading steps.
        ok = (len(traj) == steps) and (frac_reached > 0.9) and (dwell > 0.45)
        status = "PASS" if ok else "FAIL"

    print(f"[{name:9s} n={n}] j_rms={jrms:6.3f} start={start:6.3f} end={end:10.3f} "
          f"reached_frac={frac_reached:6.3f} dwell_lo40={dwell:5.2f} eps={res.calibrated_eps_tilde:.3g} "
          f"steps={len(traj)} -> {status}   csv={csv_path.name}")
    return ok, csv_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/sched_sanity"))
    ap.add_argument("--plot", action="store_true", help="also render J_perp(t) plots")
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    csvs = []
    for name, maker in CLASSES.items():
        J = maker(args.n, args.seed)
        ok, csv_path = run_one(name, J, args.steps, args.outdir, args.seed)
        all_ok = all_ok and ok
        csvs.append(csv_path)

    if args.plot:
        from plot_schedule_trajectory import main as plot_main  # local import
        for c in csvs:
            plot_main([str(c)])

    print()
    print("PRE-FLIGHT:", "OK — safe to submit HPC job." if all_ok else
          "FAILED — do NOT submit; trajectory looks frozen/linear (see CSVs/plots).")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
