"""
benchmark.py — MPI-parallel comprehensive benchmark driver (qanneal optimal schedule).

Compares 4 solvers on 7 problem classes:
    sqa-std    SQA, standard coupled (beta, Gamma) ladder of length num_steps
    sqa-opt    SQA, adaptive J_perp schedule (two-phase, budget-calibrated)
    sqapt-std  SQAPT, standard schedule (pt_steps = num_steps)
    sqapt-opt  SQAPT, adaptive J_perp schedule (budget-calibrated)
All four use the SAME sweep budget (num_steps * sweeps_per_step) for a fair comparison.

Robust MPI design (no send/recv, deadlock-free): each rank owns a disjoint slice of the
(problem, n, instance) work list and writes its own CSV part file, flushing after every
solver. Crash-safe and resumable — a rank skips (problem,n,instance,solver) rows already in
its part file. At the very end rank 0 concatenates all part files into one CSV.

Usage:  srun --mpi=pmi2 -n 56 python -u benchmark.py --output-dir results --tag $SLURM_JOB_ID
"""
import argparse
import csv
import glob
import os
import time
import numpy as np
from mpi4py import MPI

import qanneal
from qanneal import DenseIsing, solve, auto_schedule_sqa
from problems import (
    sk_spin_glass, ea3d_spin_glass, random_3regular_maxcut,
    weighted_maxcut, rfim_2d, planted_partition, number_partitioning,
)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

# --- fixed run parameters (equal sweep budget for all solvers) -------------------------------
N_INSTANCES = 20
N_READS     = 20
N_SWEEPS    = 20     # Metropolis sweeps per annealing step (proven adequate in prior runs)
N_STEPS     = 150    # annealing steps
WORLDLINE   = 3
M_TROTTER   = 32
REPLICAS_SQA   = 4
REPLICAS_SQAPT = 8

SOLVERS = ["sqa-std", "sqa-opt", "sqapt-std", "sqapt-opt"]

# Put every instance at a common coupling scale so the optimal J_perp schedule has a quantum
# transition to navigate (the raw problems span j_rms ~ 0.1 to 16; below ~2.9 the SQAPT ladder's
# start exceeds 5*j_rms and the adaptive schedule degenerates). Rescaling (J, h, gs) by one global
# constant is an EXACT energy rescaling: it leaves the argmin and all relative structure unchanged,
# only fixing the energy units. Reported energies are therefore in these common units.
TARGET_JRMS = 5.0

def rescale_to_jrms(J, h, gs, target=TARGET_JRMS):
    iu = np.triu_indices(J.shape[0], 1)
    nz = J[iu][J[iu] != 0.0]
    jr = float(np.sqrt(np.mean(nz ** 2))) if nz.size else 0.0
    if jr <= 0.0:
        return J, h, gs
    c = target / jr
    return J * c, h * c, (None if gs is None else gs * c)

# Problem registry. Dense classes (sk/wmaxcut/planted) capped at n=256 for runtime; sparse
# classes (ea3d/w3reg/rfim2d) and small numpart go larger. gs_threshold: relative closeness
# to the reference energy counted as a "success".
PROBLEMS = [
    {"name": "sk",      "gen": lambda n, r: sk_spin_glass(n, r),
     "n_values": [32, 64, 128, 256], "gs_threshold": 0.01},
    {"name": "ea3d",    "gen": lambda n, r: ea3d_spin_glass(int(round(n ** (1 / 3))), r),
     "n_values": [27, 64, 125, 216], "gs_threshold": 0.01},
    {"name": "w3reg",   "gen": lambda n, r: random_3regular_maxcut(n, r),
     "n_values": [32, 64, 128, 256, 512], "gs_threshold": 0.01},
    {"name": "wmaxcut", "gen": lambda n, r: weighted_maxcut(n, r),
     "n_values": [32, 64, 128, 256], "gs_threshold": 0.01},
    {"name": "rfim2d",  "gen": lambda n, r: rfim_2d(int(round(np.sqrt(n))), 0.5, r),
     "n_values": [36, 64, 100, 144, 196], "gs_threshold": 0.01},
    {"name": "planted", "gen": lambda n, r: planted_partition(n, r),
     "n_values": [32, 64, 128, 256], "gs_threshold": 0.001},
    {"name": "numpart", "gen": lambda n, r: number_partitioning(n, r),
     "n_values": [16, 24, 32, 48], "gs_threshold": 0.001},
]

CSV_FIELDS = [
    "problem", "n", "instance", "seed", "solver", "read",
    "best_energy", "mean_energy", "min_energy_across_reads", "gs_energy",
    "energy_gap", "success", "wall_seconds", "n_sweeps_total",
]


def run_solver(solver, ham, seed):
    """Run one solver with N_READS internal reads. Returns (per_read_energies, total_wall)."""
    common = dict(reads=N_READS, sweeps_per_beta=N_SWEEPS, trotter_slices=M_TROTTER,
                  worldline_sweeps=WORLDLINE, progress=False, seed=seed)
    t0 = time.perf_counter()
    if solver == "sqa-std":
        r = solve(ham, method="sqa", schedule_type="standard",
                  schedule=auto_schedule_sqa(steps=N_STEPS),
                  replicas=REPLICAS_SQA, **common)
    elif solver == "sqa-opt":
        r = solve(ham, method="sqa", schedule_type="optimal",
                  optimal_num_steps=N_STEPS, optimal_eps_tilde=0.0,
                  replicas=REPLICAS_SQA, **common)          # j_perp_end auto = max(start, 5*j_rms)
    elif solver == "sqapt-std":
        r = solve(ham, method="sqapt", schedule_type="standard",
                  pt_steps=N_STEPS, swap_interval=1, replicas=REPLICAS_SQAPT, **common)
    elif solver == "sqapt-opt":
        r = solve(ham, method="sqapt", schedule_type="optimal",
                  optimal_num_steps=N_STEPS, optimal_eps_tilde=0.0,
                  swap_interval=1, replicas=REPLICAS_SQAPT, **common)
    else:
        raise ValueError(solver)
    return [float(e) for e in r.energies], time.perf_counter() - t0


def build_work_list():
    """(problem, n, instance, seed) tuples, ordered small-n first so partial runs still scale."""
    work = []
    for prob in PROBLEMS:
        for n in prob["n_values"]:
            for inst in range(N_INSTANCES):
                seed = (hash((prob["name"], n, inst)) % (2 ** 31))
                work.append((prob["name"], n, inst, seed))
    work.sort(key=lambda w: (w[1], w[0], w[2]))   # by n, then problem, then instance
    return work


def main():
    global N_INSTANCES, N_READS, N_SWEEPS, N_STEPS, PROBLEMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--tag", default="run")
    # Optional overrides (production defaults come from the module constants above).
    ap.add_argument("--instances", type=int, default=N_INSTANCES)
    ap.add_argument("--reads", type=int, default=N_READS)
    ap.add_argument("--sweeps", type=int, default=N_SWEEPS)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--problems", nargs="+", default=None, help="subset of problem names")
    ap.add_argument("--max-n", type=int, default=None, help="cap problem sizes (for smoke tests)")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    N_INSTANCES, N_READS, N_SWEEPS, N_STEPS = args.instances, args.reads, args.sweeps, args.steps
    if args.problems:
        PROBLEMS = [p for p in PROBLEMS if p["name"] in args.problems]
    if args.max_n is not None:
        for p in PROBLEMS:
            p["n_values"] = [n for n in p["n_values"] if n <= args.max_n] or [min(p["n_values"])]

    part_path = os.path.join(args.output_dir, f"benchmark_{args.tag}_rank{RANK}.csv")

    # Resume: which (problem,n,instance,solver) already have all reads in this rank's part file.
    done = set()
    if os.path.exists(part_path):
        with open(part_path) as f:
            for row in csv.DictReader(f):
                done.add((row["problem"], row["n"], row["instance"], row["solver"]))

    file_exists = os.path.exists(part_path)
    fh = open(part_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader(); fh.flush()

    prob_lookup = {p["name"]: p for p in PROBLEMS}
    work = build_work_list()
    my_work = [w for i, w in enumerate(work) if i % SIZE == RANK]

    if RANK == 0:
        print(f"[rank0] qanneal {qanneal.__version__} | {SIZE} ranks | {len(work)} work items "
              f"(~{len(work)//SIZE}/rank) | reads={N_READS} steps={N_STEPS} sweeps={N_SWEEPS}",
              flush=True)

    t_start = time.perf_counter()
    for wi, (prob_name, n, inst, seed) in enumerate(my_work):
        prob = prob_lookup[prob_name]
        # Generate the instance (skip whole item on generator failure, e.g. odd-n 3-regular).
        try:
            rng = np.random.default_rng(seed)
            J, h, gs = prob["gen"](n, rng)
            J, h, gs = rescale_to_jrms(J, h, gs)
            ham = DenseIsing(np.ascontiguousarray(h, dtype=float),
                             np.ascontiguousarray(J, dtype=float))
            n_act = J.shape[0]
        except Exception as exc:
            print(f"[rank{RANK}] SKIP gen {prob_name} n={n} inst={inst}: {exc}", flush=True)
            continue

        for solver in SOLVERS:
            if (prob_name, str(n_act), str(inst), solver) in done:
                continue
            try:
                energies, wall = run_solver(solver, ham, seed + (hash(solver) % 9973))
            except Exception as exc:
                print(f"[rank{RANK}] FAIL {prob_name} n={n_act} inst={inst} {solver}: {exc}",
                      flush=True)
                continue
            best_e = min(energies)
            mean_e = float(np.mean(energies))
            per_read_wall = wall / max(len(energies), 1)
            ref = gs if gs is not None else best_e     # unknown gs -> use best across reads
            for read_idx, e in enumerate(energies):
                gap = abs(e - ref) / (abs(ref) + 1e-12)
                if gs is not None:
                    success = 1 if gap <= prob["gs_threshold"] else 0
                else:
                    success = -1
                writer.writerow({
                    "problem": prob_name, "n": n_act, "instance": inst, "seed": seed,
                    "solver": solver, "read": read_idx,
                    "best_energy": e, "mean_energy": mean_e, "min_energy_across_reads": best_e,
                    "gs_energy": ("nan" if gs is None else gs),
                    "energy_gap": gap, "success": success,
                    "wall_seconds": per_read_wall, "n_sweeps_total": N_SWEEPS * N_STEPS,
                })
            fh.flush()
        if RANK == 0 and (wi + 1) % 5 == 0:
            el = time.perf_counter() - t_start
            print(f"[rank0] {wi+1}/{len(my_work)} items, {el/60:.1f} min elapsed", flush=True)

    fh.close()
    COMM.Barrier()

    # Rank 0 stitches all part files into one CSV (pure disk read; no MPI comm).
    if RANK == 0:
        final = os.path.join(args.output_dir, f"benchmark_{args.tag}.csv")
        parts = sorted(glob.glob(os.path.join(args.output_dir, f"benchmark_{args.tag}_rank*.csv")))
        nrows = 0
        with open(final, "w", newline="") as out:
            w = csv.DictWriter(out, fieldnames=CSV_FIELDS); w.writeheader()
            for p in parts:
                with open(p) as f:
                    for row in csv.DictReader(f):
                        w.writerow(row); nrows += 1
        print(f"[rank0] DONE. Merged {len(parts)} parts -> {final} ({nrows} rows)", flush=True)


if __name__ == "__main__":
    main()
