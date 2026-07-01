#!/usr/bin/env python3
"""
benchmark_v7.py — focused SQA-std vs SQA-opt re-run with the fixed 0.6.1 schedule.

qanneal >= 0.6.1: beta_end = max(C/j_rms, beta_min) and the two-phase SQA-opt protocol
(phase 1 ramps beta at fixed J_perp; phase 2 = calibrated adaptive J_perp). This is the
SQA half of the paper's table; the SQAPT half comes from v6 (already clean).

Solvers (equal sweep budget num_steps*sweeps):
  sqa-std : standard SQA, schedule = auto_schedule_sqa(steps=num_steps)
  sqa-opt : SQA run_optimal, budget-calibrated, two-phase (beta ramp 0.3 + adaptive 0.7)

Problems: wmaxcut, sk, ea3d, w3reg at n in {64,128,256}; numpart at n in {25,28,30}
(number partitioning is hard only at small n with wide integers).
"""
from __future__ import annotations
import argparse, csv, logging, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    rank, size, _mpi = _comm.Get_rank(), _comm.Get_size(), True
except ImportError:
    _comm, rank, size, _mpi = None, 0, 1, False

log = logging.getLogger("bench")
def _barrier():
    if _mpi: _comm.Barrier()
def _gather(o):
    return _comm.gather(o, root=0) if _mpi else [o]

from problems import make_problem, j_rms
import qanneal
from qanneal import DenseIsing, solve, auto_schedule_sqa

PROBLEMS = ["wmaxcut", "sk", "ea3d", "w3reg", "numpart"]
SOLVERS = ["sqa-std", "sqa-opt"]
# numpart is hard only at small n (wide integers); the rest scale up.
N_BY_PROBLEM = {"numpart": [25, 28, 30]}


@dataclass
class RunResult:
    problem: str
    n: int
    instance: int
    solver: str
    schedule_type: str
    reads: int
    num_steps: int
    j_rms: float
    beta_end: float
    best_energy: float
    energies: List[float]
    wall_sec: float

    @property
    def success_prob(self) -> float:
        thr = abs(self.best_energy) * 0.01 if self.best_energy != 0 else 0.05
        return sum(1 for e in self.energies if e <= self.best_energy + thr) / len(self.energies)

    @property
    def tts_99(self) -> float:
        p = self.success_prob
        if p <= 0: return float("inf")
        if p >= 1: return 1.0
        return math.ceil(math.log(1 - 0.99) / math.log(1 - p))


def run_solver(ham, J, solver, args):
    from qanneal import optimal_j_perp_params
    jr = j_rms(J)
    jpe = max(5.0 * jr, 3.0)
    beta_end = optimal_j_perp_params(ham, trotter_slices=args.slices)[0]
    common = dict(reads=args.reads, trotter_slices=args.slices,
                  worldline_sweeps=args.worldline_sweeps, sweeps_per_beta=args.sweeps,
                  replicas=args.replicas_sqa, progress=False, seed=args.seed)
    t0 = time.perf_counter()
    if solver == "sqa-std":
        r = solve(ham, method="sqa", schedule_type="standard",
                  schedule=auto_schedule_sqa(steps=args.num_steps), **common)
    else:  # sqa-opt
        r = solve(ham, method="sqa", schedule_type="optimal", optimal_num_steps=args.num_steps,
                  optimal_eps_tilde=0.0, optimal_j_perp_end=jpe, optimal_alpha=args.alpha,
                  optimal_calib_probes=args.calib_probes, optimal_calib_sweeps=args.calib_sweeps,
                  optimal_beta_ramp_fraction=args.beta_ramp_fraction, **common)
    wall = time.perf_counter() - t0
    return float(r.best_energy), [float(e) for e in r.energies], wall, jr, beta_end


def run_instance(problem, n, inst, args):
    seed_base = args.base_seed + inst * 997 + n * 31 + hash(problem) % 10000
    h, J, c, n_act = make_problem(problem, n, seed_base)
    ham = DenseIsing(h, J, c)
    out = []
    for s in args.solvers:
        try:
            be, es, wall, jr, beta_end = run_solver(ham, J, s, args)
            out.append(RunResult(problem=problem, n=n_act, instance=inst, solver=s,
                                  schedule_type=("optimal" if s.endswith("opt") else "standard"),
                                  reads=args.reads, num_steps=args.num_steps, j_rms=round(jr, 4),
                                  beta_end=round(beta_end, 4), best_energy=be, energies=es,
                                  wall_sec=wall))
            log.info("[r%d] %-8s n=%d inst=%d %-8s -> %.4g sp=%.3f beta=%.2f %.1fs",
                     rank, problem, n_act, inst, s, be, out[-1].success_prob, beta_end, wall)
        except Exception as exc:
            log.error("[r%d] %-8s n=%d inst=%d %s FAILED: %s", rank, problem, n, inst, s, exc)
    return out


def write_csv(results, path):
    if not results: return
    fields = ["problem", "n", "instance", "solver", "schedule_type", "reads", "num_steps",
              "j_rms", "beta_end", "best_energy", "success_prob", "tts_99", "wall_sec"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = asdict(r); row.pop("energies")
            row["success_prob"] = round(r.success_prob, 4)
            row["tts_99"] = r.tts_99 if math.isfinite(r.tts_99) else -1
            w.writerow(row)


def print_summary(results):
    from collections import defaultdict
    tab = defaultdict(list)
    for r in results:
        tab[(r.problem, r.n, r.solver)].append(r)
    print("\n" + "=" * 92)
    print(f"  v7 SQA opt-vs-std — qanneal {qanneal.__version__}  ({size} ranks)")
    print("=" * 92)
    print(f"{'problem':<9} {'n':>4} {'solver':<8} {'MeanE':>13} {'SuccP':>8} {'MeanTTS':>9} {'Wall':>8} {'beta':>6}")
    print("-" * 92)
    for (prob, n, s), rl in sorted(tab.items()):
        me = np.mean([r.best_energy for r in rl]); mp = np.mean([r.success_prob for r in rl])
        tv = [r.tts_99 for r in rl if math.isfinite(r.tts_99)]; mt = np.mean(tv) if tv else float("inf")
        mw = np.mean([r.wall_sec for r in rl]); be = rl[0].beta_end
        ts = f"{mt:9.1f}" if math.isfinite(mt) else "      inf"
        print(f"{prob:<9} {n:>4} {s:<8} {me:13.4g} {mp:8.3f} {ts} {mw:7.1f}s {be:6.2f}")
    print("=" * 92)
    print("\n  SQA opt vs std  (Delta% success prob)")
    print(f"{'problem':<9} {'n':>4} {'Std SuccP':>10} {'Opt SuccP':>10} {'Delta%':>8}")
    print("-" * 50)
    for prob in sorted(set(k[0] for k in tab)):
        for n in sorted(set(k[1] for k in tab if k[0] == prob)):
            std = tab.get((prob, n, "sqa-std"), []); opt = tab.get((prob, n, "sqa-opt"), [])
            if not std or not opt: continue
            ss = np.mean([r.success_prob for r in std]); so = np.mean([r.success_prob for r in opt])
            d = (so - ss) / max(ss, 1e-9) * 100
            flag = "opt" if so > ss * 1.05 else ("std" if so < ss * 0.95 else "~")
            print(f"{prob:<9} {n:>4} {ss:10.4f} {so:10.4f} {d:+7.1f}% {flag}")
    print("=" * 92)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--problems", nargs="+", default=PROBLEMS)
    p.add_argument("--n-list", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--numpart-n-list", type=int, nargs="+", default=[25, 28, 30])
    p.add_argument("--instances", type=int, default=30)
    p.add_argument("--solvers", nargs="+", default=SOLVERS)
    p.add_argument("--reads", type=int, default=15)
    p.add_argument("--num-steps", type=int, default=150, dest="num_steps")
    p.add_argument("--sweeps", type=int, default=20)
    p.add_argument("--worldline-sweeps", type=int, default=3, dest="worldline_sweeps")
    p.add_argument("--slices", type=int, default=32)
    p.add_argument("--replicas-sqa", type=int, default=4, dest="replicas_sqa")
    p.add_argument("--alpha", type=float, default=15 / 14)
    p.add_argument("--calib-probes", type=int, default=12, dest="calib_probes")
    p.add_argument("--calib-sweeps", type=int, default=10, dest="calib_sweeps")
    p.add_argument("--beta-ramp-fraction", type=float, default=0.3, dest="beta_ramp_fraction")
    p.add_argument("--base-seed", type=int, default=42, dest="base_seed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="results", dest="out_dir")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("qanneal %s | ranks=%d | v7 SQA opt-vs-std | instances=%d",
                 qanneal.__version__, size, args.instances)
    _barrier()
    work = []
    for prob in args.problems:
        ns = args.numpart_n_list if prob == "numpart" else args.n_list
        for n in ns:
            for i in range(args.instances):
                work.append((prob, n, i))
    my = [w for k, w in enumerate(work) if k % size == rank]
    if rank == 0:
        log.info("Total work: %d items (~%d/rank)", len(work), max(1, len(work) // size))
    mine = []
    for prob, n, inst in my:
        mine.extend(run_instance(prob, n, inst, args))
    _barrier()
    alll = _gather(mine)
    if rank == 0:
        allr = [r for lst in (alll or []) for r in (lst or [])]
        import os, datetime
        job = os.environ.get("SLURM_JOB_ID", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        write_csv(allr, out_dir / f"v7_{job}.csv")
        log.info("Results -> %s", out_dir / f"v7_{job}.csv")
        print_summary(allr)


if __name__ == "__main__":
    main()
