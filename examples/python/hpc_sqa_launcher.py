#!/usr/bin/env python3
"""
hpc_sqa_launcher.py  —  HPC-ready Simulated Quantum Annealing for huge sparse Ising problems.

Simulated Quantum Annealing (SQA / SQAPT / CT-PIMC) is the best classical analog
of a real quantum annealer.  For huge problems (n = 1 000 – 100 000 spins) we need:

  1. SparseIsing representation  — O(n + |E|) memory instead of O(n²)
  2. Embarrassingly-parallel reads spread across MPI ranks or CPU workers
  3. Problem-adaptive schedules  — auto_schedule_sqa_tuned / auto_ladder_sqa_tuned

Usage modes
-----------
1. Single-node, multiprocessing  (no MPI needed):
   python hpc_sqa_launcher.py --n 5000 --method sqapt --reads 64 --workers 8

2. Multi-node MPI  (mpi4py required):
   mpirun -n 32 python hpc_sqa_launcher.py --n 50000 --method sqa --reads 8 --mpi

3. SLURM multi-node (see scripts/slurm/run_sqa_hpc.sh)

4. SLURM job array (no mpi4py, see scripts/slurm/run_sqa_array.sh):
   sbatch run_sqa_array.sh
   # after all tasks complete:
   python merge_array_results.py results/run_*.json

5. Scaling benchmark:
   python hpc_sqa_launcher.py --scaling --method sqapt --mode fast

Problem types
-------------
  random_sparse   random ±1 Ising, k-regular random graph  [default]
  chimera         D-Wave Chimera C_{m,m,4} topology
  maxcut          Max-Cut on random k-regular graph
  from_file       load from .npz  {h: (n,), edges: (m,3)}
                            or .json {h:[…], edges:[[i,j,J],…]}

Outputs
-------
  <out>.json           summary (energy, time, statistics)
  <out>_spins.npy      best spin configuration  int8 array shape (n,)
  <out>_energy_dist.png energy histogram across reads  [if matplotlib available]
  <out>_scaling.json/png  [--scaling mode only]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── qlatannealv4 path injection ───────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_V4_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_V4_PY_PATH = os.path.join(_V4_ROOT, "python")

if _V4_PY_PATH not in sys.path:
    sys.path.insert(0, _V4_PY_PATH)

from qanneal import (  # noqa: E402
    SparseIsing,
    SparseEdge,
    solve,
    auto_schedule_sqa_tuned,
    auto_ladder_sqa_tuned,
    auto_schedule_sa_tuned,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

try:
    from mpi4py import MPI  # type: ignore
    HAS_MPI = True
except Exception:
    HAS_MPI = False


# ─────────────────────────────────────────────────────────────────────────────
# SparseProblem  —  picklable wrapper around SparseIsing data
# (SparseIsing itself is a pybind11 object without pickle support)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SparseProblem:
    """Picklable representation of a sparse Ising problem."""
    h: np.ndarray        # shape (n,)   local fields
    edges: np.ndarray    # shape (m, 3) columns: [i, j, J_ij]
    n: int
    c: float = 0.0       # constant energy offset

    def to_ising(self) -> SparseIsing:
        """Reconstruct the SparseIsing C++ object."""
        sparse_edges = [
            SparseEdge(int(e[0]), int(e[1]), float(e[2]))
            for e in self.edges
        ]
        return SparseIsing(self.h.astype(float), sparse_edges, self.n, self.c)

    def n_edges(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict:
        return {
            "h": self.h.tolist(),
            "edges": self.edges.tolist(),
            "n": self.n,
            "c": self.c,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SparseProblem":
        return cls(
            h=np.array(d["h"], dtype=float),
            edges=np.array(d["edges"], dtype=float),
            n=int(d["n"]),
            c=float(d.get("c", 0.0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Problem generators
# ─────────────────────────────────────────────────────────────────────────────

def random_sparse_ising(n: int, degree: int, seed: int,
                        scale: float = 1.0) -> SparseProblem:
    """
    Random Ising on a k-regular random graph (degree neighbors per spin).

    Memory: O(n × degree) — suitable for n up to 100 000+ spins.
    For D-Wave-like problems use degree ≈ 6-15.
    """
    rng = np.random.default_rng(seed)
    h = rng.uniform(-scale, scale, n)

    # Build edge set via random-offset neighbour sampling (O(n × degree))
    edge_dict: Dict[Tuple[int, int], float] = {}
    offsets = rng.integers(1, max(n, 2), size=(n, degree * 2))
    for i in range(n):
        added = 0
        for d in range(degree * 2):
            if added >= degree:
                break
            j = int((i + int(offsets[i, d])) % n)
            key = (min(i, j), max(i, j))
            if key[0] == key[1] or key in edge_dict:
                continue
            edge_dict[key] = float(rng.uniform(-scale, scale))
            added += 1

    if not edge_dict:
        edge_arr = np.zeros((0, 3), dtype=float)
    else:
        keys = list(edge_dict.keys())
        vals = [edge_dict[k] for k in keys]
        edge_arr = np.array([[i, j, v] for (i, j), v in zip(keys, vals)], dtype=float)

    return SparseProblem(h=h, edges=edge_arr, n=n)


def chimera_ising(m: int, seed: int, scale: float = 1.0) -> SparseProblem:
    """
    D-Wave Chimera C_{m,m,4} topology.

    n = 8 * m * m  qubits.
    Each unit cell: 4 horizontal × 4 vertical qubits, fully bipartite (16 couplings).
    Inter-cell edges connect same-type qubits in adjacent cells.
    """
    rng = np.random.default_rng(seed)
    n = 8 * m * m
    h = rng.uniform(-scale, scale, n)

    def qubit(row: int, col: int, side: int, k: int) -> int:
        # side 0 = horizontal, side 1 = vertical
        return (row * m + col) * 8 + side * 4 + k

    edges = []
    for row in range(m):
        for col in range(m):
            # Intra-cell: bipartite 4×4
            for k0 in range(4):
                for k1 in range(4):
                    q0 = qubit(row, col, 0, k0)
                    q1 = qubit(row, col, 1, k1)
                    edges.append((min(q0, q1), max(q0, q1),
                                  float(rng.uniform(-scale, scale))))
            # Inter-cell horizontal (same row, next column)
            if col + 1 < m:
                for k in range(4):
                    q0 = qubit(row, col, 0, k)
                    q1 = qubit(row, col + 1, 0, k)
                    edges.append((min(q0, q1), max(q0, q1),
                                  float(rng.uniform(-scale, scale))))
            # Inter-cell vertical (same column, next row)
            if row + 1 < m:
                for k in range(4):
                    q0 = qubit(row, col, 1, k)
                    q1 = qubit(row + 1, col, 1, k)
                    edges.append((min(q0, q1), max(q0, q1),
                                  float(rng.uniform(-scale, scale))))

    edge_arr = np.array(edges, dtype=float) if edges else np.zeros((0, 3), dtype=float)
    return SparseProblem(h=h, edges=edge_arr, n=n)


def maxcut_ising(n: int, degree: int, seed: int) -> SparseProblem:
    """
    Max-Cut as Ising: J_ij = -1 for all edges (maximise cut = minimise Ising energy).
    Uses the same k-regular random graph structure as random_sparse_ising.
    """
    rng = np.random.default_rng(seed)
    h = np.zeros(n)

    edge_dict: Dict[Tuple[int, int], float] = {}
    offsets = rng.integers(1, max(n, 2), size=(n, degree * 2))
    for i in range(n):
        added = 0
        for d in range(degree * 2):
            if added >= degree:
                break
            j = int((i + int(offsets[i, d])) % n)
            key = (min(i, j), max(i, j))
            if key[0] == key[1] or key in edge_dict:
                continue
            edge_dict[key] = -1.0   # ferromagnetic → maximise cut
            added += 1

    if not edge_dict:
        edge_arr = np.zeros((0, 3), dtype=float)
    else:
        keys = list(edge_dict.keys())
        edge_arr = np.array([[i, j, edge_dict[(i, j)]] for (i, j) in keys], dtype=float)

    return SparseProblem(h=h, edges=edge_arr, n=n)


def load_problem(path: str) -> SparseProblem:
    """Load problem from .npz {h, edges} or .json {h:[…], edges:[[i,j,J],…]}."""
    if path.endswith(".npz"):
        data = np.load(path)
        h = data["h"]
        edges = data["edges"].astype(float)
        return SparseProblem(h=h.astype(float), edges=edges, n=len(h))
    if path.endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        h = np.array(d["h"], dtype=float)
        edges = np.array(d["edges"], dtype=float)
        return SparseProblem(h=h, edges=edges, n=len(h))
    raise ValueError(f"Unsupported format '{path}'. Use .npz or .json")


def save_problem(prob: SparseProblem, path: str) -> None:
    """Save problem to .npz for later reuse."""
    np.savez_compressed(path, h=prob.h, edges=prob.edges)


# ─────────────────────────────────────────────────────────────────────────────
# Solver helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RankResult:
    rank: int
    n_spins: int
    n_edges: int
    best_energy: float
    best_spins: List[int]
    all_energies: List[float]
    wall_time_s: float
    method: str
    reads: int


def _build_schedule(ising: SparseIsing, method: str, mode: str):
    """Return a problem-tuned schedule object."""
    if method == "ctpimc":
        return auto_schedule_sqa_tuned(ising, mode=mode, gamma_end_scale=0.04)
    if method == "sqapt":
        return auto_ladder_sqa_tuned(ising, replicas=8, mode=mode)
    if method == "sqa":
        return auto_schedule_sqa_tuned(ising, mode=mode)
    return auto_schedule_sa_tuned(ising, mode=mode)


def _solve_one_rank(prob_dict: dict, method: str, reads: int,
                    mode: str, seed: int, rank: int) -> RankResult:
    """
    Single-rank solve.  Accepts a picklable dict so it can be called from
    ProcessPoolExecutor or deserialized from MPI broadcast.
    """
    # Re-inject path when running inside a subprocess
    if _V4_PY_PATH not in sys.path:
        sys.path.insert(0, _V4_PY_PATH)
    from qanneal import solve as _solve, SparseIsing as _SI, SparseEdge as _SE
    from qanneal import auto_schedule_sqa_tuned as _sqa_sch
    from qanneal import auto_ladder_sqa_tuned as _ladder_sch
    from qanneal import auto_schedule_sa_tuned as _sa_sch

    prob = SparseProblem.from_dict(prob_dict)
    ising = prob.to_ising()
    n = int(ising.size())

    # Problem-adaptive schedule
    if method == "ctpimc":
        schedule = _sqa_sch(ising, mode=mode, gamma_end_scale=0.04)
    elif method == "sqapt":
        schedule = _ladder_sch(ising, replicas=8, mode=mode)
    elif method == "sqa":
        schedule = _sqa_sch(ising, mode=mode)
    else:
        schedule = _sa_sch(ising, mode=mode)

    # Offset seed per rank so each rank explores independently
    rank_seed = seed + rank * reads

    t0 = time.perf_counter()
    res = _solve(
        ising,
        method=method,
        reads=reads,
        schedule=schedule,
        seed=rank_seed,
        replicas=8 if method == "sqapt" else 1,
        progress=False,
    )
    dt = time.perf_counter() - t0

    return RankResult(
        rank=rank,
        n_spins=n,
        n_edges=prob.n_edges(),
        best_energy=float(res.best_energy),
        best_spins=list(map(int, res.best_sample)),
        all_energies=[float(e) for e in res.energies],
        wall_time_s=dt,
        method=method,
        reads=reads,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Execution backends
# ─────────────────────────────────────────────────────────────────────────────

def run_standalone(prob: SparseProblem, method: str, reads: int, mode: str,
                   seed: int, workers: int = 1) -> RankResult:
    """
    Standalone (no MPI) execution.

    workers=1  → sequential (good for large n where C++ parallelism already
                             fills the CPU via OpenMP or the reads loop)
    workers>1  → multiprocessing, splits reads across CPU workers
    """
    prob_dict = prob.to_dict()

    if workers <= 1:
        return _solve_one_rank(prob_dict, method, reads, mode, seed, rank=0)

    import concurrent.futures

    # Distribute reads evenly across workers
    chunks = [reads // workers + (1 if i < reads % workers else 0)
              for i in range(workers)]
    chunks = [c for c in chunks if c > 0]
    n_workers = len(chunks)

    results: List[RankResult] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [
            ex.submit(_solve_one_rank, prob_dict, method, c, mode, seed, w)
            for w, c in enumerate(chunks)
        ]
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())

    best = min(results, key=lambda r: r.best_energy)
    all_energies = [e for r in results for e in r.all_energies]
    best.all_energies = all_energies
    best.reads = reads
    return best


def run_mpi(prob: SparseProblem, method: str, reads: int, mode: str,
            seed: int) -> Optional[RankResult]:
    """
    MPI-parallel execution (requires mpi4py).

    Strategy  — embarrassingly parallel:
      • Rank 0 builds the problem, broadcasts serialised dict to all ranks.
      • Each rank r runs `reads` independent SQA reads with seed = seed + r*reads.
      • All ranks call comm.allreduce to find the global minimum energy.
      • Rank 0 collects every RankResult, merges energy lists, returns best.
      • Non-root ranks return None.

    Scaling: total_reads = mpi_size × reads.
    Wall time ≈ single-rank time (perfectly parallel).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Broadcast serialised problem to all ranks
    prob_dict = comm.bcast(prob.to_dict() if rank == 0 else None, root=0)

    local = _solve_one_rank(prob_dict, method, reads, mode, seed, rank)

    all_results: Optional[List[RankResult]] = comm.gather(local, root=0)

    if rank == 0:
        best = min(all_results, key=lambda r: r.best_energy)
        best.all_energies = [e for r in all_results for e in r.all_energies]
        best.reads = reads * size
        return best
    return None  # non-root ranks


# ─────────────────────────────────────────────────────────────────────────────
# Scaling benchmark
# ─────────────────────────────────────────────────────────────────────────────

# Problem sizes for the scaling study
SCALING_SIZES = [100, 250, 500, 1000, 2000, 5000, 10_000]


def run_scaling(method: str, mode: str, reads: int, degree: int,
                seed: int, out_prefix: str) -> None:
    """
    Benchmark wall-time and best energy vs problem size.
    Generates a scaling.json and scaling.png.
    """
    print(f"\nScaling benchmark: {method.upper()} [{mode}] degree={degree}")
    print(f"{'n':>8}  {'n_edges':>9}  {'best_E':>12}  {'time_s':>8}")
    print("-" * 44)

    records = []
    for n in SCALING_SIZES:
        prob = random_sparse_ising(n, degree=degree, seed=seed)
        prob_dict = prob.to_dict()
        r = _solve_one_rank(prob_dict, method, reads, mode, seed, rank=0)
        print(f"{n:>8}  {prob.n_edges():>9}  {r.best_energy:>12.4f}  {r.wall_time_s:>8.2f}s")
        records.append({
            "n": n,
            "n_edges": prob.n_edges(),
            "best_energy": r.best_energy,
            "wall_time_s": r.wall_time_s,
            "reads": reads,
            "method": method,
            "mode": mode,
        })

    json_path = f"{out_prefix}_scaling.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nScaling data  → {json_path}")

    if not HAS_PLT:
        return

    ns = [r["n"] for r in records]
    times = [r["wall_time_s"] for r in records]
    energies = [r["best_energy"] for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.loglog(ns, times, "o-", color="#4c8bf5", lw=2, ms=7)
    # Fit O(n) and O(n log n) reference lines
    n_arr = np.array(ns, dtype=float)
    t0 = times[0] / ns[0]
    ax1.loglog(ns, t0 * n_arr, "--", color="#aaa", lw=1, label="O(n)")
    ax1.loglog(ns, t0 * n_arr * np.log2(np.maximum(n_arr, 2)), ":",
               color="#888", lw=1, label="O(n log n)")
    ax1.set_xlabel("Number of spins (n)", fontsize=11)
    ax1.set_ylabel("Wall time (s)", fontsize=11)
    ax1.set_title(f"{method.upper()} scaling  [{mode}]", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.semilogx(ns, energies, "s-", color="#c84b31", lw=2, ms=7)
    ax2.set_xlabel("Number of spins (n)", fontsize=11)
    ax2.set_ylabel("Best energy found", fontsize=11)
    ax2.set_title("Solution quality vs problem size", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = f"{out_prefix}_scaling.png"
    fig.savefig(png_path, dpi=180)
    print(f"Scaling plot  → {png_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Output / reporting
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(result: RankResult, prob: SparseProblem,
                  args: argparse.Namespace) -> None:
    payload: Dict = {
        "problem_type": args.problem_type,
        "n_spins": int(result.n_spins),
        "n_edges": int(result.n_edges),
        "method": result.method,
        "mode": args.mode,
        "reads": result.reads,
        "seed": args.seed,
        "best_energy": result.best_energy,
        "wall_time_s": result.wall_time_s,
        "energy_mean": float(np.mean(result.all_energies)),
        "energy_std": float(np.std(result.all_energies)),
        "energy_min": float(np.min(result.all_energies)),
        "energy_max": float(np.max(result.all_energies)),
        "spins_file": f"{args.out}_spins.npy",
    }
    json_path = f"{args.out}.json"
    spins_path = f"{args.out}_spins.npy"

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    np.save(spins_path, np.array(result.best_spins, dtype=np.int8))

    print("\nResults")
    print(f"  n_spins     : {result.n_spins:,}")
    print(f"  n_couplings : {result.n_edges:,}")
    print(f"  method      : {result.method.upper()}")
    print(f"  total reads : {result.reads}")
    print(f"  best_energy : {result.best_energy:.6f}")
    print(f"  mean±std    : {payload['energy_mean']:.4f} ± {payload['energy_std']:.4f}")
    print(f"  wall_time   : {result.wall_time_s:.2f} s")
    print(f"  JSON        → {json_path}")
    print(f"  spins       → {spins_path}")

    if HAS_PLT and len(result.all_energies) >= 4:
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = min(40, len(result.all_energies))
        ax.hist(result.all_energies, bins=bins,
                color="#4c8bf5", edgecolor="white", linewidth=0.5)
        ax.axvline(result.best_energy, color="#cc3d2b", lw=2, linestyle="--",
                   label=f"Best: {result.best_energy:.4f}")
        ax.set_xlabel("Energy", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(
            f"{result.method.upper()} energy distribution  —  n={result.n_spins:,} spins",
            fontsize=12,
        )
        ax.legend()
        fig.tight_layout()
        png_path = f"{args.out}_energy_dist.png"
        fig.savefig(png_path, dpi=180)
        print(f"  energy dist → {png_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HPC-ready SQA launcher for huge sparse Ising problems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    grp = p.add_argument_group("Problem")
    grp.add_argument("--problem-type", default="random_sparse",
                     choices=["random_sparse", "chimera", "maxcut", "from_file"],
                     help="Problem topology (default: random_sparse)")
    grp.add_argument("--problem-file", default=None,
                     help="Path to .npz/.json (required with --problem-type from_file)")
    grp.add_argument("--n", type=int, default=1000,
                     help="Number of spins for random problems (default: 1000)")
    grp.add_argument("--degree", type=int, default=10,
                     help="Number of neighbours per spin — k-regular graph (default: 10)")
    grp.add_argument("--chimera-m", type=int, default=4,
                     help="Chimera grid size m → n=8m² qubits (default: 4 → n=128)")
    grp.add_argument("--seed", type=int, default=42)

    grp2 = p.add_argument_group("Solver")
    grp2.add_argument("--method", default="sqapt",
                      choices=["sa", "sqa", "sqapt", "ctpimc"],
                      help="Annealing method (default: sqapt — best QA analog)")
    grp2.add_argument("--reads", type=int, default=32,
                      help="Independent reads per rank/worker (default: 32)")
    grp2.add_argument("--mode", default="balanced",
                      choices=["fast", "balanced", "accurate"],
                      help="Schedule quality preset (default: balanced)")

    grp3 = p.add_argument_group("Execution")
    grp3.add_argument("--mpi", action="store_true",
                      help="Use mpi4py for MPI-parallel execution")
    grp3.add_argument("--workers", type=int, default=1,
                      help="CPU workers for standalone multiprocessing (default: 1)")

    grp4 = p.add_argument_group("Output")
    grp4.add_argument("--out", default="hpc_sqa_result",
                      help="Output prefix (default: hpc_sqa_result)")
    grp4.add_argument("--save-problem", action="store_true",
                      help="Save generated problem to <out>_problem.npz")

    grp5 = p.add_argument_group("Scaling mode")
    grp5.add_argument("--scaling", action="store_true",
                      help="Run wall-time scaling benchmark (ignores --n)")
    grp5.add_argument("--scaling-reads", type=int, default=8,
                      help="Reads per size-point in scaling mode (default: 8)")

    return p


def main() -> None:
    args = _parser().parse_args()

    # ── MPI init ──────────────────────────────────────────────────────────────
    use_mpi = args.mpi and HAS_MPI
    if args.mpi and not HAS_MPI:
        print("WARNING: --mpi requested but mpi4py is not installed. "
              "Falling back to standalone mode.")
    rank = 0
    mpi_size = 1
    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        mpi_size = comm.Get_size()

    # ── Scaling benchmark ─────────────────────────────────────────────────────
    if args.scaling:
        if rank == 0:
            run_scaling(
                method=args.method,
                mode=args.mode,
                reads=args.scaling_reads,
                degree=args.degree,
                seed=args.seed,
                out_prefix=args.out,
            )
        return

    # ── Print mode info ───────────────────────────────────────────────────────
    if rank == 0:
        total_reads = mpi_size * args.reads if use_mpi else (
            max(1, args.workers) * args.reads if args.workers > 1 else args.reads
        )
        mode_label = (
            f"MPI ({mpi_size} ranks × {args.reads} reads = {mpi_size * args.reads} total)"
            if use_mpi else
            f"Standalone ({args.workers} workers × reads/worker = {total_reads} total)"
            if args.workers > 1 else
            f"Sequential ({args.reads} reads)"
        )
        print(f"Mode   : {mode_label}")
        print(f"Method : {args.method.upper()}  [{args.mode}]")

    # ── Build / load problem ──────────────────────────────────────────────────
    prob: Optional[SparseProblem] = None
    if rank == 0:
        if args.problem_type == "from_file":
            if args.problem_file is None:
                raise ValueError("--problem-file required with --problem-type from_file")
            prob = load_problem(args.problem_file)
        elif args.problem_type == "chimera":
            prob = chimera_ising(args.chimera_m, args.seed)
            print(f"Chimera C_{{{args.chimera_m},{args.chimera_m},4}}: "
                  f"n={prob.n:,}  couplings={prob.n_edges():,}")
        elif args.problem_type == "maxcut":
            prob = maxcut_ising(args.n, args.degree, args.seed)
            print(f"MaxCut: n={prob.n:,}  edges={prob.n_edges():,}  degree≈{args.degree}")
        else:
            prob = random_sparse_ising(args.n, args.degree, args.seed)
            print(f"Random sparse Ising: n={prob.n:,}  "
                  f"couplings={prob.n_edges():,}  degree≈{args.degree}")

        if args.save_problem:
            prob_path = f"{args.out}_problem.npz"
            save_problem(prob, prob_path)
            print(f"Problem saved → {prob_path}")

    # ── Solve ─────────────────────────────────────────────────────────────────
    if use_mpi:
        result = run_mpi(prob, args.method, args.reads, args.mode, args.seed)
        if rank != 0:
            return  # non-root ranks are done
    else:
        result = run_standalone(prob, args.method, args.reads, args.mode,
                                args.seed, workers=args.workers)

    # ── Save & report (rank 0 only) ───────────────────────────────────────────
    _save_results(result, prob, args)


if __name__ == "__main__":
    main()
