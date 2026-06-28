# qanneal User Guide

This guide explains how to define optimization problems, configure schedules, run SA/SQA/SQAPT/CT-PIMC, and extract detailed state traces.

## 1. Install and Build

From the repository root:

```bash
python -m pip install . --no-build-isolation
```

For active development:

```bash
python -m pip install -e . --no-build-isolation
```

C++ build/test:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## 2. Problem Definitions

### 2.1 QUBO convention in qanneal

qanneal uses:

\[
E(x)=\sum_i\sum_j Q_{ij}x_ix_j,\quad x_i\in\{0,1\}
\]

- `Q[i,i]`: linear terms
- `Q[i,j]` for `i != j`: interaction terms
- If you start from an upper-triangular QUBO form, mirror off-diagonal weights to keep pair strengths explicit.

### 2.2 Ising convention

\[
E(s)=\sum_i h_is_i + \sum_{i<j} J_{ij}s_is_j + c,\quad s_i\in\{-1,+1\}
\]

Mapping used internally for QUBO→Ising:

`x = (s + 1) / 2`

## 3. Constructing Models

### 3.1 QUBO from dense matrix

```python
import numpy as np
from qanneal import QUBO

Q = np.array([
    [1.0, -2.0],
    [-2.0, 3.0],
], dtype=float)
qubo = QUBO(Q)
ising = qubo.to_ising()
```

### 3.2 QUBO from sparse entries

```python
from qanneal import QUBO

entries = [
    (0, 0, 1.0),
    (1, 1, 3.0),
    (0, 1, -2.0),
    (1, 0, -2.0),
]
qubo = QUBO(entries, n=2)
```

### 3.3 QUBO from dict

```python
entries = {(0, 0): 1.0, (1, 1): 3.0, (0, 1): -2.0, (1, 0): -2.0}
qubo = QUBO(entries, n=2)
```

### 3.4 QUBO from dimod BQM

```python
import dimod
from qanneal import QUBO

bqm = dimod.BinaryQuadraticModel({0: 1.0, 1: 3.0}, {(0, 1): -2.0}, vartype="BINARY")
qubo = QUBO(bqm)
```

### 3.5 Ising directly

```python
import numpy as np
from qanneal import DenseIsing

h = np.array([0.3, -0.8, 0.1], dtype=float)
J = np.array([
    [0.0, 1.2, -0.4],
    [1.2, 0.0, 0.6],
    [-0.4, 0.6, 0.0],
], dtype=float)
ising = DenseIsing(h, J, c=0.0)
```

## 4. Schedule Configuration

qanneal supports two schedule families.

### 4.1 Legacy fixed schedules

- `auto_schedule_sa(...)`: linear beta ramp
- `auto_schedule_sqa(...)`: linear beta + gamma ramps

### 4.2 Tuned schedules (recommended)

- `auto_schedule_sa_tuned(problem, mode=...)`
- `auto_schedule_sqa_tuned(problem, mode=...)`
- `auto_ladder_sqa_tuned(problem, replicas, mode=...)` for SQAPT

Modes:

- `fast`: quick iteration
- `balanced`: default production mode
- `accurate`: stronger final freezing and longer ramps

Tuned helpers estimate a characteristic `|ΔE|` scale from random flips, then derive beta/gamma ranges from acceptance-target heuristics.

## 5. Running Solvers

### 5.1 One-call solver

```python
from qanneal import solve

res = solve(
    ising,
    method="sqa",
    reads=16,
    sweeps_per_beta=60,
    worldline_sweeps=4,
    cluster_sweeps=1,
    trotter_slices=24,
    replicas=4,
    progress=False,
)
print(res.best_energy)
print(res.best_sample)
```

`method` options:

- `"sa"`
- `"sqa"`
- `"sqapt"`
- `"ctpimc"`

### 5.2 SQAPT (quantum parallel tempering)

```python
from qanneal import solve, auto_ladder_sqa_tuned

ladder = auto_ladder_sqa_tuned(ising, replicas=8, mode="balanced")

res = solve(
    ising,
    method="sqapt",
    reads=8,
    sweeps_per_beta=50,
    worldline_sweeps=4,
    cluster_sweeps=1,
    trotter_slices=24,
    replicas=8,
    pt_steps=70,
    swap_interval=1,
    schedule=ladder,
    progress=False,
)
```

SQAPT performs local SQA updates in each ladder replica and proposes swaps between neighboring `(beta, gamma)` replicas.

### 5.3 CT-PIMC

```python
import numpy as np
from qanneal import DenseIsing, SQASchedule, CTPIMCAnnealer

h = np.array([0.2, -0.3, 0.1])
J = np.array([[0.0, -1.0, 0.2], [-1.0, 0.0, 0.5], [0.2, 0.5, 0.0]])
ising = DenseIsing(h, J, c=0.0)

schedule = SQASchedule.from_vectors(
    betas=np.linspace(0.5, 4.0, 40).tolist(),
    gammas=np.linspace(2.0, 0.05, 40).tolist(),
)

annealer = CTPIMCAnnealer(ising, schedule, qubits_per_update=1, qubits_per_chain=1)
res = annealer.run(sweeps_per_beta=50, reads=1)
```

## 6. Physical Meaning of Key Parameters

### 6.1 `beta` (`β`)

- Inverse temperature: `β = 1 / (k_B T)`
- Higher `β` means lower temperature and stronger rejection of uphill moves.

### 6.2 `gamma` (`Γ`)

- Transverse field strength controlling quantum fluctuations.
- Large `Γ`: stronger tunneling-like effects.
- Small `Γ`: recovery toward classical Ising landscape.

### 6.3 `trotter_slices`

- Number of imaginary-time slices in discrete-time SQA.
- Larger values reduce Trotter discretization error but increase runtime roughly linearly.

### 6.4 `worldline_sweeps`

- Proposals that flip one spin through all slices (same site across imaginary time).
- Helps cross barriers that single-slice moves struggle with.

### 6.5 `cluster_sweeps`

- Swendsen-Wang-style cluster updates along imaginary time for each site.
- Improves mixing in stiff low-temperature / high-coupling regimes.

### 6.6 `continuous_time_slices`

- Optional large-slice override in SQA to approximate the continuous-time limit.
- Memory and runtime increase with this value.

### 6.7 `replicas` (SQA/SQAPT)

- SQA: independent copies in one run (multi-start effect).
- SQAPT: number of points in the `(beta, gamma)` ladder.

### 6.8 `pt_steps`, `swap_interval` (SQAPT)

- `pt_steps`: number of local-update / swap epochs.
- `swap_interval`: attempt swaps every `swap_interval` steps.

## 7. States, Samples, and Traces

### 7.1 Spin and bit conversion

- qanneal state vectors are spins in `{-1, +1}`.
- QUBO bits are `x = (s + 1) // 2`.

```python
bits = ((res.best_sample + 1) // 2).astype(int)
```

### 7.2 What `SolveResult` contains

- `samples`: one best sample per read
- `energies`: corresponding energies
- `best_sample`, `best_energy`: global best over reads
- `trace`: energy trace from first read
- `var_order`: variable mapping for BQM/graph inputs

### 7.3 Observer APIs

Classical SA:

- `MetricsObserver`
- `StateTraceObserver`

SQA:

- `SQAMetricsObserver`
- `SQAStateTraceObserver`

`SQAStateTraceObserver` stores per-sweep replica/slice state snapshots and metadata (`beta`, `gamma`, phase, energies).

## 8. Parallelism and Performance

### 8.1 OpenMP parallelized C++ paths

Parallel loops are enabled in:

- `ReplicaAnnealer`
- `SQAAnnealer` (parallel path when no sweep observer is attached)
- `SQAParallelTemperingAnnealer`

Build:

```bash
cmake -S . -B build -DQANNEAL_ENABLE_OPENMP=ON
cmake --build build
```

Runtime thread control:

```bash
export OMP_NUM_THREADS=8
```

### 8.2 Python benchmark parallelism

Benchmark scripts parallelize reads/tasks via process pools:

- `examples/python/number_partition_benchmark.py --jobs N`
- `examples/python/number_partition_compare_all.py --jobs N`

Both scripts include thread fallback if process pools are unavailable.

### 8.3 MPI

```bash
cmake -S . -B build -DQANNEAL_ENABLE_MPI=ON
cmake --build build
mpirun -n 4 build/qanneal_mpi_example
```

## 9. Optimal Adaptive J⊥ Schedule *(new in 0.4.0)*

### 9.1 When to use it

The standard schedule has a fixed Γ(t) ramp — it does not adapt to the problem. The optimal schedule
measures how close the system is to the quantum critical point at each step and **automatically slows
down near criticality**. Use it when:

- You care about solution quality more than wall-clock time.
- The standard schedule finds good solutions sometimes but misses the global optimum.
- You suspect the system is getting stuck near the quantum phase transition.

### 9.2 One-call usage

```python
from qanneal import solve

result = solve(
    problem,
    method="sqapt",
    schedule_type="optimal",
    replicas=8,
    reads=4,
    trotter_slices=32,
    worldline_sweeps=3,
    cluster_sweeps=1,
    optimal_eps_tilde=0.02,  # adiabaticity — smaller = better but slower
    optimal_num_steps=300,
)
print(result.best_energy)
```

All `optimal_*` parameters are optional and auto-computed from the problem's energy scale if omitted.

### 9.3 Parameter guidance

| Parameter | Effect | Starting point |
|-----------|--------|---------------|
| `optimal_eps_tilde` | Controls overall speed. Large → fast but non-adiabatic. Small → slow but adiabatic. | `0.05` (fast), `0.02` (balanced), `0.005` (accurate) |
| `optimal_num_steps` | Max adaptive steps. If J⊥ reaches `j_perp_end` before this, the run stops early. | 100–500 depending on problem size |
| `optimal_alpha` | Universality exponent. Rarely needs changing. | `15/14` (default, 1-D quantum Ising) |
| `sweeps_per_beta` | Sweeps per adaptive step (passed as `sweeps_per_step`). More sweeps → better χ_B estimate per step. | 10–30 |
| `replicas` | More replicas → better χ_B estimate near criticality for SQAPT. | 4–8 |

### 9.4 Inspecting the J⊥ trace (low-level API)

```python
from qanneal import SQAAnnealer, SQASchedule, optimal_j_perp_params
import matplotlib.pyplot as plt

dummy = SQASchedule.from_vectors([3.0], [0.01])
ann = SQAAnnealer(ising, dummy, trotter_slices=32, replicas=4)

beta, jp_start, jp_end = optimal_j_perp_params(ising)
result = ann.run_optimal(
    beta=beta, j_perp_start=jp_start, j_perp_end=jp_end,
    eps_tilde=0.05, num_steps=200, sweeps_per_step=20,
    worldline_sweeps=3, cluster_sweeps=1,
)

steps = list(range(len(result.j_perp_trace)))
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(steps, result.j_perp_trace)
ax1.set_ylabel("J⊥ (Trotter coupling)")
ax1.set_title("Optimal adaptive schedule")

ax2.plot(steps, result.energy_trace)
ax2.set_ylabel("Energy")
ax2.set_xlabel("Adaptive step")
plt.tight_layout()
plt.show()
```

A correctly calibrated run shows:
- J⊥ increasing slowly near the middle of the schedule (critical point region, large χ_B)
- J⊥ increasing faster at the beginning and end (away from criticality, small χ_B)
- Energy decreasing monotonically as J⊥ increases toward `jp_end`

### 9.5 Troubleshooting optimal schedule

| Symptom | Cause | Fix |
|---------|-------|-----|
| J⊥ reaches end in very few steps | `eps_tilde` too large | Reduce `optimal_eps_tilde` by 5–10× |
| J⊥ barely moves after many steps | `eps_tilde` too small | Increase `optimal_eps_tilde` by 5–10× |
| Schedule is uniform (no slowdown near critical point) | Too few `replicas` or `sweeps_per_step` | Use `replicas≥4`, `sweeps_per_step≥10` |
| Energy doesn't improve vs standard schedule | `optimal_num_steps` too small | Increase so the run actually finishes |

## 10. Recommended Workflow for New Problems

1. Encode and sanity-check small instances with brute force.
2. Start with `method="sa"`, tuned `mode="fast"` for quick checks.
3. Move to `method="sqa"`, mode `"balanced"`.
4. Add `method="sqapt"` when landscapes are rugged or multimodal.
5. Try `schedule_type="optimal"` when you want the best quality for a given compute budget.
6. Track `best_energy`, traces, and run-to-run variance across reads.
7. Use `--fair` in benchmark scripts for time/quality comparisons.

## 11. Troubleshooting

- `ModuleNotFoundError: qanneal._qanneal`:
  - reinstall from the same repository root (`python -m pip install . --no-build-isolation`)
  - avoid mixing stale site-packages with source `PYTHONPATH` overlays
- `CTPIMCAnnealer is not available`:
  - reinstall qanneal from a build that includes current bindings
- OpenMP not detected on macOS:
  - install libomp and rebuild (Homebrew: `brew install libomp`)

## 12. Related docs

- `docs/api.md`
- `docs/optimal_schedule.md`
- `docs/ctpimc.md`
- `docs/sqa_trace_parameters.md`
- `docs/number_partition_parallel.md`
