# qanneal

**Research-grade Ising/QUBO optimizer** — version 0.6.1

Four annealing engines from classical SA to the closest available CPU simulation of quantum annealing, plus a theoretically-grounded **optimal adaptive J⊥ schedule** derived from the local adiabaticity condition.

| Method | Short name | What it simulates |
|--------|-----------|-------------------|
| Simulated Annealing | `sa` | Classical thermal fluctuations |
| Simulated Quantum Annealing | `sqa` | Discrete-time path-integral (Trotterized QA) |
| SQA + Parallel Tempering | `sqapt` | QA with replica exchange on a (β, Γ) ladder |
| Continuous-Time PIMC | `ctpimc` | Continuous-time path-integral (worldline sampling) |

All engines share a unified `solve()` Python API and a C++17 core with pybind11 bindings.

---

## Install

```bash
# From repo root (recommended)
python -m pip install . --no-build-isolation

# Editable/development
python -m pip install -e . --no-build-isolation

# Convenience scripts
./setup.sh        # macOS / Linux
setup.bat         # Windows cmd
./setup.ps1       # Windows PowerShell
```

Requires: Python ≥ 3.11, numpy, C++17 compiler.
Optional: OpenMP (parallel replicas), MPI (multi-node), matplotlib (plots).

---

## Quickstart

### Solve a QUBO in one call

```python
import numpy as np
from qanneal import solve

# Encode a 3-variable QUBO: minimise x₀ + x₂ − 2x₀x₁ − 2x₁x₂
Q = np.array([
    [ 1.0, -2.0,  0.0],
    [-2.0,  0.0, -2.0],
    [ 0.0, -2.0,  1.0],
], dtype=float)

result = solve(Q, method="sqapt", reads=16, seed=0, return_bits=True)
print(result.best_sample)   # array of bits {0, 1}
print(result.best_energy)   # minimum QUBO energy found
```

### Number partition (classic NP-hard)

```python
import numpy as np
from qanneal import DenseIsing, solve

# Partition numbers [3, 5, 7, 11, 13] into two groups with equal sum
nums = np.array([3, 5, 7, 11, 13], dtype=float)
n = len(nums)

# Ising encoding: spin +1 → group A, spin -1 → group B
h = np.zeros(n)
J = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        J[i, j] = J[j, i] = 2.0 * nums[i] * nums[j]

ising = DenseIsing(h, J, c=float(np.dot(nums, nums)))
result = solve(ising, method="sqa", reads=20, seed=42)
spins = result.best_sample        # +1 or -1
diff  = abs(float(np.dot(nums, spins)))
print(f"Partition diff: {diff}")  # 0.0 = perfect split
```

---

## The Four Methods — When to Use Which

### `sa` — Simulated Annealing
Classical Metropolis algorithm with a temperature schedule.
- **Use when**: fast results, simple landscapes, baseline comparisons.
- **Control**: `sweeps_per_beta` (thermal equilibration per temperature step).

### `sqa` — Simulated Quantum Annealing
Quantum Monte Carlo in the Suzuki–Trotter formulation. The spin system is replicated
across `trotter_slices` imaginary-time slices; flips along the time dimension mimic
quantum tunneling through energy barriers.
- **Use when**: landscapes have tall narrow barriers that classical SA misses.
- **Key extra controls**: `trotter_slices`, `worldline_sweeps`, `cluster_sweeps`.

### `sqapt` — SQA + Parallel Tempering *(recommended default)*
Multiple SQA replicas run simultaneously at different `(β, Γ)` points on a ladder.
Adjacent replicas periodically swap configurations, letting solutions discovered at
high fluctuations (large Γ, low β) flow toward low-energy states at strong freezing.
- **Use when**: rugged/multi-modal landscapes, moderate to hard combinatorial problems.
- **Key extra controls**: `replicas` (ladder length), `pt_steps`, `swap_interval`.

### `ctpimc` — Continuous-Time PIMC
Samples worldlines in continuous imaginary time using Swendsen–Wang cluster updates.
No Trotter discretisation error; better mixing in the high-Γ regime.
- **Use when**: you want a closer analog to D-Wave sampling; density-matrix–level statistics.
- **Key extra controls**: `ctpimc_qubits_per_update`, `ctpimc_qubits_per_chain`.

---

## Core Concepts

### Energy conventions

**QUBO** (binary variables x ∈ {0, 1}):
```
E(x) = Σᵢ Σⱼ Qᵢⱼ xᵢ xⱼ
```
- `Q[i,i]`: linear (bias) term for variable i.
- `Q[i,j]` (i ≠ j): quadratic coupling. Include both Q[i,j] and Q[j,i] for symmetric problems.

**Ising** (spins s ∈ {−1, +1}):
```
E(s) = Σᵢ hᵢ sᵢ  +  Σᵢ<ⱼ Jᵢⱼ sᵢ sⱼ  +  c
```
- `h`: local magnetic fields.
- `J`: pairwise coupling matrix.
- `c`: constant energy offset (irrelevant for optimization, useful for energy tracking).

**Conversion**: `x = (s + 1) / 2` — QUBO uses bits, Ising uses spins.
`solve(..., return_bits=True)` converts the result for you.

### Hamiltonian models

| Class | Memory | Best for |
|-------|--------|---------|
| `DenseIsing(h, J, c)` | O(n²) | n ≤ ~3 000, fully connected |
| `SparseIsing(h, edges, n, c)` | O(n + \|E\|) | sparse graphs, n up to 100 000+ |
| `QUBO(Q)` | O(n²) | binary variables; call `.to_ising()` |

---

## Optimal Adaptive Schedule *(calibrated since 0.5.0; SQA two-phase since 0.6.1)*

The optimal schedule paces the transverse-field ramp using the bond susceptibility
`χ_B = Var(B)`, a proxy for how close the Suzuki–Trotter path integral is to its quantum
critical point. It implements the **local adiabaticity condition** (Roland & Cerf, 2002)
for the path integral: move J⊥ slowly where the system is critical, quickly where it is not.

**χ_B is large near the quantum transition and collapses toward zero in the ordered phase**
(it does *not* diverge). The schedule therefore *dwells* in the low-J⊥ critical region and
crosses the ordered region quickly — a fast-slow-fast trajectory.

### How it works (budget-calibrated, the default)

Pass `optimal_eps_tilde=0.0` (the default) to request **per-instance budget calibration**. A
short pilot pre-pass measures `χ_B(J)` at a grid of J⊥ points, then the run follows the
inverse-CDF of `∫ χ_B(J)^α dJ`, which guarantees — by construction, independent of the χ_B
shape — that the schedule (a) consumes exactly `optimal_num_steps`, (b) reaches `j_perp_end`,
and (c) dwells where χ_B is large. This replaces the old fixed-`ε̃` update, which could
"freeze" (barely move J⊥) when χ_B was large.

- `α` (`optimal_alpha`) — universality-class exponent, default `15/14` (1-D quantum Ising).
- `j_perp_end` — auto-set to `max(j_perp_start, 5·j_rms)` from the coupling scale.
- All schedule parameters adapt to *your* problem's coupling scale; nothing is hard-coded.

**SQA runs a two-phase protocol** (`optimal_beta_ramp_fraction`, default 0.3): phase 1 ramps
β from `beta_ramp_start` up to a cold `β = max(4/j_rms, 1)` at fixed J⊥ (thermal anneal),
then phase 2 fixes β and runs the calibrated adaptive J⊥ ramp. Without phase 1, single-
temperature SQA has no thermal annealing and underperforms.

> **When does it help?** The advantage appears for genuinely hard, frustrated Ising problems
> whose coupling scale puts them near a quantum transition (`j_rms ≳ j_perp_start ≈ 2.9`). Use
> `scripts/sanity_schedule.py` to check a new instance is in the favorable regime first.

### Quickstart

```python
from qanneal import solve

result = solve(
    problem,
    method="sqa",             # or "sqapt"
    schedule_type="optimal",
    reads=15,
    replicas=4,
    optimal_eps_tilde=0.0,    # 0.0 = per-instance budget calibration (recommended)
    optimal_num_steps=150,
    # optional overrides — all auto-computed from the problem scale if omitted:
    # optimal_j_perp_end=25.0,          # default max(j_perp_start, 5*j_rms)
    # optimal_beta_ramp_fraction=0.3,   # SQA two-phase split
    # optimal_alpha=15/14,
    # optimal_debug_csv="schedule.csv", # per-step J_perp / chi_B diagnostics
)
print(result.best_energy, result.best_sample)
```

### SQAPT + optimal schedule

```python
result = solve(
    problem,
    method="sqapt",
    schedule_type="optimal",
    replicas=8,
    reads=15,
    trotter_slices=32,
    worldline_sweeps=3,
    cluster_sweeps=0,         # >0 adds Swendsen–Wang cluster updates (SQAPT-SW)
    swap_interval=1,
    optimal_eps_tilde=0.0,    # calibrated
    optimal_num_steps=150,
)
```

### J⊥ schedule helpers

```python
from qanneal import j_perp_from_beta_gamma, optimal_j_perp_params, j_rms_from_problem

jp = j_perp_from_beta_gamma(beta=3.0, gamma=0.5, trotter_slices=32)  # 0.5*ln(1/tanh(βγ/M))
beta, jp_start, jp_end = optimal_j_perp_params(problem, trotter_slices=32)  # β=max(4/j_rms,1)
jr = j_rms_from_problem(problem)   # RMS of the present couplings (ndarray/DenseIsing/BQM/graph)
```

### Direct C++ / low-level API

```python
from qanneal import SQAAnnealer, SQASchedule

dummy = SQASchedule.from_vectors([1.0], [1.0])  # only used for construction
ann = SQAAnnealer(ising, dummy, trotter_slices=32, replicas=4)
result = ann.run_optimal(
    beta=1.0,                # phase-2 (cold) inverse temperature
    j_perp_start=0.1, j_perp_end=25.0,
    eps_tilde=0.0,           # <=0 -> budget calibration
    alpha=15/14, num_steps=150, sweeps_per_step=20,
    worldline_sweeps=3, cluster_sweeps=0,
    calib_probes=12, calib_sweeps=10,   # pilot pre-pass grid
    beta_ramp_fraction=0.3,             # two-phase thermal ramp
)
# result.j_perp_trace, result.chi_B_trace — per-step schedule diagnostics
# result.calibrated_eps_tilde, result.final_j_perp — calibration read-outs
# result.best_state, result.best_energy
```

See `docs/optimal_schedule.md` for the full derivation, calibration math, and benchmark
methodology, and `benchmarks/schedule/` for the reproducible study.

---

## Schedule Design

Every annealer needs a **schedule** — a sequence of `(β, Γ)` or just `β` values.

### Problem-adaptive schedules *(recommended)*

```python
from qanneal import auto_schedule_sa_tuned, auto_schedule_sqa_tuned, auto_ladder_sqa_tuned

# For SA
schedule = auto_schedule_sa_tuned(ising, mode="balanced")

# For SQA / CT-PIMC
schedule = auto_schedule_sqa_tuned(ising, mode="balanced")

# For SQAPT — returns a (β, Γ) ladder with `replicas` points
ladder = auto_ladder_sqa_tuned(ising, replicas=8, mode="balanced")
```

**How tuning works**: The helper probes `~256` random single-spin flips, computes the
75th-percentile |ΔE|, and derives `β_start`/`β_end` from physical acceptance-target
heuristics. This avoids manual tuning across problem sizes and coupling scales.

| `mode` | Steps | Exploration | Use when |
|--------|-------|------------|---------|
| `"fast"` | 30–40 | High | Prototyping, large sweeps |
| `"balanced"` | 60–80 | Medium | Default production |
| `"accurate"` | 110–130 | Low | Best solution quality |

### Fixed schedules *(legacy)*

```python
from qanneal import AnnealSchedule, SQASchedule
import numpy as np

# SA: linear β ramp
schedule = AnnealSchedule.linear(beta_start=0.1, beta_end=5.0, steps=60)

# SQA: paired (β, Γ) — geometric decay for Γ
schedule = SQASchedule.from_vectors(
    betas=np.linspace(0.1, 5.0, 60).tolist(),
    gammas=np.geomspace(5.0, 0.01, 60).tolist(),
)
```

---

## Parameter Reference

### Common to all methods

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `reads` | 1 | Independent runs; best over all reads is returned |
| `sweeps_per_beta` | 20 | Metropolis sweeps per temperature step |
| `schedule` | auto | Schedule object; auto-selected per method if None |
| `seed` | None | RNG seed for reproducibility |
| `progress` | True | Show tqdm progress bar |
| `backend` | `"cpu"` | Compute backend |
| `return_bits` | False | Return {0,1} bits instead of {−1,+1} spins |

### SQA and SQAPT (`method="sqa"` or `"sqapt"`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `trotter_slices` | 32 | Number of imaginary-time slices. More slices → less Trotter error, more memory/time |
| `replicas` | 1 | Parallel SQA chains (SQA) or PT ladder rungs (SQAPT) |
| `worldline_sweeps` | 5 | Flips one spin through all time-slices at once (improves mixing) |
| `cluster_sweeps` | 0 | Swendsen–Wang cluster updates along imaginary time |
| `continuous_time_slices` | 0 | If > 0, approximates continuous-time limit within SQA |

### SQAPT only (`method="sqapt"`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `pt_steps` | 50 | Number of local-update + swap epochs |
| `swap_interval` | 1 | Attempt replica swap every N steps |
| `pt_betas` | None | Explicit β ladder (overrides schedule) |
| `pt_gammas` | None | Explicit Γ ladder (must pair with pt_betas) |

### CT-PIMC only (`method="ctpimc"`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ctpimc_qubits_per_update` | 1 | Spins updated per cluster proposal |
| `ctpimc_qubits_per_chain` | 1 | Chain length for multi-qubit proposals |

---

## Accessing Results

```python
result = solve(problem, method="sqapt", reads=32, ...)

result.best_sample    # np.ndarray, shape (n,), dtype int — best spin/bit configuration
result.best_energy    # float — lowest energy found across all reads
result.samples        # list of n arrays — one per read
result.energies       # list of floats — one per read
result.trace          # list of floats — energy trace from the first read
result.var_order      # variable ordering (relevant for BQM/graph inputs)

# Convert spins to bits manually
bits = ((result.best_sample + 1) // 2).astype(int)
```

### Observers (low-level diagnostics)

```python
from qanneal import Annealer, MetricsObserver, AnnealSchedule
import numpy as np

ising = ...
schedule = AnnealSchedule.linear(0.1, 5.0, 60)
obs = MetricsObserver()
ann = Annealer(ising, schedule)
res = ann.run(sweeps_per_beta=40, observer=obs)

import matplotlib.pyplot as plt
plt.plot(obs.energy_trace)
plt.xlabel("Temperature step"); plt.ylabel("Energy"); plt.show()
```

SQA observers (`SQAMetricsObserver`, `SQAStateTraceObserver`) capture per-sweep replica/slice
state snapshots — see `docs/sqa_trace_parameters.md` for the full field list.

---

## Parallelism

### OpenMP (automatic, within a run)

Enabled by default (`QANNEAL_ENABLE_OPENMP=ON`). Parallelizes:
- `ReplicaAnnealer` — across replicas
- `SQAAnnealer` — across replicas × slices (when no sweep-level observer is attached)
- `SQAParallelTemperingAnnealer` — across ladder replicas

Control at runtime:
```bash
export OMP_NUM_THREADS=8
```

### Multi-process / multi-node

For large problems or many reads, use the HPC launcher:

```bash
# Single node, multiprocessing (no extra deps)
python examples/python/hpc_sqa_launcher.py \
  --n 5000 --method sqapt --reads 64 --workers 8

# Multi-node MPI (requires mpi4py)
mpirun -n 32 python examples/python/hpc_sqa_launcher.py \
  --n 50000 --method sqa --reads 8 --mpi

# SLURM job array (most portable, no mpi4py needed)
sbatch scripts/slurm/run_sqa_array.sh
python examples/python/merge_array_results.py results/run_*.json
```

### C++ MPI (build-time)

```bash
cmake -S . -B build -DQANNEAL_ENABLE_MPI=ON
cmake --build build
mpirun -n 8 build/qanneal_mpi_example
```

---

## Notebooks

Interactive Jupyter notebooks in `notebooks/`:

| Notebook | What you'll learn |
|----------|------------------|
| `01_quickstart.ipynb` | First problem, SA vs SQA comparison |
| `02_sqa_physics.ipynb` | Trotter slices, Γ schedules, worldline sweeps |
| `03_sqapt_and_ctpimc.ipynb` | Replica exchange, (β,Γ) ladders, CT-PIMC |
| `04_large_problems.ipynb` | SparseIsing, Chimera/MaxCut, HPC scaling |

---

## Examples

```bash
# Tuned SQAPT demo
python examples/python/tuned_sqapt_demo.py --mode balanced

# Number partition benchmark (SA / SQA / SQAPT / CT-PIMC vs D-Wave SDK)
python examples/python/number_partition_benchmark.py \
  --with-sqapt --with-ctpimc --schedule-mode balanced --jobs 8

# Full comparative benchmark vs D-Wave
python examples/python/dwave_comparative_benchmark.py --n 40 --reads 16

# HPC scaling benchmark
python examples/python/hpc_sqa_launcher.py --scaling --method sqapt --mode fast

# Interactive graph problem editor
python examples/python/graph_editor_gui.py
```

---

## Documentation

| File | Content |
|------|---------|
| `docs/api.md` | Complete API reference (all classes, parameters, return types) |
| `docs/optimal_schedule.md` | Optimal adaptive J⊥ schedule — physics, algorithm, and parameter guide |
| `docs/overview.md` | Architecture, data flow, component map |
| `docs/user_guide.md` | Step-by-step usage guide with physical intuition |
| `docs/ctpimc.md` | CT-PIMC algorithm details |
| `docs/sqa_trace_parameters.md` | SQA observer field guide |

---

## Build Options

```bash
cmake -S . -B build \
  -DQANNEAL_ENABLE_OPENMP=ON   \  # parallel loops (default ON)
  -DQANNEAL_ENABLE_MPI=OFF     \  # distributed anneal (default OFF)
  -DQANNEAL_BUILD_TESTS=ON     \  # unit tests
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

---

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
The CT-PIMC engine uses D-Wave's localPIMC (Apache-2.0), credited in `NOTICE`.
