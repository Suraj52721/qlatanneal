# qanneal Architecture Overview

How the pieces fit together, from problem encoding to result extraction.

---

## System Flow

```
User problem
    │
    ├─ QUBO(Q)          ─── .to_ising() ──────────┐
    ├─ DenseIsing(h,J)  ─────────────────────────────► Hamiltonian
    ├─ SparseIsing(h,E) ─────────────────────────────►  (energy/delta_energy)
    ├─ np.ndarray / dict / BQM / nx.Graph  ─ auto ──┘
    │
    ▼
solve(problem, method=...) ── schedule auto-tuned from problem
    │
    ├─ "sa"      ──► Annealer           ──► AnnealResult
    ├─ "sqa"     ──► SQAAnnealer        ──► SQAResult
    ├─ "sqapt"   ──► SQAPTAnnealer      ──► SQAParallelTemperingResult
    └─ "ctpimc"  ──► CTPIMCAnnealer     ──► CTPIMCResult
             │
             └── Observer (optional) ──► traces, diagnostics
```

---

## Core Components

### 1. Hamiltonian Models

| Class | Storage | `energy()` | `delta_energy()` |
|-------|---------|-----------|-----------------|
| `DenseIsing` | n×n J matrix | O(n²) | O(n) via local-field cache |
| `SparseIsing` | edge list + adjacency | O(\|E\|) | O(degree) |
| `QUBO` | n×n Q matrix | — | Converts to `DenseIsing` |

**Local-field caching**: `compute_local_fields()` pre-computes `fᵢ = hᵢ + Σⱼ Jᵢⱼ sⱼ`.
After each spin flip, only O(n) (dense) or O(degree) (sparse) updates are needed.
This is the key performance optimisation — O(n) per flip instead of O(n²).

**SIMD optimisation**: `DenseIsing::compute_local_fields()` uses a branch-free inner loop
that exposes the DAXPY pattern to compiler auto-vectorisers (AVX2, NEON).

### 2. Schedules

| Class | What it specifies |
|-------|-----------------|
| `AnnealSchedule` | β sequence (for SA, ReplicaAnnealer, ParallelTempering) |
| `SQASchedule` | (β, Γ) pairs (for SQA, SQAPT, CT-PIMC) |

**Auto-tuning pipeline** (`auto_schedule_sa_tuned`, `auto_schedule_sqa_tuned`):
1. Sample `probes=256` random spin flips on random states.
2. Compute 75th-percentile |ΔE| → characteristic scale Δ.
3. Derive β from acceptance targets: `β = −ln(p) / Δ`.
4. Derive Γ as a multiple of Δ, then decay geometrically.

### 3. Annealer Classes

#### `Annealer` — Classical SA
Single Metropolis chain. One spin flipped per proposal.
```
for β in schedule.betas:
    for sweep in sweeps_per_beta:
        for spin in range(n):
            ΔE = delta_energy(flip=spin)
            if ΔE ≤ 0 or random() < exp(−β × ΔE):
                apply_flip(spin)
```

#### `ReplicaAnnealer` — Parallel SA
`replicas` independent Annealer chains. OpenMP-parallelised across replicas.
Returns global best and per-replica results.

#### `ParallelTemperingAnnealer` — Classical PT
One replica per β in the ladder. After every `swap_interval` steps, adjacent replicas
propose to swap configurations with acceptance:
```
ΔS = (βᵢ − βⱼ)(Eⱼ − Eᵢ)
accept = ΔS ≤ 0  or  random() < exp(ΔS)
```

#### `SQAAnnealer` — Simulated Quantum Annealing

The Suzuki–Trotter decomposition maps the quantum problem to a classical one with M extra dimensions:

```
SQAState layout: [replica][slice][spin]   (contiguous buffer)

For each (β, Γ) step:
  1. Slice sweeps: single-spin Metropolis in each classical slice
     ΔE_classical = delta_energy(flip)
     ΔE_quantum   = delta_trotter(flip, slice)  using J_⊥ coupling

  2. Worldline sweeps: flip one spin THROUGH ALL slices
     ΔE = Σ_slices delta_energy(flip)  (no Trotter coupling cost!)

  3. Cluster sweeps: Swendsen-Wang on the time direction
     Bond activation probability: p = 1 - exp(−2 β J_⊥ δ_{sᵢ,t, sᵢ,t+1})
```

**Trotter coupling**: `J_⊥(β, Γ, M) = ½ ln(1 / tanh(βΓ/M))`
- Large Γ → strong J_⊥ → spins are correlated across slices → tunneling-like
- Γ → 0 → J_⊥ → ∞ → all slices must align → classical frozen state

#### `SQAParallelTemperingAnnealer` — SQAPT

R independent SQA replicas at different (βᵣ, Γᵣ) points. After each epoch, adjacent
replicas attempt swaps using the classical-action difference:
```
S_r = (β_r / M) × Σ_slices E_classical[slice] − J_⊥(β_r, Γ_r, M) × Σ_slices overlap[t, t+1]
accept swap(r, r+1) if ΔS ≤ 0 or random() < exp(ΔS)
```

#### `CTPIMCAnnealer` — Continuous-Time PIMC

Uses D-Wave's `localPIMC` (Swendsen-Wang cluster updates on worldlines).
No Trotter discretisation: worldlines are continuous curves in imaginary time.
At each schedule step, couplings are rescaled by β:
```
invTempJ_ij = β × J_ij
invTempH_i  = β × h_i
invTempGamma = β × Γ
```

### 4. Backend

`Backend` is an abstract interface isolating the algorithm from compute kernels:
- `CPUBackend` wraps `Hamiltonian` for CPU computation.
- Future: `CUDABackend` (stub in build system, kernels not yet wired).

### 5. Observers

Callbacks attached to the annealing loop for diagnostics:

| Observer | Records | Passed to |
|---------|---------|----------|
| `MetricsObserver` | energy, magnetization per step | `Annealer.run()` |
| `StateTraceObserver` | full State per sweep | `Annealer.run()` |
| `SQAMetricsObserver` | energy, magnetization per step | `SQAAnnealer.run()` |
| `SQAStateTraceObserver` | full SQAState per sweep + metadata | `SQAAnnealer.run()` |

All observers support a `stride` parameter to sample every N sweeps (reduces memory).

**Important**: Attaching a `SQASweepObserver` disables the OpenMP parallel path in `SQAAnnealer`.
For performance benchmarking, run without observer first, then attach for diagnostics.

---

## Data Structures

### `State`
```cpp
struct State {
    std::vector<int8_t> spins;   // ±1 values
    size_t size();
    static State random(size_t n, RNG& rng);
};
```

### `SQAState`
Stores the full Trotter path for all replicas:
```cpp
// Layout: [replica][slice][spin]  — one contiguous buffer
class SQAState {
    size_t index(size_t r, size_t t, size_t i);  // r*slices*spins + t*spins + i
    int8_t& at(size_t r, size_t t, size_t i);
    int8_t* slice_ptr(size_t r, size_t t);       // for fast sweep
    State slice_state(size_t r, size_t t);       // extract as State
};
```

### Result hierarchy
```
AnnealResult               ← Annealer
MultiAnnealResult          ← ReplicaAnnealer
  └── ReplicaResult[]
ParallelTemperingResult    ← ParallelTemperingAnnealer
SQAResult                  ← SQAAnnealer
SQAParallelTemperingResult ← SQAParallelTemperingAnnealer
CTPIMCResult               ← CTPIMCAnnealer
SolveResult                ← solve()   (Python wrapper)
```

---

## Python Layer

### `solve()` — Problem normalisation pipeline

```python
_normalize_problem(problem)
    ├── isinstance(problem, DenseIsing/SparseIsing) → passthrough
    ├── isinstance(problem, QUBO)               → .to_ising()
    ├── isinstance(problem, np.ndarray)          → QUBO(Q).to_ising()
    ├── isinstance(problem, dict/list)           → QUBO(entries, n).to_ising()
    ├── isinstance(problem, dimod.BQM)           → QUBO(bqm).to_ising()
    └── isinstance(problem, nx.Graph)            → _qubo_from_graph().to_ising()
```

### Schedule auto-selection in `solve()`
```
schedule is None:
    method="sa"     → auto_schedule_sa()
    method="sqa"    → auto_schedule_sqa()
    method="sqapt"  → auto_ladder_sqa_tuned(ising, replicas=max(2, replicas))
    method="ctpimc" → auto_schedule_sqa_tuned(ising, gamma_end_scale=0.04)
```

---

## Parallelism Model

```
┌─ Python process ─────────────────────────────────────┐
│  solve(reads=R)                                       │
│    for r in range(R):          ← sequential reads    │
│      annealer.run()                                   │
│        └─ SQAAnnealer (OpenMP) ─────────────────────►│
│             for replica in range(replicas):  ← OMP   │
│               for slice in range(slices):            │
│                 sweep()                              │
└───────────────────────────────────────────────────────┘

┌─ MPI / multiprocessing (hpc_sqa_launcher.py) ────────┐
│  rank/worker 0: seed=0    → R reads                  │
│  rank/worker 1: seed=R    → R reads                  │
│  ...                                                  │
│  rank/worker K: seed=K*R  → R reads                  │
│                                                       │
│  allreduce(min_energy) → global best                  │
└───────────────────────────────────────────────────────┘
```

---

## Repository Map

```
qlatannealv4/
├── include/qanneal/          C++ public headers
│   ├── hamiltonian.hpp       Abstract base class
│   ├── dense_ising.hpp       DenseIsing
│   ├── sparse_ising.hpp      SparseIsing, SparseEdge
│   ├── qubo.hpp              QUBO
│   ├── schedule.hpp          AnnealSchedule
│   ├── sqa_schedule.hpp      SQASchedule
│   ├── annealer.hpp          Annealer + AnnealResult
│   ├── replica_annealer.hpp  ReplicaAnnealer
│   ├── parallel_tempering.hpp
│   ├── sqa_annealer.hpp      SQAAnnealer + SQAResult
│   ├── sqa_parallel_tempering.hpp  SQAParallelTemperingAnnealer
│   ├── ctpimc_annealer.hpp   CTPIMCAnnealer
│   ├── metrics_observer.hpp  All observer classes
│   ├── sqa_observer.hpp      SQA observer interfaces
│   ├── backend.hpp           Backend abstraction
│   ├── state.hpp             State
│   ├── sqa_state.hpp         SQAState
│   ├── metrics.hpp           magnetization(), overlap()
│   └── mpi/                  MPI distributed anneal
│       ├── mpi_context.hpp   MPIContext RAII wrapper
│       └── mpi_replica.hpp   run_replica_anneal()
│
├── src/                      C++ implementations
│   ├── annealer.cpp
│   ├── dense_ising.cpp       SIMD-friendly local field ops
│   ├── sparse_ising.cpp      Adjacency-list operations
│   ├── qubo.cpp              QUBO→Ising conversion
│   ├── replica_annealer.cpp  OpenMP parallel
│   ├── parallel_tempering.cpp
│   ├── sqa_annealer.cpp      Trotter sweeps
│   ├── sqa_parallel_tempering.cpp
│   ├── ctpimc_annealer.cpp   D-Wave localPIMC wrapper
│   └── mpi_replica.cpp       MPI allreduce
│
├── python/
│   ├── bindings.cpp          pybind11 bindings
│   └── qanneal/
│       ├── __init__.py       Public API exports
│       ├── solver.py         solve(), schedule helpers
│       └── gui.py            Graph editor
│
├── notebooks/                Jupyter examples
│   ├── 01_quickstart.ipynb
│   ├── 02_sqa_physics.ipynb
│   ├── 03_sqapt_and_ctpimc.ipynb
│   └── 04_large_problems.ipynb
│
├── examples/
│   ├── python/               Runnable scripts + benchmarks
│   └── mpi/                  C++ MPI example
│
├── scripts/slurm/            SLURM job scripts
│   ├── run_sa_mpi.sh
│   ├── run_sqa_hpc.sh        Multi-node MPI
│   └── run_sqa_array.sh      Job array (no mpi4py)
│
├── tests/                    C++ unit tests
├── docs/                     Documentation
└── CMakeLists.txt
```

---

## Extension Points

- **New Hamiltonians**: Subclass `Hamiltonian` and implement `energy()`, `delta_energy()`,
  `compute_local_fields()`, `update_local_fields_after_flip()`.
- **New backends**: Implement the `Backend` interface (e.g., GPU kernel).
- **New update rules**: Extend `SQAAnnealer::run()` with new sweep types.
- **New observers**: Inherit from `Observer`/`SQAObserver` and implement `record()`.
- **New problem formats**: Add a branch to `_normalize_problem()` in `solver.py`.
