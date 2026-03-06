# qanneal API Reference

Complete reference for every public class, function, and parameter.

---

## Table of Contents

1. [Model Classes](#model-classes)
2. [Schedule Classes](#schedule-classes)
3. [Schedule Helpers](#schedule-helpers)
4. [Annealer Classes](#annealer-classes)
5. [Result Classes](#result-classes)
6. [Observer Classes](#observer-classes)
7. [High-Level Solver](#high-level-solver)
8. [Utility Functions](#utility-functions)
9. [C++ API Summary](#c-api-summary)

---

## Model Classes

All model classes implement the `Hamiltonian` interface and are accepted
wherever a `problem` argument is required.

---

### `DenseIsing(h, J, c=0.0)`

Fully-connected Ising model stored as a dense n×n coupling matrix.

**Energy**: `E(s) = Σᵢ hᵢ sᵢ + Σᵢ<ⱼ Jᵢⱼ sᵢ sⱼ + c`

**Parameters**

| Argument | Type | Description |
|----------|------|-------------|
| `h` | `np.ndarray` shape `(n,)` | Local magnetic fields (longitudinal). Positive → spin prefers −1. |
| `J` | `np.ndarray` shape `(n, n)` | Pairwise coupling matrix. Only upper triangle i<j is used. Set symmetric: J[i,j]=J[j,i]. Ferromagnetic < 0, antiferromagnetic > 0. |
| `c` | `float` | Constant energy offset (default 0.0). Does not affect optimization; useful for computing exact QUBO energies. |

**Methods**

| Method | Returns | Description |
|--------|---------|-------------|
| `size()` | `int` | Number of spins n |
| `energy(spins)` | `float` | Total energy for a spin list {−1,+1} |
| `delta_energy(spins, flip)` | `float` | Energy change if spin at index `flip` is flipped |

**Complexity**: `energy()` O(n²), `delta_energy()` O(n) using cached local fields.

**Example**
```python
import numpy as np
from qanneal import DenseIsing

h = np.array([0.5, -0.3, 0.0])
J = np.array([[0.0,  1.2, -0.4],
              [1.2,  0.0,  0.8],
              [-0.4, 0.8,  0.0]])
ising = DenseIsing(h, J, c=0.0)
print(ising.size())             # 3
print(ising.energy([1, -1, 1])) # compute energy
```

---

### `SparseIsing(h, edges, n, c=0.0)`

Sparse Ising model stored as an edge list. Scales to n ≥ 100 000 spins.

**Energy**: Same as DenseIsing but only edges in the edge list contribute.

**Parameters**

| Argument | Type | Description |
|----------|------|-------------|
| `h` | `np.ndarray` shape `(n,)` | Local fields |
| `edges` | `list[SparseEdge]` | Edge list. Each entry is a `SparseEdge(i, j, value)`. |
| `n` | `int` | Total number of spins |
| `c` | `float` | Constant offset |

**Methods**: Same as `DenseIsing`.

**Complexity**: `energy()` O(|E|), `delta_energy()` O(degree).

**Example**
```python
from qanneal import SparseIsing, SparseEdge
import numpy as np

n = 5
h = np.zeros(n)
edges = [
    SparseEdge(0, 1,  1.0),
    SparseEdge(1, 2, -0.5),
    SparseEdge(2, 3,  0.8),
    SparseEdge(3, 4,  1.2),
]
ising = SparseIsing(h, edges, n)
```

---

### `SparseEdge(i, j, value)`

Simple struct holding one edge of a `SparseIsing` model.

| Field | Type | Description |
|-------|------|-------------|
| `i` | `int` | First spin index (0-based) |
| `j` | `int` | Second spin index (0-based) |
| `value` | `float` | Coupling J_{ij} |

Convention: only one direction needed (i < j); the model symmetrizes internally.

---

### `QUBO(Q)` / `QUBO(entries, n)` / `QUBO(bqm)`

Quadratic unconstrained binary optimization model.

**Energy**: `E(x) = Σᵢ Σⱼ Qᵢⱼ xᵢ xⱼ`,  x ∈ {0, 1}

**Constructors**

| Signature | Description |
|-----------|-------------|
| `QUBO(Q: np.ndarray)` | Dense n×n matrix. `Q[i,i]` = linear term, `Q[i,j]` = coupling. |
| `QUBO(entries: list[tuple[int,int,float]], n: int)` | Sparse list of `(i, j, value)` triples. |
| `QUBO(entries: dict[tuple[int,int], float], n: int)` | Dict keyed by `(i, j)` pairs. |
| `QUBO(bqm: dimod.BinaryQuadraticModel)` | Accepts dimod BINARY or SPIN BQM (auto-converted). |

**Methods**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_ising()` | `DenseIsing` | Convert to Ising. Symmetrizes Q and applies the exact substitution x=(s+1)/2. |
| `size()` | `int` | Number of binary variables |

**QUBO→Ising conversion** (exact, no approximation):
```
hᵢ   = ½ Σⱼ (Qᵢⱼ + Qⱼᵢ) / 2
Jᵢⱼ  = (Qᵢⱼ + Qⱼᵢ) / 4          (for i ≠ j)
c    = Σᵢ Qᵢᵢ/4 + Σᵢⱼ Qᵢⱼ/4
```

**Example**
```python
import numpy as np
from qanneal import QUBO

Q = np.array([[1.0, -2.0], [-2.0, 3.0]])
qubo = QUBO(Q)
ising = qubo.to_ising()          # DenseIsing
bits = [0, 1]
print(qubo.size())               # 2
```

---

## Schedule Classes

---

### `AnnealSchedule`

Classical annealing schedule — a sequence of inverse temperatures β.

**Constructor (static factory)**

```python
AnnealSchedule.linear(beta_start, beta_end, steps)
```

| Argument | Type | Description |
|----------|------|-------------|
| `beta_start` | `float` | Starting β (hot, high acceptance). Typically 0.1–1.0. |
| `beta_end` | `float` | Final β (cold, strong rejection). Typically 2.0–20.0 depending on coupling scale. |
| `steps` | `int` | Number of temperature steps. Each step runs `sweeps_per_beta` Metropolis sweeps. |

**Attributes**
- `betas`: `list[float]` — the β sequence

**Physical meaning**: β = 1/(k_B T). High β = low temperature = frozen state.
The schedule should start hot enough that nearly all moves are accepted, and end cold
enough that virtually no uphill moves are accepted.

---

### `SQASchedule`

Quantum annealing schedule — paired sequences of (β, Γ) values.

```python
SQASchedule.from_vectors(betas: list[float], gammas: list[float])
```

| Argument | Description |
|----------|-------------|
| `betas` | Inverse temperature ramp. Same interpretation as classical SA. |
| `gammas` | Transverse field strength ramp. Should **start high** (strong quantum fluctuations) and **decrease** to near zero (classical limit). Geometric decay is recommended. |

**Attributes**
- `betas`: `list[float]`
- `gammas`: `list[float]`

**Physical meaning**: Γ controls quantum tunneling. Large Γ lets the system tunnel
through classical energy barriers. As Γ → 0 the system freezes into a classical spin
configuration. The optimal annealing path follows the quantum critical point.

**Recommended gamma decay**: geometric (`np.geomspace`), not linear. Linear decay
wastes most schedule steps at ineffective mid-Γ values.

```python
import numpy as np
from qanneal import SQASchedule

schedule = SQASchedule.from_vectors(
    betas  = np.linspace(0.1, 5.0, 80).tolist(),
    gammas = np.geomspace(5.0, 0.01, 80).tolist(),  # geometric!
)
```

---

## Schedule Helpers

These functions build schedules automatically from a problem instance.

---

### `auto_schedule_sa(steps=50, beta_start=0.1, beta_end=4.0)`

Fixed linear β schedule. Simple starting point.

Returns: `AnnealSchedule`

---

### `auto_schedule_sqa(steps=50, beta_start=0.1, beta_end=4.0, gamma_start=5.0, gamma_end=0.01)`

Fixed (β, Γ) schedule with geometric gamma decay.

Returns: `SQASchedule`

---

### `auto_schedule_sa_tuned(problem, mode="balanced", steps=None, probes=256, seed=1234, n=None, beta_end_scale=1.0)`

**Problem-adaptive SA schedule.** Estimates the energy scale from the problem and
derives β_start/β_end from physical acceptance targets.

**How it works**:
1. Probe `probes` random single-spin flips on random states.
2. Compute the 75th percentile of |ΔE| → this is the characteristic energy scale `Δ`.
3. Set `β_start = −ln(p_start) / Δ` where `p_start` is the target acceptance at the hot end.
4. Set `β_end   = −ln(p_end)   / Δ` where `p_end` is the target acceptance at the cold end.

| `mode` | `p_start` | `p_end` | `steps` | Use case |
|--------|-----------|---------|---------|---------|
| `fast` | 0.85 | 0.22 | 30 + 6√n | Prototyping |
| `balanced` | 0.88 | 0.10 | 60 + 6√n | Production |
| `accurate` | 0.90 | 0.04 | 110 + 6√n | Best quality |

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `problem` | — | Any supported problem type (DenseIsing, SparseIsing, QUBO, array, dict, BQM, graph) |
| `mode` | `"balanced"` | Quality preset |
| `steps` | None | Override step count (None = auto from mode) |
| `probes` | 256 | Number of random flip samples for energy scale estimation |
| `seed` | 1234 | RNG seed for probe sampling |
| `beta_end_scale` | 1.0 | Multiplicative scale on β_end (>1 = colder final state) |

Returns: `AnnealSchedule`

---

### `auto_schedule_sqa_tuned(problem, mode="balanced", steps=None, probes=256, seed=1234, n=None, beta_end_scale=1.0, gamma_end_scale=1.0)`

**Problem-adaptive SQA schedule.** Like `auto_schedule_sa_tuned` but also calibrates Γ.

- `gamma_start = gamma_mult × Δ` where gamma_mult ∈ {2.0, 3.0, 4.5} for fast/balanced/accurate.
- `gamma_end   = gamma_start × gamma_final_ratio × gamma_end_scale`
- gammas are spaced **geometrically** (not linearly).

**Extra parameter**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma_end_scale` | 1.0 | Multiplicative scale on γ_end. Set to 0.04 for CT-PIMC (needs γ→0 for clean classical projection). |

Returns: `SQASchedule`

---

### `auto_ladder_sqa_tuned(problem, replicas=8, mode="balanced", probes=256, seed=1234, n=None)`

**Problem-adaptive SQAPT ladder.** Builds a `(β, Γ)` ladder for parallel tempering.
Each rung of the ladder is a different (β, Γ) point; adjacent rungs can swap configurations.

- β ladder: geometric from β_min (hot) to β_max (cold).
- Γ ladder: **inverse** geometric — high Γ at low β, low Γ at high β. This ensures each rung is physically meaningful.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `replicas` | 8 | Number of ladder rungs (PT replicas). More rungs → better coverage but more memory. |
| `mode` | `"balanced"` | Sets β_max and gamma_max from problem scale. |

Returns: `SQASchedule` (betas and gammas are the ladder points, not a time sequence)

**Example**
```python
from qanneal import auto_ladder_sqa_tuned, solve

ladder = auto_ladder_sqa_tuned(ising, replicas=8, mode="balanced")
result = solve(ising, method="sqapt", replicas=8, schedule=ladder, reads=16)
```

---

## Annealer Classes

Low-level API. Most users should use `solve()` instead.

---

### `Annealer(hamiltonian, schedule, backend="cpu")`

Classical single-chain simulated annealing.

**Algorithm**: For each β in schedule, run `sweeps_per_beta` Metropolis sweeps.
Each sweep proposes random spin flips and accepts with probability min(1, exp(−β ΔE)).

```python
ann = Annealer(ising, schedule)
ann.set_seed(42)
result = ann.run(sweeps_per_beta=50, observer=None)
```

**`run(sweeps_per_beta, observer=None)` → `AnnealResult`**

| Parameter | Description |
|-----------|-------------|
| `sweeps_per_beta` | Metropolis sweeps per temperature step |
| `observer` | Optional `MetricsObserver` or `StateTraceObserver` |

---

### `ReplicaAnnealer(hamiltonian, schedule, replicas, backend="cpu")`

Multiple independent SA chains run in parallel (OpenMP).

```python
ann = ReplicaAnnealer(ising, schedule, replicas=16)
result = ann.run(sweeps_per_beta=50)
result.global_best_energy   # best across all replicas
result.replicas             # list of per-replica ReplicaResult
```

**`run(sweeps_per_beta)` → `MultiAnnealResult`**

---

### `ParallelTemperingAnnealer(hamiltonian, betas, backend="cpu")`

Classical replica exchange on a β ladder.

```python
import numpy as np
betas = np.linspace(0.1, 5.0, 8).tolist()
ann = ParallelTemperingAnnealer(ising, betas)
result = ann.run(sweeps_per_step=50, steps=100, swap_interval=1)
result.swap_acceptance_trace   # fraction of accepted swaps per step
```

**`run(sweeps_per_step, steps, swap_interval=1)` → `ParallelTemperingResult`**

---

### `SQAAnnealer(hamiltonian, schedule, trotter_slices, replicas=1, backend="cpu")`

Simulated quantum annealing via Suzuki–Trotter decomposition.

The system is replicated into `trotter_slices` imaginary-time slices. Metropolis flips
are proposed both within slices (classical Ising energy) and along the imaginary-time
direction (Trotter coupling = quantum tunneling).

**Trotter coupling strength**: `J_⊥ = ½ ln(1 / tanh(βΓ/M))` where M = trotter_slices.

```python
ann = SQAAnnealer(ising, schedule, trotter_slices=32, replicas=4)
result = ann.run(sweeps_per_beta=60, worldline_sweeps=4, cluster_sweeps=1)
result.best_state.spins    # best spin configuration found
result.energy_trace        # energy at each temperature step
```

**`run(sweeps_per_beta, worldline_sweeps, cluster_sweeps=0, continuous_time_slices=0, observer=None)` → `SQAResult`**

| Parameter | Description |
|-----------|-------------|
| `sweeps_per_beta` | Single-spin Metropolis sweeps per temperature step |
| `worldline_sweeps` | Worldline (imaginary-time direction) sweeps — flip one spin through all slices |
| `cluster_sweeps` | Swendsen–Wang cluster updates along imaginary time |
| `continuous_time_slices` | Overrides `trotter_slices` with a larger value to approximate continuous time |
| `observer` | Optional `SQAMetricsObserver` or `SQAStateTraceObserver` |

**Tuning guide**:
- Start with `trotter_slices=16-32`, `worldline_sweeps=3-5`.
- Add `cluster_sweeps=1` if convergence is slow on strongly frustrated instances.
- Increase `trotter_slices` if you observe Trotter-error artifacts (energy mismatch between slice energies).

---

### `SQAParallelTemperingAnnealer(hamiltonian, betas, gammas, trotter_slices, backend="cpu")`

SQA with replica exchange on a (β, Γ) ladder — the closest CPU simulation of a quantum annealer.

Each replica is an independent SQA system at a different (β, Γ) point. Adjacent replicas
periodically swap configurations using a quantum action-based acceptance criterion.

```python
ladder = auto_ladder_sqa_tuned(ising, replicas=8, mode="balanced")
ann = SQAParallelTemperingAnnealer(
    ising,
    ladder.betas,
    ladder.gammas,
    trotter_slices=24,
)
result = ann.run(
    sweeps_per_step=50,
    worldline_sweeps=4,
    steps=80,
    swap_interval=1,
    cluster_sweeps=1,
)
result.swap_acceptance_trace   # good swap acceptance is 0.2–0.5
```

**`run(sweeps_per_step, worldline_sweeps, steps, swap_interval=1, cluster_sweeps=0, continuous_time_slices=0)` → `SQAParallelTemperingResult`**

---

### `CTPIMCAnnealer` (two modes)

#### General mode: `CTPIMCAnnealer(ising, schedule, qubits_per_update=1, qubits_per_chain=1)`

Continuous-time PIMC for arbitrary DenseIsing or SparseIsing.

```python
from qanneal import CTPIMCAnnealer

annealer = CTPIMCAnnealer(ising, schedule, qubits_per_update=1)
result = annealer.run(sweeps_per_beta=100, reads=4)
```

#### Lattice mode: `CTPIMCAnnealer(Lperiodic, inv_temp_over_J, gamma_over_J, initial_condition=0, qubits_per_update=1, qubits_per_chain=1)`

For D-Wave-style cylindrical lattice experiments.

| Parameter | Description |
|-----------|-------------|
| `Lperiodic` | Lattice period (must be multiple of 6) |
| `inv_temp_over_J` | β/J — temperature in units of coupling J |
| `gamma_over_J` | Γ/J — transverse field in units of coupling |
| `initial_condition` | 0 = random, 1 = ferromagnetic, −1 = antiferromagnetic |

**`run(sweeps_per_beta, reads=1)` → `CTPIMCResult`**

| Parameter | Description |
|-----------|-------------|
| `sweeps_per_beta` | Worldline cluster sweeps per schedule step |
| `reads` | Independent runs; best is returned |

**Important**: Γ must be > 0. Use `gamma_end_scale=0.04` in `auto_schedule_sqa_tuned`
so γ reaches near-zero at the end for clean classical projection.

---

## Result Classes

---

### `AnnealResult`
From `Annealer.run()`.

| Field | Type | Description |
|-------|------|-------------|
| `best_state` | `State` | State with lowest energy found |
| `best_energy` | `float` | Lowest energy found |
| `energy_trace` | `list[float]` | Energy at each temperature step |

---

### `MultiAnnealResult`
From `ReplicaAnnealer.run()`.

| Field | Type | Description |
|-------|------|-------------|
| `global_best_state` | `State` | Best state across all replicas |
| `global_best_energy` | `float` | Best energy |
| `replicas` | `list[ReplicaResult]` | Per-replica results |
| `average_energy_trace` | `list[float]` | Mean energy across replicas |
| `average_magnetization_trace` | `list[float]` | Mean magnetization |

---

### `ParallelTemperingResult`
From `ParallelTemperingAnnealer.run()`.

| Field | Type | Description |
|-------|------|-------------|
| `best_state` | `State` | Global best |
| `best_energy` | `float` | Global best energy |
| `final_states` | `list[State]` | Final state of each replica |
| `final_energies` | `list[float]` | Final energy of each replica |
| `average_energy_trace` | `list[float]` | Average energy trace |
| `swap_acceptance_trace` | `list[float]` | Fraction of accepted swaps per step |

---

### `SQAResult`
From `SQAAnnealer.run()`.

| Field | Type | Description |
|-------|------|-------------|
| `best_state` | `State` | Best classical projection (best Trotter slice) |
| `best_energy` | `float` | Energy of best_state |
| `energy_trace` | `list[float]` | Energy trace |

---

### `SQAParallelTemperingResult`
From `SQAParallelTemperingAnnealer.run()`. Same fields as `ParallelTemperingResult`.

---

### `CTPIMCResult`
From `CTPIMCAnnealer.run()`.

| Field | Type | Description |
|-------|------|-------------|
| `best_state` | `State` | Best state found |
| `best_energy` | `float` | Best energy found |
| `energy_trace` | `list[float]` | Energy at each schedule step (average of reads) |

---

### `State`

Low-level spin state container.

```python
state.spins     # list[int8] of ±1 values
state.size()    # int — number of spins
```

---

### `SolveResult`
From `solve()`. High-level wrapper.

| Field | Type | Description |
|-------|------|-------------|
| `method` | `str` | Which method was used |
| `samples` | `list[np.ndarray]` | Best spin/bit array from each read |
| `energies` | `list[float]` | Best energy from each read |
| `best_sample` | `np.ndarray` | Global best spin/bit array |
| `best_energy` | `float` | Global best energy |
| `trace` | `list[float]` | Energy trace from the first read |
| `var_order` | `list` | Variable ordering (for BQM/graph inputs) |
| `return_bits` | `bool` | Whether samples are bits {0,1} or spins {−1,+1} |

---

## Observer Classes

Observers hook into the annealing loop to record diagnostics without modifying the core algorithm.

### `MetricsObserver` (classical SA)

```python
obs = MetricsObserver()
annealer.run(sweeps_per_beta=40, observer=obs)

obs.energy_trace          # list[float] — energy after each temperature step
obs.magnetization_trace   # list[float] — mean spin after each step
obs.clear()               # reset for reuse
```

---

### `StateTraceObserver` (classical SA)

Records full spin states at every sweep.

```python
obs = StateTraceObserver(stride=1)   # record every sweep
annealer.run(sweeps_per_beta=40, observer=obs)

obs.state_trace    # list of State objects
obs.energy_trace   # list[float] per sweep
obs.step_trace     # temperature step index per record
obs.sweep_trace    # sweep index per record
obs.beta_trace     # β value per record
```

---

### `SQAMetricsObserver`

SQA equivalent of `MetricsObserver`.

```python
obs = SQAMetricsObserver()
sqa_annealer.run(..., observer=obs)

obs.energy_trace          # averaged over slices and replicas
obs.magnetization_trace
```

---

### `SQAStateTraceObserver`

Rich per-sweep trace for SQA. See `docs/sqa_trace_parameters.md` for complete field descriptions.

```python
obs = SQAStateTraceObserver(stride=1)
sqa_annealer.run(..., observer=obs)

obs.avg_energy_trace       # average energy per sweep
obs.replica_energy_trace   # per-replica energies
obs.state_trace            # full SQA state snapshots
obs.beta_trace             # β at each recorded sweep
obs.gamma_trace            # Γ at each recorded sweep
obs.phase_trace            # SQASweepPhase enum: Slice / Worldline / Cluster
obs.step_trace             # temperature step index
obs.replica_trace          # replica index
obs.sweep_trace            # sweep index within step
# dimensions
obs.replicas               # int
obs.slices                 # int
obs.spins                  # int
```

---

## High-Level Solver

### `solve(problem, method="sqa", **kwargs)` → `SolveResult`

One-call interface. Accepts any supported problem type, auto-selects schedules,
runs multiple reads, and returns the best result.

**Supported problem types** (auto-detected):
- `DenseIsing`, `SparseIsing` — passed directly
- `QUBO` — converted via `to_ising()`
- `np.ndarray` shape (n,n) — interpreted as QUBO matrix
- `dict{(i,j): float}` — sparse QUBO dict
- `list[(i,j,float)]` — sparse QUBO entry list
- `dimod.BinaryQuadraticModel` — converted (requires dimod)
- `networkx.Graph` — node `bias`, edge `weight` attributes (requires networkx)

**Full parameter table**

| Parameter | Default | Methods | Description |
|-----------|---------|---------|-------------|
| `method` | `"sqa"` | all | `"sa"`, `"sqa"`, `"sqapt"`, `"ctpimc"` |
| `reads` | 1 | all | Independent runs; best is returned |
| `sweeps_per_beta` | 20 | all | Sweeps per temperature step |
| `schedule` | None | all | Schedule object; auto-selected if None |
| `seed` | None | all | RNG seed (int); None = random |
| `backend` | `"cpu"` | all | Compute backend |
| `progress` | True | all | Show tqdm progress bar |
| `n` | None | all | Hint for problem size (dict/list inputs) |
| `return_bits` | False | all | Return {0,1} bits instead of {−1,+1} spins |
| `trotter_slices` | 32 | sqa, sqapt | Imaginary-time slices |
| `replicas` | 1 | sqa, sqapt | SQA replicas (SQA) or PT ladder size (SQAPT) |
| `worldline_sweeps` | 5 | sqa, sqapt | Time-direction sweeps per step |
| `cluster_sweeps` | 0 | sqa, sqapt | Swendsen–Wang cluster sweeps |
| `continuous_time_slices` | 0 | sqa, sqapt | CT approximation slice count |
| `pt_steps` | 50 | sqapt | Local-update + swap epochs |
| `swap_interval` | 1 | sqapt | Steps between swap attempts |
| `pt_betas` | None | sqapt | Explicit β ladder (overrides schedule) |
| `pt_gammas` | None | sqapt | Explicit Γ ladder (must pair with pt_betas) |
| `ctpimc_qubits_per_update` | 1 | ctpimc | Update cluster size |
| `ctpimc_qubits_per_chain` | 1 | ctpimc | Chain length for lattice mode |

---

## Utility Functions

```python
from qanneal import magnetization, overlap

magnetization(spins)      # float — Σ sᵢ / n
overlap(spins_a, spins_b) # float — Σ aᵢ bᵢ / n
```

Both accept `list`, `np.ndarray`, or a `State` object.

---

## C++ API Summary

Umbrella include:
```cpp
#include "qanneal/core.hpp"
```

### Key C++ classes

| Class | Header |
|-------|--------|
| `DenseIsing` | `dense_ising.hpp` |
| `SparseIsing`, `SparseEdge` | `sparse_ising.hpp` |
| `QUBO` | `qubo.hpp` |
| `AnnealSchedule` | `schedule.hpp` |
| `SQASchedule` | `sqa_schedule.hpp` |
| `Annealer` | `annealer.hpp` |
| `ReplicaAnnealer` | `replica_annealer.hpp` |
| `ParallelTemperingAnnealer` | `parallel_tempering.hpp` |
| `SQAAnnealer` | `sqa_annealer.hpp` |
| `SQAParallelTemperingAnnealer` | `sqa_parallel_tempering.hpp` |
| `CTPIMCAnnealer` | `ctpimc_annealer.hpp` |
| `State` | `state.hpp` |
| `SQAState` | `sqa_state.hpp` |
| `MetricsObserver`, `StateTraceObserver` | `metrics_observer.hpp` |
| `SQAMetricsObserver`, `SQAStateTraceObserver` | `metrics_observer.hpp` |
| `MPIContext` | `mpi/mpi_context.hpp` |
| `run_replica_anneal()` | `mpi/mpi_replica.hpp` |

### Build options

| CMake Flag | Default | Effect |
|-----------|---------|--------|
| `QANNEAL_ENABLE_OPENMP` | ON | Parallel loops in Replica/SQA/SQAPT annealers |
| `QANNEAL_ENABLE_MPI` | OFF | Distributed `run_replica_anneal()` |
| `QANNEAL_BUILD_TESTS` | ON | Build C++ unit tests |
| `QANNEAL_BUILD_PYTHON` | OFF* | Build pybind11 Python extension (*forced ON by scikit-build) |
