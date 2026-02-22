# API Summary

This is a concise reference for the public Python and C++ APIs.

## Python API

Import:
```python
from qanneal import (
    QUBO, DenseIsing, SparseIsing, SparseEdge,
    AnnealSchedule, SQASchedule,
    Annealer, ReplicaAnnealer, ParallelTemperingAnnealer, SQAAnnealer,
    MetricsObserver, StateTraceObserver,
    SQAMetricsObserver, SQAStateTraceObserver,
)
```

### Models

- `QUBO(Q: ndarray)`  
  `Q` is a dense `N×N` matrix (diagonal = linear terms, off-diagonal = couplings).
- `QUBO(entries: list[tuple[i, j, value]], n: int)`  
  Sparse-style constructor; entries are summed into a dense matrix.
  `to_ising() -> DenseIsing`
- `DenseIsing(h: ndarray, J: ndarray, c: float = 0.0)`
- `SparseIsing(h: ndarray, edges: list[SparseEdge], n: int, c: float = 0.0)`
- `SparseEdge(i: int, j: int, value: float)`

### Schedules

- `AnnealSchedule.linear(beta_start, beta_end, steps)`
- `AnnealSchedule.from_betas(betas)`
- `SQASchedule.from_vectors(betas, gammas)`

### Annealers

- `Annealer(hamiltonian, schedule, backend="cpu")`
- `ReplicaAnnealer(hamiltonian, schedule, replicas, backend="cpu")`
- `ParallelTemperingAnnealer(hamiltonian, betas, backend="cpu")`
- `SQAAnnealer(hamiltonian, schedule, trotter_slices, replicas=1, backend="cpu")`

### Run methods

- `Annealer.run(sweeps_per_beta, observer=None) -> AnnealResult`
- `ReplicaAnnealer.run(sweeps_per_beta) -> MultiAnnealResult`
- `ParallelTemperingAnnealer.run(sweeps_per_step, steps, swap_interval=1) -> ParallelTemperingResult`
- `SQAAnnealer.run(sweeps_per_beta, worldline_sweeps, observer=None) -> SQAResult`

### Observers

- `MetricsObserver`: `energy_trace`, `magnetization_trace`
- `StateTraceObserver`: `state_trace`, `energy_trace`, `step_trace`, `sweep_trace`, `beta_trace`
- `SQAMetricsObserver`: `energy_trace`, `magnetization_trace`
- `SQAStateTraceObserver`: full sweep-level traces including replica and slice states

### Results

- `AnnealResult`: `best_state`, `best_energy`, `energy_trace`
- `MultiAnnealResult`: `replicas`, `global_best_state`, `global_best_energy`, `average_energy_trace`
- `SQAResult`: `best_state`, `best_energy`, `energy_trace`

## C++ API

Include headers from:

- `include/qanneal/*.hpp`

Core types:

- `qanneal::State`
- `qanneal::DenseIsing`, `qanneal::SparseIsing`, `qanneal::QUBO`
- `qanneal::Annealer`, `qanneal::SQAAnnealer`
- `qanneal::Observer`, `qanneal::SQAObserver`
- `qanneal::MetricsObserver`, `qanneal::StateTraceObserver`
- `qanneal::SQAMetricsObserver`, `qanneal::SQAStateTraceObserver`

Backends:

- `qanneal::Backend` interface
- `qanneal::CPUBackend` implementation
