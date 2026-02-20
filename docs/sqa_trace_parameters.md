# SQA Parameters, Traces, and Physical Meaning

This guide explains the simulated quantum annealing (SQA) parameters in `qanneal`, what they mean physically, and how to interpret the sweep‑level traces captured by `SQAStateTraceObserver`.

**Overview**
SQA maps a quantum transverse‑field Ising model to a classical Ising model in an extra (imaginary time) dimension using the Trotter decomposition. Each physical spin becomes a ring of `trotter_slices` classical spins coupled along the imaginary‑time direction. The anneal then updates these classical spins using Monte Carlo sweeps.

**Core Parameters**
- `betas`: Inverse temperature schedule. Larger `beta` means lower temperature, stronger preference for lower energy states. Physically, you are cooling the system.
- `gammas`: Transverse‑field schedule. Larger `gamma` means stronger quantum fluctuations; smaller `gamma` reduces quantum tunneling and approaches a classical Ising model.
- `steps`: The number of annealing steps. This is simply `len(betas)` (and must match `len(gammas)`).
- `trotter_slices`: Number of imaginary‑time slices in the Trotter decomposition. More slices means a more accurate quantum‑to‑classical mapping but higher cost. Physically, this is the discretization of imaginary time.
- `replicas`: Number of independent replicas of the full Trotter system. This is like running multiple independent anneals in parallel inside one run.
- `sweeps_per_beta`: Number of *slice updates* per step. Each sweep tries to update every spin in every slice for each replica using local Metropolis moves. This is the main sampling work at each `(beta, gamma)`.
- `worldline_sweeps`: Number of *worldline updates* per step. Each sweep tries to flip a spin consistently across all slices in a replica. This helps move through correlated imaginary‑time configurations.
- `observer.stride`: Only record every N sweeps to reduce memory. `stride=1` records every sweep.

**Physical Meaning of Beta and Gamma Together**
- The effective coupling between slices is controlled by both `beta` and `gamma`. Increasing `beta` at fixed `gamma` makes the classical energy dominate. Increasing `gamma` at fixed `beta` makes the imaginary‑time coupling stronger and promotes tunneling.
- Typical anneals start with high `gamma` and low `beta`, then lower `gamma` and raise `beta`.

**Sweep Phases**
`SQAStateTraceObserver` records data during two phases.
- `SQASweepPhase.SLICE`: Local single‑slice Metropolis updates. This explores local configurations in each slice.
- `SQASweepPhase.WORLDLINE`: Global updates that flip a spin across all slices. This changes worldlines and helps with quantum‑like collective moves.

**Trace Fields (What You Get Back)**
All of these are stored on the observer as arrays of equal length.
- `step_trace`: Step index in the anneal schedule (0 to `steps-1`).
- `sweep_trace`: Sweep index inside the step.
- `phase_trace`: 0 for `SLICE`, 1 for `WORLDLINE`.
- `beta_trace`: The beta value used for that sweep.
- `gamma_trace`: The gamma value used for that sweep.
- `avg_energy_trace`: Average energy across all replicas and slices for that sweep.
- `replica_energy_trace`: Per‑replica average energies for that sweep. Shape is `[records][replicas]`.
- `state_trace`: Full Trotter state captured at that sweep.

**State Trace Layout**
`state_trace` stores a flattened array of spins with length:
```
replicas * slices * spins
```
The flattening order is:
```
for replica in replicas:
  for slice in slices:
    for spin in spins:
      append spin
```
You can reshape it into a 3D tensor:
```python
states = np.array(obs.state_trace, dtype=np.int8)
states = states.reshape(len(obs.state_trace), obs.replicas, obs.slices, obs.spins)
```

**Memory Notes**
Recording every sweep can be large. For a rough estimate:
```
records ≈ steps * (replicas * sweeps_per_beta + replicas * worldline_sweeps)
bytes ≈ records * replicas * slices * spins
```
If this is too big, increase `observer.stride` or reduce slices/replicas.

**Minimal Example**
```python
import numpy as np
from qanneal import SQASchedule, SQAAnnealer, SQAStateTraceObserver

steps = 50
betas = np.linspace(0.1, 4.0, steps).tolist()
gammas = np.linspace(5.0, 0.01, steps).tolist()
schedule = SQASchedule.from_vectors(betas, gammas)

obs = SQAStateTraceObserver()
obs.stride = 1

annealer = SQAAnnealer(ising, schedule, trotter_slices=32, replicas=4)
result = annealer.run(sweeps_per_beta=20, worldline_sweeps=5, observer=obs)

print("records:", len(obs.state_trace))
print("replicas:", obs.replicas, "slices:", obs.slices, "spins:", obs.spins)
```

**Practical Tips**
- Increase `trotter_slices` for better quantum fidelity but expect slower runs.
- Increase `replicas` for better statistics.
- Use `worldline_sweeps` when you expect strong correlations in imaginary time.
- Start with a shorter schedule (fewer steps) for debugging, then scale up.
