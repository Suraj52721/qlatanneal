# CT‑PIMC (Continuous‑Time Path‑Integral Monte Carlo)

This document describes the continuous‑time PIMC engine (`CTPIMCAnnealer`), how it maps to physics, and how to use it.

## Physical model

We target the transverse‑field Ising model

```
H = 1/2 * Σ_{i,j} J_ij σ_i^z σ_j^z + Σ_i (h_i σ_i^z − Γ σ_i^x)
```

The algorithm samples worldlines in **continuous imaginary time** (no finite Trotter slices). The key inputs are:

- `β` (beta): inverse temperature, `β = 1 / (k_B T)`
- `Γ` (gamma): transverse field strength (quantum fluctuations)
- `h_i, J_ij`: longitudinal fields and couplers

Internally, the continuous‑time PIMC uses **Swendsen‑Wang‑style cluster updates** over worldline segments.

## General vs lattice modes

### General mode

Use this for arbitrary Dense/Sparse Ising models. The schedule `(β, Γ)` is applied by **rescaling couplings**:

```
invTempJ_ij = β * J_ij
invTempH_i  = β * h_i
invTempGamma = β * Γ
```

At each schedule step:

1. Rescale couplings using the current `(β, Γ)`.
2. Run `sweeps_per_beta` updates.
3. Record the projected sample from the current worldlines.

**Energy reporting:** energies are reported in the *original Ising energy units* (not scaled by β).

### Lattice mode

This mode mirrors the D‑Wave localPIMC experiments (triangular and square‑octagonal cylindrical lattices).
You construct the annealer with:

- `Lperiodic`: lattice period (must be a multiple of 6 for the included lattice geometry)
- `inv_temp_over_J`: `β / J` (temperature in units of coupling)
- `gamma_over_J`: `Γ / J`
- `qubits_per_update` and `qubits_per_chain` for the chain update rule

**Energy reporting:** energies are returned in units of `J` (the lattice’s coupling scale).

## Parameters and meanings

- `sweeps_per_beta`:
  The number of sweeps per schedule step. Larger values improve equilibration but increase runtime.

- `reads`:
  Number of independent runs. The best result is returned; the energy trace is averaged across reads.

- `qubits_per_update`:
  Size of the cluster update unit (1 for single‑qubit updates; larger for chain updates in lattice mode).

- `qubits_per_chain`:
  Length of the chain used by the multi‑qubit updates in lattice mode.

- `schedule`:
  A `SQASchedule` object with matched `betas` and `gammas` lists.

**Important:** `Γ` should not be zero. Use a small value (e.g., `1e‑3`) instead of exactly zero.

## Usage examples

### General Ising

```python
import numpy as np
from qanneal import DenseIsing, SQASchedule, CTPIMCAnnealer

h = np.array([0.2, -0.3, 0.1])
J = np.array([[0.0, -1.0, 0.2],
              [-1.0, 0.0, 0.5],
              [0.2, 0.5, 0.0]])

ising = DenseIsing(h, J)

schedule = SQASchedule.from_vectors(
    betas=np.linspace(0.5, 4.0, 40).tolist(),
    gammas=np.linspace(2.0, 0.05, 40).tolist(),
)

annealer = CTPIMCAnnealer(ising, schedule, qubits_per_update=1, qubits_per_chain=1)
res = annealer.run(sweeps_per_beta=50, reads=4)
print(res.best_energy)
```

### Lattice mode

```python
from qanneal import CTPIMCAnnealer

annealer = CTPIMCAnnealer(
    Lperiodic=12,
    inv_temp_over_J=2.0,
    gamma_over_J=0.6,
    initial_condition=0,
    qubits_per_update=1,
    qubits_per_chain=1,
)

res = annealer.run(sweeps_per_beta=32768)
print(res.best_energy)
```

## When to use CT‑PIMC vs SQA

- **CT‑PIMC** is preferable when you want continuous‑time sampling and SW‑style updates.
- **SQA** is preferable when you need explicit Trotter slices, replica structure, and sweep‑level traces.

In practice, CT‑PIMC can mix faster on some problems, but it is **not** guaranteed to outperform SQA across all cases.

## Notes and limitations

- The implementation uses D‑Wave’s localPIMC code (Apache‑2.0).
- Gamma must be positive; extremely large gamma may reduce the effectiveness of the cluster updates.
- Lattice mode assumes the specific cylindrical lattices used in the D‑Wave experiments.
