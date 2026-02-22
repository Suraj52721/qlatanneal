# qanneal

Research-grade simulated quantum annealing toolkit (CPU-first, CUDA-ready).

**What it is**
- Classical simulated annealing (SA) and simulated quantum annealing (SQA) for dense and sparse Ising/QUBO models.
- C++ core with Python bindings (pybind11).
- Sweep-level tracing for reproducibility and diagnostics.
- MPI/SLURM scaffolding for multi-process runs.

---

**Physics and Models**

**QUBO**
- Binary variables: `x_i ∈ {0,1}`
- Energy:
  `E(x) = Σ_i Σ_j Q_ij x_i x_j`
- Diagonal `Q[i,i]` = linear terms, off-diagonal `Q[i,j]` = pairwise couplings.

**Ising**
- Spins: `s_i ∈ {-1,+1}`
- Energy:
  `E(s) = Σ_i h_i s_i + Σ_ij J_ij s_i s_j + c`
- QUBO→Ising mapping: `x_i = (1 + s_i)/2`

**SQA (path-integral picture)**
- Transverse-field Ising model mapped to `M` Trotter slices (imaginary time).
- Effective inter-slice coupling:
  `J_perp = 0.5 * log(coth(βΓ/M))`
- Two update phases per step:
  slice updates and worldline updates.

---

**Install**

**macOS / Linux**
```bash
./setup.sh
```

**Windows (PowerShell)**
```powershell
.\setup.ps1
```

**Windows (Command Prompt)**
```bat
setup.bat
```

**Pure pip (all platforms)**
```bash
python -m pip install . --no-build-isolation
```

**From PyPI (after first release)**
```bash
python -m pip install qanneal
```

**Windows prerequisites**
- Visual Studio Build Tools with **Desktop development with C++**
- CMake (e.g., `winget install Kitware.CMake`)

---

**Quickstart (QUBO → SQA)**
```python
import numpy as np
from qanneal import QUBO, SQASchedule, SQAAnnealer

Q = np.array([[1.0, -1.0],
              [-1.0, 2.0]], dtype=float)

qubo = QUBO(Q)
ising = qubo.to_ising()

betas = np.linspace(0.1, 4.0, 50).tolist()
gammas = np.linspace(5.0, 0.01, 50).tolist()
schedule = SQASchedule.from_vectors(betas, gammas)

annealer = SQAAnnealer(ising, schedule, trotter_slices=32, replicas=4, backend="cpu")
result = annealer.run(sweeps_per_beta=20, worldline_sweeps=5)
print(result.best_energy)
```

**QUBO from sparse entries**
```python
from qanneal import QUBO

entries = [
    (0, 0, 1.0),   # linear term on x0
    (1, 1, 2.0),   # linear term on x1
    (0, 1, -1.5),  # coupling x0*x1
]

qubo = QUBO(entries, n=2)
ising = qubo.to_ising()
```

**QUBO from dict**
```python
from qanneal import QUBO

entries = {
    (0, 0): 1.0,
    (1, 1): 2.0,
    (0, 1): -1.5,
}

qubo = QUBO(entries, n=2)
ising = qubo.to_ising()
```

**QUBO from dimod BQM**
```python
import dimod
from qanneal import QUBO

bqm = dimod.BinaryQuadraticModel({0: 1.0, 1: 2.0}, {(0, 1): -1.5}, vartype="BINARY")
qubo = QUBO(bqm)
ising = qubo.to_ising()
```

---

**Core Parameters (SQA)**

**Schedule**
- `betas`: inverse temperature values (cooling).
- `gammas`: transverse-field values (quantum fluctuations).
- `steps`: `len(betas)`, must equal `len(gammas)`.

**Geometry**
- `trotter_slices`: number of imaginary-time slices.
- `replicas`: independent replicas in a single run.

**Monte Carlo**
- `sweeps_per_beta`: slice-update sweeps per step.
- `worldline_sweeps`: worldline-update sweeps per step.

**Tracing**
- `SQAStateTraceObserver.stride`: record every N sweeps.

---

**Quality-of-life helpers**

`solve()` runs a full SA/SQA solve with progress + logging and multiple reads:

```python
import numpy as np
from qanneal import solve

Q = np.array([[1.0, -1.0], [-1.0, 2.0]], dtype=float)
result = solve(Q, method="sqa", reads=10, return_bits=True)
print(result.best_energy)
print(result.best_sample)
```

Auto schedules:
```python
from qanneal import auto_schedule_sa, auto_schedule_sqa

sa_schedule = auto_schedule_sa(steps=50)
sqa_schedule = auto_schedule_sqa(steps=50)
```

`solve()` accepts:
- `numpy` QUBO matrix
- `dict` or `list` entries
- `dimod` BQM (if installed)
- `networkx` graph (if installed)
- `DenseIsing` / `SparseIsing`

---

**Observers and Traces**

**Classical SA**
- `MetricsObserver`: energy + magnetization traces.
- `StateTraceObserver`: sweep-level states and energies.

**SQA**
- `SQAMetricsObserver`: energy + magnetization traces.
- `SQAStateTraceObserver`: full sweep-level state trace,
  per-replica energies, and phase (slice vs worldline).

See `docs/sqa_trace_parameters.md` for a full explanation.

---

**Examples**
```bash
python examples/python/sa_multi.py
python examples/python/sqa_basic.py
python examples/python/metrics_plot.py
python examples/python/parallel_tempering.py
python examples/python/sqa_trace_full.py
```

---

**Build (C++ core)**
```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

**CMake presets**
```bash
cmake --preset cpu-only
cmake --build --preset cpu-only
ctest --preset cpu-only
```

**MPI build**
```bash
cmake -S . -B build -DQANNEAL_ENABLE_MPI=ON
cmake --build build
mpirun -n 4 build/qanneal_mpi_example
```

**SLURM scripts**
- `scripts/slurm/run_sa_mpi_srun.sh`
- `scripts/slurm/run_sa_mpi_mpirun.sh`

---

**Docs**
- `docs/overview.md`
- `docs/api.md`
- `docs/sqa_trace_parameters.md`
- `docs/latex/qanneal_technical_report.tex`

---

**Release (PyPI wheels)**
1. Update version in `pyproject.toml`.
2. Tag and push:
```bash
git tag v0.1.0
git push origin v0.1.0
```
3. GitHub Actions builds wheels and publishes to PyPI.

---

**License**
Apache-2.0 (see `LICENSE`). Portions derived from `sqaod` with attribution in `NOTICE`.
