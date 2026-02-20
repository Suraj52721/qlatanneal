# qanneal

Research-grade simulated quantum annealing toolkit (CPU-first, CUDA-ready).

## Quick install (recommended)

### macOS / Linux

```bash
./setup.sh
```

### Windows (PowerShell)

```powershell
.\setup.ps1
```

### Windows (Command Prompt)

```bat
setup.bat
```

If you prefer pure pip:

```bash
python -m pip install . --no-build-isolation
```

## Install from PyPI (after first release)

```bash
python -m pip install qanneal
```

### Windows prerequisites

- Install **Visual Studio Build Tools** with the **Desktop development with C++** workload.
- Install **CMake** (e.g., `winget install Kitware.CMake`).

## Windows + VS Code quickstart

1. Install **VS Code** and the **Python** extension.
2. Open the project folder in VS Code.
3. Open a terminal in VS Code (``Ctrl+` ``).
4. Create/activate a venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

5. Run the setup:

```powershell
.\setup.ps1
```

6. Run an example:

```powershell
python examples\python\sqa_basic.py
```

## Build (C++ core)

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## Install as a pip package (manual)

From a clone:

```bash
python -m pip install -U pip
python -m pip install . --no-build-isolation
```

Build a wheel locally:

```bash
python -m pip install -U build
python -m build .
```

The wheel/sdist will be in `dist/`.

If you want to install directly from Git:

```bash
python -m pip install "git+https://your-repo-url.git"
```

### CMake presets (CPU-only)

Requires CMake 3.19+ for presets.

```bash
cmake --preset cpu-only
cmake --build --preset cpu-only
ctest --preset cpu-only
```

### CPU + MPI preset (OpenMPI recommended)

```bash
cmake --preset cpu-mpi
cmake --build --preset cpu-mpi
```

If CMake cannot find OpenMPI, set `QANNEAL_MPI_HOME` or `MPI_HOME`:

```bash
cmake -S . -B build -DQANNEAL_ENABLE_MPI=ON -DQANNEAL_MPI_HOME=/path/to/openmpi
```

## Python examples

```bash
python examples/python/sa_multi.py
python examples/python/sqa_basic.py
python examples/python/metrics_plot.py
python examples/python/parallel_tempering.py
```

## Release (PyPI wheels)

1. Update the version in `pyproject.toml`.
2. Commit and tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

3. GitHub Actions will build wheels for Linux/macOS/Windows and publish to PyPI.

### Notes
- Publishing uses GitHub Actions OIDC. Ensure PyPI is configured to trust this repo.
- You can also run the publish workflow manually from GitHub Actions.

### Optional MPI build

```bash
cmake -S . -B build -DQANNEAL_ENABLE_MPI=ON
cmake --build build
```

Run MPI example:

```bash
mpirun -n 4 build/qanneal_mpi_example
```

### SLURM examples (OpenMPI)

Use either launcher style depending on your cluster policy:

- `qanneal/scripts/slurm/run_sa_mpi_srun.sh` (srun)
- `qanneal/scripts/slurm/run_sa_mpi_mpirun.sh` (mpirun)

The original `qanneal/scripts/slurm/run_sa_mpi.sh` remains as a simple srun starter.

## Roadmap

- Core Ising/QUBO models
- Classical and SQA annealers
- Observer and metrics API
- CUDA backend (optional)
- Python bindings (pybind11)
- MPI / SLURM examples

## License

Apache-2.0 (see `LICENSE`). Portions derived from the `sqaod` project with attribution in `NOTICE`.
