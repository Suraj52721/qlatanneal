# Python examples

Run from the repo root after installing the Python package:

```bash
python -m pip install -e .
```

---

## Basic demos

| Script | What it shows |
|--------|--------------|
| `sa_multi.py` | Multiple SA reads, best-result extraction |
| `sqa_basic.py` | Minimal SQA run with Trotter slices |
| `metrics_plot.py` | Energy / magnetization trace via `MetricsObserver` |
| `parallel_tempering.py` | Classical parallel tempering (PT) |
| `tuned_sqapt_demo.py` | Auto-tuned SQAPT with swap-acceptance monitoring |

```bash
python examples/python/sa_multi.py
python examples/python/sqa_basic.py
python examples/python/metrics_plot.py
python examples/python/parallel_tempering.py
python examples/python/tuned_sqapt_demo.py --mode balanced
```

---

## Number partition benchmarks

| Script | What it shows |
|--------|--------------|
| `number_partition_benchmark.py` | SA / SQA / SQAPT / CT-PIMC vs brute-force on number partition |
| `number_partition_compare_all.py` | Scaling comparison across many random instances |

```bash
python examples/python/number_partition_benchmark.py \
  --with-sqapt --with-ctpimc --schedule-mode balanced --jobs 8

python examples/python/number_partition_compare_all.py \
  --with-sqapt --schedule-mode balanced --instances 16 --jobs 8
```

---

## D-Wave comparative benchmark

Full parallel benchmark of qlatannealv4 (SA, SQA, SQAPT, CT-PIMC) versus
D-Wave SDK (dwave-NEAL, dwave-PIMC) on random Ising spin-glass instances.

Saves results to `dwave_comparative_results_v4.json` and eight PNG figures:
`benchmark_v4_accuracy.png`, `benchmark_v4_walltime.png`,
`benchmark_v4_power.png`, `benchmark_v4_success_rate.png`,
`benchmark_v4_tts.png`, `benchmark_v4_pareto.png`,
`benchmark_v4_boxplot.png`, `benchmark_v4_radar.png`.

```bash
# Full benchmark (n = 20–90, 15 seeds each, all workers)
python examples/python/dwave_comparative_benchmark.py

# Quick run with fewer instances
python examples/python/dwave_comparative_benchmark.py --quick --workers 4

# Disable CT-PIMC (faster)
python examples/python/dwave_comparative_benchmark.py --no-pimc --workers 8
```

---

## Comprehensive multi-family benchmark (`crazy_benchmark.py`)

Head-to-head across five problem families (SK spin glass, frustrated magnet,
MaxCut, dense QUBO, number partition) with optional fair compute-budget mode.

```bash
# Defaults (balanced mode, 10 instances per family)
python examples/python/crazy_benchmark.py

# More statistics
python examples/python/crazy_benchmark.py --instances 20 --reads 50

# Fair compute budget (equal spin-update operations per method)
python examples/python/crazy_benchmark.py --fair

# Thorough + fair
python examples/python/crazy_benchmark.py --fair --accurate

# Quick (skip SQAPT and dwave-NEAL)
python examples/python/crazy_benchmark.py --no-sqapt --no-neal
```

Outputs: `crazy_benchmark_dashboard.png`, `crazy_benchmark_scaling.png`,
`crazy_benchmark_results.json`, `crazy_benchmark_summary.csv`,
`crazy_benchmark_traces.png`.

---

## Spin-glass benchmark

SA / SQA / SQAPT on random ±J spin-glass instances across problem sizes.

```bash
python examples/python/spin_glass_benchmark.py \
  --n 48 --instances 12 --reads 32 --schedule-mode balanced
```

---

## HPC launcher (`hpc_sqa_launcher.py`)

For large sparse problems (n = 1 000 – 100 000 spins).
Supports single-node multiprocessing, MPI multi-node, and SLURM array jobs.

```bash
# Single node, 8 CPU workers
python examples/python/hpc_sqa_launcher.py \
  --n 5000 --method sqapt --reads 64 --workers 8

# Multi-node MPI (requires mpi4py)
mpirun -n 32 python examples/python/hpc_sqa_launcher.py \
  --n 50000 --method sqa --reads 8 --mpi

# Scaling benchmark (n = 100 to 10 000)
python examples/python/hpc_sqa_launcher.py \
  --scaling --method sqapt --mode fast

# Custom problem type
python examples/python/hpc_sqa_launcher.py \
  --problem-type chimera --chimera-m 4 --method sqapt --reads 16
```

Outputs: `<out>_result.json` / `<out>_result.png` (single run)
or `<out>_scaling.json` / `<out>_scaling.png` (scaling mode).

## Merging SLURM array results

```bash
# After running scripts/slurm/run_sqa_array.sh
python examples/python/merge_array_results.py results/run_*.json
```

---

## Interactive graph editor

```bash
python examples/python/graph_editor_gui.py
```

---

See the companion `.md` files in this folder for short explanations of the
basic demo scripts.
