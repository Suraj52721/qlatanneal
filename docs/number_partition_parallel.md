# Number Partition Parallelization Guide

This guide covers the parallel benchmark paths for number partitioning in qanneal.

## Scripts

- Single-instance benchmark with read-level parallelism:
  `/Users/satyanveshi/Desktop/SQOAD/qlatanneal/examples/python/number_partition_benchmark.py`
- Multi-instance detailed comparison (all methods):
  `/Users/satyanveshi/Desktop/SQOAD/qlatanneal/examples/python/number_partition_compare_all.py`

## 1) Fast single-instance benchmark

Use process-parallel reads with `--jobs`.

```bash
python /Users/satyanveshi/Desktop/SQOAD/qlatanneal/examples/python/number_partition_benchmark.py \
  --n 30 --reads 64 --steps 80 --sweeps 100 \
  --worldline 6 --slices 24 --replicas 8 \
  --cluster 2 --ct-slices 0 --jobs 8 \
  --schedule-mode balanced --with-sqapt --pt-steps 80 --swap-interval 1 \
  --with-ctpimc
```

Outputs:
- `<out>.json` raw results
- `<out>.png` accuracy/runtime plots

Notes:
- `--jobs` parallelizes **qanneal reads** via processes.
- `--schedule-mode` controls tuned auto-schedule generation:
  - `legacy`: fixed linear schedules (`auto_schedule_sa/sqa`)
  - `fast|balanced|accurate`: scale-aware tuned schedules
- `--with-sqapt` enables SQA quantum parallel tempering (`method="sqapt"`).
- `--pt-steps` and `--swap-interval` control SQAPT ladder evolution.
- D-Wave NEAL/PIMC calls are executed once each per benchmark run.
- If process pools are blocked by the environment, the script automatically falls back to thread-level parallel execution.

## 2) Detailed all-method comparison

Run many random instances and compare all methods.

```bash
python /Users/satyanveshi/Desktop/SQOAD/qlatanneal/examples/python/number_partition_compare_all.py \
  --n 30 --instances 24 --reads 32 --steps 60 --sweeps 80 \
  --worldline 5 --slices 16 --replicas 4 --cluster 1 \
  --schedule-mode balanced --with-sqapt --pt-steps 60 --swap-interval 1 \
  --jobs 10 --out number_partition_compare_all
```

Outputs:
- `number_partition_compare_all.json` full per-run records
- `number_partition_compare_all.csv` aggregate summary table
- `number_partition_compare_all_summary.png` aggregate plots

Metrics in CSV:
- `median_diff`, `mean_diff`
- `median_time_s`, `mean_time_s`
- `oracle_hit_rate`
- `median_gap_to_oracle`
- includes `qanneal-SQAPT` rows when `--with-sqapt` is set

## Oracle definition

Per instance, the oracle is:
1. Exact brute-force optimum if `n <= bf_max_n`.
2. Otherwise best diff observed across all successful methods.

## Dependency matrix

Required:
- `qanneal`
- `numpy`

Optional:
- `matplotlib` for plots
- `dwave-neal` (`import neal`) for NEAL
- `dwave-samplers` for Path-Integral sampler
- `dimod`

Methods with missing optional dependencies are marked failed and excluded from aggregate stats.

## Parallelization strategy

### `number_partition_benchmark.py`
- Parallel axis: read index (`reads`)
- Granularity: one qanneal solve per process
- Good when: one large instance, high `reads`

### `number_partition_compare_all.py`
- Parallel axis: `(instance, method)` task grid
- Granularity: one method solve on one instance per process
- Good when: cross-method study on many instances
- If process pools are blocked, it automatically falls back to thread-level parallel execution.

## Practical tuning

- Start with `jobs = physical_cores`.
- If system is memory-constrained, reduce `jobs` first.
- SQA runtime scales strongly with `slices * replicas * sweeps`.
- SQAPT runtime scales strongly with `pt_steps * slices * replicas * sweeps`.
- Use `--fair` in single-instance benchmark when comparing runtime/quality tradeoffs across methods.
