# number_partition_compare_all.py

Detailed multi-instance comparison across:

- `qanneal-SA`
- `qanneal-SQA`
- `qanneal-SQAPT` (optional, `--with-sqapt`)
- `qanneal-CTPIMC`
- `dwave-NEAL` (if installed)
- `dwave-PIMC` (if installed)
- `greedy`
- `karmarkar-karp`
- `random`

## Run

```bash
python examples/python/number_partition_compare_all.py \
  --n 30 --instances 20 --reads 32 --steps 60 --sweeps 80 \
  --worldline 5 --slices 16 --replicas 4 --cluster 1 \
  --schedule-mode balanced --with-sqapt --pt-steps 60 --swap-interval 1 \
  --jobs 8
```

## Output files

- `number_partition_compare_all.json`
- `number_partition_compare_all.csv`
- `number_partition_compare_all_summary.png` (if matplotlib is installed)

## Important flags

- `--jobs`: process workers for task-level parallelism.
- `--bf-max-n`: brute-force cutoff for exact oracle.
- `--reads`: best-of-reads per stochastic method.
- `--schedule-mode`: `legacy|fast|balanced|accurate` for qanneal auto schedules.
- `--with-sqapt`: include quantum parallel tempering lane.
- `--pt-steps`, `--swap-interval`: SQAPT ladder evolution controls.

## Interpretation

- Lower `median_diff` is better.
- Lower `median_time_s` is better.
- Higher `oracle_hit_rate` is better.
