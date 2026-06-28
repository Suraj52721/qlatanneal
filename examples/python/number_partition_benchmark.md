# number_partition_benchmark.py

Single-instance number partition benchmark across qanneal, D-Wave samplers, and heuristic baselines.

Methods:

- `qanneal-SA`
- `qanneal-SQA`
- `qanneal-SQAPT` (optional, `--with-sqapt`)
- `qanneal-CTPIMC` (optional, `--with-ctpimc`)
- `dwave-NEAL` (if installed)
- `dwave-PIMC` (if installed)
- `karmarkar-karp`
- `greedy`

## Run

```bash
python examples/python/number_partition_benchmark.py \
  --n 30 --reads 64 --steps 80 --sweeps 100 \
  --worldline 6 --slices 24 --replicas 8 --cluster 2 \
  --schedule-mode balanced --with-sqapt --pt-steps 80 --swap-interval 1 \
  --with-ctpimc --jobs 8 --out number_partition_results
```

## Output

- `number_partition_results.json`
- `number_partition_results.png` (if matplotlib is installed)

## Useful flags

- `--schedule-mode`: `legacy|fast|balanced|accurate`
- `--fair`: normalize effective spin-update budget across methods
- `--budget-updates`: explicit fair budget override
- `--jobs`: process-level parallel reads for qanneal methods
