# tuned_sqapt_demo.py

This script demonstrates three qanneal workflows on the same random QUBO:

- SA with `auto_schedule_sa_tuned`
- SQA with `auto_schedule_sqa_tuned`
- SQA quantum parallel tempering (`method="sqapt"`) with `auto_ladder_sqa_tuned`

Run:

```bash
python examples/python/tuned_sqapt_demo.py \
  --n 20 --mode balanced --reads 16 \
  --sa-steps 80 --sqa-steps 90 --pt-steps 70 \
  --sweeps 60 --worldline 4 --slices 24 --replicas 8
```

Output:

- best Ising and QUBO energies for SA/SQA/SQAPT
- best bitstring for SQAPT
- brute-force optimum when `n <= 22`

Use `--mode fast` for quick iteration and `--mode accurate` for stronger annealing schedules.
