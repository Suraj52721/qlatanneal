# Examples

This folder contains small runnable examples for both C++ and Python.

## Python examples

- `python/sa_multi.py`: Multi-replica classical simulated annealing.
- `python/sqa_basic.py`: Simulated quantum annealing with user schedules.
- `python/metrics_plot.py`: Energy/magnetization tracking and plotting.
- `python/parallel_tempering.py`: Classical parallel tempering (replica exchange).
- `python/tuned_sqapt_demo.py`: Tuned schedules + quantum parallel tempering (`method="sqapt"`).
- `python/number_partition_benchmark.py`: Single-instance benchmark (qanneal + D-Wave + heuristics).
- `python/number_partition_compare_all.py`: Multi-instance aggregate comparison.

See `python/*.md` files for short explanations.

## C++ examples

- `sqa_quantum_parallel_tempering.cpp`: Quantum parallel tempering in C++ (`SQAParallelTemperingAnnealer`).
- `mpi/sa_mpi.cpp`: MPI distributed replicas (CPU).

Build with `-DQANNEAL_ENABLE_MPI=ON` and run via `mpirun` or `srun`.
