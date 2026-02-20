# Examples

This folder contains small runnable examples for both C++ and Python.

## Python examples

- `python/sa_multi.py`: Multi-replica classical simulated annealing.
- `python/sqa_basic.py`: Simulated quantum annealing with user schedules.
- `python/metrics_plot.py`: Energy/magnetization tracking and plotting.
- `python/parallel_tempering.py`: Parallel tempering (replica exchange).

See `python/*.md` files for short explanations.

## C++ examples

- `mpi/sa_mpi.cpp`: MPI distributed replicas (CPU).

Build with `-DQANNEAL_ENABLE_MPI=ON` and run via `mpirun` or `srun`.
