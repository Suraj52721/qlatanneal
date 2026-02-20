#!/bin/bash
#SBATCH -J qanneal_sa
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=compute

# Example SLURM launcher for the MPI SA demo (srun).
# OpenMPI example. Adjust module name for your cluster environment.
# module load openmpi

srun qanneal/build/qanneal_mpi_example
