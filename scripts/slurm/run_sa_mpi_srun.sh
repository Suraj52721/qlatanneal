#!/bin/bash
#SBATCH -J qanneal_sa_srun
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=compute

# OpenMPI example. Adjust module name for your cluster.
# module load openmpi

srun qanneal/build/qanneal_mpi_example
