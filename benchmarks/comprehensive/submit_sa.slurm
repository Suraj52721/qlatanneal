#!/bin/bash
#SBATCH --job-name=qanneal_sa_baseline
#SBATCH --partition=nandaq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --output=logs/sa_%j.out
#SBATCH --error=logs/sa_%j.err

mkdir -p logs results

module load anaconda3/2022.05 intel/2017
source /opt/apps/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate qanneal_bench
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd /home/shassan/SA/test/comprehensive_benchmark

echo "Job $SLURM_JOB_ID started at $(date) on $(hostname)  ranks=$SLURM_NTASKS"
echo "qanneal: $(python -c 'import qanneal; print(qanneal.__version__)')"
echo "SA classical baseline, ALL 7 problem classes, same sweep budget (150 steps x 20 sweeps)"
echo "as the other 4 solvers. SA is a single classical chain (no replicas/Trotter), so this"
echo "should run substantially faster than the QA solver jobs."

srun --mpi=pmi2 -n $SLURM_NTASKS python3 -u benchmark.py \
    --output-dir results --tag sa_$SLURM_JOB_ID \
    --solvers sa-std

echo "Done at $(date)"
