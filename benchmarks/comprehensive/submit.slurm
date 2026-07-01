#!/bin/bash
#SBATCH --job-name=qanneal_comprehensive
#SBATCH --partition=nandaq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=logs/comprehensive_%j.out
#SBATCH --error=logs/comprehensive_%j.err

mkdir -p logs results plots

module load anaconda3/2022.05 intel/2017
source /opt/apps/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate qanneal_bench
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # qanneal 0.6.1 conda libgomp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK                 # one thread-team per rank (no oversubscription)
export PYTHONUNBUFFERED=1

cd /home/shassan/SA/test/comprehensive_benchmark

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)  ranks=$SLURM_NTASKS"
echo "qanneal: $(python -c 'import qanneal; print(qanneal.__version__)')"

srun --mpi=pmi2 -n $SLURM_NTASKS python3 -u benchmark.py \
    --output-dir results --tag $SLURM_JOB_ID

echo "Finished at $(date)"
