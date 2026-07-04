#!/bin/bash
#SBATCH --job-name=qanneal_v8_resume2
#SBATCH --partition=nandaq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --output=logs/v8resume2_%j.out
#SBATCH --error=logs/v8resume2_%j.err

mkdir -p logs results

module load anaconda3/2022.05 intel/2017
source /opt/apps/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate qanneal_bench
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd /home/shassan/SA/test/comprehensive_benchmark

echo "RESUME(2) job $SLURM_JOB_ID started at $(date)  ranks=$SLURM_NTASKS"
echo "qanneal: $(python -c 'import qanneal; print(qanneal.__version__)')"
echo "Reusing tag v8_3396184. Only 3 instances remain: wmaxcut n=512 sqapt-opt (x2),"
echo "planted n=512 sqapt-opt (x1) -- each ~2.7h based on v7's sqapt-opt-n512 wall time"
echo "(~490s/read x 20 reads). 6h wall time gives comfortable margin even if serialized."

srun --mpi=pmi2 -n $SLURM_NTASKS python3 -u benchmark.py \
    --output-dir results --tag v8_3396184 \
    --problems wmaxcut planted rfim2d

echo "Done at $(date)"
