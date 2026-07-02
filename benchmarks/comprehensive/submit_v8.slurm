#!/bin/bash
#SBATCH --job-name=qanneal_v8_targeted
#SBATCH --partition=nandaq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=2
#SBATCH --time=16:00:00
#SBATCH --output=logs/v8_%j.out
#SBATCH --error=logs/v8_%j.err

mkdir -p logs results

module load anaconda3/2022.05 intel/2017
source /opt/apps/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate qanneal_bench
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # qanneal 0.6.1 conda libgomp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK                 # avoid rank x all-cores oversubscription
export PYTHONUNBUFFERED=1

cd /home/shassan/SA/test/comprehensive_benchmark

echo "Job $SLURM_JOB_ID started at $(date) on $(hostname)  ranks=$SLURM_NTASKS"
echo "qanneal: $(python -c 'import qanneal; print(qanneal.__version__)')"
echo "v8 targeted re-run: wmaxcut (no rescale, n<=512), planted (p_in=0.55/p_out=0.45, n<=512),"
echo "rfim2d (h_strength=1.5, n<=196). sk/ea3d/w3reg kept from the existing clean v7 run."

# --problems filters the existing (already-tested) driver's registry -- no separate script needed.
srun --mpi=pmi2 -n $SLURM_NTASKS python3 -u benchmark.py \
    --output-dir results --tag v8_$SLURM_JOB_ID \
    --problems wmaxcut planted rfim2d

echo "Done at $(date)"
