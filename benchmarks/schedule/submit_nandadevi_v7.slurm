#!/bin/bash
#SBATCH -J v7_sqa_nd
#SBATCH -o v7_%j.out
#SBATCH -e v7_%j.err
#SBATCH -p nandaq
#SBATCH -N 2
#SBATCH --ntasks-per-node=28
#SBATCH -c 2
#SBATCH -t 12:00:00

module load anaconda3/2022.05 intel/2017
source /opt/apps/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate qanneal_bench
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # qanneal 0.6.1 conda libgomp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK                 # avoid rank x all-cores oversubscription
export PYTHONUNBUFFERED=1

cd /home/shassan/SA/test/schedule_benchmark

echo "=== v7: SQA-std vs SQA-opt (fixed 0.6.1 schedule) on nandadevi ==="
echo "qanneal: $(python -c 'import qanneal;print(qanneal.__version__)' 2>&1)"
echo "Job $SLURM_JOB_ID  ranks=$SLURM_NTASKS"
echo "problems: wmaxcut sk ea3d w3reg (n=64,128,256) + numpart (n=25,28,30)  instances=30 reads=15"
date

srun --mpi=pmi2 -n $SLURM_NTASKS python3 -u benchmark_v7.py \
    --instances 30 \
    --reads 15 \
    --num-steps 150 \
    --sweeps 20 \
    --worldline-sweeps 3 \
    --slices 32 \
    --replicas-sqa 4 \
    --calib-probes 12 \
    --calib-sweeps 10 \
    --beta-ramp-fraction 0.3 \
    --out-dir results

echo "=== Done ==="
date
