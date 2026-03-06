#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sqa_array.sh  —  SLURM job array for SQA (no mpi4py required)
#
# Strategy: embarrassingly-parallel job array.
#   • Each array task is an independent Python process (no MPI needed).
#   • Task i runs with seed = BASE_SEED + i, writing results/run_<i>.json.
#   • After all tasks complete, merge with merge_array_results.py.
#
# This approach requires ONLY a standard Python install with qlatannealv4.
# Use this when mpi4py is unavailable or when you want maximum node flexibility.
#
# Usage:
#   sbatch run_sqa_array.sh
#
# Then merge:
#   python examples/python/merge_array_results.py results/run_*.json --out merged_best
#
# Override parameters:
#   sbatch --export=N_SPINS=20000,METHOD=sqa,READS=16 run_sqa_array.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=qanneal_sqa_array
#SBATCH --array=0-31                   # 32 independent tasks (0..31)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # multiprocessing workers per task
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --output=logs/sqa_array_%A_%a.out
#SBATCH --error=logs/sqa_array_%A_%a.err

# ── Environment ───────────────────────────────────────────────────────────────
# module load python/3.11
# source /path/to/venv/bin/activate

# ── Parameters ────────────────────────────────────────────────────────────────
N_SPINS=${N_SPINS:-10000}
DEGREE=${DEGREE:-10}
METHOD=${METHOD:-sqapt}          # sa | sqa | sqapt | ctpimc
MODE=${MODE:-balanced}           # fast | balanced | accurate
READS=${READS:-8}                # reads per task
WORKERS=${WORKERS:-4}            # multiprocessing workers within one task
BASE_SEED=${BASE_SEED:-1000}

PROBLEM_TYPE=${PROBLEM_TYPE:-random_sparse}
PROBLEM_FILE=${PROBLEM_FILE:-}

# Task-specific seed ensures full diversity across tasks
TASK_SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))
OUT="results/run_${SLURM_ARRAY_TASK_ID}"

# ── Script path ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/../../examples/python/hpc_sqa_launcher.py"

mkdir -p logs results

echo "Task ${SLURM_ARRAY_TASK_ID}: seed=${TASK_SEED}  method=${METHOD}  n=${N_SPINS}"

# ── Build command ─────────────────────────────────────────────────────────────
CMD_ARGS=(
    --problem-type "${PROBLEM_TYPE}"
    --n "${N_SPINS}"
    --degree "${DEGREE}"
    --method "${METHOD}"
    --mode "${MODE}"
    --reads "${READS}"
    --workers "${WORKERS}"
    --seed "${TASK_SEED}"
    --out "${OUT}"
)

if [ -n "${PROBLEM_FILE}" ]; then
    CMD_ARGS+=(--problem-file "${PROBLEM_FILE}")
fi

python "${LAUNCHER}" "${CMD_ARGS[@]}"
