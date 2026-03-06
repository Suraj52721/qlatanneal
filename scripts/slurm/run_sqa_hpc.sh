#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sqa_hpc.sh  —  Multi-node MPI SQA job for huge sparse Ising problems
#
# Strategy: N_RANKS MPI processes × READS_PER_RANK independent SQA reads.
#           Total reads = N_RANKS × READS_PER_RANK (perfectly parallel).
#           Each rank solves the same problem with a different random seed.
#
# Requirements:
#   - Python environment with qlatannealv4 built in-place (python/qanneal)
#   - mpi4py installed in the same Python environment
#   - OpenMPI or MPICH available (load appropriate module below)
#
# Usage:
#   sbatch run_sqa_hpc.sh
#   # or override parameters:
#   sbatch --export=N_SPINS=50000,METHOD=sqapt,READS=16 run_sqa_hpc.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=qanneal_sqa_hpc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8            # 8 MPI ranks per node (1 per core)
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --output=logs/sqa_hpc_%j.out
#SBATCH --error=logs/sqa_hpc_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
# Adjust module names for your cluster:
# module load openmpi/4.1
# module load python/3.11
# source /path/to/venv/bin/activate

# ── Parameters (override via --export or edit here) ──────────────────────────
N_SPINS=${N_SPINS:-10000}        # spins in random sparse Ising problem
DEGREE=${DEGREE:-10}             # neighbours per spin (k-regular graph)
METHOD=${METHOD:-sqapt}          # sa | sqa | sqapt | ctpimc
MODE=${MODE:-balanced}           # fast | balanced | accurate
READS=${READS:-8}                # independent reads PER RANK
SEED=${SEED:-42}
OUT=${OUT:-results/hpc_run_${SLURM_JOB_ID}}

PROBLEM_TYPE=${PROBLEM_TYPE:-random_sparse}
PROBLEM_FILE=${PROBLEM_FILE:-}   # only used when PROBLEM_TYPE=from_file

# ── Script path ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/../../examples/python/hpc_sqa_launcher.py"

# ── Create output directory ───────────────────────────────────────────────────
mkdir -p logs
mkdir -p "$(dirname "${OUT}")"

# ── Reporting ─────────────────────────────────────────────────────────────────
N_RANKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
TOTAL_READS=$((N_RANKS * READS))
echo "=============================="
echo " qanneal SQA HPC run"
echo "=============================="
echo "  Nodes       : ${SLURM_NNODES}"
echo "  MPI ranks   : ${N_RANKS}"
echo "  Method      : ${METHOD} [${MODE}]"
echo "  Spins       : ${N_SPINS}  degree=${DEGREE}"
echo "  Reads/rank  : ${READS}   (total: ${TOTAL_READS})"
echo "  Output      : ${OUT}"
echo "=============================="

# ── Build command ─────────────────────────────────────────────────────────────
CMD_ARGS=(
    --problem-type "${PROBLEM_TYPE}"
    --n "${N_SPINS}"
    --degree "${DEGREE}"
    --method "${METHOD}"
    --mode "${MODE}"
    --reads "${READS}"
    --seed "${SEED}"
    --out "${OUT}"
    --mpi
)

# Append problem file if provided
if [ -n "${PROBLEM_FILE}" ]; then
    CMD_ARGS+=(--problem-file "${PROBLEM_FILE}")
fi

# ── Launch ────────────────────────────────────────────────────────────────────
srun python "${LAUNCHER}" "${CMD_ARGS[@]}"

# ── Summary ───────────────────────────────────────────────────────────────────
if [ -f "${OUT}.json" ]; then
    echo ""
    echo "Result summary:"
    python - <<'EOF'
import json, sys, os
path = os.environ.get("OUT", "hpc_sqa_result") + ".json"
try:
    d = json.load(open(path))
    print(f"  n_spins     : {d.get('n_spins', '?'):,}")
    print(f"  method      : {d.get('method','?').upper()}")
    print(f"  total reads : {d.get('reads','?')}")
    print(f"  best_energy : {d.get('best_energy','?'):.6f}")
    print(f"  mean±std    : {d.get('energy_mean','?'):.4f} ± {d.get('energy_std','?'):.4f}")
except Exception as e:
    print(f"  (could not parse result: {e})")
EOF
fi
