#!/bin/bash
#SBATCH --account=def-amadou
#SBATCH --job-name=cosmap_bench
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

EXPERIMENT_DIR="/home/fenosoa/links/projects/def-amadou/fenosoa/Notebook_cosmap/CosMAP-dr/experiments"
SEED="${BENCHMARK_SEED:-42}"

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5
module load scipy-stack
module load faiss/1.12.0

source "$SCRATCH/balsac_venv/bin/activate"
cd "$EXPERIMENT_DIR"

# Prevent BLAS/OpenMP libraries from oversubscribing the five allocated CPUs.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VECLIB_MAXIMUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$SEED"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"

echo "Started:           $(date --iso-8601=seconds)"
echo "Host:              $(hostname)"
echo "Job ID:            $SLURM_JOB_ID"
echo "Array task ID:     $SLURM_ARRAY_TASK_ID"
echo "CPUs per task:     $SLURM_CPUS_PER_TASK"
echo "Seed:              $SEED"
echo "Python:            $(which python)"
echo "Working directory: $(pwd)"
echo

nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

PYTHON_ARGS=(
    run_benchmark_task.py
    --index "$SLURM_ARRAY_TASK_ID"
    --seed "$SEED"
)

if [[ "${BENCHMARK_FORCE:-0}" == "1" ]]; then
    PYTHON_ARGS+=(--force)
fi

srun --cpu-bind=cores python -u "${PYTHON_ARGS[@]}"

echo
echo "Finished: $(date --iso-8601=seconds)"
