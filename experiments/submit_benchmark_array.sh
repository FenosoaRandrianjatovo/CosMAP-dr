#!/bin/bash
# Usage:
#   bash submit_benchmark_array.sh [SEED] [MAX_PARALLEL]
#
# Examples:
#   bash submit_benchmark_array.sh
#   bash submit_benchmark_array.sh 42 3
#   BENCHMARK_FORCE=1 bash submit_benchmark_array.sh 42 3

set -euo pipefail

EXPERIMENT_DIR="/home/fenosoa/links/projects/def-amadou/fenosoa/Notebook_cosmap/CosMAP-dr/experiments"
SEED="${1:-42}"
MAX_PARALLEL="${2:-3}"
N_DATASETS=12

cd "$EXPERIMENT_DIR"
mkdir -p logs

if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
    echo "Error: SEED must be an integer; received '$SEED'." >&2
    exit 2
fi

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MAX_PARALLEL must be a positive integer; received '$MAX_PARALLEL'." >&2
    exit 2
fi

echo "Submitting $N_DATASETS datasets with seed $SEED."
echo "At most $MAX_PARALLEL array tasks will run concurrently."

sbatch \
    --array="0-$((N_DATASETS - 1))%${MAX_PARALLEL}" \
    --export="ALL,BENCHMARK_SEED=${SEED}" \
    benchmark_array.sbatch


# bash submit_benchmark_array.sh 42 3