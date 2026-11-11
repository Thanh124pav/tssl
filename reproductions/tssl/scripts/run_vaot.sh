#!/usr/bin/env bash
# Train VAOT on one dataset/action with the paper settings. Re-running resumes
# from <out>/ckpts/last.ckpt.
#   run_vaot.sh pouring pouring 40
#   run_vaot.sh penn_action baseball_pitch 20
# Extra args go to vaot_train.py; on a large GPU add `--chunk 0 --workers 10`
# to run the exact upstream forward. CONDA_ENV overrides the env name.
set -euo pipefail
DATASET=$1; ACTION=$2; FRAMES=$3; shift 3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-deeplearning}"
OUT=$ROOT/data/vaot_runs/$DATASET/$ACTION
mkdir -p "$OUT"
python "$ROOT/scripts/vaot_train.py" \
  --data_path "$ROOT/data/final/vaot/$DATASET/$ACTION" \
  --out "$OUT" --num_frames "$FRAMES" "$@" 2>&1 | grep --line-buffered -v -i "warn\|weights_only" >> "$OUT/train.log"
