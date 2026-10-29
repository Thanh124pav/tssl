#!/bin/bash
# E15a: fit ceiling before integrating boundary-aware splines into training.
set -euo pipefail
cd /mnt/data/thanhpv/tssl
source reproductions/env.sh
PY=/mnt/data/thanhpv/miniconda3/envs/acvcl/bin/python
CKPT=reproductions/tssl/artifacts/runs/E00_baseline_FSeval_s1/checkpoints/last.ckpt
OUT=reproductions/tssl/artifacts/diagnostics/e15_boundary_fit
mkdir -p "$OUT"

for SPACE in X posterior; do
  for SOURCE in gt asot posterior; do
    for RATIO in 0.25 0.5; do
      TAG=${RATIO/./p}
      TARGET="$OUT/${SPACE}_${SOURCE}_ratio${TAG}.json"
      [[ -f "$TARGET" ]] && continue
      "$PY" reproductions/tssl/scripts/eval_boundary_aware_bspline_fit.py \
        --ckpt "$CKPT" --space "$SPACE" --boundary-source "$SOURCE" \
        --capacity-ratio "$RATIO" --degree 3 --tol 0.05 --min-segment 4 \
        --videos 20 --out "$TARGET"
    done
  done
done
echo '=== E15 BOUNDARY-AWARE FIT CEILING COMPLETE ==='
