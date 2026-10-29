#!/bin/bash
# E14: fixed-checkpoint, one-factor-at-a-time B-spline fit audit.
set -euo pipefail

cd /mnt/data/thanhpv/tssl
source reproductions/env.sh
PY=/mnt/data/thanhpv/miniconda3/envs/acvcl/bin/python
CKPT=reproductions/tssl/artifacts/runs/E00_baseline_FSeval_s1/checkpoints/last.ckpt
OUT=reproductions/tssl/artifacts/diagnostics/e14_bspline_fit
mkdir -p "$OUT"

CONFIGS=(
  "1 0.05 32"
  "2 0.05 32"
  "3 0.05 32"
  "3 0.025 32"
  "3 0.10 32"
  "3 0.05 16"
  "3 0.05 64"
)

for SPACE in X logits; do
  for CONFIG in "${CONFIGS[@]}"; do
    read -r DEG TOL KMAX <<< "$CONFIG"
    TAG_TOL=${TOL/./p}
    TARGET="$OUT/${SPACE}_p${DEG}_tol${TAG_TOL}_k${KMAX}.json"
    if [[ -f "$TARGET" ]]; then
      echo "Skipping existing $TARGET"
      continue
    fi
    echo "Auditing space=$SPACE degree=$DEG tol=$TOL kmax=$KMAX"
    "$PY" reproductions/tssl/scripts/eval_bspline_fit_quality.py \
      --ckpt "$CKPT" --space "$SPACE" --degree "$DEG" --tol "$TOL" \
      --kmax "$KMAX" --videos 20 --out "$TARGET"
  done
done

# Capacity recovery, activated because the complete kmax<=64 grid saturated
# on every video and no video met tol=0.05. Keep degree/tolerance fixed.
for SPACE in X logits; do
  for KMAX in 96 128 192 256; do
    TARGET="$OUT/${SPACE}_p3_tol0p05_k${KMAX}.json"
    if [[ -f "$TARGET" ]]; then
      echo "Skipping existing $TARGET"
      continue
    fi
    echo "Capacity audit space=$SPACE degree=3 tol=0.05 kmax=$KMAX"
    "$PY" reproductions/tssl/scripts/eval_bspline_fit_quality.py \
      --ckpt "$CKPT" --space "$SPACE" --degree 3 --tol 0.05 \
      --kmax "$KMAX" --videos 20 --out "$TARGET"
  done
done

# The action-space hypothesis concerns probabilities on the action simplex,
# not only the unnormalised E10a logits.
for KMAX in 32 64 96 128 192 256; do
  TARGET="$OUT/posterior_p3_tol0p05_k${KMAX}.json"
  if [[ ! -f "$TARGET" ]]; then
    echo "Posterior audit degree=3 tol=0.05 kmax=$KMAX"
    "$PY" reproductions/tssl/scripts/eval_bspline_fit_quality.py \
      --ckpt "$CKPT" --space posterior --degree 3 --tol 0.05 \
      --kmax "$KMAX" --videos 20 --out "$TARGET"
  fi
done
TARGET="$OUT/posterior_p1_tol0p05_k64.json"
if [[ ! -f "$TARGET" ]]; then
  "$PY" reproductions/tssl/scripts/eval_bspline_fit_quality.py \
    --ckpt "$CKPT" --space posterior --degree 1 --tol 0.05 \
    --kmax 64 --videos 20 --out "$TARGET"
fi
echo '=== E14 B-SPLINE FIT AUDIT COMPLETE ==='
