#!/bin/bash
# E-phase pipeline: sequential GPU experiments on FSeval
# Runs: E01_s0 (already running), E01_s1, E03_gauss, E04_bezier
# Each run uses the isolated runner with wandb logging
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"

# Common FSeval baseline args
COMMON="-d FSeval -ac all -c 12 -ne 30 --rho 0.15 -lat 0.11 -r 0.04 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0"

echo ""
echo "=== E01 B-spline FSeval seed 1 ==="
python3 train.py $COMMON --seed 1 \
  --bspline --bspline-tol 0.05 --bspline-kmax 0 --bspline-warmup-epochs 5 \
  --bspline-space X --bspline-blend-max 1.0 --bspline-blend-rampup 0 \
  --wandb --wandb-project vasot-repro --wandb-tags E01 bspline FSeval seed1 \
  --run-id E01_bspline_FSeval_s1

echo ""
echo "=== E03 Gaussian FSeval seed 0 (sigma=2.0) ==="
python3 train.py $COMMON --seed 0 \
  --gauss-smooth --gauss-sigma 2.0 --gauss-blend-max 1.0 --gauss-warmup-epochs 5 \
  --wandb --wandb-project vasot-repro --wandb-tags E03 gaussian FSeval seed0 \
  --run-id E03_gauss_s2_FSeval_s0

echo ""
echo "=== E03 Gaussian FSeval seed 1 (sigma=2.0) ==="
python3 train.py $COMMON --seed 1 \
  --gauss-smooth --gauss-sigma 2.0 --gauss-blend-max 1.0 --gauss-warmup-epochs 5 \
  --wandb --wandb-project vasot-repro --wandb-tags E03 gaussian FSeval seed1 \
  --run-id E03_gauss_s2_FSeval_s1

echo ""
echo "=== E04 Bezier FSeval seed 0 (nseg=4, adaptive) ==="
python3 train.py $COMMON --seed 0 \
  --bezier --bezier-nseg 4 --bezier-tol 0.05 --bezier-maxseg 32 --bezier-warmup-epochs 5 \
  --bezier-space X --bezier-blend-max 1.0 --bezier-blend-rampup 0 \
  --wandb --wandb-project vasot-repro --wandb-tags E04 bezier FSeval seed0 \
  --run-id E04_bezier_FSeval_s0

echo ""
echo "=== E04 Bezier FSeval seed 1 (nseg=4, adaptive) ==="
python3 train.py $COMMON --seed 1 \
  --bezier --bezier-nseg 4 --bezier-tol 0.05 --bezier-maxseg 32 --bezier-warmup-epochs 5 \
  --bezier-space X --bezier-blend-max 1.0 --bezier-blend-rampup 0 \
  --wandb --wandb-project vasot-repro --wandb-tags E04 bezier FSeval seed1 \
  --run-id E04_bezier_FSeval_s1

echo ""
echo "=== Baseline FSeval seed 1 (for paired comparison) ==="
python3 train.py $COMMON --seed 1 \
  --wandb --wandb-project vasot-repro --wandb-tags baseline FSeval seed1 \
  --run-id E00_baseline_FSeval_s1

echo ""
echo "=== ALL E-PIPELINE RUNS COMPLETE ==="
