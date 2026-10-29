#!/bin/bash
# E07a — Hypothesis A direct intervention: ε (entropy) ablation on FSeval
# Softens T* at the OT solver level without changing the representation.
# Compares against baseline (eps=0.07) and the smoothing variants (E01/E03/E04).
#
# 4 runs: eps ∈ {0.15, 0.30}, seeds {0, 1}
# Baseline eps=0.07 is already captured in E00_baseline_FSeval_s1 and baseline_s0.
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"

# Common FSeval baseline args — must match configs/vasot/baselines/fs_eval.yaml
COMMON="-d FSeval -ac all -c 12 -ne 30 --rho 0.15 -lat 0.11 -r 0.04 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0"

for EPS in 0.15 0.30; do
  TAG_EPS=$(python3 -c "print('${EPS}'.replace('.','p'))")
  for SEED in 0 1; do
    RID="E07a_eps${TAG_EPS}_FSeval_s${SEED}"
    echo ""
    echo "=== $RID (eps_train=$EPS, seed=$SEED) ==="
    python3 train.py $COMMON --seed $SEED --eps-train $EPS \
      --wandb --wandb-project vasot-repro \
      --wandb-tags E07a eps_ablation FSeval "seed${SEED}" "eps${TAG_EPS}" hypothesis_A \
      --run-id "$RID"
  done
done

echo ""
echo "=== E07a ALL RUNS COMPLETE ==="
