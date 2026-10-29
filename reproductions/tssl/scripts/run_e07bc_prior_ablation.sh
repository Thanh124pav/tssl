#!/bin/bash
# E07b — Hypothesis C direct intervention: rho (temporal prior scale) ablation
# E07c — Hypothesis C direct intervention: radius_gw (structural C^x band) ablation
#
# Baseline rho=0.15, radius_gw=0.04 (from configs/vasot/baselines/fs_eval.yaml).
# E07a already ruled out A (ε knob) as actionable. If C is the dominant harmful
# prior, dialing rho down or radius up should recover or improve MoF.
#
# 8 runs total, 2 seeds each, serialised on one GPU.
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"

# Common FSeval baseline (match configs/vasot/baselines/fs_eval.yaml, minus the knob under test)
COMMON="-d FSeval -ac all -c 12 -ne 30 -lat 0.11 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0"

# ---- E07b: rho ablation (temporal prior R scale) ---------------------------
for RHO in 0.0 0.05; do
  TAG_RHO=$(python3 -c "print('${RHO}'.replace('.','p'))")
  for SEED in 0 1; do
    RID="E07b_rho${TAG_RHO}_FSeval_s${SEED}"
    echo ""
    echo "=== $RID (rho=$RHO, radius_gw=0.04 baseline, seed=$SEED) ==="
    python3 train.py $COMMON --seed $SEED --rho $RHO --radius-gw 0.04 \
      --wandb --wandb-project vasot-repro \
      --wandb-tags E07b rho_ablation FSeval "seed${SEED}" "rho${TAG_RHO}" hypothesis_C \
      --run-id "$RID"
  done
done

# ---- E07c: radius_gw ablation (C^x structural band width) ------------------
for RADIUS in 0.12 0.50; do
  TAG_R=$(python3 -c "print('${RADIUS}'.replace('.','p'))")
  for SEED in 0 1; do
    RID="E07c_r${TAG_R}_FSeval_s${SEED}"
    echo ""
    echo "=== $RID (radius_gw=$RADIUS, rho=0.15 baseline, seed=$SEED) ==="
    python3 train.py $COMMON --seed $SEED --rho 0.15 --radius-gw $RADIUS \
      --wandb --wandb-project vasot-repro \
      --wandb-tags E07c radius_ablation FSeval "seed${SEED}" "r${TAG_R}" hypothesis_C \
      --run-id "$RID"
  done
done

echo ""
echo "=== E07bc ALL RUNS COMPLETE ==="
