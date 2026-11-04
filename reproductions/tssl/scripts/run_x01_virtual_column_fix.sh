#!/bin/bash
# X01 — paired-seed impact of the virtual-column off-by-one found in the
# 2026-11-04 code audit.
#
# train.py prepends the virtual alignment ROW but appends the virtual COLUMN,
# while align_ot returns T[:, 1:, 1:]. The dropped column is therefore a real Y
# frame and the virtual column survives inside the pseudo-label, shifting
# opt_codes one frame against `codes`. Two identical videos align with a mean
# -0.95 frame offset instead of the identity
# (scripts/test_diagnostics.py :: "solver: virtual column placement").
#
# Both arms are run fresh here, same seeds, same config, one factor changed, so
# the comparison is paired. The frozen E00 baseline is NOT reused as the control
# and NOT overwritten.
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"

# configs/vasot/baselines/fs_eval.yaml, identical to E00_baseline_FSeval_s1
COMMON="-d FSeval -ac all -c 12 -ne 30 -lat 0.11 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 --rho 0.15 --radius-gw 0.04"

for SEED in 0 1 2; do
  RID="X01a_vcol_published_FSeval_s${SEED}"
  if [[ -d "$VASOT_ARTIFACTS_DIR/runs/$RID" ]]; then
    echo "Skipping existing $RID"
  else
    echo ""
    echo "=== $RID (published padding, control arm, seed=$SEED) ==="
    python3 train.py $COMMON --seed "$SEED" \
      --wandb --wandb-project vasot-repro \
      --wandb-tags X01 virtual_column published_padding FSeval "seed${SEED}" audit \
      --run-id "$RID"
  fi

  RID="X01b_vcol_fixed_FSeval_s${SEED}"
  if [[ -d "$VASOT_ARTIFACTS_DIR/runs/$RID" ]]; then
    echo "Skipping existing $RID"
  else
    echo ""
    echo "=== $RID (virtual column prepended, treatment arm, seed=$SEED) ==="
    python3 train.py $COMMON --seed "$SEED" --fix-virtual-column \
      --wandb --wandb-project vasot-repro \
      --wandb-tags X01 virtual_column fixed_padding FSeval "seed${SEED}" audit \
      --run-id "$RID"
  fi
done

echo ""
echo "=== X01 COMPLETE ==="
