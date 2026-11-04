#!/bin/bash
# A18 — does a template that actually ADAPTS do better than the random fixed one?
#
# A14 section 8: the template parameter never receives a gradient, because the
# plan that uses it is built inside torch.no_grad. So every A14/A15 arm ran with
# a random, never-learned template, and A15's "frozen" control was a duplicate
# run rather than an ablation. The +5.5 pp was obtained with a random template.
#
# This is the comparison that was supposed to exist. Each slot is EMA-updated
# toward the mean embedding of the frames assigned to it, which is how the repo
# learns its other codebook-like objects and what A09 used for the centres.
#
# PRE-REGISTERED: if the adaptive template matches the random one, the partition
# genuinely does not need to carry information and "coarsen the target" is the
# whole story. If it beats it, there IS something to learn in the shared
# coordinate and A14's original framing was right for the wrong reason.
set -euo pipefail
cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"; conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"
COMMON="-d FSeval -ac all -c 12 -ne 30 -lat 0.11 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 --rho 0.15 --radius-gw 0.04 --template-slots 64 --template-warmup-epochs 5"
for M in 0.9 0.99; do
  TAG=$(python3 -c "print('${M}'.replace('.','p'))")
  for SEED in 0 1 2; do
    RID="A18_ema${TAG}_FSeval_s${SEED}"
    [[ -d "$VASOT_ARTIFACTS_DIR/runs/$RID" ]] && { echo "skip $RID"; continue; }
    echo ""; echo "=== $RID ==="
    python3 train.py $COMMON --seed "$SEED" --template-ema "$M" \
      --wandb --wandb-project vasot-repro \
      --wandb-tags A18 template_ema FSeval "ema${TAG}" "seed${SEED}" --run-id "$RID"
  done
done
echo ""; echo "=== A18 COMPLETE ==="
