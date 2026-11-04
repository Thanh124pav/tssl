#!/bin/bash
# A17 — if the alignment target's frame-level content is HARMFUL, how far does
# "coarsen it" go? The limit of coarsening is removal.
#
# A14 section 7 measured that training on a target 94% different from the
# pairwise plan and 150x less accurate IMPROVES segmentation (+5.5 pp MoF,
# purity +0.066, ECE -0.070). A15's frozen arm then showed the template content
# is irrelevant. The natural end of that line: turn the alignment loss DOWN.
#
# beta is VASOT's alignment-loss weight (default 1.0). beta=0 is segmentation
# only: the joint objective's alignment half removed entirely.
#
# PRE-REGISTERED: if beta=0 matches the blur and template arms, the alignment
# branch contributes nothing positive to segmentation on this setup, and the
# whole A14/A15/A16 family is better described as "the alignment loss was
# hurting". If beta=0 is clearly WORSE than blur, then the alignment loss does
# contribute and what helps is coarsening it, not removing it -- which is the
# more interesting outcome and the one that keeps the joint formulation.
set -euo pipefail
cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"; conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"
COMMON="-d FSeval -ac all -c 12 -ne 30 -lat 0.11 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 --rho 0.15 --radius-gw 0.04"
for B in 0.0 0.25 0.5; do
  TAG=$(python3 -c "print('${B}'.replace('.','p'))")
  for SEED in 0 1 2; do
    RID="A17_beta${TAG}_FSeval_s${SEED}"
    [[ -d "$VASOT_ARTIFACTS_DIR/runs/$RID" ]] && { echo "skip $RID"; continue; }
    echo ""; echo "=== $RID ==="
    python3 train.py $COMMON --seed "$SEED" --beta "$B" \
      --wandb --wandb-project vasot-repro \
      --wandb-tags A17 alignment_weight FSeval "beta${TAG}" "seed${SEED}" --run-id "$RID"
  done
done
echo ""; echo "=== A17 COMPLETE ==="
