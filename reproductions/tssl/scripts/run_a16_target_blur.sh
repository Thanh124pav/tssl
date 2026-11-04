#!/bin/bash
# A16 — the parsimonious form: blur the alignment target, no template at all.
#
# A15's frozen arm matched A14's learned template exactly on seed 0 (0.5153 vs
# 0.5153), so the template's content is doing nothing and the active ingredient
# is the BLOCK STRUCTURE of the pseudo-label (A14 section 7: block-variance
# explained 1.0000, support 12.4 frames ~ one median action).
#
# If that is right, a box blur of the existing pairwise target reproduces the
# gain with no slots, no parameters and three lines. W=6 gives a 13-frame window,
# matching the induced plan's measured 12.4-frame support.
#
# PRE-REGISTERED: if blur matches the template arms, the template apparatus is
# unnecessary and A14/A15 should be reported as "coarsen the alignment target".
# If blur does NOT match, something about the template matters after all and the
# block-structure explanation is incomplete.
set -euo pipefail
cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"; conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"
COMMON="-d FSeval -ac all -c 12 -ne 30 -lat 0.11 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 --rho 0.15 --radius-gw 0.04"
for W in 3 6 12; do
  for SEED in 0 1 2; do
    RID="A16_blur${W}_FSeval_s${SEED}"
    [[ -d "$VASOT_ARTIFACTS_DIR/runs/$RID" ]] && { echo "skip $RID"; continue; }
    echo ""; echo "=== $RID ==="
    python3 train.py $COMMON --seed "$SEED" --align-target-blur "$W" \
      --wandb --wandb-project vasot-repro \
      --wandb-tags A16 target_blur FSeval "w${W}" "seed${SEED}" --run-id "$RID"
  done
done
echo ""; echo "=== A16 COMPLETE ==="
