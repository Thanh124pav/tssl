#!/bin/bash
# A15 — what is the active ingredient in A14?
#
# A14 gave +5.5 pp MoF on 3/3 seeds, but its L=12 control matched L=64, which
# was the pre-registered falsification of the granularity hypothesis. So the
# gain comes from something else. Three one-factor ablations:
#
#   soft     soft slot assignment, so composition is NOT exact (0.006 vs 3e-08).
#            If the gain survives, exactness is not the ingredient.
#   frozen   the template is never learned. If a random fixed template works as
#            well, the FACTORING is the ingredient and the template's content is
#            not -- the same control logic that decided G01.
#   blend05  half pairwise, half induced: a dose-response check.
set -euo pipefail
cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"; conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"
COMMON="-d FSeval -ac all -c 12 -ne 30 -lat 0.11 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 --rho 0.15 --radius-gw 0.04 --template-slots 64 --template-warmup-epochs 5"
run () { local TAG=$1; shift
  for SEED in 0 1 2; do
    local RID="A15_${TAG}_FSeval_s${SEED}"
    [[ -d "$VASOT_ARTIFACTS_DIR/runs/$RID" ]] && { echo "skip $RID"; continue; }
    echo ""; echo "=== $RID ==="
    python3 train.py $COMMON --seed "$SEED" "$@" --wandb --wandb-project vasot-repro \
      --wandb-tags A15 template_ablation FSeval "$TAG" "seed${SEED}" --run-id "$RID"
  done; }
run soft    --template-soft
run frozen  --template-frozen
run blend05 --template-blend 0.5
echo ""; echo "=== A15 COMPLETE ==="
