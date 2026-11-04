#!/bin/bash
# A14 — factor alignment through an ordered template of L slots.
#
# Fixes the one multi-video defect measured tonight: A02 found the pairwise plan
# disagrees with its own composition by 18.5 frames, with only 39% of frames
# agreeing within 4. Factoring through a shared template makes composition exact
# (verified to 3e-08 in scripts/test_ray_geometry.py), so cycle-consistency is
# structural in the TRAINING TARGET.
#
# L >> n_clusters on purpose. E13 already factored through the K=12 action
# posterior and barely helped; A05's granularity result explains why 12 slots
# cannot carry frame correspondence. The L=12 arm below is that control.
#
# PRE-REGISTERED, before any run:
#  * Primary: eval-time cycle error must fall. The template only shapes the
#    pseudo-label; whether the LEARNED model then produces consistent pairwise
#    plans is an empirical question, not a consequence. If cycle error does not
#    move, the mechanism did not transfer and the arm fails.
#  * Reported together, per A07: MoF, many-to-one purity, ECE, cycle error.
#  * L=12 is expected to behave like E13 (little change). If L=64 and L=12 are
#    the same, granularity is not the active ingredient and the design is wrong.
#  * Resolution floor ~5 pp on MoF unless the paired std is small.
# Baseline arm is A01_random_FSeval_s{0,1,2}, identical config, not re-run.
set -euo pipefail
cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"; conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"
COMMON="-d FSeval -ac all -ne 30 -lat 0.11 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 --rho 0.15 --radius-gw 0.04"

run () { # run <tag> <clusters> <extra...>
  local TAG=$1; local K=$2; shift 2
  for SEED in 0 1 2; do
    local RID="A14_${TAG}_FSeval_s${SEED}"
    if [[ -d "$VASOT_ARTIFACTS_DIR/runs/$RID" ]]; then echo "skip $RID"; continue; fi
    echo ""; echo "=== $RID ==="
    python3 train.py $COMMON -c "$K" --seed "$SEED" "$@" \
      --wandb --wandb-project vasot-repro \
      --wandb-tags A14 template FSeval "$TAG" "seed${SEED}" \
      --run-id "$RID"
  done
}

run tmpl64        12 --template-slots 64 --template-warmup-epochs 5
run tmpl12_ctrl   12 --template-slots 12 --template-warmup-epochs 5
run tmpl64_K48    48 --template-slots 64 --template-warmup-epochs 5
echo ""; echo "=== A14 COMPLETE ==="
