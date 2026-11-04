#!/bin/bash
# A10 — first training runs of the A09 ray/sphere geometry objectives.
# Arms are one-factor-at-a-time; the baseline arm is A01_random_FSeval_s{0,1,2},
# whose config is identical to COMMON below, so it is not re-run.
set -euo pipefail
cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"; conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"
COMMON="-d FSeval -ac all -c 12 -ne 30 -lat 0.11 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 --rho 0.15 --radius-gw 0.04"
GEO="--ray-geometry --ray-videos 3 --ray-warmup-epochs 10 --ray-weight 0.1 --ray-group-weight 0.1"

run () {  # run <tag> <extra flags...>
  local TAG=$1; shift
  for SEED in 0 1 2; do
    local RID="A10_${TAG}_FSeval_s${SEED}"
    if [[ -d "$VASOT_ARTIFACTS_DIR/runs/$RID" ]]; then echo "skip $RID"; continue; fi
    echo ""; echo "=== $RID ==="
    python3 train.py $COMMON --seed "$SEED" "$@" \
      --wandb --wandb-project vasot-repro \
      --wandb-tags A10 ray_geometry FSeval "$TAG" "seed${SEED}" \
      --run-id "$RID"
  done
}

run sphere_full   $GEO --ray-mode sphere
run sphere_nocentre $GEO --ray-mode sphere --no-ray-centre-update
run ray_full      $GEO --ray-mode ray
echo ""; echo "=== A10 COMPLETE ==="
