#!/bin/bash
# E11: separate the two uses of rho in VASOT on FSeval.
# Four runs (two settings x two seeds), sequential in one allocated job.
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR=/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts

COMMON=(-d FSeval -ac all -c 12 -ne 30 --rho 0.15 -lat 0.11
        -r 0.04 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0)

for VARIANT in seg0 align0; do
  case "$VARIANT" in
    seg0) OVERRIDE=(--rho-seg 0.0) ;;
    align0) OVERRIDE=(--rho-align 0.0) ;;
  esac
  for SEED in 0 1; do
    RID="E11_FSeval_${VARIANT}_s${SEED}"
    RUN_DIR="$VASOT_ARTIFACTS_DIR/runs/$RID"
    if [[ -f "$RUN_DIR/metrics.json" ]] &&
       [[ "$(jq -r .status "$RUN_DIR/manifest.json")" == completed ]]; then
      echo "Skipping completed $RID"
      continue
    fi
    if [[ -e "$RUN_DIR/manifest.json" ]]; then
      echo "Existing incomplete run $RID requires a new run ID; stopping" >&2
      exit 1
    fi
    echo "Starting $RID"
    python3 train.py "${COMMON[@]}" "${OVERRIDE[@]}" --seed "$SEED" \
      --wandb --wandb-project vasot-repro \
      --wandb-tags E11 rho_split FSeval "$VARIANT" "seed${SEED}" \
      --run-id "$RID"
  done
done

echo '=== E11 ALL RUNS COMPLETE ==='
