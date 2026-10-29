#!/bin/bash
# E12: paired FS Mid test of the FSeval-validated segmentation-prior change.
# Four sequential 30-epoch runs, two seeds, fixed README/PROTOCOL FS config.
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR=/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts

COMMON=(-d FS -ac all -c 19 -ne 30 --rho 0.15 -lat 0.15
        -r 0.04 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0)

for SEED in 0 1; do
  for VARIANT in baseline seg0; do
    RID="E12_FS_mid_${VARIANT}_s${SEED}"
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
    EXTRA=()
    if [[ "$VARIANT" == seg0 ]]; then
      EXTRA=(--rho-seg 0.0)
    fi
    echo "Starting $RID"
    python3 train.py "${COMMON[@]}" "${EXTRA[@]}" --seed "$SEED" \
      --wandb --wandb-project vasot-repro \
      --wandb-tags E12 FS_mid rho_seg0 "$VARIANT" "seed${SEED}" \
      --run-id "$RID"
    python3 -c 'import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
bad = [key for key, value in state.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
if bad:
    raise SystemExit(f"Non-finite checkpoint tensors: {bad[:5]} (count={len(bad)})")
print("Checkpoint finite:", sys.argv[1])' "$RUN_DIR/checkpoints/last.ckpt"
  done
done
echo '=== E12 FS MID COMPLETE ==='
