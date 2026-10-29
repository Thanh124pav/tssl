#!/bin/bash
# E10c: third-seed confirmation of the only E10 variant passing the E00 gate.
# Baseline and latent-12/blend-0.3 B-spline use identical frozen FSeval config.
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR=/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts

COMMON=(-d FSeval -ac all -c 12 -ne 30 --rho 0.15 -lat 0.11
        -r 0.04 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0
        --seed 2 --wandb --wandb-project vasot-repro)

for VARIANT in baseline bspl_latent_dz12_b0p3; do
  RID="E10c_${VARIANT}_FSeval_s2"
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
  if [[ "$VARIANT" == bspl_latent_dz12_b0p3 ]]; then
    EXTRA=(--bspline --bspline-space latent --latent-dim 12
           --bspline-tol 0.05 --bspline-kmax 0
           --bspline-warmup-epochs 5 --bspline-blend-max 0.3
           --bspline-blend-rampup 5)
  fi
  echo "Starting $RID"
  python3 train.py "${COMMON[@]}" "${EXTRA[@]}" \
    --wandb-tags E10c seed2 "$VARIANT" --run-id "$RID"
  python3 -c 'import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
bad = [key for key, value in state.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
if bad:
    raise SystemExit(f"Non-finite checkpoint tensors: {bad[:5]} (count={len(bad)})")
print("Checkpoint finite:", sys.argv[1])' "$RUN_DIR/checkpoints/last.ckpt"
done

echo '=== E10c SEED-2 VALIDATION COMPLETE ==='
