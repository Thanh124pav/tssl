#!/bin/bash
# Third-seed confirmation for E11's segmentation-prior intervention.
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR=/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts

RID=E11c_FSeval_seg0_s2
RUN_DIR="$VASOT_ARTIFACTS_DIR/runs/$RID"
if [[ -f "$RUN_DIR/metrics.json" ]] &&
   [[ "$(jq -r .status "$RUN_DIR/manifest.json")" == completed ]]; then
  echo "Skipping completed $RID"
  exit 0
fi
if [[ -e "$RUN_DIR/manifest.json" ]]; then
  echo "Existing incomplete run $RID requires a new run ID; stopping" >&2
  exit 1
fi

python3 train.py -d FSeval -ac all -c 12 -ne 30 --rho 0.15 --rho-seg 0.0 \
  -lat 0.11 -r 0.04 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 \
  --seed 2 --wandb --wandb-project vasot-repro \
  --wandb-tags E11c rho_split FSeval seg0 seed2 \
  --run-id "$RID"
python3 -c 'import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
bad = [key for key, value in state.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
if bad:
    raise SystemExit(f"Non-finite checkpoint tensors: {bad[:5]} (count={len(bad)})")
print("Checkpoint finite:", sys.argv[1])' "$RUN_DIR/checkpoints/last.ckpt"
echo '=== E11c SEED-2 VALIDATION COMPLETE ==='
