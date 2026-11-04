#!/bin/bash
# A13 — give the encoder a temporal window.
#
# Tests A12's prediction directly: within-action phase is decodable from one raw
# frame at R2 0.103 and from a +-4 frame window at 0.314, so the single-frame
# input, not the projection or the latent space, is the binding constraint. The
# +-4 optimum is a 9-frame window against an 11-frame median FSeval action.
#
# Pre-registered before any run: the mechanism check is that the phase
# decodability of the TRAINED embedding rises from 0.055 toward 0.314. MoF alone
# does not settle it (A07), so purity and ECE are reported beside it. The
# baseline arm is A01_random_FSeval_s{0,1,2}, identical config, not re-run.
set -euo pipefail
cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"; conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"
COMMON="-d FSeval -ac all -c 12 -ne 30 -lat 0.11 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 --rho 0.15 --radius-gw 0.04 -bs 1"
for W in 4 2 8; do
  for SEED in 0 1 2; do
    RID="A13_w${W}_FSeval_s${SEED}"
    if [[ -d "$VASOT_ARTIFACTS_DIR/runs/$RID" ]]; then echo "skip $RID"; continue; fi
    echo ""; echo "=== $RID (input window +-${W}, seed ${SEED}) ==="
    python3 train.py $COMMON --seed "$SEED" --input-window "$W" \
      --wandb --wandb-project vasot-repro \
      --wandb-tags A13 input_window FSeval "w${W}" "seed${SEED}" \
      --run-id "$RID"
  done
done
echo ""; echo "=== A13 COMPLETE ==="
