#!/bin/bash
# A05 — how does segmentation granularity K affect training?
#
# FSeval has 12 GT eval actions and the baseline uses K=12. This sweeps K far
# past the number of GT classes, up to K=512 > the 256 sampled frames per video.
#
# Protocol warning, which the analysis must respect: metrics.eval_mof assigns
# clusters to GT classes ONE-TO-ONE (scipy linear_sum_assignment over the
# contingency table). With K > #GT, at most #GT clusters can ever be matched and
# every frame in an unmatched cluster counts as an error. So a falling MoF at
# large K is partly mechanical. eval_a04_segment_confidence.py recomputes a
# many-to-one (purity) accuracy on the same checkpoints to separate "the
# representation got worse" from "the protocol penalises K".
#
# Padding: published behaviour (--fix-virtual-column OFF) so these runs stay
# comparable with every earlier packet. X01 measures the padding fix separately.
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"

COMMON="-d FSeval -ac all -ne 30 -lat 0.11 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 --rho 0.15 --radius-gw 0.04"

for K in 12 24 48 128 256 512; do
  for SEED in 0 1; do
    RID="A05_K${K}_FSeval_s${SEED}"
    if [[ -d "$VASOT_ARTIFACTS_DIR/runs/$RID" ]]; then
      echo "Skipping existing $RID"
      continue
    fi
    echo ""
    echo "=== $RID (K=$K, seed=$SEED) ==="
    python3 train.py $COMMON -c "$K" --seed "$SEED" \
      --wandb --wandb-project vasot-repro \
      --wandb-tags A05 granularity FSeval "K${K}" "seed${SEED}" action_space \
      --run-id "$RID"
  done
done

echo ""
echo "=== A05 TRAINING COMPLETE ==="
