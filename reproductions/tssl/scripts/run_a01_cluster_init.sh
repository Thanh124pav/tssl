#!/bin/bash
# A01 — does a different action-prototype initialisation help learning?
#
# Motivated by G00: the learned prototypes stay at cosine 0.87 (FSeval) / 0.98
# (YTI coffee) from their OWN random initialisation, while two different random
# initialisations sit at 0.24-0.27 from each other. If the codebook barely
# moves, the initialisation largely IS the final codebook, and changing it
# should change the result.
#
# Arms (all unit-norm, none uses labels or GT):
#   random      published behaviour: normalised Gaussian
#   orthogonal  orthonormal rows (K=12 <= d=40, so exactly orthogonal)
#   data        K randomly sampled frame embeddings at init
#   temporal    mean embedding of each of K equal relative-time bins
#   kmeans      k-means centroids of the frame embeddings at init
#
# Padding: published behaviour, for comparability with earlier packets.
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"

COMMON="-d FSeval -ac all -c 12 -ne 30 -lat 0.11 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 --rho 0.15 --radius-gw 0.04"

for INIT in random orthogonal data temporal kmeans; do
  for SEED in 0 1 2; do
    RID="A01_${INIT}_FSeval_s${SEED}"
    if [[ -d "$VASOT_ARTIFACTS_DIR/runs/$RID" ]]; then
      echo "Skipping existing $RID"
      continue
    fi
    echo ""
    echo "=== $RID (cluster-init=$INIT, seed=$SEED) ==="
    python3 train.py $COMMON --seed "$SEED" --cluster-init "$INIT" \
      --wandb --wandb-project vasot-repro \
      --wandb-tags A01 cluster_init FSeval "init_${INIT}" "seed${SEED}" action_space \
      --run-id "$RID"
  done
done

echo ""
echo "=== A01 TRAINING COMPLETE ==="
