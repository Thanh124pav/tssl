#!/bin/bash
# E10b — B-spline smoothing in a dedicated learnable latent space on FSeval.
#
# A new (latent_dim, D_out) learnable basis `latent_basis` is added to the
# model (unit-norm rows, grad flows through the smoothing path and through
# the per-step renormalization that already covers `clusters`). The forward
# path becomes:
#     Z        = features @ latent_basis.T    # (B, T, latent_dim)
#     Z_smooth = fit_bspline(Z)
#     smoothed = Z_smooth @ latent_basis      # (B, T, D_out)
#     features = normalize((1 - blend) * features + blend * smoothed)
#
# Difference vs E10a (logits-space):
#   logits-space reuses the `clusters` (shape (n_clusters, D)) matrix, so the
#   subspace it projects into is the action codebook — tied to the OT cost.
#   latent-space has its own basis learned purely through the smoothing path
#   (plus a shared per-step row-norm renorm), so it can learn a different
#   subspace that is "smoothness-friendly" rather than "action-separating".
#
# Grid: latent_dim ∈ {12, 24}, blend_max ∈ {0.3, 1.0}, seeds {0, 1}
#       = 2 × 2 × 2 = 8 runs. latent_dim=12 is apples-to-apples with E10a
#       (same rank as logits), latent_dim=24 gives extra representational room.
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"

# Same FSeval baseline as E07b/E10a (configs/vasot/baselines/fs_eval.yaml)
COMMON="-d FSeval -ac all -c 12 -ne 30 --rho 0.15 -lat 0.11 -r 0.04 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0"

for DZ in 12 24; do
  for BLEND in 0.3 1.0; do
    TAG_B=$(python3 -c "print('${BLEND}'.replace('.','p'))")
    for SEED in 0 1; do
      RID="E10b_bspl_latent_dz${DZ}_b${TAG_B}_FSeval_s${SEED}"
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
      echo ""
      echo "=== $RID (bspline latent, dz=$DZ, blend_max=$BLEND, seed=$SEED) ==="
      python3 train.py $COMMON --seed $SEED \
        --bspline --bspline-space latent --latent-dim $DZ \
        --bspline-tol 0.05 --bspline-kmax 0 \
        --bspline-warmup-epochs 5 \
        --bspline-blend-max $BLEND --bspline-blend-rampup 5 \
        --wandb --wandb-project vasot-repro \
        --wandb-tags E10b bspline latent FSeval "seed${SEED}" "blend${TAG_B}" "dz${DZ}" \
        --run-id "$RID"
      # Guard against NaN checkpoints like E04 Bézier produced.
      python3 -c 'import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
bad = [k for k, v in state.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
if bad:
    raise SystemExit(f"Non-finite checkpoint tensors: {bad[:5]} (count={len(bad)})")
print("Checkpoint finite:", sys.argv[1])' "$RUN_DIR/checkpoints/last.ckpt"
    done
  done
done

echo ""
echo "=== E10b ALL RUNS COMPLETE ==="
