#!/bin/bash
# E10a — B-spline smoothing in logits-space (action-space) on FSeval.
#
# Motivation. The FSeval X-space B-spline runs in E01 reduced MoF by 11.95 pp.
# Logits-space tests whether smoothing action scores instead of frame
# embeddings changes that result. It is a new hypothesis, not a recovery of
# the failed X-space fit.
#
# Logits-space is smaller (D=12 for FSeval = n_clusters) and the "piecewise
# constant" action-sequence inductive bias is a plausible match for cubic
# B-spline. Code is already present as --bspline-space logits:
#     L        = features @ clusters.T     # (N, n_clusters)
#     L_smooth = fit_bspline(L)
#     smoothed = L_smooth @ clusters
#
# Grid: 2 blend values × 2 seeds = 4 runs.
#   blend_max ∈ {0.3, 1.0}  → test both "regulariser" and "replace".
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"

# Same FSeval baseline as E07a/E07b (configs/vasot/baselines/fs_eval.yaml)
COMMON="-d FSeval -ac all -c 12 -ne 30 --rho 0.15 -lat 0.11 -r 0.04 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0"

for BLEND in 0.3 1.0; do
  TAG_B=$(python3 -c "print('${BLEND}'.replace('.','p'))")
  for SEED in 0 1; do
    RID="E10a_bspl_logits_b${TAG_B}_FSeval_s${SEED}"
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
    echo "=== $RID (bspline logits, blend_max=$BLEND, seed=$SEED) ==="
    python3 train.py $COMMON --seed $SEED \
      --bspline --bspline-space logits --bspline-tol 0.05 --bspline-kmax 0 \
      --bspline-warmup-epochs 5 \
      --bspline-blend-max $BLEND --bspline-blend-rampup 5 \
      --wandb --wandb-project vasot-repro \
      --wandb-tags E10a bspline logits FSeval "seed${SEED}" "blend${TAG_B}" \
      --run-id "$RID"
    python3 -c 'import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
bad = [key for key, value in state.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
if bad:
    raise SystemExit(f"Non-finite checkpoint tensors: {bad[:5]} (count={len(bad)})")
print("Checkpoint finite:", sys.argv[1])' "$RUN_DIR/checkpoints/last.ckpt"
  done
done

echo ""
echo "=== E10a ALL RUNS COMPLETE ==="
