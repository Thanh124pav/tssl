#!/bin/bash
# Finalize overnight sweep:
#   1. wait for the dispatcher to exit
#   2. run post-training diagnostics on all overnight checkpoints
#   3. compile SESSION_REPORT_OVERNIGHT.md
#   4. commit + push to tssl
set -u
cd "$(dirname "$0")/src"
source /mnt/data/thanhpv/tssl/reproductions/env.sh

LOGS=/mnt/data/thanhpv/tssl/reproductions/tssl/logs/overnight
WANDB_ROOT=/mnt/data/thanhpv/tssl/outputs/wandb/vasot-repro
DIAG_OUT=/mnt/data/thanhpv/tssl/reproductions/tssl/outputs/overnight_diag
mkdir -p "$DIAG_OUT"

# 1. wait for dispatcher
while pgrep -f overnight_sweep.sh >/dev/null 2>&1; do sleep 60; done
echo "[finalize] dispatcher exited at $(date)"

# 2. diagnostics: find each run's ckpt by wandb-run-name match
tsv="$LOGS/_status.tsv"
while IFS=$'\t' read -r variant dataset seed start end exit mof f1 miou wandb; do
    [ "$variant" = "variant" ] && continue
    [ "$exit" = "0" ] || continue
    tag="${variant}__${dataset}__s${seed}"
    # Each wandb run saves a ckpt under $WANDB_ROOT/<run_id>/checkpoints
    run_id="${wandb##runs/}"
    ckpt_dir="$WANDB_ROOT/$run_id/checkpoints"
    [ -d "$ckpt_dir" ] || continue
    ckpt=$(ls -1t "$ckpt_dir"/*.ckpt 2>/dev/null | head -1)
    [ -z "$ckpt" ] && continue
    case "$dataset" in
        FSeval) dsflag="FSeval";;
        FSmid) dsflag="FS";;
        bf_coffee) dsflag="Breakfast"; act=coffee;;
        bf_cereals) dsflag="Breakfast"; act=cereals;;
        bf_scrambledegg) dsflag="Breakfast"; act=scrambledegg;;
    esac
    if [ "$dsflag" = "Breakfast" ]; then
        python diagnose.py --ckpt "$ckpt" --dataset "$dsflag" --activity "$act" \
            --out "$DIAG_OUT/$tag" > "$LOGS/diag_$tag.log" 2>&1 &
    else
        python diagnose.py --ckpt "$ckpt" --dataset "$dsflag" --activity all \
            --out "$DIAG_OUT/$tag" > "$LOGS/diag_$tag.log" 2>&1 &
    fi
    # cap diag parallelism
    while (( $(jobs -r | wc -l) >= 6 )); do wait -n; done
done < "$tsv"
wait
echo "[finalize] diagnostics done"

# 3. aggregate
python aggregate_results.py "$tsv" > "$LOGS/_summary.md"
python compile_final_report.py "$tsv" "$DIAG_OUT" \
    > /mnt/data/thanhpv/tssl/reports/SESSION_REPORT_OVERNIGHT_2026-10-28.md

# 4. commit + push
cd /mnt/data/thanhpv/tssl
git add reports/SESSION_REPORT_OVERNIGHT_2026-10-28.md reports/OVERNIGHT_PLAN_2026-10-27.md
git commit -m "$(cat <<'EOF'
docs: overnight sweep results — 8 smoothing variants × 5 datasets × 2 seeds

Variant sweep over the discrete→continuous representation family (B-spline X,
B-spline logits, Gaussian σ∈{2,5}, arc-length re-param, combinations) on FS
Mid, FSeval, Breakfast {coffee, cereals, scrambledegg}, two seeds each.
Post-training diagnostics (T* entropy, boundary z-score, index-vs-semantic
correlation, codebook spectrum) collected on every surviving checkpoint.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
git push -u origin tssl
echo "[finalize] pushed"
