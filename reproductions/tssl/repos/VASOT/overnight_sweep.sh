#!/bin/bash
# Overnight variant sweep for VAOT/VASOT limitation study.
#
# Launches a grid of (variant × dataset × seed), 4-concurrent on the shared
# H100. All training logs land in logs/overnight; a final aggregation step
# parses them into a Markdown report.
#
# RULE.md compliance: assumes the caller already holds a Slurm compute
# allocation on worker-0. Does not request GPUs or spawn beyond MAX_PAR
# processes. All output under /mnt/data (HPS), none in $HOME.
#
# Usage:
#   bash overnight_sweep.sh > logs/overnight/_dispatcher.log 2>&1
set -u                                  # fail on undefined vars (but not on command errors — one failed run must not abort the sweep)

cd "$(dirname "$0")/src"
source /mnt/data/thanhpv/acvcl/reproductions/env.sh

OUT=/mnt/data/thanhpv/acvcl/reproductions/tssl/logs/overnight
mkdir -p "$OUT"
STATUS="$OUT/_status.tsv"
if [ ! -s "$STATUS" ]; then
    printf 'variant\tdataset\tseed\tstart\tend\texit\tmof\tf1\tmiou\twandb\n' > "$STATUS"
fi

# Resume safely: keep the existing manifest and skip only successful cells.
# Interrupted/failed cells have no exit=0 row and will be retried.
declare -A DONE=()
while IFS=$'\t' read -r v d s _start _end exit_code _rest; do
    [ "$v" = "variant" ] && continue
    [ "$exit_code" = "0" ] && DONE["$v|$d|$s"]=1
done < "$STATUS"

MAX_PAR=${MAX_PAR:-4}
pids=()
wait_slot() {
    if (( ${#pids[@]} >= MAX_PAR )); then
        wait -n
        new=()
        for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && new+=("$p"); done
        pids=("${new[@]}")
    fi
}

# ---------- dataset recipes ----------
# Each recipe is: dataset_id, flags for train.py (dataset-specific hparams),
# log-friendly short name.
declare -A DS_FLAGS=(
    [FSeval]="-d FSeval -ac all -c 12 -ne 30 --rho 0.15 -at 0.3 -lat 0.11 -r 0.02 -vf 10 -lr 1e-3 -wd 1e-4 -ua"
    [FSmid]="-d FS -ac all -c 19 -ne 100 --rho 0.15 -at 0.3 -lat 0.15 -r 0.02 -vf 20 -lr 1e-3 -wd 1e-4 -ua"
    [bf_coffee]="-d Breakfast -ac coffee -c 7 -ne 50 -s --rho 0.2 -lat 0.1 -r 0.02 -ae 0.6 -at 0.5 -lr 1e-3 -wd 1e-4 -vf 10"
    [bf_cereals]="-d Breakfast -ac cereals -c 5 -ne 50 -s --rho 0.2 -lat 0.1 -r 0.02 -ae 0.6 -at 0.5 -lr 1e-3 -wd 1e-4 -vf 10"
    [bf_scrambledegg]="-d Breakfast -ac scrambledegg -c 12 -ne 50 -s --rho 0.2 -lat 0.1 -r 0.02 -ae 0.6 -at 0.5 -lr 1e-3 -wd 1e-4 -vf 10"
)
DATASETS=("FSeval" "FSmid" "bf_coffee" "bf_cereals" "bf_scrambledegg")

# ---------- variant recipes ----------
declare -A V_FLAGS=(
    [baseline]=""
    [E1_arc]="--arc-reparam --arc-warmup-epochs 0"
    [E2_bsx_b03]="--bspline --bspline-space X --bspline-blend-max 0.3 --bspline-warmup-epochs 5 --bspline-tol 0.05 --bspline-kmax 32"
    [E2_bsx_b01]="--bspline --bspline-space X --bspline-blend-max 0.1 --bspline-warmup-epochs 5 --bspline-tol 0.05 --bspline-kmax 32"
    [E2_bsl_b03]="--bspline --bspline-space logits --bspline-blend-max 0.3 --bspline-warmup-epochs 5 --bspline-tol 0.03 --bspline-kmax 24"
    [E2d_gauss2]="--gauss-smooth --gauss-sigma 2 --gauss-blend-max 0.3 --gauss-warmup-epochs 5"
    [E2d_gauss5]="--gauss-smooth --gauss-sigma 5 --gauss-blend-max 0.3 --gauss-warmup-epochs 5"
    [E3_arc_bsx]="--arc-reparam --arc-warmup-epochs 0 --bspline --bspline-space X --bspline-blend-max 0.3 --bspline-warmup-epochs 10 --bspline-tol 0.05 --bspline-kmax 32"
)
VARIANTS=("baseline" "E1_arc" "E2_bsx_b03" "E2_bsx_b01" "E2_bsl_b03" "E2d_gauss2" "E2d_gauss5" "E3_arc_bsx")

SEEDS=(0 1)

run_one() {
    local variant="$1" dataset="$2" seed="$3"
    if [ "${DONE["$variant|$dataset|$seed"]:-0}" = "1" ]; then
        echo "[skip] completed: $variant / $dataset / seed $seed"
        return
    fi
    local tag="${variant}__${dataset}__s${seed}"
    local log="$OUT/${tag}.log"
    local start=$(date +%s)
    local cmd=(python -u train.py ${DS_FLAGS[$dataset]} -g 0 --seed "$seed"
               ${V_FLAGS[$variant]}
               --wandb --wandb-project vasot-repro
               --wandb-tags overnight "$variant" "$dataset" "s$seed"
               --wandb-run-name "${tag}")
    (
        "${cmd[@]}" >"$log" 2>&1
        local exit=$?
        local end=$(date +%s)
        local mof=$(grep "test_mof_full" "$log" | grep -oE "0\.[0-9]+" | head -1)
        local f1=$(grep "test_f1_full" "$log" | grep -oE "0\.[0-9]+" | head -1)
        local miou=$(grep "test_miou_full" "$log" | grep -oE "0\.[0-9]+" | head -1)
        local wb=$(grep -oE "runs/[A-Za-z0-9]+" "$log" | head -1)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$variant" "$dataset" "$seed" "$start" "$end" "$exit" "${mof:-NA}" "${f1:-NA}" "${miou:-NA}" "${wb:-NA}" >> "$STATUS"
    ) &
    pids+=($!)
    wait_slot
}

# ------- launch grid -------
for variant in "${VARIANTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_one "$variant" "$dataset" "$seed"
        done
    done
done
wait
echo "All runs done. Status at $STATUS"
# ------- aggregate -------
python aggregate_results.py "$STATUS" > "$OUT/_summary.md"
echo "Summary written to $OUT/_summary.md"
