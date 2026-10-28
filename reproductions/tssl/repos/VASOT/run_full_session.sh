#!/bin/bash
# Full session runner: reproduction (DA + YTI) + diagnostics + smoothing sweep
# Runs up to MAX_PAR concurrent processes on one H100.
set -u

cd "$(dirname "$0")/src"
source /mnt/data/thanhpv/miniconda3/etc/profile.d/conda.sh
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh

LOGDIR="/mnt/data/thanhpv/tssl/reproductions/tssl/logs"
CKPTDIR="/mnt/data/thanhpv/tssl/reproductions/tssl/ckpts"
DIAGDIR="/mnt/data/thanhpv/tssl/reproductions/tssl/outputs"
STATUS="$LOGDIR/session_full/_status.tsv"
mkdir -p "$LOGDIR/session_full" "$CKPTDIR" "$DIAGDIR"

if [ ! -s "$STATUS" ]; then
    printf 'phase\tvariant\tdataset\tseed\tstart\tend\texit\tmof\tf1\tmiou\n' > "$STATUS"
fi

# Resume: skip successful cells
declare -A DONE=()
while IFS=$'\t' read -r ph v d s _rest; do
    [ "$ph" = "phase" ] && continue
    # parse exit code (7th field)
    exit_code=$(echo "$_rest" | cut -f3)
    [ "$exit_code" = "0" ] && DONE["$ph|$v|$d|$s"]=1
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

parse_metrics() {
    # Extract test_mof_full, test_f1_full, test_miou_full from log
    local log="$1"
    local mof=$(grep -oP 'test_mof_full\s*│\s*\K[0-9.]+' "$log" | tail -1)
    local f1=$(grep -oP 'test_f1_full\s*│\s*\K[0-9.]+' "$log" | tail -1)
    local miou=$(grep -oP 'test_miou_full\s*│\s*\K[0-9.]+' "$log" | tail -1)
    echo "${mof:-NA}	${f1:-NA}	${miou:-NA}"
}

run_one() {
    local phase="$1"; local variant="$2"; local dataset="$3"; local seed="$4"
    shift 4
    local key="$phase|$variant|$dataset|$seed"
    [ "${DONE[$key]:-}" = "1" ] && return 0

    local logfile="$LOGDIR/session_full/${phase}_${variant}_${dataset}_s${seed}.log"
    local start=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    echo "[$(date +%H:%M:%S)] START $key" >&2
    VASOT_CKPT_DIR="$CKPTDIR/${phase}_${variant}_${dataset}_s${seed}" \
        python train.py "$@" > "$logfile" 2>&1
    local ec=$?
    local end=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    local metrics
    if [ "$ec" -eq 0 ]; then
        metrics=$(parse_metrics "$logfile")
    else
        metrics="NA	NA	NA"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$variant" "$dataset" "$seed" "$start" "$end" "$ec" "$metrics" >> "$STATUS"
    echo "[$(date +%H:%M:%S)] DONE  $key (exit=$ec)" >&2
}

# =============================================================================
# PHASE 1: Reproduce DA and YTI baselines (paper hparams)
# =============================================================================
echo "=== Phase 1: Desktop Assembly + YTI reproduction ==="

# Desktop Assembly: paper Table 5: rho=0.25, lat=0.16, r=0.02, K=22, ne=30
wait_slot
run_one repro baseline desktop_assembly 0 \
    -d desktop_assembly -ac all -c 22 -ne 30 -g 0 --seed 0 \
    --rho 0.25 -lat 0.16 -vf 5 -lr 1e-3 -wd 1e-4 -ls 512 128 40 -ua -r 0.02 &
pids+=($!)

# YTI: 5 activities from run_yti.sh
yti_acts=(changing_tire coffee cpr jump_car repot)
yti_ks=(11 10 7 12 8)
for i in "${!yti_acts[@]}"; do
    wait_slot
    run_one repro baseline "YTI_${yti_acts[$i]}" 0 \
        -d YTI -ac "${yti_acts[$i]}" -c "${yti_ks[$i]}" -ne 30 -g 0 --seed 0 \
        --rho 0.15 -lat 0.12 -r 0.04 -lr 1e-4 -wd 1e-5 -ls 3000 32 32 -x -1 -vf 5 &
    pids+=($!)
done

# Wait for Phase 1 to complete
wait "${pids[@]}" 2>/dev/null
pids=()
echo "=== Phase 1 done ==="

# =============================================================================
# PHASE 2: Smoothing variant sweep (datasets: FSeval, FS, BF-coffee, BF-cereals, BF-scrambledegg, DA, YTI-coffee)
# Continue overnight sweep that was interrupted — add DA and YTI
# =============================================================================
echo "=== Phase 2: Smoothing variant sweep ==="

# Common hparams per dataset
# Format: dataset|activity|K|ne|extra_args
declare -a CONFIGS=(
    "FSeval|all|12|30|--rho 0.15 -lat 0.11 -lr 1e-3 -wd 1e-4 -ua"
    "FS|all|19|100|--rho 0.15 -lat 0.15 -lr 1e-3 -wd 1e-4 -ua"
    "Breakfast|coffee|7|50|--rho 0.2 -lat 0.1 -r 0.02 -ae 0.6 -at 0.5 -lr 1e-3 -wd 1e-4 -s"
    "Breakfast|cereals|5|50|--rho 0.2 -lat 0.1 -r 0.02 -ae 0.6 -at 0.5 -lr 1e-3 -wd 1e-4 -s"
    "Breakfast|scrambledegg|12|50|--rho 0.2 -lat 0.1 -r 0.02 -ae 0.6 -at 0.5 -lr 1e-3 -wd 1e-4 -s"
    "desktop_assembly|all|22|30|--rho 0.25 -lat 0.16 -lr 1e-3 -wd 1e-4 -ls 512 128 40 -ua -r 0.02"
    "YTI|coffee|10|30|--rho 0.15 -lat 0.12 -r 0.04 -lr 1e-4 -wd 1e-5 -ls 3000 32 32 -x -1"
)

# Variants and their extra flags
declare -A VARIANTS=(
    [baseline]=""
    [E1_arc]="--arc-reparam --arc-warmup-epochs 0"
    [E2_bsx_b03]="--bspline --bspline-space X --bspline-blend-max 0.3 --bspline-warmup-epochs 5"
    [E2_bsx_b01]="--bspline --bspline-space X --bspline-blend-max 0.1 --bspline-warmup-epochs 5"
    [E2_bsl_b03]="--bspline --bspline-space logits --bspline-blend-max 0.3 --bspline-warmup-epochs 5"
    [E2d_gauss2]="--gauss-smooth --gauss-sigma 2 --gauss-blend-max 0.3 --gauss-warmup-epochs 5"
    [E2d_gauss5]="--gauss-smooth --gauss-sigma 5 --gauss-blend-max 0.3 --gauss-warmup-epochs 5"
    [E3_arc_bsx]="--arc-reparam --arc-warmup-epochs 0 --bspline --bspline-space X --bspline-blend-max 0.3 --bspline-warmup-epochs 5"
)

SEEDS=(0 1)

for cfg_str in "${CONFIGS[@]}"; do
    IFS='|' read -r ds ac K ne extra <<< "$cfg_str"
    ds_label="${ds}"
    [ "$ac" != "all" ] && ds_label="${ds}_${ac}"

    for vname in baseline E1_arc E2_bsx_b03 E2_bsx_b01 E2_bsl_b03 E2d_gauss2 E2d_gauss5 E3_arc_bsx; do
        vflags="${VARIANTS[$vname]}"
        for seed in "${SEEDS[@]}"; do
            wait_slot
            run_one sweep "$vname" "$ds_label" "$seed" \
                -d "$ds" -ac $ac -c "$K" -ne "$ne" -g 0 --seed "$seed" \
                -vf 5 $extra $vflags &
            pids+=($!)
        done
    done
done

wait "${pids[@]}" 2>/dev/null
pids=()
echo "=== Phase 2 done ==="

echo "=== All phases complete. Status: $STATUS ==="
