#!/bin/bash
# Breakfast re-run with paper Table 5 hyperparameters:
#   r=0.02 (was 0.04 in run_bf.sh), rest unchanged.
# 4-concurrent on 1 H100 (14K-param MLP, GPU-bound only by dataloader).
set -euo pipefail

cd "$(dirname "$0")/src"

LOG_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/logs/paper_hparams"
mkdir -p "$LOG_DIR"

actions=("pancake" "salat" "friedegg" "scrambledegg" "sandwich" "juice" "milk" "tea" "cereals" "coffee")
clusters=(14 8 9 12 9 8 5 7 5 7)

run_one() {
    local act=$1 K=$2
    python -u train.py -d Breakfast -ac "$act" -c "$K" -ne 50 -g 0 --seed 0 -s \
        --rho 0.2 -lat 0.1 -r 0.02 -ae 0.6 -at 0.5 -lr 1e-3 -wd 1e-4 -vf 5 \
        --wandb --wandb-project vasot-repro --wandb-tags paper_hparams breakfast r0.02 \
        > "$LOG_DIR/Breakfast_${act}_K${K}_r0.02.log" 2>&1
}

MAX_PAR=4
pids=()
for i in "${!actions[@]}"; do
    run_one "${actions[$i]}" "${clusters[$i]}" &
    pids+=($!)
    if (( ${#pids[@]} >= MAX_PAR )); then
        wait -n
        new_pids=()
        for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && new_pids+=("$p"); done
        pids=("${new_pids[@]}")
    fi
done
wait
echo "Breakfast paper-hparams re-run done."
