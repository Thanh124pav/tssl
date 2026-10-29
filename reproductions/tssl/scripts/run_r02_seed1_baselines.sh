#!/bin/bash
# Paired seed-1 controls for the E08 rho=0 cross-dataset test.
# Run one group at a time; each group stays within the four-run packet cap.
set -euo pipefail

GROUP=${1:?usage: run_r02_seed1_baselines.sh first|second}
case "$GROUP" in
  first) RUNS=(DA changing_tire coffee cpr) ;;
  second) RUNS=(jump_car repot) ;;
  *) echo "Unknown group: $GROUP" >&2; exit 2 ;;
esac

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR=/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts

for ACT in "${RUNS[@]}"; do
  if [[ "$ACT" == DA ]]; then
    RID=R02_DA_baseline_s1
    ARGS=(-d desktop_assembly -ac all -c 22 -ne 30 --rho 0.25 -lat 0.16
          -r 0.02 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0
          -ls 512 128 40)
  else
    case "$ACT" in
      changing_tire) K=11 ;;
      coffee) K=10 ;;
      cpr) K=7 ;;
      jump_car) K=12 ;;
      repot) K=8 ;;
    esac
    RID="R02_YTI_${ACT}_baseline_s1"
    ARGS=(-d YTI -ac "$ACT" -c "$K" -ne 30 --rho 0.15 -lat 0.12
          -r 0.04 -ae 0.6 -at 0.3 -lr 1e-4 -wd 1e-5 -vf 5 -g 0
          -x -1 -ls 3000 32 32)
  fi
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
  echo "Starting $RID"
  python3 train.py "${ARGS[@]}" --seed 1 --wandb --wandb-project vasot-repro \
    --wandb-tags R02 baseline seed1 "$ACT" E08_control --run-id "$RID"
done
