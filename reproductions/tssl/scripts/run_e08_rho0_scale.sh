#!/bin/bash
# E08 — Scale rho=0 (winning Hypothesis-C intervention from E07b) to DA and YTI.
# Gate: E07b passed on FSeval with +7.95 pp mean MoF (both seeds improved).
# Scope here: canonical datasets of VASOT paper. Each dataset uses its own
# frozen baseline config (configs/vasot/baselines/*.yaml), overriding only rho.
#
# Scope split in two sections — DA first (2 runs), YTI second (10 runs).
# Total up to 12 runs serialised on one GPU; ~45-90 min depending on YTI timing.
set -euo pipefail

cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate acvcl
source /mnt/data/thanhpv/tssl/reproductions/env.sh
export VASOT_ARTIFACTS_DIR="/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts"

# -------- E08a: Desktop Assembly, rho=0, 2 seeds -----------------------------
# Baseline (configs/vasot/baselines/desktop_assembly.yaml):
#   rho=0.25, lat=0.16, radius_gw=0.02, c=22, layers=[512,128,40]
DA_COMMON="-d desktop_assembly -ac all -c 22 -ne 30 -lat 0.16 -r 0.02 -ae 0.6 -at 0.3 -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 -ls 512 128 40"
for SEED in 0 1; do
  RID="E08a_DA_rho0_s${SEED}"
  echo ""
  echo "=== $RID (rho=0, seed=$SEED) ==="
  python3 train.py $DA_COMMON --seed $SEED --rho 0 \
    --wandb --wandb-project vasot-repro \
    --wandb-tags E08 E08a DA rho0 "seed${SEED}" hypothesis_C_scale \
    --run-id "$RID"
done

# -------- E08b: YTI 5 activities, rho=0, 2 seeds -----------------------------
# Baseline (configs/vasot/baselines/yti.yaml):
#   rho=0.15, lat=0.12, radius_gw=0.04, layers=[3000,32,32], exclude=-1
YTI_COMMON="-d YTI -ne 30 -lat 0.12 -r 0.04 -ae 0.6 -at 0.3 -lr 1e-4 -wd 1e-5 -vf 5 -g 0 -x -1 -ls 3000 32 32"
declare -A YTI_K=( [changing_tire]=11 [coffee]=10 [cpr]=7 [jump_car]=12 [repot]=8 )

for ACT in changing_tire coffee cpr jump_car repot; do
  K=${YTI_K[$ACT]}
  for SEED in 0 1; do
    RID="E08b_YTI_${ACT}_rho0_s${SEED}"
    echo ""
    echo "=== $RID (rho=0, k=$K, seed=$SEED) ==="
    python3 train.py $YTI_COMMON -ac $ACT -c $K --seed $SEED --rho 0 \
      --wandb --wandb-project vasot-repro \
      --wandb-tags E08 E08b YTI "$ACT" rho0 "seed${SEED}" hypothesis_C_scale \
      --run-id "$RID"
  done
done

echo ""
echo "=== E08 ALL RUNS COMPLETE ==="
