#!/bin/bash
# ACT then Diffusion Policy, sequential on the MIG slice (RULE.md §2).
# Step counts are reduced from lerobot's 100k default to fit the remaining
# allocation, so both are PARTIAL reproductions; everything else (policy
# defaults, optimiser, RGB observations, EMA for DP) is lerobot's own.
set -u

ACVCL=/mnt/data/thanhpv/acvcl
OUT=$ACVCL/outputs/reproductions/robot_learning
mkdir -p "$OUT"

ACT_STEPS="${ACT_STEPS:-15000}"
DP_STEPS="${DP_STEPS:-15000}"
ACT_TIMEOUT="${ACT_TIMEOUT:-3300}"
DP_TIMEOUT="${DP_TIMEOUT:-3300}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"

cd "$ACVCL"
source reproductions/robot_learning/env_lerobot.sh

echo "############### act_aloha_sim_insertion_human (${ACT_STEPS} steps) :: $(date -Is) ###############"
timeout "$ACT_TIMEOUT" lerobot-train \
  --policy.type=act --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --env.type=aloha --env.task=AlohaInsertion-v0 \
  --output_dir="$OUT/act_aloha_insertion" \
  --steps="$ACT_STEPS" --batch_size=8 --num_workers=4 --seed=1000 \
  --log_freq=200 --save_freq="$ACT_STEPS" --env_eval_freq="$ACT_STEPS" \
  --eval.n_episodes="$EVAL_EPISODES" --eval.batch_size=1 \
  --wandb.enable=true --wandb.project=acvcl-reproduction \
  2>&1 | grep -vE "it/s\]|it/s," | tail -45
echo "--- act exit=${PIPESTATUS[0]} $(date -Is) ---"

echo "############### diffusion_pusht (${DP_STEPS} steps) :: $(date -Is) ###############"
timeout "$DP_TIMEOUT" lerobot-train \
  --policy.type=diffusion --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --env.type=pusht --ema.enable=true \
  --output_dir="$OUT/diffusion_pusht" \
  --steps="$DP_STEPS" --batch_size=64 --num_workers=4 --seed=1000 \
  --log_freq=200 --save_freq="$DP_STEPS" --env_eval_freq="$DP_STEPS" \
  --eval.n_episodes="$EVAL_EPISODES" --eval.batch_size=1 \
  --wandb.enable=true --wandb.project=acvcl-reproduction \
  2>&1 | grep -vE "it/s\]|it/s," | tail -45
echo "--- dp exit=${PIPESTATUS[0]} $(date -Is) ---"

echo "ROBOT SUITE DONE $(date -Is)"
