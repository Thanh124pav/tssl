#!/bin/bash
# Remaining papers back-to-back, one at a time on the MIG slice (RULE.md §2).
# Each stage is wall-clock bounded so a slow one cannot consume the rest of the
# Slurm allocation; bounds are generous, not expected to trigger.
set -u

ACVCL=/mnt/data/thanhpv/acvcl
OUT=$ACVCL/outputs/reproductions
mkdir -p "$OUT/visual_debiasing" "$OUT/robot_learning" "$OUT/logs"

CNC_TIMEOUT="${CNC_TIMEOUT:-3600}"
NSF_TIMEOUT="${NSF_TIMEOUT:-1800}"
ACT_TIMEOUT="${ACT_TIMEOUT:-2700}"
DP_TIMEOUT="${DP_TIMEOUT:-2700}"
NSF_EPOCHS="${NSF_EPOCHS:-30}"
ACT_STEPS="${ACT_STEPS:-15000}"
DP_STEPS="${DP_STEPS:-15000}"

stage() { echo; echo "############### $1 :: $(date -Is) ###############"; }

# ------------------------------------------------- 2. Correct-N-Contrast
stage "cnc_waterbirds (official 5 epochs)"
(
  cd "$ACVCL" && source reproductions/env.sh
  timeout "$CNC_TIMEOUT" python reproductions/visual_debiasing/run_cnc.py \
    --run-name waterbirds_cnc_seed42 \
    --group visual_debiasing \
    --tags "cnc,waterbirds,official-config" \
    --summary-out "$OUT/visual_debiasing/waterbirds_cnc_seed42.json" \
    -- --dataset waterbirds --arch resnet50_pt --train_encoder \
    --pretrained_spurious_path "./model/waterbirds/waterbirds_erm_regularized.pt" \
    --num_anchor 17 --num_positive 17 --num_negative 17 --num_negative_easy 17 \
    --batch_factor 32 --optim sgd --lr 1e-4 --momentum 0.9 \
    --weight_decay 1e-3 --weight_decay_c 1e-3 --target_sample_ratio 1 \
    --temperature 0.1 --max_epoch 5 --no_projection_head --contrastive_weight 0.75 \
    --log_visual_interval 10000 --checkpoint_interval 10000 --verbose \
    --log_loss_interval 10 --replicate 42 --seed 42
  echo "--- cnc exit=$? $(date -Is) ---"
) 2>&1 | tail -70

# ------------------------------------------------- 4. Let Samples Speak (ERM stage)
# train_sup.sh is deliberately NOT used: it picks a GPU from nvidia-smi output,
# overrides CUDA_VISIBLE_DEVICES and backgrounds with nohup. Same hyperparameters,
# invoked directly. Epochs reduced from 100 -> $NSF_EPOCHS to fit the allocation.
stage "nsf_waterbirds_erm (${NSF_EPOCHS} of 100 epochs)"
(
  cd "$ACVCL/third_party/nsf_debiasing" && source "$ACVCL/reproductions/env.sh"
  export WANDB_PROJECT=acvcl-reproduction
  export WANDB_NAME="waterbirds_nsf_erm_seed1"
  export WANDB_RUN_GROUP=visual_debiasing
  export WANDB_TAGS="nsf,waterbirds,erm-stage,partial"
  timeout "$NSF_TIMEOUT" python train_supervised.py \
    --output_dir="$OUT/visual_debiasing/nsf_waterbirds_erm" \
    --num_epochs="$NSF_EPOCHS" --eval_freq=1 --save_freq=100 --seed=1 \
    --weight_decay=1e-4 --batch_size=32 --init_lr=3e-3 \
    --scheduler=cosine_lr_scheduler --data_dir=waterbirds \
    --data_transform=AugWaterbirdsCelebATransform \
    --dataset=SpuriousCorrelationDataset --model=imagenet_resnet50_pretrained
  echo "--- nsf exit=$? $(date -Is) ---"
) 2>&1 | tail -50

# ------------------------------------------------- 5. ACT
stage "act_aloha_sim_insertion_human (${ACT_STEPS} steps)"
(
  cd "$ACVCL" && source reproductions/robot_learning/env_lerobot.sh
  timeout "$ACT_TIMEOUT" lerobot-train \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha --env.task=AlohaInsertion-v0 \
    --output_dir="$OUT/robot_learning/act_aloha_insertion" \
    --steps="$ACT_STEPS" --batch_size=8 --num_workers=4 --seed=1000 \
    --log_freq=200 --save_freq="$ACT_STEPS" --env_eval_freq="$ACT_STEPS" \
    --eval.n_episodes=10 --eval.batch_size=1 \
    --wandb.enable=true --wandb.project=acvcl-reproduction
  echo "--- act exit=$? $(date -Is) ---"
) 2>&1 | tail -50

# ------------------------------------------------- 6. Diffusion Policy
stage "diffusion_pusht (${DP_STEPS} steps)"
(
  cd "$ACVCL" && source reproductions/robot_learning/env_lerobot.sh
  timeout "$DP_TIMEOUT" lerobot-train \
    --policy.type=diffusion \
    --dataset.repo_id=lerobot/pusht \
    --env.type=pusht --ema.enable=true \
    --output_dir="$OUT/robot_learning/diffusion_pusht" \
    --steps="$DP_STEPS" --batch_size=64 --num_workers=4 --seed=1000 \
    --log_freq=200 --save_freq="$DP_STEPS" --env_eval_freq="$DP_STEPS" \
    --eval.n_episodes=10 --eval.batch_size=1 \
    --wandb.enable=true --wandb.project=acvcl-reproduction
  echo "--- dp exit=$? $(date -Is) ---"
) 2>&1 | tail -50

echo
echo "PIPELINE DONE $(date -Is)"
