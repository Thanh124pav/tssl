#!/bin/bash
# Visual-debiasing reproductions, strictly one at a time on the MIG slice.
# Official commands from each repo's README; only paths and logging are ours.
set -u

cd /mnt/data/thanhpv/acvcl
source reproductions/env.sh

# Absolute: the runners chdir into their own repo before writing summaries.
OUT=/mnt/data/thanhpv/acvcl/outputs/reproductions/visual_debiasing
mkdir -p "$OUT"
export WANDB_PROJECT=acvcl-reproduction

stage() { echo "=============== $1 :: $(date -Is) ==============="; }

# ---------------------------------------------------------------- CnC, priority 2
# README command for Waterbirds, resnet50_pt, 5 epochs, starting from the
# authors' own regularized ERM checkpoint.
stage "cnc_waterbirds"
python reproductions/visual_debiasing/run_cnc.py \
  --run-name waterbirds_cnc_seed42 \
  --group visual_debiasing \
  --tags "cnc,waterbirds,official-config" \
  --summary-out "$OUT/waterbirds_cnc_seed42.json" \
  -- --dataset waterbirds --arch resnet50_pt --train_encoder \
  --pretrained_spurious_path "./model/waterbirds/waterbirds_erm_regularized.pt" \
  --num_anchor 17 --num_positive 17 --num_negative 17 --num_negative_easy 17 \
  --batch_factor 32 --optim sgd --lr 1e-4 --momentum 0.9 \
  --weight_decay 1e-3 --weight_decay_c 1e-3 --target_sample_ratio 1 \
  --temperature 0.1 --max_epoch 5 --no_projection_head --contrastive_weight 0.75 \
  --log_visual_interval 10000 --checkpoint_interval 10000 --verbose \
  --log_loss_interval 10 --replicate 42 --seed 42 \
  2>&1 | tail -60
echo "--- exit=${PIPESTATUS[0]} $(date -Is) ---"

# ---------------------------------------------------------------- LfF, priority 3
# script/train.sh: vanilla baseline first, then LfF, both on
# ColoredMNIST-Skewed0.01-Severity4 (the repo's own choice: skewed3 severity4).
cd /mnt/data/thanhpv/acvcl
stage "lff_vanilla_colored_mnist"
python reproductions/visual_debiasing/run_lff.py \
  --run-name cmnist_vanilla_skew0.01_sev4_seed1 \
  --script train_vanilla.py \
  --tags "lff,colored_mnist,vanilla-baseline,official-config" \
  --summary-out "$OUT/cmnist_vanilla.json" \
  -- with server_user colored_mnist skewed3 severity4 \
  2>&1 | tail -30
echo "--- exit=${PIPESTATUS[0]} $(date -Is) ---"

cd /mnt/data/thanhpv/acvcl
stage "lff_colored_mnist"
python reproductions/visual_debiasing/run_lff.py \
  --run-name cmnist_lff_skew0.01_sev4_seed1 \
  --script train.py \
  --tags "lff,colored_mnist,official-config" \
  --summary-out "$OUT/cmnist_lff.json" \
  -- with server_user colored_mnist skewed3 severity4 \
  2>&1 | tail -30
echo "--- exit=${PIPESTATUS[0]} $(date -Is) ---"

echo "DEBIAS SUITE DONE $(date -Is)"
