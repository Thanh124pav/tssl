#!/bin/bash
# Official CORDS CIFAR-10 configs at fraction 0.1, run strictly one at a time
# so only a single heavy job occupies the MIG slice (RULE.md section 2).
set -u

cd /mnt/data/thanhpv/acvcl
source reproductions/env.sh

OUT=outputs/reproductions/data_selection
FRACTION="${FRACTION:-0.1}"
EPOCHS="${EPOCHS:-300}"
SEED="${SEED:-42}"
mkdir -p "$OUT"

run_one() {
  local name="$1" config="$2"
  local run_name="cifar10_${name}_frac${FRACTION}_seed${SEED}"
  echo "=============== ${run_name} :: $(date -Is) ==============="
  python reproductions/data_selection/run_cords.py \
    --config "third_party/cords/configs/SL/${config}" \
    --fraction "$FRACTION" --epochs "$EPOCHS" --seed "$SEED" \
    --results-dir "${OUT}/${run_name}" \
    --run-name "$run_name" \
    --group data_selection \
    --tags "cords,cifar10,official-config,frac${FRACTION}" \
    2>&1 | tail -40
  echo "--- exit=${PIPESTATUS[0]} finished $(date -Is) ---"
}

run_one glister   config_glister_cifar10.py
run_one gradmatchpb config_gradmatchpb_cifar10.py
run_one random    config_random_cifar10.py

echo "ALL CIFAR10 RUNS DONE $(date -Is)"
