#!/usr/bin/env bash
# Generic queue over run_mrt_matrix.sh: ARMS x BENCH x SEEDS, JOBS in parallel.
#   ARMS="dc dc_local" [SEEDS="0 1 2"] [BENCH="fseval da ..."] [JOBS=7] bash run_arm_queue.sh
# Seed is the outer loop, so every arm gets its seed-0 answer first. Safe to
# re-run and to run several pools at once (finished runs and locked runs are skipped).
cd "$(dirname "$0")"
BENCH=${BENCH:-"fseval da yti:changing_tire yti:coffee yti:cpr yti:jump_car yti:repot"}
: "${ARMS:?set ARMS}"
{
for s in ${SEEDS:-0 1 2}; do for b in $BENCH; do for a in $ARMS; do echo $b $a $s; done; done; done
} | xargs -P "${JOBS:-7}" -L 1 bash run_mrt_matrix.sh || true
