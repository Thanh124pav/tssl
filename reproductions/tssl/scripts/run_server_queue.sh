#!/usr/bin/env bash
# Full hub/local matrix on the server: 7 arms x 7 benchmarks x 3 seeds, JOBS in
# parallel (default 7). Seed 0 and the proposal arm go first; safe to re-run.
cd "$(dirname "$0")"
BENCH="fseval da yti:changing_tire yti:coffee yti:cpr yti:jump_car yti:repot"
ARMS="hub_local baseline hub local tmpl hub_local_raw local_raw"
{
for s in 0 1 2; do for a in $ARMS; do for b in $BENCH; do echo $b $a $s; done; done; done
} | xargs -P "${JOBS:-7}" -L 1 bash run_mrt_matrix.sh || true
