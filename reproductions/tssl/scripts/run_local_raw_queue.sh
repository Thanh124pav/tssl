#!/usr/bin/env bash
# Third worker: local loss on un-normalised residuals, seed 0.
cd "$(dirname "$0")"
{
for b in fseval yti:changing_tire yti:coffee yti:cpr yti:jump_car yti:repot da; do for a in hub_local_raw local_raw; do echo $b $a 0; done; done
} | xargs -P 1 -L 1 bash run_mrt_matrix.sh || true
