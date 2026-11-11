#!/usr/bin/env bash
# Queue for the intra-action (local) loss, alone and on top of the prototype hub.
cd "$(dirname "$0")"
YTI="yti:changing_tire yti:coffee yti:cpr yti:jump_car yti:repot"
{
for b in fseval $YTI da; do for a in hub_local local; do echo $b $a 0; done; done
for s in 1 2; do for b in fseval da; do for a in hub_local local; do echo $b $a $s; done; done; done
} | xargs -P "${JOBS:-2}" -L 1 bash run_mrt_matrix.sh || true
