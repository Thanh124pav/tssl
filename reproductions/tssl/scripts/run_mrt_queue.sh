#!/usr/bin/env bash
# MRT matrix queue, JOBS runs in parallel (default 2); safe to re-run.
# Arms: baseline, tmpl (random fixed template hub, A14), hub (action prototypes as hub).
cd "$(dirname "$0")"
YTI="yti:changing_tire yti:coffee yti:cpr yti:jump_car yti:repot"
BF="bf:coffee bf:cereals bf:tea bf:milk bf:juice bf:sandwich bf:scrambledegg bf:friedegg bf:salat bf:pancake"
{
for b in fseval da $YTI; do for a in baseline hub tmpl; do echo $b $a 0; done; done
for s in 1 2; do for b in fseval da; do for a in baseline hub tmpl; do echo $b $a $s; done; done; done
for b in $BF; do for a in baseline hub tmpl; do echo $b $a 0; done; done
for s in 1 2; do for b in $YTI; do for a in baseline hub tmpl; do echo $b $a $s; done; done; done
} | xargs -P "${JOBS:-2}" -L 1 bash run_mrt_matrix.sh || true
