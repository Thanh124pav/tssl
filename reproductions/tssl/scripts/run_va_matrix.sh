#!/usr/bin/env bash
# VA metrics (eval_va.py) for every finished run of the hub/local matrix.
# One JSON per benchmark and seed under artifacts/runs/current/va/; existing
# files are skipped. JOBS evaluations in parallel (default 4).
cd "$(dirname "$0")"
py=${TSSL_PYTHON:-python}
runs=../artifacts/runs/current
mkdir -p $runs/va
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
one() {
  bench=$1; seed=$2; tag=${bench//:/_}; out=$runs/va/${tag}_s$seed.json
  [ -s "$out" ] && return 0
  case "$bench" in
    fseval) ds=(--dataset FSeval) ;;
    da) ds=(--dataset desktop_assembly) ;;
    yti:*) ds=(--dataset YTI --activity "${bench#*:}") ;;
  esac
  ck=()
  for a in baseline hub tmpl local hub_local local_raw hub_local_raw; do
    c=$runs/mrt_${tag}_${a}_s$seed/checkpoints/last.ckpt
    # only finished runs (row in the tsv); wait until all 7 arms are there
    grep -q "^$bench	$a	$seed	" $runs/mrt_matrix.tsv && [ -f "$c" ] && ck+=("$a=$c")
  done
  [ ${#ck[@]} -lt 7 ] && return 0
  "$py" eval_va.py "${ds[@]}" --ckpts "${ck[@]}" --out "$out" > "$runs/va/${tag}_s$seed.log" 2>&1 \
    || echo "FAILED va $bench $seed" >&2
}
export -f one; export py runs
for s in 0 1 2; do for b in fseval da yti:changing_tire yti:coffee yti:cpr yti:jump_car yti:repot; do echo $b $s; done; done \
  | xargs -P "${JOBS:-4}" -L 1 bash -c 'one "$0" "$1"'
