#!/usr/bin/env bash
# One VASOT run with the published recipe of a benchmark plus one MRT arm.
#   run_mrt_matrix.sh <bench> <arm> [seed=0]
#   bench: fseval | da | yti:<activity> | bf:<activity> (-r 0.02) | bf4:<activity> (-r 0.04)
#   arm:   baseline | tmpl | hub   (dc, hub_dc: dropped after YTI results)
# Appends "bench arm seed mof f1 miou" to artifacts/runs/current/mrt_matrix.tsv;
# a run already in the table is skipped.
set -euo pipefail
bench=$1; arm=$2; seed=${3:-0}
root=$(cd "$(dirname "$0")/.." && pwd)
py=${TSSL_PYTHON:-/home/pavt1024/miniconda3/envs/deeplearning/bin/python}
tsv=$root/artifacts/runs/current/mrt_matrix.tsv
if grep -q "^$bench	$arm	$seed	" "$tsv" 2>/dev/null; then exit 0; fi
act=${bench#*:}
case "$bench" in
  fseval) k=12; cfg=(-d FSeval -ac all -c 12 -ne 30 --rho 0.15 -lat 0.11 -lr 1e-3 -wd 1e-4 -ua) ;;
  da) k=22; cfg=(-d desktop_assembly -ac all -c 22 -ne 30 --rho 0.25 -lat 0.16 -lr 1e-3 -wd 1e-4 -r 0.02 -ls 512 128 40 -ua) ;;
  yti:*)
    case "$act" in changing_tire) k=11 ;; coffee) k=10 ;; cpr) k=7 ;; jump_car) k=12 ;; repot) k=8 ;; esac
    cfg=(-d YTI -ac "$act" -c "$k" -ne 30 --rho 0.15 -lat 0.12 -r 0.04 -lr 1e-4 -wd 1e-5 -ls 3000 32 32 -x -1) ;;
  bf:*|bf4:*)
    bfr=0.02; [ "${bench%%:*}" = bf4 ] && bfr=0.04   # bf4 = upstream run_bf.sh radius
    case "$act" in coffee|tea) k=7 ;; cereals|milk) k=5 ;; juice|salat) k=8 ;; sandwich|friedegg) k=9 ;;
      scrambledegg) k=12 ;; pancake) k=14 ;; esac
    cfg=(-d Breakfast -ac "$act" -c "$k" -ne 50 -s --rho 0.2 -lat 0.1 -r "$bfr" -ae 0.6 -at 0.5 -lr 1e-3 -wd 1e-4) ;;
  *) echo "unknown bench $bench" >&2; exit 2 ;;
esac
case "$arm" in
  baseline) extra=() ;;
  dc) extra=(--decenter 1.0) ;;
  local) extra=(--local-weight 1.0 --local-warmup-epochs 5) ;;
  local_raw) extra=(--local-weight 1.0 --local-warmup-epochs 5 --local-raw) ;;
  hub_local_raw) extra=(--template-slots "$k" --template-warmup-epochs 5 --align-hub clusters --local-weight 1.0 --local-warmup-epochs 5 --local-raw) ;;
  hub_local) extra=(--template-slots "$k" --template-warmup-epochs 5 --align-hub clusters --local-weight 1.0 --local-warmup-epochs 5) ;;
  tmpl) extra=(--template-slots "$k" --template-warmup-epochs 5) ;;
  hub) extra=(--template-slots "$k" --template-warmup-epochs 5 --align-hub clusters) ;;
  hub_dc) extra=(--template-slots "$k" --template-warmup-epochs 5 --align-hub clusters --decenter 1.0) ;;
  dc_local) extra=(--decenter 1.0 --local-weight 1.0 --local-warmup-epochs 5) ;;
  blur3) extra=(--align-target-blur 3) ;;
  blur6) extra=(--align-target-blur 6) ;;
  blur12) extra=(--align-target-blur 12) ;;
  blur6_local) extra=(--align-target-blur 6 --local-weight 1.0 --local-warmup-epochs 5) ;;
  blur12_local) extra=(--align-target-blur 12 --local-weight 1.0 --local-warmup-epochs 5) ;;
  pow05) extra=(--align-target-power 0.5) ;;
  pow2) extra=(--align-target-power 2.0) ;;
  local_pow2) extra=(--align-target-power 2.0 --local-weight 1.0 --local-warmup-epochs 5) ;;
  # spread regularisers from prior work (spread_reg.py); weights put each at
  # 0.3-0.6x the gradient norm of the alignment loss (calib_spread.py)
  vcreg) extra=(--spread-reg vc --spread-weight 3.0) ;;
  unif) extra=(--spread-reg unif --spread-weight 1.0) ;;
  koleo) extra=(--spread-reg koleo --spread-weight 0.2) ;;
  ldreg) extra=(--spread-reg ldreg --spread-weight 0.1) ;;
  koleo_res) extra=(--spread-reg koleo_res --spread-weight 0.2 --spread-warmup-epochs 5) ;;
  tneg) extra=(--spread-reg tneg --spread-weight 0.5) ;;
  csls) extra=(--align-csls 10) ;;
  # combinations: which spread arm keeps its gain next to local / target blur
  vcreg1) extra=(--spread-reg vc --spread-weight 1.0) ;;
  vcreg_blur6) extra=(--spread-reg vc --spread-weight 3.0 --align-target-blur 6) ;;
  csls_local) extra=(--align-csls 10 --local-weight 1.0 --local-warmup-epochs 5) ;;
  csls_blur6_local) extra=(--align-csls 10 --align-target-blur 6 --local-weight 1.0 --local-warmup-epochs 5) ;;
  tneg_blur6) extra=(--spread-reg tneg --spread-weight 0.5 --align-target-blur 6) ;;
  ldreg_local) extra=(--spread-reg ldreg --spread-weight 0.1 --local-weight 1.0 --local-warmup-epochs 5) ;;
  # lower-weight VICReg, alone and next to the alignment configuration
  vcreg03) extra=(--spread-reg vc --spread-weight 0.3) ;;
  vcreg06) extra=(--spread-reg vc --spread-weight 0.6) ;;
  vcreg1_csls_local) extra=(--spread-reg vc --spread-weight 1.0 --align-csls 10 --local-weight 1.0 --local-warmup-epochs 5) ;;
  vcreg1_local) extra=(--spread-reg vc --spread-weight 1.0 --local-weight 1.0 --local-warmup-epochs 5) ;;
  vcreg1_csls_blur6_local) extra=(--spread-reg vc --spread-weight 1.0 --align-csls 10 --align-target-blur 6 --local-weight 1.0 --local-warmup-epochs 5) ;;
  vcreg03_csls_blur6_local) extra=(--spread-reg vc --spread-weight 0.3 --align-csls 10 --align-target-blur 6 --local-weight 1.0 --local-warmup-epochs 5) ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac
out=$root/artifacts/runs/current/mrt_${bench//:/_}_${arm}_s${seed}
mkdir -p "$out"
mkdir "$out.lock" 2>/dev/null || exit 0   # another worker owns this run
trap 'rmdir "$out.lock"' EXIT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$root/repos/VASOT"
"$py" src/train.py -p "$root/data/final" "${cfg[@]}" -vf 5 -g 0 --seed "$seed" "${extra[@]}" \
  --run-dir "$out" > "$out.log" 2>&1 || { echo "FAILED $bench $arm $seed" >&2; exit 1; }
m() { grep -E "test_$1_full" "$out.log" | tail -1 | tr -s ' ' | cut -d' ' -f4; }
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$bench" "$arm" "$seed" "$(m mof)" "$(m f1)" "$(m miou)" >> "$tsv"
