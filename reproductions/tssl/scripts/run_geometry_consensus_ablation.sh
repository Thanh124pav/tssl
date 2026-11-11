#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 || $# > 6 )); then
  echo "Usage: $0 {fseval|da|yti|breakfast} {control|geometry|geometry_light|spectot_tar|consensus|both|both_light} [seed=0] [epochs=30] [n_videos=4] [activity]" >&2
  exit 2
fi
benchmark=$1
arm=$2
seed=${3:-0}
epochs=${4:-30}
n_videos=${5:-4}
activity=${6:-}
if (( n_videos < 3 )); then
  echo "n_videos must be at least 3" >&2
  exit 2
fi
root=$(cd "$(dirname "$0")/.." && pwd)
python_bin=${TSSL_PYTHON:-/home/pavt1024/miniconda3/envs/deeplearning/bin/python}
suffix=${activity:+_${activity}}
out=$root/artifacts/runs/current/geometry_consensus_${benchmark}${suffix}_${arm}_n${n_videos}_s${seed}
mkdir -p "$out"
case "$benchmark" in
  fseval) dataset=FSeval; activity=all; config=(-c 12 --rho 0.15 -lat 0.11 -ua); optim=(-lr 1e-3 -wd 1e-4) ;;
  da) dataset=desktop_assembly; activity=all; config=(-c 22 --rho 0.25 -lat 0.16 -r 0.02 -ls 512 128 40 -ua); optim=(-lr 1e-3 -wd 1e-4) ;;
  yti)
    dataset=YTI; activity=${activity:-changing_tire}
    case "$activity" in
      changing_tire) k=11 ;; coffee) k=10 ;; cpr) k=7 ;;
      jump_car) k=12 ;; repot) k=8 ;;
      *) echo "Unknown YTI activity: $activity" >&2; exit 2 ;;
    esac
    config=(-c "$k" --rho 0.15 -lat 0.12 -r 0.04 -ls 3000 32 32 -x -1)
    optim=(-lr 1e-4 -wd 1e-5) ;;
  breakfast)
    dataset=Breakfast; activity=${activity:-coffee}
    case "$activity" in
      coffee|tea) k=7 ;; cereals|milk) k=5 ;; juice|salat) k=8 ;;
      sandwich|friedegg) k=9 ;; scrambledegg) k=12 ;; pancake) k=14 ;;
      *) echo "Unknown Breakfast activity: $activity" >&2; exit 2 ;;
    esac
    config=(-c "$k" -s --rho 0.2 -lat 0.1 -r 0.02 -ae 0.6 -at 0.5)
    optim=(-lr 1e-3 -wd 1e-4) ;;
  *) echo "Unknown benchmark: $benchmark" >&2; exit 2 ;;
esac
case "$arm" in
  control) proposal=() ;;
  geometry) proposal=(--geometry-weight 0.5 --geometry-repulsion-weight 0.1) ;;
  geometry_light) proposal=(--geometry-weight 0.05 --geometry-repulsion-weight 0.01) ;;
  spectot_tar) proposal=(--spectot-tar-weight 0.1) ;;
  consensus) proposal=(--soft-consensus-weight 0.01 --soft-consensus-blend 0.5) ;;
  both) proposal=(--geometry-weight 0.5 --geometry-repulsion-weight 0.1
                  --soft-consensus-weight 0.01 --soft-consensus-blend 0.5) ;;
  mrt_dc) proposal=(--decenter 1.0) ;;
  mrt_noise) proposal=(--ot-noise 0.3) ;;
  mrt) proposal=(--decenter 1.0 --ot-noise 0.3) ;;
  both_light) proposal=(--geometry-weight 0.05 --geometry-repulsion-weight 0.01
                        --soft-consensus-weight 0.01 --soft-consensus-blend 0.5) ;;
  *) echo "Unknown arm: $arm" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$root/repos/VASOT"
"$python_bin" src/train.py -p "$root/data/final" -d "$dataset" -ac "$activity" \
  "${config[@]}" -ne "$epochs" -g 0 --seed "$seed" -vf 5 \
  "${optim[@]}" --proposal-multiview --proposal-videos "$n_videos" \
  "${proposal[@]}" --run-dir "$out" > "$out.log" 2>&1
echo "$out"
