# Source before any reproduction run.
# Caps CPU threads below the Slurm allocation (RULE.md section 2) and keeps
# every cache and output on /mnt/data rather than $HOME (RULE.md section 3).

export ACVCL_ROOT=/mnt/data/thanhpv/acvcl
export CONDA_ROOT=/mnt/data/thanhpv/miniconda3
export ACVCL_ENV="$CONDA_ROOT/envs/acvcl"
export PATH="$ACVCL_ENV/bin:$PATH"

export HF_HOME=/mnt/data/thanhpv/cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PIP_CACHE_DIR=/mnt/data/thanhpv/cache/pip
export XDG_CACHE_HOME=/mnt/data/thanhpv/cache/xdg
export TORCH_HOME=/mnt/data/thanhpv/cache/torch

export WANDB_DIR="$ACVCL_ROOT/outputs/wandb"
export WANDB_CACHE_DIR=/mnt/data/thanhpv/cache/wandb
export WANDB_CONFIG_DIR=/mnt/data/thanhpv/cache/wandb/config
export WANDB_API_KEY="$(tr -d ' \t\r\n' < /home/thanhpv/wandb.txt)"

_alloc_cpus="${SLURM_CPUS_PER_TASK:-4}"
_threads=$(( _alloc_cpus / 2 ))
[ "$_threads" -lt 1 ] && _threads=1
export OMP_NUM_THREADS="$_threads"
export MKL_NUM_THREADS="$_threads"
export OPENBLAS_NUM_THREADS="$_threads"
export NUMEXPR_NUM_THREADS="$_threads"
export ACVCL_NUM_WORKERS=4

mkdir -p "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$TORCH_HOME"
