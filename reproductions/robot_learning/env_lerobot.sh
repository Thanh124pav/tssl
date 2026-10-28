# Source before ACT / Diffusion Policy runs.
# Separate env from `acvcl`: lerobot needs python>=3.12 and torch>=2.7.

export ACVCL_ROOT=/mnt/data/thanhpv/acvcl
export LEROBOT_ENV=/mnt/data/thanhpv/miniconda3/envs/lerobot
export PATH="$LEROBOT_ENV/bin:$PATH"

export HF_HOME=/mnt/data/thanhpv/cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR=/mnt/data/thanhpv/cache/pip
export XDG_CACHE_HOME=/mnt/data/thanhpv/cache/xdg
export TORCH_HOME=/mnt/data/thanhpv/cache/torch

export WANDB_DIR="$ACVCL_ROOT/outputs/wandb"
export WANDB_CACHE_DIR=/mnt/data/thanhpv/cache/wandb
export WANDB_CONFIG_DIR=/mnt/data/thanhpv/cache/wandb/config
export WANDB_API_KEY="$(tr -d ' \t\r\n' < /home/thanhpv/wandb.txt)"

# MIG mode disables NVIDIA EGL for GL rendering, so MuJoCo (gym-aloha) must
# fall back to software rendering or env creation fails.
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

_alloc_cpus="${SLURM_CPUS_PER_TASK:-4}"
_threads=$(( _alloc_cpus / 2 ))
[ "$_threads" -lt 1 ] && _threads=1
export OMP_NUM_THREADS="$_threads"
export MKL_NUM_THREADS="$_threads"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
