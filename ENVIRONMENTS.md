# ENVIRONMENTS

Conda lives at `/mnt/data/thanhpv/miniconda3` (already installed, on HPS — not
`$HOME`). Every cache is redirected to `/mnt/data` by `reproductions/env.sh`,
which must be sourced before any run:

```bash
cd /mnt/data/thanhpv/acvcl && source reproductions/env.sh
```

That script sets `HF_HOME`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME`, `TORCH_HOME`,
the `WANDB_*` paths and `W&B` credentials, and caps
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` to half of `SLURM_CPUS_PER_TASK`
(8 of the 16 allocated CPUs) with `ACVCL_NUM_WORKERS=4` for DataLoaders.

## `acvcl` — shared env for data selection + visual debiasing

Python 3.10.22. Covers CORDS (GLISTER/GradMatch), Correct-N-Contrast,
Learning from Failure and nsf_debiasing: one env instead of four, since all
four are plain torch/torchvision image pipelines.

| package | version | note |
|---|---|---|
| torch | 2.4.1+cu121 | bf16 verified on the H100 MIG 3g.40gb slice |
| torchvision | 0.19.1+cu121 | paired build; ABI checked via `torchvision.io` |
| numpy | 1.26.4 | pinned `<2` for numba / apricot / older repo APIs |
| transformers | 4.46.3 | see deviation below |
| datasets | 3.1.0 | version that keeps `transformers` 4.46 consistent |
| huggingface-hub | 0.36.2 | |
| scikit-learn | 1.7.2 | |
| numba | 0.68.0 | |
| apricot-select | 0.6.1 | CORDS submodular selection |
| ray | 2.59.0 | only imported at CORDS module load |
| wandb | 0.31.0 | |
| timm, umap-learn, sacred, h5py, gdown, opencv-python-headless, scikit-image, scipy, pandas, matplotlib, dotmap, pyarrow | current | |

`pip check` reports no broken requirements.

Created with:

```bash
conda create -y -n acvcl python=3.10 pip
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 apricot-select numba scipy scikit-learn scikit-image \
  tqdm matplotlib pandas pyyaml dotmap ray transformers==4.46.3 datasets==3.1.0 \
  huggingface_hub pyarrow timm umap-learn wandb sacred opencv-python-headless h5py gdown
```

### Version deviations and why

- **transformers pinned to 4.46.3, not 5.x.** transformers 5.20.1 installs
  cleanly but reports "PyTorch was not found" against torch 2.4.1 and has
  dropped `ViTFeatureExtractor`, which CORDS imports at module load. 4.46.3
  restores both.
- **numpy pinned `<2`.** CORDS, LfF and nsf_debiasing predate the numpy 2 API
  removals.
- **CORDS is not `pip install -e .`'d.** Its `setup.py` pins
  `torchtext~=0.10.1` (which forces torch 1.9, unusable on an H100) and
  `pyyaml~=5.4.1` (no wheel for Python 3.10). It is used from
  `third_party/cords` on `PYTHONPATH` instead, with
  `reproductions/data_selection/cords_compat.py` stubbing the three imports
  that only NLP/MILO paths need (`torchtext`, `sentence_transformers`,
  `submodlib`) and restoring the `sklearn.datasets.load_boston` attribute
  that scikit-learn removed in 1.2.

## `lerobot` — not yet created

ACT and Diffusion Policy need their own env: lerobot requires
`python>=3.12` and `torch>=2.7,<2.12`, both incompatible with `acvcl`. This is
a real dependency conflict, not a convenience split.

## `mate` — pre-existing, left untouched

`/mnt/data/thanhpv/miniconda3/envs/mate` (Python 3.9, torch 2.1.2+cu121)
belongs to another project. Its torchvision 0.14.1 is ABI-mismatched against
its torch (`torchvision.io` fails with an undefined symbol), and it lacks
sklearn/transformers/timm, so it was not reused and not modified.
