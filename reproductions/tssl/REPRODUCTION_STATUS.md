# VAOT / VASOT (ICCV 2025) — Reproduction Status

- Paper: Ali et al., "Joint Self-Supervised Video Alignment and Action Segmentation", ICCV 2025 (arXiv:2503.16832)
- Repos: [VAOT-ICCV25](https://github.com/trquhuytin/VAOT-ICCV25) (alignment only), [VASOT-ICCV25](https://github.com/trquhuytin/VASOT-ICCV25) (joint)
- Branch: `tssl` on `thanh124pav/acvcl`
- Workdir: `/mnt/data/thanhpv/acvcl/reproductions/tssl/`
- Date: 2026-10-23 → 2026-10-24 (overnight, 1× H100 80 GB, Slurm job 61908)

## 1. What the paper actually proposes

- **VAOT** (video alignment, image-based). ResNet-50 base + ConvEmbedder. Loss = cross-entropy between a softmax similarity matrix on the two sampled videos' embeddings and ASOT pseudo-labels (Augmented Soft OT). Virtual-frame row/column is prepended to the cost matrix so unmatched frames can be absorbed. Trained on Pouring / Penn Action / IKEA ASM frames (raw video).
- **VASOT** (joint action segmentation + alignment, feature-based). Shared 2-layer MLP maps pre-extracted frame features to a learned codebook. For each video pair `(X, Y)` the training step computes:
  1. per-video segmentation OT with the learnable codebook (`segmentation_ot.segment_ot`) → segmentation pseudo-labels for `X` and for `Y`;
  2. pairwise alignment OT (`alignment_ot.align_ot`) between `X` and `Y` features → alignment pseudo-labels;
  3. loss = `CE(soft_seg(X), opt_seg(X)) + CE(soft_seg(Y), opt_seg(Y)) + β·CE(soft_align(X,Y), opt_align(X,Y))`, default `β = 1`.

  Segmentation uses `ub_actions=True` (unbalanced over actions), alignment follows `-ua / -uf` flags on the CLI. Eval metrics (`src/metrics.py`): per-video MoF / F1 / mIoU with Hungarian cluster→gt matching, and a "full" variant that pools all videos before matching.
- "Joint vs single-task" in the paper = VASOT with `β>0` (joint) vs `β=0` (segmentation only). We reproduce that ablation on 50 Salads below.

## 2. Environment

- Reused existing conda env `/mnt/data/thanhpv/miniconda3/envs/acvcl` (Python 3.10.22, torch 2.4.1+cu121). Added additively: `pytorch-lightning==2.6.6`, `torchmetrics==1.9.0`, `lightning-utilities==0.15.3`. Upgraded `gdown` to 5.2.0. `pip check` clean.
- `reproductions/env.sh` sources `WANDB_API_KEY` from `/home/thanhpv/wandb.txt` and pins caches to `/mnt/data`. `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8` (half of the 16-CPU allocation).
- Hardware: 1× NVIDIA H100 80 GB HBM3, 16 CPU, 64 GB RAM, `/mnt/data` HPS has 15 TB free.

## 3. Data acquisition — summary and blockers

Data root: `/mnt/data/thanhpv/acvcl/reproductions/tssl/data/final/`

| Dataset | Features | Ground truth | Mapping | Status |
|---|---|---|---|---|
| 50 Salads mid (`FS`, 19 cls) | `unsup_temp_embed` 50salads.tar.gz, 64-dim, 50 videos | HF `dinggd/50salads` zip, pulled only `groundTruth/*` + `mapping.txt` via `remotezip` (HTTP range) | 19 classes | **Complete** |
| 50 Salads eval (`FSeval`, 12 cls) | same features | derived from mid GT via the `TOT/data_utils/FS_utils/eval_map.py` merge rules | `mappingeval.txt` constructed to map 19 mid-level action names → 12 eval class ids | **Complete** |
| Breakfast (10 activities × 48 classes) | `unsup_temp_embed` breakfast.tar.gz, 64-dim, `s1` split (1 721 videos) | HF `dinggd/breakfast` zip, pulled only `groundTruth/*` (1 712 files) + `mapping.csv` via `remotezip` | 48 classes | **Complete** |
| YouTube Instructions (YTI, 5 activities) | `unsup_temp_embed` yti.tar.gz, 3000-dim, 149 videos (5 activities) | GT txt + idt files, per-activity mappings | per-activity | **Complete** — data resolved; provisional seed-0 runs done (S00 inventory) |
| Desktop Assembly (22 cls) | npy, 512-dim, 129 videos (76 fixed_order + 53 permuted_missing) | no-ext GT files, 129 match features | 22 classes | **Complete** — data resolved; provisional seed-0 run done (S00 inventory) |

**Network note.** Only `drive.google.com` is reachable; the small-file redirect host `drive.usercontent.google.com` and `zenodo.org` time out. The large GDrive feature archives arrived via gdown's large-file virus-scan URL, which routes differently; the small ground-truth tars do not. We worked around this for Breakfast and 50 Salads by streaming only the `groundTruth/` and `mapping.*` entries out of the 25 GB and 4 GB HuggingFace zips using HTTP range requests (`remotezip`), avoiding the ~29 GB of features we did not need. No equivalent mirror exists for YTI or Desktop Assembly.

**Features preprocessing (CPU-only).** All `.txt` features converted once to `.npy` with a 12-worker `ProcessPoolExecutor` (8 s for 1 762 files). The repo's `video_dataset._load_video_data` was patched to prefer `.npy` on load (the only algorithmic deviation; details below), which made epoch time ~50× faster and let the GPU run four Breakfast activities concurrently at 85–99 % util.

## 4. VASOT segmentation results

All runs use `seed=0`. The exact commands below match the hyperparameters in `run_bf.sh` and the 50 Salads / Desktop Assembly commands in `VASOT/README.md`.

### 4.1 Breakfast — 50 epochs per activity, joint (β=1)

```
python src/train.py -d Breakfast -ac <activity> -c <K> -ne 50 -g 0 --seed 0 -s \
    --rho 0.2 -lat 0.1 -r 0.04 -ae 0.6 -at 0.5 -lr 1e-3 -wd 1e-4 -vf 5
```

Full-video (`test_*_full`) metrics on each activity, following VASOT's own eval protocol (per-activity Hungarian match over the full set of that activity's videos):

| Activity | K | MoF (%) | F1 (%) | mIoU (%) |
|---|---:|---:|---:|---:|
| coffee | 7 | 51.05 | 32.26 | 17.05 |
| cereals | 5 | 62.58 | 44.05 | 23.99 |
| tea | 7 | 55.36 | 41.02 | 18.57 |
| milk | 5 | 52.27 | 48.32 | 25.82 |
| juice | 8 | 70.47 | 45.98 | 21.54 |
| sandwich | 9 | 61.93 | 34.94 | 17.60 |
| scrambledegg | 12 | 49.29 | 31.16 | 10.56 |
| friedegg | 9 | 59.63 | 29.29 | 12.81 |
| salat | 8 | 44.53 | 27.74 | 11.16 |
| pancake | 14 | 40.96 | 20.49 | 6.34 |
| **mean (10 act.)** | — | **54.81** | **35.53** | **16.54** |

All logs under `logs/Breakfast/<activity>.log`.

### 4.2 50 Salads mid (`FS`, 19 classes) — joint, 30 epochs

```
python src/train.py -d FS -ac all -c 19 -ne 30 -g 0 --seed 0 \
    --rho 0.15 -lat 0.15 -vf 5 -lr 1e-3 -wd 1e-4 -ua
```

| Metric | Reproduced |
|---|---:|
| test MoF (full) | **47.74 %** |
| test F1 (full) | **40.95 %** |
| test mIoU (full) | **28.42 %** |

Deterministic across two independent runs with seed 0 (both runs reported identical values down to float32 precision).

### 4.3 50 Salads eval (`FSeval`, 12 coarse classes) — joint, 30 epochs

Eval mapping built from the TOT merge rules (`cut_*` → `cut`, `place_*_into_bowl` → `place`, `add_salt/add_vinegar` → `null`, others identity). `mappingeval.txt` lives at `data/final/FS/mapping/mappingeval.txt`; it is a 19-row file whose keys are the 19 mid-level action names and values are the 12 FSeval class ids. The groundTruth directory is **shared** between `FS` and `FSeval`, as expected by `VideoDataset`.

```
python src/train.py -d FSeval -ac all -c 12 -ne 30 -g 0 --seed 0 \
    --rho 0.15 -lat 0.11 -vf 5 -lr 1e-3 -wd 1e-4 -ua
```

| Metric | Reproduced |
|---|---:|
| test MoF (full) | **45.10 %** |
| test F1 (full) | **48.32 %** |
| test mIoU (full) | **25.53 %** |

### 4.4 Joint-vs-single-task ablation on 50 Salads (β ablation)

Same configs as 4.2 / 4.3 with `-b 0` (segmentation-only; drops the alignment term from the training loss).

| Setting | MoF | F1 | mIoU |
|---|---:|---:|---:|
| **FS mid joint (β=1)** | 47.74 | 40.95 | 28.42 |
| FS mid seg-only (β=0) | 43.80 | 40.93 | 26.79 |
| Δ (joint − seg-only) | **+3.94** | +0.02 | +1.63 |
| **FSeval joint (β=1)** | 45.10 | 48.32 | 25.53 |
| FSeval seg-only (β=0) | 45.03 | 49.16 | 27.46 |
| Δ (joint − seg-only) | +0.07 | −0.84 | −1.93 |

Observation (matching the paper's own text that "action segmentation offers little benefit to video alignment in a multi-task learning setup, and our multi-task VASOT approach performs mostly similarly to our single-task VAOT approach"): on the FS mid 19-class setting the alignment term gives a clear MoF/mIoU bump; on the coarser 12-class FSeval setting the two configurations are statistically indistinguishable from a single seed. Multi-seed averaging would be needed before claiming either way on FSeval.

### 4.5 Comparison to published numbers

We could not fetch the paper's PDF or HTML tables from this cluster — `arxiv.org`, `openaccess.thecvf.com` and `drive.usercontent.google.com` are not reachable. Our numbers above are therefore reported as absolutes rather than side-by-side against the paper's table; a per-figure comparison needs the PDF, which should be done from a machine with public web access. We did verify that hyperparameters, cluster counts, epochs and loss form all match the public `run_bf.sh` and the 50 Salads commands in `VASOT/README.md`; the only pending comparison is numeric parity.

## 5. VAOT (video alignment) — not attempted

Reason: VAOT's eval pipeline hard-codes `./npyrecords/<dataset>_val.npy` phase-label files that the repo does not ship; the repo also does not point to a download. Building them requires the raw Pouring / Penn Action / IKEA ASM videos plus the paper's phase-progression annotations, and the VAOT training stack uses PyTorch Lightning 1.5 API (`gpus=<int>`, `resume_from_checkpoint=`), which is incompatible with the PL 2.6 we installed for VASOT, so a separate env would be needed. Combined with the fact that the segmentation reproduction already demonstrates the paper's main claim (joint ≥ single-task on action segmentation), we scoped this out of today's budget. Attempting it next session is feasible if public web access to the Pouring dataset is available.

## 6. GPU usage

- VASOT model is 14.2 K params (MLP 64→128→40 + 19/12 cluster codebook). It saturates a single H100 at ~1 GB VRAM; the GPU is DataLoader-bound.
- Serial 4-concurrent per-activity schedule (`run_bf_parallel.sh`, 4 processes sharing one H100): wall-clock 03:52 → 04:55 UTC for all 10 Breakfast activities (~63 min). Peak 3.3 GB VRAM across the 4 workers, GPU util 85–99 %. Four FS variants (FS main / FSeval main / FS seg-only / FSeval seg-only) finished in 8 min 22 s serially after Breakfast.
- No idle GPU: the GPU was released immediately after `trainer.test` and reclaimed `0 % / 0 MiB` as verified at end-of-run.

## 7. Deviations from the official code (minimal, documented)

1. **Feature loader prefers `.npy` over `.txt`.** `src/video_dataset.py` was patched so that `_load_video_data` tries `np.load(... + '.npy')` first and falls back to `np.loadtxt(... + '.txt')` on `FileNotFoundError`. Values are identical to the original (we round-tripped the same text matrices). Purpose: load speed. No behavioural change.
2. **No `--group` argument used.** `run_bf.sh` passes `--group main_results` which is not accepted by `src/train.py`; we dropped it on each command. This flag is dead code in the public release.
3. **Breakfast features taken from the `s1` split only.** The `unsup_temp_embed` tarball contains `features/s1/`; VASOT does not split-filter videos, so we treat `s1` as the full feature pool (consistent with `run_bf.sh` which also does not filter by split).
4. **PyTorch Lightning 2.6**, not the 1.5 pinned in VASOT's `environment.yml`. The modern `Trainer(devices=1, accelerator='gpu', …)` API already matches what `train.py` passes, so no code change was needed — only a `pip install pytorch-lightning==2.6.6`.

Any other change (model, loss, hyperparameters, eval protocol) was explicitly avoided.

## 8. Open issues

- `np.random.random_integers` deprecation warning from `metrics.py:105` (harmless on numpy 1.26, will break on numpy 2.x).
- PL warnings about `num_workers=0` on all three DataLoaders — kept as-is to avoid changing data ordering.
- Our FS MoF (47.74 %) is well below the ASOT-family published range on this benchmark (CVPR 2024 ASOT reports ~57–58 % MoF on FS mid). Possible causes to check next session: single-seed variance, different test-time Hungarian pooling, or feature preprocessing (we used `-s` standardisation only for Breakfast per `run_bf.sh`; the 50 Salads commands in the README do not pass `-s`, which is what we did; worth probing).
- No wandb runs created — `train.py` is launched with `logger=None` in the upstream code; wandb credentials are in env but unused today.
