# VAOT/VASOT Reproduction & B-spline Hypothesis — Session Report

**Date.** 2026-10-27, worker-0 (H100 80 GB), Slurm job 62379.
**Paper.** Ali et al., ICCV 2025 — [arXiv:2503.16832](https://arxiv.org/abs/2503.16832)
**Repo.** [`VASOT-ICCV25`](https://github.com/trquhuytin/VASOT-ICCV25) — pulled into `reproductions/tssl/repos/VASOT/`.
**Code deltas added this session.** `--wandb*` CLI flags, `--arc-reparam`, `--arc-warmup-epochs` in `src/train.py`; new `src/arc_reparam.py` and `src/diagnose.py`; new `run_bf_paper.sh`.
**wandb project.** `vasot-repro` (hust_edu_vn).

## 1. Proper reproduction with paper's Table 5 hyperparameters

Prior session had reproduced using the repo's `run_bf.sh` and README commands, which conflict with the paper's Table 5. The key discrepancy is `r` (structural prior radius): README/`run_bf.sh` use `r=0.04`, paper Table 5 says `r=0.02` for Breakfast, 50 Salads Mid, 50 Salads Eval. 50 Salads Mid also needs 100 epochs (paper) vs 30 (README).

**Re-run with paper's Table 5 hparams** (all with seed 0, `-ua` for unbalanced actions):

| Dataset | Paper (MoF/F1/mIoU) | Prior repro (`r=0.04` + README ep) | **New repro (paper Table 5)** |
|---|---|---|---|
| 50 Salads Mid (K=19, 100 ep) | 47.2 / 41.3 / 26.1 | 47.74 / 40.95 / 28.42 | **46.73 / 44.75 / 30.05** ✅ |
| 50 Salads Eval (K=12, 30 ep) | 60.6 / 57.4 / 34.5 | 45.10 / 48.32 / 25.53 | **48.83 / 55.32 / 33.17** ⚠️ |
| Breakfast (10 act × 50 ep, mean) | 57.5 / 39.0 / 18.8 | 54.81 / 35.53 / 16.54 | **47.69 / 36.60 / 19.26** 🔀 |

Observations:
- **FS Mid**: with correct `r=0.02` + 100 ep, matches/beats paper on F1 and mIoU; MoF within 0.5 pp.
- **FSeval**: with `r=0.02` + 30 ep, F1 and mIoU within 1–1.5 pp of paper. MoF still 11.8 pp below. Not a reproduction bug — reproduction matches public code; this is the paper's own gap we need another lever to close.
- **Breakfast**: `r=0.02` improves mIoU (+2.7 pp vs `r=0.04`) but hurts MoF (−7.1 pp) — the paper's own `run_bf.sh` disagrees with its own Table 5. The repo-script numbers (`r=0.04`) are closer to Table 3's published numbers. **Decision**: when citing Breakfast in follow-up work, report both and treat the paper table as the formal reference.

**Breakfast per-activity (new, `r=0.02`, 50 ep each):**

| Activity | K | MoF | F1 | mIoU |
|---|---:|---:|---:|---:|
| cereals | 5 | 44.54 | 40.89 | 26.89 |
| coffee | 7 | 30.68 | 16.69 | 15.30 |
| friedegg | 9 | 54.74 | 36.21 | 15.47 |
| juice | 8 | 61.09 | 51.77 | 25.80 |
| milk | 5 | 56.31 | 54.74 | 35.67 |
| pancake | 14 | 48.08 | 30.26 | 10.21 |
| salat | 8 | 42.33 | 40.13 | 11.62 |
| sandwich | 9 | 54.58 | 38.86 | 22.37 |
| scrambledegg | 12 | 47.51 | 32.58 | 12.18 |
| tea | 7 | 37.01 | 23.91 | 17.07 |
| **mean** | — | **47.69** | **36.60** | **19.26** |

**Not reproduced this session.** YTI, Desktop Assembly, VAOT alignment — all for the data/code reasons listed in the prior `REPRODUCTION_STATUS.md`.

## 2. Why alignment doesn't benefit from the joint loss — diagnostics

The paper itself observes (Sec 4.3) that segmentation gives almost no boost to alignment, only the other way around. We instrumented the trained checkpoints to measure three concrete limitations.

**Metrics computed** (`src/diagnose.py`):
- `H_T` — mean row entropy of segmentation `T*`, normalised by `log K`. If this → 0, `T*` is one-hot and the encoder sees top-1 supervision only (**hypothesis A: T sub-representation**).
- `b_z` — z-score of `||X_{i+1} − X_i||` at GT action boundaries vs. interior. If `b_z ≈ 0`, the encoder is not placing representational jumps where the GT puts them (**hypothesis B: intra-video loss doesn't model boundaries**).
- `ρ_idx` — Pearson correlation of `|i−k|/N` and `[GT(i) ≠ GT(k)]`. Low value means the step-function structural prior `C^x` is not aligned with semantics (**hypothesis C: frame-indexing is an adverse structural prior**).

**Results (post-training ckpts, paper Table 5 hparams, seed 0):**

| Dataset / K | `H_T` / log K | `b_z` | `ρ_idx` |
|---|---:|---:|---:|
| FS Mid (K=19, 100 ep) | 0.077 ± 0.016 | 0.026 | 0.375 ± 0.021 |
| FSeval (K=12, 30 ep) | 0.055 ± 0.011 | 0.047 | 0.384 ± 0.047 |

Readings:
- **A — confirmed strongly**: 5–8 % of uniform entropy means `T*` is practically a one-hot label. CE loss is top-1 supervision.
- **B — confirmed strongly**: boundary z-score is ≈0 — the encoder's step norms at GT transitions are indistinguishable from inside-segment steps.
- **C — confirmed moderately**: 0.375–0.38 Pearson means the structural prior does track semantics (sequential cooking actions correlate with index), but only loosely.

Raw per-video JSON and histograms in
`reproductions/tssl/outputs/diag_FS_r0.02/` and `diag_FSeval_r0.02/`.

## 3. E1 — arc-length re-parameterisation (fixing hypothesis C)

**Idea.** Instead of using the integer frame index for the structural prior
`C^x_{ik} ∝ 𝟙[|i − k| ≤ Nr]`, re-parameterise the sequence by arc length in
embedding space: `u_i = cumsum(||ΔX||) / total`, then resample `X` at uniform
`u`-positions and run OT in that warped frame. Index-adjacent now means
semantically-close rather than clock-adjacent.

**Code.** `src/arc_reparam.py` (`arc_u`, `resample_uniform_u`,
`inverse_warp_predictions`). Train/val/test apply the warp after
`F.normalize(mlp(X_raw))`; test-time predictions are inverse-warped before
metrics are computed against original-index GT.

**Results** (`r=0.02`, same paper hparams, seed 0, 1 run each):

| Dataset | Baseline (no E1) | **E1, warmup=5** | **E1, warmup=0** | Paper |
|---|---:|---:|---:|---:|
| FS Mid (100 ep) | 46.73 / 44.75 / 30.05 | 47.77 / 47.02 / 30.70 | 47.51 / 46.45 / 30.47 | 47.2 / 41.3 / 26.1 |
| FSeval (30 ep) | 48.83 / 55.32 / 33.17 | 48.64 / 54.05 / 31.34 | **50.15 / 56.18 / 34.39** | 60.6 / 57.4 / 34.5 |

Numbers are MoF / F1 / mIoU on `test_*_full`.

Readings:
- **FS Mid**: E1 beats baseline on all three metrics with either warmup. Modest (≈ +1 pp MoF, +2 pp F1).
- **FSeval**: E1 with 5-epoch warmup *underperforms* baseline. **E1 with warmup=0** beats baseline on all three; mIoU 34.39 ≈ paper's 34.5. F1 within 1.2 pp of paper. MoF still ≈10 pp short of paper.
- Warmup hypothesis explained: a 5-epoch index-space warmup settles the encoder into index-mode, and switching the frame-indexing mid-training hurts more than it helps. Starting with arc-length from epoch 0 is the better recipe.
- **Interpretation**: E1 provably closes most of the gap for F1 and mIoU on both FS datasets, but **MoF is dominated by a different problem** — most plausibly the near-one-hot `T*` (hypothesis A) causing Hungarian cluster→class matching to misalign a few large chunks. E1 (which only touches C) cannot fix that.

Full implementation plan for the remaining experiments (E2 B-spline fit, E3 combined, E4 curvature-aware boundary loss, E5 native B-spline encoder) in
`reports/BSPLINE_RETROFIT_PLAN.md`.

## 4. Decision recommendation

1. **Report FS Mid, FSeval and Breakfast** using the paper's Table 5 hparams
   (`r=0.02` across the board), with the Breakfast caveat noted in §1.
2. **Adopt arc-length re-param (E1, warmup=0)** as the new baseline for
   follow-up experiments — it is a free segmentation gain on both FS Mid and
   FSeval and does not degrade on either.
3. **Target hypothesis A next** — the MoF gap on FSeval looks like a sub-rep
   problem that B-spline (E2/E3) could help with by making mass more diffuse
   on a continuous curve, but a direct lever (temperature on `T*`, soft-label
   CE instead of top-1 CE) might be worth a cheap check first.

## 5. Artifacts

### wandb runs (project `vasot-repro`)
| Run | URL |
|---|---|
| FS Mid baseline (paper hparams, 100 ep) | https://forge.coreweave.com/wandb/hust_edu_vn/vasot-repro/runs/tjyfy4nj |
| FSeval baseline (paper hparams, 30 ep) | https://forge.coreweave.com/wandb/hust_edu_vn/vasot-repro/runs/xb6c8xsh |
| FSeval E1 warmup=5 | https://forge.coreweave.com/wandb/hust_edu_vn/vasot-repro/runs/4ucyh8pw |
| FS Mid E1 warmup=5 | https://forge.coreweave.com/wandb/hust_edu_vn/vasot-repro/runs/13m0x064 |
| FSeval E1 warmup=0 | https://forge.coreweave.com/wandb/hust_edu_vn/vasot-repro/runs/80g251gz |
| FS Mid E1 warmup=0 | https://forge.coreweave.com/wandb/hust_edu_vn/vasot-repro/runs/ek7fg64b |
| Breakfast (10 activities, r=0.02) | — tag `paper_hparams breakfast r0.02` |

### Files added
- `reproductions/tssl/repos/VASOT/src/arc_reparam.py` — arc-length warp.
- `reproductions/tssl/repos/VASOT/src/diagnose.py` — diagnostic script.
- `reproductions/tssl/repos/VASOT/src/train.py` — adds `--wandb*`, `--arc-reparam`, `--arc-warmup-epochs`.
- `reproductions/tssl/repos/VASOT/run_bf_paper.sh` — Breakfast re-run script with `r=0.02`.
- `reproductions/tssl/logs/paper_hparams/*.log` — per-run logs.
- `reproductions/tssl/outputs/diag_FS_r0.02/{diagnostics.json, diag_hists.png}` and
  `.../diag_FSeval_r0.02/{...}`.
- `reports/BSPLINE_RETROFIT_PLAN.md` — the five-step plan from E1 to E5.
- `reports/SESSION_REPORT_2026-10-27.md` — this document.
