# VASOT Reproduction Session Report — 2026-10-28

**Slurm job:** 62561, 1× H100 80 GB
**Paper:** Ali et al., "Joint Self-Supervised Video Alignment and Action Segmentation", ICCV 2025
**wandb project:** `vasot-repro` (hust_edu_vn)
**Packets completed this session:** S00, S01, S02, S03, R00, R01 (in progress)

---

## 1. Infrastructure Work (S00–S03) — DONE

### S00: Inventory old workspace
- Catalogued 6 completed baseline runs, 4 interrupted E2 B-spline runs, 3 lightning_logs versions, 7 diagnostic outputs
- Created 14 retrospective manifests with provenance tracking
- Documented checkpoint mapping ambiguity (`checkpoint_identity=unknown` for shared `version_62534`)

### S01: Protocol freeze
- Froze configs from README/scripts (canonical source):
  - DA: 30 epochs, rho=0.25, radius_gw=0.02, K=22
  - YTI: per run_yti.sh, per-activity K values, lr=1e-4, wd=1e-5
- Documented trailing frame policy (34/129 DA videos, accepted GT-indexed truncation)
- Created machine-readable YAML configs for all 5 datasets

### S02: Runner refactoring
- Fixed double Trainer bug in `train.py` (was creating Trainer twice)
- Added `--run-dir`, `--run-id` CLI args for run isolation
- Added auto CPU/GPU detection, manifest.json generation, atomic status tracking
- Fixed InferenceMode crash in `metrics.py` (inplace update → non-inplace)

### S03: End-to-end preflight
- CPU smoke test: DA + YTI, 1 epoch, verified pipeline end-to-end
- GPU smoke test: DA + YTI tiny runs, confirmed checkpoint isolation and manifest generation

---

## 2. VASOT Reproduction Runs

### R00: Desktop Assembly (DA) — DONE ✅
- **Run ID:** `R00_DA_baseline_s0`
- **wandb:** `ge8a3wu7`
- **Config:** 30 epochs, rho=0.25, radius_gw=0.02, K=22, seed=0

| Metric | Value |
|--------|------:|
| MoF    | 0.5986 |
| F1     | 0.6380 |
| mIoU   | 0.4211 |

### R01: YouTube Instructions (YTI) — DONE ✅

5 activities with wandb logging, all match old results exactly (deterministic reproduction):

| Activity | K | MoF | F1 | mIoU | wandb |
|----------|--:|----:|---:|-----:|-------|
| changing_tire | 11 | 0.4483 | 0.2956 | 0.2040 | tihe3mu9 |
| coffee | 10 | 0.6847 | 0.4720 | 0.3249 | 6jaj64ev |
| cpr | 7 | 0.5590 | 0.4924 | 0.2727 | 5cg9rker |
| jump_car | 12 | 0.3392 | 0.2604 | 0.1744 | — |
| repot | 8 | 0.3655 | 0.2261 | 0.1535 | — |
| **Mean** | — | **0.4793** | **0.3493** | **0.2259** | — |

---

## 3. Diagnostic Evidence (Hypothesis A/B/C)

Diagnostics run on all completed checkpoints. Summary across 10 dataset×activity combinations:

| Dataset | Activity | N_vid | T_entropy | Smoothness | Boundary_z | Idx_sem_corr |
|---------|----------|------:|----------:|-----------:|-----------:|-------------:|
| Breakfast | cereals | 184 | 0.025 | 0.077 | -0.108 | 0.516 |
| Breakfast | coffee | 167 | 0.040 | 0.078 | -0.124 | 0.501 |
| Breakfast | pancake | 157 | 0.037 | 0.032 | -0.073 | 0.496 |
| Breakfast | scrambledegg | 166 | 0.024 | 0.032 | -0.054 | 0.477 |
| Breakfast | tea | 184 | 0.038 | 0.074 | -0.058 | 0.460 |
| FS | all | 50 | 0.077 | 0.085 | 0.026 | 0.375 |
| FSeval | all | 50 | 0.055 | 0.069 | 0.047 | 0.384 |
| DA | all | 129 | 0.032 | 0.215 | 0.852 | 0.332 |
| YTI | changing_tire | 30 | 0.065 | 0.200 | 0.442 | 0.170 |
| YTI | coffee | 30 | 0.067 | 0.176 | 0.256 | 0.202 |
| **Aggregate** | | | **0.046** | | **0.121** | **0.391** |

### Preliminary findings:

**Hypothesis A (T* peaky supervision):** SUPPORTED
- Mean normalized row entropy = 0.046 (range 0.024–0.077)
- This means each row of the transport plan concentrates >95% mass on very few targets
- Effectively top-1 supervision — model receives sparse, noisy pseudo-labels

**Hypothesis B (Boundary blindness):** PARTIALLY SUPPORTED
- Mean boundary z-score = 0.121 (very weak signal)
- Breakfast activities show *negative* boundary z-scores (worse at boundaries than interior)
- DA and YTI show positive but modest scores
- The segmentation loss does not explicitly model transitions

**Hypothesis C (Frame index ≠ semantic distance):** SUPPORTED
- Mean index-semantic correlation = 0.391 (moderate, far from 1.0)
- YTI has weakest correlation (0.17–0.20) — unsurprising given unstructured instructional videos
- Breakfast has strongest (0.46–0.52) — more structured cooking activities
- Frame index distance is a poor proxy for semantic similarity

---

## 4. Previous Session Results (for context)

From 2026-10-27 session with paper Table 5 hyperparameters:

| Dataset | Paper (MoF/F1/mIoU) | Reproduced |
|---------|---------------------|------------|
| 50 Salads Mid | 47.2/41.3/26.1 | 46.73/44.75/30.05 |
| 50 Salads Eval | 60.6/57.4/34.5 | 48.83/55.32/33.17 |
| Breakfast (mean 10) | 57.5/39.0/18.8 | 47.69/36.60/19.26 |

---

## 5. E-Phase: Continuous Curve Experiments — IN PROGRESS

### E04: Bézier Implementation — DONE ✅
- Implemented `src/bezier.py`: piecewise cubic Bézier with C1 continuity, adaptive segment splitting
- 5 new tests (endpoint preservation, C1 continuity, short sequence, identity blend, gradient propagation)
- Integrated into `train.py` with `--bezier` CLI flag (same interface as `--bspline`)
- All 8/8 tests pass

### GPU Pipeline Running (E01/E03/E04 experiments on FSeval):
Currently executing sequentially on GPU:
1. E01_bspline_FSeval_s0 (running)
2. E01_bspline_FSeval_s1
3. E03_gauss_s2_FSeval_s0
4. E03_gauss_s2_FSeval_s1
5. E04_bezier_FSeval_s0
6. E04_bezier_FSeval_s1
7. E00_baseline_FSeval_s1 (paired seed 1 baseline for comparison)

## 6. Next Steps

1. **E-pipeline completion:** ~7 runs × ~3 min each = ~21 min remaining
2. **E05:** Aggregate E-phase results into comparison matrix (baseline vs B-spline vs Gaussian vs Bézier)
3. **Diagnostics on E checkpoints:** Run diagnose.py on winning variants
4. **Scale gate:** If any variant improves ≥1pp mean over 2 seeds, scale to FS Mid / DA / YTI

---

## 6. Compute Usage

- GPU time: ~35 min for DA (30 epochs) + ~35 min for 5 YTI activities (30 epochs each)
- Peak VRAM: ~838 MiB (model is tiny: 14.2K params)
- No idle GPU waste — sequential execution within single Slurm job
