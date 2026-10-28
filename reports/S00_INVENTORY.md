# S00 — Artifact Inventory & Migration Report

**Date:** 2026-10-28
**Packet:** S00
**Status:** DONE

## 1. Workspace mapping

| Location | Purpose | Size |
|---|---|---|
| `/mnt/data/thanhpv/tssl` | New repo (working tree) | — |
| `/mnt/data/thanhpv/acvcl/reproductions/tssl` | Old workspace (artifact source) | — |
| `…/acvcl/…/data/final/` | Data root (symlinked from VASOT) | ~1.1 GB |
| `…/acvcl/…/repos/VASOT/src/lightning_logs/` | Checkpoints (3 versions) | ~16 MB |
| `…/acvcl/…/logs/session_full/` | Training logs (18 files) | ~108 KB |
| `…/acvcl/…/outputs/` | Diagnostic outputs (7 datasets) | ~700 KB |

Data symlink: `reproductions/tssl/repos/VASOT/data → /mnt/data/thanhpv/acvcl/reproductions/tssl/data/final`

## 2. Completed runs — VASOT DA/YTI baseline (Slurm job 62534)

All 6 runs exit code 0, seed 0, 30 epochs. Ran 2026-10-28 12:12–12:24 UTC.

| Run ID | Dataset | MoF | F1 | mIoU |
|---|---|---:|---:|---:|
| repro_baseline_desktop_assembly_s0 | DA | 0.5986 | 0.6380 | 0.4211 |
| repro_baseline_YTI_cpr_s0 | YTI CPR | 0.5590 | 0.4924 | 0.2727 |
| repro_baseline_YTI_coffee_s0 | YTI coffee | 0.6847 | 0.4720 | 0.3249 |
| repro_baseline_YTI_changing_tire_s0 | YTI changing tire | 0.4483 | 0.2956 | 0.2040 |
| repro_baseline_YTI_jump_car_s0 | YTI jump car | 0.3392 | 0.2604 | 0.1744 |
| repro_baseline_YTI_repot_s0 | YTI repot | 0.3655 | 0.2261 | 0.1535 |

**Provenance issues:**
- All runs shared `lightning_logs/version_62534` → checkpoint-to-run mapping is ambiguous.
- Config came from version_62534 hparams: `layer_sizes=[512,128,40]`, `n_clusters=22`, `rho=0.25`, `radius_gw=0.02`, `n_epochs=30`.
- Protocol question: paper may use 100 epochs / rho=0.2 for DA. Needs S01 resolution.

## 3. Completed runs — Sweep baseline + E1 arc (Slurm job 62534)

| Run ID | Dataset | Seed | MoF | F1 | mIoU |
|---|---|---:|---:|---:|---:|
| sweep_baseline_FSeval_s0 | FSeval | 0 | 0.4510 | 0.4832 | 0.2553 |
| sweep_baseline_FSeval_s1 | FSeval | 1 | 0.4945 | 0.5165 | 0.2841 |
| sweep_E1_arc_FSeval_s0 | FSeval | 0 | 0.4850 | 0.5247 | 0.2821 |
| sweep_E1_arc_FSeval_s1 | FSeval | 1 | 0.4448 | 0.4389 | 0.2500 |

## 4. Interrupted runs — E2 B-spline (SIGTERM at ~epoch 16/30)

| Run ID | Variant | Seed | Status |
|---|---|---:|---|
| sweep_E2_bsx_b03_FSeval_s0 | bspline blend=0.3 | 0 | SIGTERM epoch 16 |
| sweep_E2_bsx_b03_FSeval_s1 | bspline blend=0.3 | 1 | SIGTERM epoch 16 |
| sweep_E2_bsx_b01_FSeval_s0 | bspline blend=0.1 | 0 | SIGTERM epoch 16 |
| sweep_E2_bsx_b01_FSeval_s1 | bspline blend=0.1 | 1 | SIGTERM epoch 16 |

**No test metrics available.** Checkpoints exist in version_62534 but cannot be mapped to specific runs.

## 5. Overnight runs (from earlier session, with wandb)

From `overnight_partial_2026-10-27.tsv`: 20 runs covering baseline + E1_arc on FSeval, FSmid, bf_coffee, bf_cereals, bf_scrambledegg, all seeds 0/1, all exit 0. Each has a wandb run ID.

## 6. Checkpoint inventory

### version_61908 (Slurm job 61908 — earlier overnight Breakfast run)
- 17 checkpoints, 2.9 MB total
- Hparams: `n_clusters=19`, `layer_sizes=[64,128,40]`, `rho=0.15`, `radius_gw=0.04`
- Matches FS mid (19 cls) config. Contains epoch 0–49 checkpoints with `-v` suffixes from concurrent runs.

### version_62379 (Slurm job 62379 — arc sweep)
- 8 checkpoints, 1.5 MB total
- Hparams: `arc_reparam=true`, `n_clusters=12`, `layer_sizes=[64,128,40]`, `radius_gw=0.02`
- Matches FSeval arc-length E1 config.

### version_62534 (Slurm job 62534 — DA/YTI + E2 session)
- 16 checkpoints, 11 MB total
- Hparams: `n_clusters=22`, `layer_sizes=[512,128,40]`, `rho=0.25`, `bspline=false`
- **Shared by all runs in this session** (6 DA/YTI + 4 sweep + 4 E2).
- Checkpoint identity: **UNKNOWN** — `-v` suffixes indicate collisions. Cannot reliably map any single checkpoint to a specific run.

### Standalone checkpoint
- `src/checkpoints/epoch=99-step=7600.ckpt` (850 KB)
- Origin: likely an earlier DA or FS run with 100 epochs (step=7600 = 100 epochs × 76 steps for DA fixed_order). Needs S01 protocol check.

## 7. Diagnostic outputs inventory

All from `diagnose.py`, stored in `outputs/diag_*`:

| Directory | Dataset | Activity | N videos | T_ent_norm | Smoothness | Bnd_z | Idx_sem_corr |
|---|---|---|---:|---:|---:|---:|---:|
| diag_FS_r0.02 | FS | all | 50 | 0.0772 | 0.0853 | 0.0264 | 0.3752 |
| diag_FSeval_r0.02 | FSeval | all | 50 | 0.0547 | 0.0688 | 0.0465 | 0.3842 |
| diag_BF_cereals_r0.04 | Breakfast | cereals | 184 | 0.0249 | 0.0772 | -0.1078 | 0.5161 |
| diag_BF_coffee_r0.04 | Breakfast | coffee | 167 | 0.0395 | 0.0780 | -0.1243 | 0.5005 |
| diag_BF_pancake_r0.04 | Breakfast | pancake | 157 | 0.0366 | 0.0319 | -0.0733 | 0.4964 |
| diag_BF_scrambledegg_r0.04 | Breakfast | scrambledegg | 166 | 0.0244 | 0.0319 | -0.0535 | 0.4769 |
| diag_BF_tea_r0.04 | Breakfast | tea | 184 | 0.0379 | 0.0736 | -0.0579 | 0.4598 |

**Interpretation preview (for D01-D03):**
- **Hypothesis A (peaky T*):** T_entropy_norm 0.024–0.077 → very peaky across all datasets. Breakfast even peakier than 50 Salads.
- **Hypothesis B (boundary blindness):** boundary_z near zero or negative → no meaningful boundary signal.
- **Hypothesis C (index prior):** idx_sem_corr 0.38–0.52 → moderate correlation, meaning index distance is only partially predictive of semantic distance.

## 8. Data status

| Dataset | Features | GT | Mapping | Videos | Status |
|---|---|---|---|---:|---|
| FS (50 Salads mid) | npy, 64-dim | txt | 19 cls | 50 | Complete |
| FSeval (50 Salads eval) | shared with FS | derived | 12 cls | 50 | Complete |
| Breakfast | npy, 64-dim | txt | 48 cls (10 act) | 1712 | Complete |
| Desktop Assembly | npy, 512-dim | no-ext files | 22 cls | 129 | Complete (129 feat, 129 GT) |
| YTI | txt, 3000-dim | txt + idt | per-activity | 149 | Complete (149 feat, 298 GT files incl idt) |

**DA note:** 76 fixed_order + 53 permuted_missing = 129 total. 34/129 have feature one frame longer than GT.
**YTI note:** `repot_0026` missing from both features and GT (filtered consistently).

## 9. Open issues for next packets

1. **S01:** Resolve DA protocol (30 vs 100 epochs, rho 0.25 vs 0.2). The standalone `epoch=99-step=7600.ckpt` suggests 100 epochs was tried.
2. **S01:** Freeze YTI aggregation method (mean-of-activities vs pooled-frame).
3. **S02:** Fix Trainer double-init and checkpoint isolation before any new training.
4. **S02:** Each run must get its own `lightning_logs/version_*` directory.
5. **R00/R01:** Use existing results if protocol matches S01 freeze; rerun only if mismatch.

## 10. Retrospective manifests

14 manifest files created in `reproductions/tssl/artifacts/manifests/`:
- 6 DA/YTI baseline (completed)
- 4 FSeval sweep baseline + E1_arc (completed)
- 4 E2 B-spline (interrupted)

Registry: `reproductions/tssl/artifacts/run_registry.jsonl`
