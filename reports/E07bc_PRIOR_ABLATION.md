# E07b / E07c — Direct interventions on Hypothesis C (frame-index prior)

**Date:** 2026-10-29
**Dataset:** 50 Salads Eval (FSeval), 30 epochs, paired seeds {0, 1}
**Baseline config:** `configs/vasot/baselines/fs_eval.yaml` (lock
`reports/E00_contract.lock`); `rho = 0.15`, `radius_gw = 0.04`.

> **2026-10-29 correction:** The FSeval numbers below are valid, but several
> causal claims in the original interpretation were not. `rho` weights the
> temporal prior in **both** segmentation and alignment. E08 found `rho=0`
> worsens Desktop Assembly and YTI seed-0 MoF; E11 is separating the two
> pathways. E09 measured actual `T_xy` and found it is not near one-hot on
> FSeval. Higher epsilon does not soften it in this solver. Bézier checkpoints
> are all-NaN and their metrics must not be treated as valid.

## Motivation

E07a (ε knob) + E01/E03/E04 (representation smoothing) motivated a test of
Hypothesis C, but do not rule out A: E09 subsequently showed that higher ε
hardens `T_xy` in this implementation, while the smoothing variants affect
`T_xa` and `T_xy` differently. The remaining diagnosis support in D03 points at
**Hypothesis C** — the frame-index prior encoded by `R` (temporal prior scaled
by `rho`) and `C^x` (structural diagonal prior with band width `radius_gw`).
Those two knobs are the direct interventions on C.

- **E07b** targets `rho` (strength of temporal prior `R`):
  `rho ∈ {0.00, 0.05}` vs baseline 0.15.
- **E07c** targets `radius_gw` (width of structural prior `C^x`):
  `radius_gw ∈ {0.12, 0.50}` vs baseline 0.04.

Baselines held in both: paired seeds {0, 1}, every other hyperparameter from
the frozen `fs_eval.yaml`.

## Results

### E07b — rho ablation

| Run ID | Seed | rho | MoF | F1 | mIoU |
|---|---:|---:|---:|---:|---:|
| baseline_s0 (ref) | 0 | 0.15 | 0.4510 | 0.4832 | 0.2553 |
| E00_baseline_FSeval_s1 (ref) | 1 | 0.15 | 0.4945 | 0.5165 | 0.2841 |
| E07b_rho0p05_FSeval_s0 | 0 | 0.05 | 0.5283 | 0.5339 | 0.3209 |
| E07b_rho0p05_FSeval_s1 | 1 | 0.05 | 0.4993 | 0.4963 | 0.2841 |
| E07b_rho0p0_FSeval_s0 | 0 | 0.00 | **0.5049** | 0.5137 | 0.3025 |
| E07b_rho0p0_FSeval_s1 | 1 | 0.00 | **0.5995** | 0.5581 | 0.3208 |

### E07c — radius_gw ablation

| Run ID | Seed | radius_gw | MoF | F1 | mIoU |
|---|---:|---:|---:|---:|---:|
| E07c_r0p12_FSeval_s0 | 0 | 0.12 | 0.4429 | 0.4295 | 0.2023 |
| E07c_r0p12_FSeval_s1 | 1 | 0.12 | 0.4456 | 0.4325 | 0.1640 |
| E07c_r0p50_FSeval_s0 | 0 | 0.50 | 0.3164 | 0.2927 | 0.2007 |
| E07c_r0p50_FSeval_s1 | 1 | 0.50 | 0.3251 | 0.3178 | 0.2073 |

### Variant summary (mean over 2 seeds)

| Variant | Mean MoF | Δ MoF (pp) | Mean mIoU | Δ mIoU (pp) | Scale gate |
|---|---:|---:|---:|---:|---|
| Baseline (rho=0.15, radius=0.04) | 0.4727 | — | 0.2697 | — | — |
| **rho = 0.00** | **0.5522** | **+7.95** | 0.3116 | **+4.20** | **PASS** |
| **rho = 0.05** | **0.5138** | **+4.10** | 0.3025 | **+3.28** | **PASS** |
| radius_gw = 0.12 (3×) | 0.4442 | −2.85 | 0.1831 | −8.66 | FAIL |
| radius_gw = 0.50 (12×) | 0.3207 | −15.20 | 0.2040 | −6.57 | FAIL |

## Headline

1. **`rho` is an effective FSeval lever, not a general solution to C.** Both `rho = 0.00` and
   `rho = 0.05` pass the ≥1 pp scale gate, with `rho = 0` giving the biggest
   margin yet seen in this project (+7.95 pp MoF, +4.20 pp mIoU on FSeval mean).
   The two-seed mean improves as `rho` decreases, although the per-seed
   response is not monotonic. E08 shows this effect does not transfer to
   Desktop Assembly or YTI seed 0.
2. **`radius_gw` is NOT the lever.** Widening `C^x`'s band by 3× moderately
   hurts (−2.85 pp); by 12× it collapses MoF to 0.32. The structural prior
   is apparently load-bearing at its tight-band baseline value and does not
   want to be relaxed by itself.
3. The two knobs behave differently, but `rho` affects both the segmentation
   and alignment temporal priors. The current ablation cannot attribute the
   FSeval gain to one pathway; E11 is designed to do that.

## Full E-phase matrix (all variants on FSeval, from `collect_e_results.py`)

| Variant | n | Mean MoF | Δ MoF (pp) | Scale gate |
|---|---:|---:|---:|---|
| Baseline | 2 | 0.4727 | — | — |
| Arc-length | 2 | 0.4649 | −0.78 | FAIL |
| B-spline | 2 | 0.3532 | −11.95 | FAIL |
| Gaussian σ=2 | 2 | 0.3673 | −10.55 | FAIL |
| Bézier 4-seg | 2 | invalid (NaN checkpoints) | — | NUMERICAL FAILURE |
| ε = 0.15 (E07a) | 2 | 0.3237 | −14.90 | FAIL |
| ε = 0.30 (E07a) | 2 | 0.2140 | −25.87 | FAIL |
| **rho = 0.05 (E07b)** | 2 | **0.5138** | **+4.10** | **PASS** |
| **rho = 0.00 (E07b)** | 2 | **0.5522** | **+7.95** | **PASS** |
| radius_gw = 0.12 (E07c) | 2 | 0.4442 | −2.85 | FAIL |
| radius_gw = 0.50 (E07c) | 2 | 0.3207 | −15.20 | FAIL |

## Why this re-frames the project

The project brief listed smoothening the discrete representation (B-spline /
Bézier / arc-length) as the proposal to test against limitations A/B/C. The
E-phase matrix now shows:

- The finite smoothing proposals tested here were below baseline on FSeval.
- Increasing training ε worsened performance but also caused solver/transport
  degeneration; it cannot test whether softer `T_xy` would help.
- `rho=0` improved FSeval but worsened DA and YTI in seed 0.

The defensible reading is narrower: FSeval is sensitive to temporal-prior
weighting, but neither the responsible pathway nor a dataset-general fix has
been established. E09 found that the original explanation via a nearly
one-hot alignment plan was incorrect on its measured FSeval protocol.

## Provenance

- Launch script: `reproductions/tssl/scripts/run_e07bc_prior_ablation.sh`
- Baseline config lock: `reports/E00_contract.lock`
  (sha256 `63148876ba0ddfe58332b066f1211349f04b16b39f9f33b6b57feb7446ea9f99`)
- Manifests: `reproductions/tssl/artifacts/runs/E07{b,c}_*/manifest.json`
- Wandb project `vasot-repro`, tags `E07b` / `E07c`, `hypothesis_C`.
- Collect script: `reproductions/tssl/scripts/collect_e_results.py`
  (updated to recognise E07a/E07b/E07c run_ids).

## Follow-ups and status

- **E08 completed:** `rho=0` loses 3.23 pp MoF on DA and 4.17 pp on the
  five-activity YTI mean over paired seeds; see `E08_RHO0_TRANSFER.md`.
- **E09 completed:** actual `T_xy` support and direct boundary metrics revise
  the A/B interpretation; see `E09_LIMITATION_RECHECK.md`. The originally
  proposed joint `rho × radius_gw` ablation was not run because E08 failed
  the cross-dataset transfer check.
- **E11 running:** separate the segmentation and alignment uses of `rho` on
  FSeval before attributing the local gain to one prior.
- **Hypothesis B intervention remains open:** E09 measures boundary errors,
  but no boundary-aware training loss has been tested.
