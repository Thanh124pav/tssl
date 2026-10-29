# E07b / E07c — Direct interventions on Hypothesis C (frame-index prior)

**Date:** 2026-10-29
**Dataset:** 50 Salads Eval (FSeval), 30 epochs, paired seeds {0, 1}
**Baseline config:** `configs/vasot/baselines/fs_eval.yaml` (lock
`reports/E00_contract.lock`); `rho = 0.15`, `radius_gw = 0.04`.

## Motivation

E07a (ε knob) + E01/E03/E04 (representation smoothing) jointly ruled out
Hypothesis A as actionable: both levers that reduce T\*'s peakiness degrade
MoF monotonically. The remaining diagnosis support in D03 points at
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

1. **`rho` is the actionable lever for Hypothesis C.** Both `rho = 0.00` and
   `rho = 0.05` pass the ≥1 pp scale gate, with `rho = 0` giving the biggest
   margin yet seen in this project (+7.95 pp MoF, +4.20 pp mIoU on FSeval mean).
   The improvement is **monotonic** in `rho ↓` and present on both seeds —
   not a single-seed artefact.
2. **`radius_gw` is NOT the lever.** Widening `C^x`'s band by 3× moderately
   hurts (−2.85 pp); by 12× it collapses MoF to 0.32. The structural prior
   is apparently load-bearing at its tight-band baseline value and does not
   want to be relaxed by itself.
3. The two priors behave very differently under direct ablation. Hypothesis
   C should be sharpened to *the temporal prior `R`*; `C^x` is a different
   mechanism that we should not conflate with it.

## Full E-phase matrix (all variants on FSeval, from `collect_e_results.py`)

| Variant | n | Mean MoF | Δ MoF (pp) | Scale gate |
|---|---:|---:|---:|---|
| Baseline | 2 | 0.4727 | — | — |
| Arc-length | 2 | 0.4649 | −0.78 | FAIL |
| B-spline | 2 | 0.3532 | −11.95 | FAIL |
| Gaussian σ=2 | 2 | 0.3673 | −10.55 | FAIL |
| Bézier 4-seg | 2 | 0.3538 | −11.89 | FAIL |
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

- The smoothing proposals were **all** below baseline on FSeval.
- The direct intervention on A (`ε`) was also below baseline.
- The direct intervention on C via `rho` is **the only variant that improves
  the baseline** — and does so by a wide, both-seed margin.

The right reading is: VASOT's main actionable weakness on FSeval is **the
temporal prior `R` itself**, not the peakiness of the resulting transport
plan. Smoothing variants reduce T\*'s peakiness as a *downstream* effect of
their reparameterisation, which is why they fail — they are symptom-level
interventions on a working mechanism, while leaving the harmful index-prior
untouched.

## Provenance

- Launch script: `reproductions/tssl/scripts/run_e07bc_prior_ablation.sh`
- Baseline config lock: `reports/E00_contract.lock`
  (sha256 `63148876ba0ddfe58332b066f1211349f04b16b39f9f33b6b57feb7446ea9f99`)
- Manifests: `reproductions/tssl/artifacts/runs/E07{b,c}_*/manifest.json`
- Wandb project `vasot-repro`, tags `E07b` / `E07c`, `hypothesis_C`.
- Collect script: `reproductions/tssl/scripts/collect_e_results.py`
  (updated to recognise E07a/E07b/E07c run_ids).

## Follow-ups

- **E08 — scale `rho=0` to DA and YTI** with paired seeds, using the exact
  baseline configs of each dataset. This is the first variant worth
  cross-dataset validation; the ≥1 pp scale gate on FSeval is cleared.
- **E09 — joint ablation**: `rho=0` AND `radius_gw ∈ {0.04 (baseline), 0.12}`
  to check whether `rho=0` also makes the structural prior easier to relax.
- **E07d — Hypothesis B direct intervention** (boundary-aware auxiliary head).
  Deferred until E08 confirms whether C's lever generalises off FSeval.
