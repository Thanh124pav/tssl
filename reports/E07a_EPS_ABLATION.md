# E07a — Entropy knob (ε) ablation: direct intervention on Hypothesis A

**Date:** 2026-10-29
**Dataset:** 50 Salads Eval (FSeval), 30 epochs, paired seeds {0, 1}
**Mechanism under test:** Hypothesis A — "T* is peaky, supervision is near top-1".
**Lever:** change the OT solver's entropy regularisation `eps_train`, holding
every other knob constant. Baseline is `eps_train = 0.07`.
**Grid:** `eps_train ∈ {0.15, 0.30}`, 2× the baseline and ~4× the baseline.

This is the counter-evidence check that D04 explicitly scheduled:

> **Counter-evidence to check:** Does higher ε (softer transport) improve
> downstream metrics? A controlled ablation on ε would distinguish
> "peaky is harmful" from "peaky is a symptom."
> — reports/D04_DIAGNOSIS_AGGREGATE.md §2

## Results

| Run ID | Seed | eps_train | MoF | F1 | mIoU |
|---|---:|---:|---:|---:|---:|
| baseline_s0 (ref) | 0 | 0.07 | 0.4510 | 0.4832 | 0.2553 |
| E00_baseline_FSeval_s1 (ref) | 1 | 0.07 | 0.4945 | 0.5165 | 0.2841 |
| E07a_eps0p15_FSeval_s0 | 0 | 0.15 | 0.3064 | 0.2359 | 0.0338 |
| E07a_eps0p15_FSeval_s1 | 1 | 0.15 | 0.3410 | 0.2578 | 0.0321 |
| E07a_eps0p30_FSeval_s0 | 0 | 0.30 | 0.2092 | 0.1194 | 0.0629 |
| E07a_eps0p30_FSeval_s1 | 1 | 0.30 | 0.2188 | 0.1300 | 0.0588 |

Variant summary (both seeds, final):

| Variant | Mean MoF | Δ MoF vs baseline | Mean mIoU | Δ mIoU vs baseline |
|---|---:|---:|---:|---:|
| Baseline (eps=0.07) | 0.4727 | — | 0.2697 | — |
| eps=0.15 (2× baseline) | 0.3237 | **−14.90 pp** | 0.0330 | **−23.67 pp** |
| eps=0.30 (~4× baseline) | 0.2140 | **−25.87 pp** | 0.0608 | **−20.89 pp** |

Monotonic in ε on both MoF and F1. mIoU collapses to ~0.03–0.06 across both
settings — the model loses the clustering structure almost completely.

## Interpretation

Softening T* at the solver level moves the task metrics in the **same direction**
as softening it at the representation level (E01 B-spline, E03 Gaussian,
E04 Bézier) — i.e. uniformly worse — and does so **monotonically** in ε.
mIoU collapses almost completely (from 0.27 to 0.03), meaning the model loses
its clustering structure before it loses frame accuracy.

Two independent levers (OT solver entropy, representation smoothing) therefore
converge on the same conclusion: making T\* less peaky harms VASOT, not helps it.

**Verdict on Hypothesis A as actionable root cause: REFUTED.**
T\*'s peakiness is correctly diagnosed as a property of the trained model
(D01, D04), but it is functional — the model needs the near-top-1 assignment
to produce coherent segments. Reducing peakiness, by any lever we have tried,
destroys mIoU before it buys anything.

This re-weights the diagnosis:

- **A (peaky T*)**: observed, but *not* a lever. Marks the model's working mode,
  not its failure mode.
- **B (tautological CE + boundary blindness)**: partially supported; needs a
  boundary-aware loss as its own direct intervention, not a smoothing surrogate.
- **C (frame-index prior)**: now the most plausible remaining root cause.
  No lever in this session has yet directly ablated `R` (temporal prior) or
  `C^x` (structural step-function prior); that is the next direct intervention
  worth running.

## Provenance

- Launch script: `reproductions/tssl/scripts/run_e07a_eps_ablation.sh`
- Baseline config lock: `reports/E00_contract.lock` (sha256:
  `63148876ba0ddfe58332b066f1211349f04b16b39f9f33b6b57feb7446ea9f99`)
- Manifests: `reproductions/tssl/artifacts/runs/E07a_eps*_FSeval_s{0,1}/manifest.json`
- Wandb project: `vasot-repro`, tags `E07a`, `eps_ablation`, `hypothesis_A`.

## Follow-ups (not launched this session)

- **E07b — C-prior ablation**: train with `rho = 0` (or very large radius `r`)
  on FSeval, 2 seeds. Directly tests whether the diagonal structural prior is
  the dominant harmful constraint.
- **E07c — R-prior ablation**: zero out the temporal prior term in the loss
  weighting; same 2 seeds. Separates "index-adjacency bias" from "diagonal
  bias" within Hypothesis C.
- **E07d — boundary-aware loss (Hypothesis B)**: add a small boundary-prediction
  head (change-point regression) and train jointly; measure boundary P/R/F1
  explicitly rather than the transport-plan proxy.
