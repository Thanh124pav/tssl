# E05 — Continuous Curve Comparison Matrix on FSeval

**Date:** 2026-10-29
**Dataset:** 50 Salads Eval (FSeval), 30 epochs, paired seeds 0/1
**Scale gate:** ≥ 1 pp MoF improvement over baseline mean across 2 seeds

Numbers below come from `reproductions/tssl/scripts/collect_e_results.py` reading
each run's `manifest.json`. The two historical results marked *(known)* were
imported from pre-S00 session logs; the other eight rows come from fully-isolated
S02 runs with manifests on disk.

## Per-run table

| Run ID | Seed | MoF | F1 | mIoU | T_entropy (norm) | Boundary z | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_s0 | 0 | 0.4510 | 0.4832 | 0.2553 | — | — | known (session log) |
| E00_baseline_FSeval_s1 | 1 | 0.4945 | 0.5165 | 0.2841 | — | — | manifest |
| arc_length_s0 | 0 | 0.4850 | — | — | — | — | known (session log) |
| arc_length_s1 | 1 | 0.4448 | — | — | — | — | known (session log) |
| E01_bspline_FSeval_s0 | 0 | 0.3383 | 0.3934 | 0.2058 | 0.1194 | -0.0237 | manifest |
| E01_bspline_FSeval_s1 | 1 | 0.3681 | 0.4043 | 0.2141 | 0.1260 | -0.0427 | manifest |
| E03_gauss_s2_FSeval_s0 | 0 | 0.3465 | 0.3967 | 0.2092 | — | — | manifest |
| E03_gauss_s2_FSeval_s1 | 1 | 0.3880 | 0.4148 | 0.2368 | — | — | manifest |
| E04_bezier_FSeval_s0 | 0 | 0.3533* | 0.2569* | 0.0294* | — | — | manifest; NaN checkpoint |
| E04_bezier_FSeval_s1 | 1 | 0.3543* | 0.2580* | 0.0295* | — | — | manifest; NaN checkpoint |

\* Both Bézier checkpoints have 13,960/13,960 non-finite model parameters.
The reported metrics are produced by an invalid model and must not be used as
valid task scores. The run status `completed` indicates process exit only.

## Variant summary (mean over 2 seeds)

| Variant | N seeds | Mean MoF | Mean F1 | Mean mIoU | Δ MoF (pp) | Scale gate |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 2 | 0.4727 | 0.4999 | 0.2697 | — | — |
| Arc-length | 2 | 0.4649 | — | — | **-0.78** | FAIL |
| B-spline | 2 | 0.3532 | 0.3988 | 0.2100 | **-11.95** | FAIL |
| Gaussian (σ=2) | 2 | 0.3673 | 0.4057 | 0.2230 | **-10.55** | FAIL |
| Bézier (4 seg) | 2 | invalid | invalid | invalid | — | NUMERICAL FAILURE |

## Interpretation

No continuous-representation variant passes the ≥ 1 pp scale gate. Valid
B-spline and Gaussian runs reduce mean MoF by 10–12 pp. Arc-length
re-parameterisation is roughly neutral (-0.78 pp) but differs by seed
(+3.4 pp on seed 0, −5.0 pp on seed 1). Bézier is numerically invalid.

Key diagnostic read-outs:

- **B-spline changes the two transport plans differently.** E09's matched
  50-pair/256-frame measurement finds `T_xa` top-1 mass 0.914→0.881, but
  alignment `T_xy` top-1 mass 0.126→0.277. The older entropy figures
  referred only to `T_xa` and used a different evaluation protocol.
- **But boundary signal weakens**: boundary z-score goes from +0.047 to
  negative values (−0.024, −0.043), meaning the softened transport blurs the
  action transitions it was supposed to help diagnose (Hypothesis B).
- **Representation geometry stays well-conditioned** (codebook condition
  3.1–3.4), so the collapse is not a numeric failure — the model is actually
  being driven toward a worse clustering.
- **Bézier (adaptive 4-segment) fails numerically** on both seeds. Every
  checkpoint parameter is NaN; the apparent mIoU ~0.03 and MoF ~0.35 come
  from degenerate argmax evaluation of invalid transport plans. The method
  needs a numerical fix before any task comparison is interpretable.

## Conclusion

The tested X-space B-spline and Gaussian settings reduce FSeval task quality;
arc-length misses the scale gate. Bézier cannot be assessed because training
becomes non-finite. The stronger mechanism claim in the original E05 draft
is withdrawn: E09 shows that B-spline/Gaussian make `T_xa` slightly softer
but `T_xy` sharper on the same 50-pair protocol. The previous claim treated
these different plans as one quantity.

This matches the direction of the D04 aggregate (A: supported, B: partial,
C: supported, with the strongest handle being C — the frame-index prior
itself).

## Decisions

- **E06 (scale winner to FS Mid / DA / YTI): SKIP.** No variant passes gate.
  Scaling any of these to DA/YTI is predictably bad research spend.
- **E07 (mechanism controls): pivot to direct interventions on A/B/C** rather
  than more smoothing families:
  - **A** — entropy-knob ablation (`eps_train`): soften T* without touching
    the representation; see whether the loss-level softening alone reproduces
    the collapse we see with structural smoothing. (E07a)
  - **B** — boundary-aware diagnostic signal: cross-check whether CE on
    `argmax(T_xa)` is actually learning transition sharpness or just
    frame-level clustering (follow-up; data already available from D02).
  - **C** — index-prior ablation on FSeval: train with `rho_gw=∞` (effectively
    disable C^x) and/or no R, to see whether the index prior is the dominant
    constraint that smoothing was trying to work around indirectly.

E07a is launched in this session; the other two are documented and queued.

## Provenance

- Collect script: `reproductions/tssl/scripts/collect_e_results.py`
- Baseline s0 and arc-length numbers imported from the S00 inventory logs
  (`reports/S00_INVENTORY.md`); all other runs have on-disk `manifest.json`.
- Diagnostics for B-spline runs:
  `reproductions/tssl/artifacts/runs/E01_bspline_FSeval_s*/diagnostics/diagnostics.json`.
