# E11 — Which temporal prior drives the FSeval rho=0 gain?

Date: 2026-10-29. VASOT uses `rho` in two different costs: segmentation
frame-to-action and inter-video alignment frame-to-frame. E11 adds independent
`--rho-seg` and `--rho-align` overrides, with their defaults falling back to
the original `--rho`. All settings below share the frozen FSeval config,
30 epochs, and paired seeds 0/1. The baseline and both-zero corners are the
existing reference/E07b runs; E11 trained the two missing corners.

| Segmentation rho | Alignment rho | MoF s0 | MoF s1 | Mean MoF | Δ vs baseline |
|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.15 | 0.4510 | 0.4945 | 0.4727 | — |
| 0.00 | 0.15 | 0.5200 | 0.6152 | 0.5676 | +9.49 pp |
| 0.15 | 0.00 | 0.4858 | 0.4790 | 0.4824 | +0.97 pp |
| 0.00 | 0.00 | 0.5049 | 0.5995 | 0.5522 | +7.95 pp |

The segmentation-only intervention improves both seeds (+6.90 and +12.07
pp), and has the best two-seed mean. The alignment-only intervention improves
seed 0 (+3.49 pp) but lowers seed 1 (−1.55 pp); its mean +0.97 pp is just
below the E00 +1 pp scale gate. Turning off alignment prior *in addition to*
segmentation prior lowers mean MoF by 1.53 pp relative to segmentation-only.
Thus the FSeval benefit of `rho=0` is dominated by the **segmentation**
frame-index prior, not by a demonstrated defect in inter-video alignment
prior `R_xy`. This is a dataset-specific two-seed result, not a general fix:
E08 found `rho=0` worsens DA and YTI means.

Mean F1/mIoU also favor segmentation-only: F1 0.5528 and mIoU 0.3232,
versus baseline 0.4998/0.2697, alignment-only 0.5203/0.2837, and
both-zero 0.5359/0.3116.

## Train-time versus inference-time effect

On seed 1, swapping only the segmentation prior at full-video test time,
without retraining, gives:

| Trained checkpoint | Eval rho_seg | MoF |
|---|---:|---:|
| baseline (trained rho_seg=0.15) | 0.15 | 0.4945 |
| baseline (same weights) | 0.00 | 0.4913 |
| seg0 (trained rho_seg=0.00) | 0.00 | 0.6152 |
| seg0 (same weights) | 0.15 | 0.5943 |

Changing the prior only at evaluation does not rescue the baseline. Restoring
the prior at evaluation lowers the seg0 checkpoint by 2.09 pp, but it remains
9.98 pp above baseline. Most of this seed-1 gain therefore comes from the
representation learned under a different segmentation pseudo-label regime,
not a post-hoc change to the decoder. This counterfactual has been run only
for seed 1, because the historical FSeval seed-0 baseline checkpoint was not
retained in the isolated artifact registry.

Raw metrics: `reproductions/tssl/artifacts/runs/E11_FSeval_*/metrics.json`,
E07b both-zero metrics, E00 seed-1 baseline, and
`reproductions/tssl/artifacts/manifests/sweep_baseline_FSeval_s0.json`.
Inference swaps: `reproductions/tssl/artifacts/diagnostics/rho_swap_FSeval_{baseline,seg0}_s1_*.json`.
Code: `reproductions/tssl/scripts/run_e11_rho_split.sh` and `eval_rho_swap.py`.

## E11c: seed-2 replication before scaling

The segmentation-only winner was rerun on seed 2 with the same frozen
FSeval config and compared with the paired E10c baseline seed 2. The
checkpoint is finite. No parameter was retuned after the two-seed screen.

| Seed | Baseline MoF | `rho_seg=0`, `rho_align=0.15` MoF | Paired Δ (pp) |
|---:|---:|---:|---:|
| 0 | 0.4510 | 0.5200 | +6.90 |
| 1 | 0.4945 | 0.6152 | +12.07 |
| 2 | 0.5266 | 0.5974 | +7.08 |
| Mean ± sample SD, N=3 | 0.4907 ± 0.0380 | 0.5775 ± 0.0506 | **+8.69** |

Seed-2 F1 also rises from 0.5420 to 0.5829 and mIoU from 0.3176 to
0.3231. All three paired MoF effects are positive and clear the E00 gate.
**Scale: FS Mid next**, with an exact paired control. This supports an
FSeval-specific harmful segmentation index prior; it does not yet show
transfer to another dataset. That follow-up is now complete in E12:
FS Mid gains only +0.18 pp mean MoF over two seeds and fails the scale gate.

E11c raw run: `reproductions/tssl/artifacts/runs/E11c_FSeval_seg0_s2/`.
Paired baseline: `reproductions/tssl/artifacts/runs/E10c_baseline_FSeval_s2/`.
Script: `reproductions/tssl/scripts/run_e11_seed2_validation.sh`.
