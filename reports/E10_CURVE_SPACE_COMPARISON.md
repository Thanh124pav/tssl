# E10 — B-spline smoothing space: logits vs dedicated latent basis

Date: 2026-10-29. FSeval, 30 epochs, paired seeds 0/1, same frozen baseline
configuration as E01/E07. E10a fits a cubic B-spline to action logits;
E10b fits it to coordinates in a separate learned latent basis. All 12 runs
completed and all 12 `last.ckpt` state dictionaries are finite. Blending
starts after epoch 5 and ramps for 5 epochs. Mean MoF baseline is 0.4727.

| Variant | MoF s0 | MoF s1 | Mean MoF | Δ vs baseline (pp) | Scale gate |
|---|---:|---:|---:|---:|---|
| Baseline | 0.4510 | 0.4945 | 0.4727 | — | — |
| E10a logits, blend 0.3 | 0.4459 | 0.4773 | 0.4616 | −1.11 | FAIL |
| E10a logits, blend 1.0 | 0.4468 | 0.3744 | 0.4106 | −6.21 | FAIL |
| E10b latent 12-D, blend 0.3 | 0.4998 | 0.4928 | 0.4963 | +2.36 | PASS screening |
| E10b latent 12-D, blend 1.0 | 0.4307 | 0.4407 | 0.4357 | −3.70 | FAIL |
| E10b latent 24-D, blend 0.3 | 0.4239 | 0.4812 | 0.4525 | −2.02 | FAIL |
| E10b latent 24-D, blend 1.0 | 0.3709 | 0.4639 | 0.4174 | −5.54 | FAIL |

Only the 12-D latent, 0.3-blend variant passes the frozen E00 scale gate:
mean MoF improves by at least 1 pp and neither seed falls more than 2 pp
below its paired baseline. Its per-seed effects are **+4.88 pp** (s0) and
**−0.17 pp** (s1), so the positive mean is driven by seed 0. Mean F1
improves from approximately 0.4999 to 0.5253 (+2.55 pp); mean mIoU from
0.2697 to 0.3053 (+3.57 pp). This was a promising screening result, not a
replicated final claim; the required seed-2 check below reverses it.

## E10c: pre-specified seed-2 confirmation

The same frozen config was run for a paired baseline and the sole screening
winner, with no retuning after seeing seeds 0/1. Both runs completed, and both
checkpoints are finite.

| Seed | Baseline MoF | Latent 12-D, blend 0.3 MoF | Paired Δ (pp) |
|---:|---:|---:|---:|
| 0 | 0.4510 | 0.4998 | +4.88 |
| 1 | 0.4945 | 0.4928 | −0.17 |
| 2 | 0.5266 | 0.4743 | **−5.23** |
| Mean ± sample SD, N=3 | 0.4907 ± 0.0380 | 0.4889 ± 0.0132 | **−0.17** |

Seed-2 F1 also falls from 0.5420 to 0.4940 and mIoU from 0.3176 to 0.3018.
The apparent two-seed benefit does **not** replicate. In particular, the
third seed violates the E00 no-collapse threshold (−5.23 pp < −2 pp), and
the three-seed mean is slightly below baseline. **Scale: SKIP** FS Mid,
DA, and YTI for this variant. This is an empirical failure of this exact
latent-basis/B-spline setting, not a general impossibility result for smooth
representations.

E10c raw runs: `reproductions/tssl/artifacts/runs/E10c_{baseline,bspl_latent_dz12_b0p3}_FSeval_s2/`.
Script: `reproductions/tssl/scripts/run_e10c_seed2_validation.sh`.

The contrast with E01's X-space B-spline (−11.95 pp MoF) is substantial:
smoothing directly in encoder space fails, while weak smoothing through a
small learned basis can help. Full replacement (blend 1.0) fails in both
logits and latent space. The experiment changes the representation space and
introduces a learned basis; it does not prove that the spline operator alone
causes the gain.

## Mechanism diagnostics for the screening winner

On 50 deterministic adjacent FSeval pairs of 256 frames per video, the
seed-1 E10b winner has `T_xa` top-1 mass 0.889 and effective support 1.61
actions; the seed-1 baseline has 0.914 and 1.50. `T_xy` has top-1 0.135 and
effective support 35.85 frames, versus baseline 0.126 and 30.87. Thus the
segmentation plan softens modestly; alignment entropy/support also increase
but its top-1 mass does not fall. The seed-0 winner has `T_xa` top-1 0.882,
support 1.68; `T_xy` top-1 0.133, support 34.80.

On the direct boundary diagnostic (not paper segment F1), seed-1 F1@3
falls from 0.381 for baseline to 0.366 for the E10b winner, while the
predicted/GT boundary count rises from 1.289 to 1.576. The winner's seed-0
boundary F1@3 is 0.372 with a count ratio of 1.668; no comparable baseline
seed-0 boundary probe is available. The task-metric gain therefore does not
demonstrate a boundary-quality gain, and limitation B remains open.

Raw runs: `reproductions/tssl/artifacts/runs/E10{a,b}_*/`.
Raw diagnostics: `reproductions/tssl/artifacts/diagnostics/{transport,boundary}_E10b_latent_dz12_b0p3_FSeval_s{0,1}.json`.
Launch scripts: `reproductions/tssl/scripts/run_e10{a,b}_bspline_*.sh`.
