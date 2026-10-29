# E08 — Does removing the index-based temporal prior transfer?

Date: 2026-10-29. This is a paired, two-seed comparison with the canonical
R00/R01 configs and new R02 seed-1 controls. `rho=0` disables the temporal
prior in **both** segmentation and alignment; it is not an isolated test of
the alignment prior. All 12 E08 runs and 6 R02 controls completed. All 18
`last.ckpt` state dictionaries have finite tensor values.

| Dataset/activity | Baseline MoF mean | rho=0 MoF mean | Δ mean (pp) | Δ s0 / s1 (pp) |
|---|---:|---:|---:|---:|
| Desktop Assembly | 0.6029 | 0.5707 | −3.23 | −8.10 / +1.65 |
| YTI changing_tire | 0.4785 | 0.4214 | −5.70 | −2.41 / −9.00 |
| YTI coffee | 0.5862 | 0.5130 | −7.33 | −18.55 / +3.90 |
| YTI cpr | 0.5079 | 0.4826 | −2.53 | +0.03 / −5.09 |
| YTI jump_car | 0.3285 | 0.2464 | −8.21 | −8.88 / −7.54 |
| YTI repot | 0.3677 | 0.3967 | +2.90 | −1.84 / +7.64 |
| YTI five-activity mean | 0.4538 | 0.4120 | −4.17 | −6.33 / −2.02 |

By contrast, the earlier E07b FSeval two-seed test found +7.95 percentage
points for `rho=0`. The FSeval gain does **not** transfer in mean to DA or
YTI. YTI mean F1 falls 0.3319→0.2776 (−5.43 pp) and mIoU falls
0.2084→0.1666 (−4.18 pp); DA mean F1 falls 0.6441→0.5967 and mIoU
0.4227→0.3794. Four of five YTI activities have negative mean MoF effects;
only repot improves. DA and coffee reverse sign between seeds, so two seeds
are still too few for a precise activity-specific effect estimate.

This weakens the practical claim that merely removing index priors solves
limitation C; it does not refute that index priors can be wrong for some
video pairs. E11 separates segmentation and alignment uses of `rho` on
FSeval. A duration-conditioned test or a learned temporal coordinate would
be needed to test the full causal claim about nonuniform action durations.

Raw metrics: `reproductions/tssl/artifacts/runs/{R00_DA_baseline_s0,R01_YTI_*_s0,R02_*_baseline_s1,E08a_*,E08b_*}/metrics.json`.
Run script: `reproductions/tssl/scripts/run_e08_rho0_scale.sh`.
