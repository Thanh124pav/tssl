# Current VASOT/VAOT reproduction and mechanism verdicts

Date: 2026-10-29. This supersedes the causal/curve conclusions in the
historical F00 snapshot. `IMPLEMENTATION.md` §7 is the live task ledger.

## Reproduction status

- VASOT Desktop Assembly and all five YTI activities have completed
  paper-protocol baseline runs for seeds 0 and 1 (R00–R02). Seed-0 DA MoF is
  0.5986; seed-0 YTI five-activity mean MoF is 0.4793. R02 supplies the
  seed-1 controls used in E08. See `reports/E08_RHO0_TRANSFER.md` and
  `reproductions/tssl/artifacts/runs/R0*/`.
- VAOT source imports and the PL 2.x port are complete, but a canonical
  paper-comparable baseline remains blocked by data: the available Pouring
  phase labels are all zero, and raw Penn Action/IKEA ASM are absent. The
  pipeline port does not cure missing labels. See `reports/V01_VAOT_DATA_AUDIT.md`
  and `reports/V02_VAOT_PL2_PORT.md`.

## Limitation hypotheses

| Hypothesis | Current verdict | Direct evidence and caveat |
|---|---|---|
| A: inter-video `T_xy` is nearly top-1 and CE gives no negative signal | **Original form refuted** | Trained FSeval `T_xy` top-1 mass 0.126, effective support 30.87/256; DA top-1 0.285, YTI coffee 0.112. `T_xa` segmentation is sharp (FSeval top-1 0.914). For a real-target row, CE logit gradient is `mP_j−T_j`, positive for unmatched columns with nonzero `P_j`; repulsion is not absent. A weaker banded-support claim remains. See E09. |
| B: frame-level pseudo-label CE misses transition quality | **Partially supported, causal mechanism open** | FSeval `rho=0` raises seed-1 frame MoF 0.494→0.599 but boundary F1@3 is 0.381→0.375. X-B-spline and Gaussian over-segment at 3.59× and 3.05× GT boundary counts, with lower boundary F1. There is no explicit boundary loss; this does not prove CE is tautological or that adding such a loss fixes the issue. YTI boundary scores are excluded due to `-1` labels. See E09. |
| C: index-based prior can force bad matches | **Context-dependent support, not a general fix** | Removing both priors (`rho=0`) improves FSeval +7.95 pp over two seeds but hurts DA −3.23 pp and YTI mean −4.17 pp. Isolating segmentation prior improves FSeval +8.69 pp over three paired seeds, but FS Mid only +0.18 pp over two seeds and fails gate. Thus the harm is real in FSeval's evaluation mapping but not robust to FS Mid or other datasets. A duration-conditioned causal test remains open. See E08, E11, E12. |

## Smooth discrete representation proposals

On FSeval, X-space B-spline is −11.95 pp MoF and Gaussian σ=2 is
−10.55 pp over paired seeds; action-logits B-spline at blend 0.3 is −1.11
pp. The 12-D dedicated latent B-spline at blend 0.3 initially passed the
two-seed screening gate (+2.36 pp) but seed 2 is −5.23 pp; its three-seed
mean is **−0.17 pp** versus baseline and it violates the no-collapse gate.
Larger blend and 24-D latent variants fail. Bézier checkpoints have
non-finite parameters and are numerical failures, not valid task scores.
**No tested smoothing proposal currently qualifies for DA/YTI scaling.**
See `reports/E05_COMPARISON_MATRIX.md` and E10.

## What remains legitimately open

1. A direct self-supervised boundary intervention for B, pre-registered
   against the current boundary F1 protocol; diagnosis alone cannot assign
   causality to the CE objective.
2. A duration-conditioned or semantic-time intervention for C; merely
   setting a global temporal-prior coefficient to zero is not a substitute.
3. VAOT canonical evaluation after a properly phase-labelled dataset is
   supplied. Current Pouring labels cannot support its phase metrics.

The E00 scale gate and its stopped variants are not relaxed by these open
questions. New interventions require their own explicit experiment packets.
