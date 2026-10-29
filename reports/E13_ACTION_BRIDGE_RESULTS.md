# E13 — Action bridge results

Date: 2026-10-29. FSeval, paired seeds 0/1, frozen baseline. No spline or
temporal warp is active. All four checkpoints are finite.

| Variant | MoF s0 | MoF s1 | Mean MoF | Δ MoF | Δ F1 | Δ mIoU |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.4510 | 0.4945 | 0.4727 | — | — | — |
| action-cost only | 0.4502 | 0.4698 | 0.4600 | −1.27 pp | −1.06 pp | −0.71 pp |
| bidirectional bridge | 0.4534 | 0.4942 | 0.4738 | +0.11 pp | +0.53 pp | +0.27 pp |

Cost-only fails the task gate and seed 1 drops 2.47 pp. Adding the
alignment-to-action auxiliary CE removes that collapse, but the resulting
+0.11 pp mean MoF remains far below the E00 +1 pp gate. **Task scale: SKIP.**

On 20 deterministic seed-1 pairs, GT same-action transport mass (an
evaluation-only proxy, not a paper VAOT metric) is 0.5670 baseline, 0.5718
cost-only and 0.5758 bidirectional. Posterior-similarity mass is
0.7068/0.7641/0.7700. Thus the bridge moves alignment in its intended action
direction, but only by +0.88 pp labelled agreement and without a meaningful
segmentation gain.

Direct boundary F1@3 is 0.381 baseline, 0.376 cost-only and 0.380
bidirectional; fragmentation ratios are 1.289/1.289/1.283. The bridge does
not solve boundary quality.

Conclusion: the proposed two-way information path is mechanically active and
stable at the calibrated weight, but action-space agreement alone is not a
sufficient task intervention. E15 may reuse the bidirectional bridge only as
a control; it must not attribute any future curve gain to the bridge without
the no-spline row.

Runs: `reproductions/tssl/artifacts/runs/E13_action_bridge_*`.
Diagnostics: `reproductions/tssl/artifacts/diagnostics/action_bridge_alignment_*`
and `boundary_action_bridge_*`.
