# E13 — Pre-registered action-bridge protocol

Date: 2026-10-29. The VASOT joint objective shares an encoder, but the
frame-to-frame alignment cost does not directly use the frame-to-action
posterior. E13 tests an explicit bridge before introducing any spline.

## Mechanism

For action posteriors `q_X`, `q_Y`, define cosine similarity in action space
and add its distance to the detached alignment-OT ground cost:

`C_xy = C_visual + rho_align R + gamma (1 - cos(q_X, q_Y))`.

This is segmentation -> alignment. The bidirectional variant also predicts a
row-normalized alignment from action similarity and applies CE against the
detached frame transport `T_xy`; this is alignment -> action space. Spline,
boundary gating and temporal warping are disabled in E13.

## Frozen screen

Fixed-checkpoint calibration on 20 deterministic FSeval pairs gives visual
cost median 0.380 (mean 0.399) and action cost median 0.904 (mean 0.715).
Therefore `gamma=0.1` is fixed before training; its median contribution is
about 0.09 and should complement rather than replace the visual cue.

| Variant | action-cost weight | auxiliary-loss weight | temperature | warmup |
|---|---:|---:|---:|---:|
| cost-only | 0.1 | 0 | 0.1 | 5 epochs |
| bidirectional | 0.1 | 0.05 | 0.1 | 5 epochs |

Screening uses paired seeds 0/1 on the frozen FSeval config. Existing paired
baselines are reused. Report MoF/F1/mIoU, direct boundary metrics, transport
support and an explicitly labelled action-consistency alignment proxy. The
proxy is not a paper VAOT metric.

Scale calibration artifact:
`reproductions/tssl/artifacts/diagnostics/action_bridge_scale_FSeval_baseline_s1.json`.
