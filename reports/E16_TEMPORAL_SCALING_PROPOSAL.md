# E16 — Segment-wise temporal scaling with curve resampling

Date: 2026-10-29. This packet treats execution speed as a nuisance variable:
the same action may occupy different fractions of two videos, while its phase
semantics should remain comparable.

## Core construction

Let self-supervised action posteriors produce soft segments and confidence at
each frame. For confident segment interiors, sample a positive duration scale
per segment (initially log-uniform in `[0.8, 1.25]`) and compose them into a
monotone piecewise time warp `w:[0,1]->[0,1]`. Re-sample the frame/action
trajectory through a fitted curve rather than duplicating or dropping integer
frames.

The first implementation must preserve action order, endpoints and total
video duration. Low-confidence boundary neighborhoods are anchors: the warp
may change the duration of an interior but may not move samples across an
anchor. This differs from global speed augmentation, which cannot model one
short action followed by one long action independently.

## Learning signals

1. **Action equivariance:** after mapping the warped posterior back through
   `w^{-1}`, it should agree with the original posterior.
2. **Representation consistency:** corresponding phase coordinates inside an
   original/warped segment should remain close; unrelated segments provide
   negatives.
3. **Alignment equivariance:** if `W_X`, `W_Y` are interpolation operators,
   the warped transport should agree with `W_X T_xy W_Y^T` up to masking and
   normalization.

B-spline is well motivated here as a differentiable local interpolant. Its
role is resampling under a known monotone warp, not assuming that the original
trajectory must be globally smooth.

## Required controls and risks

- global video speed warp;
- hard segment-wise warp from argmax labels;
- confidence-gated soft segment warp;
- linear interpolation versus B-spline resampling.

The principal risk is error amplification: a false pseudo-boundary causes
frames from different actions to be treated as one scalable segment. E16 is
therefore gated on E13 action-consistency and E15 boundary reliability. GT
boundaries may evaluate warp leakage but may not define training warps.

Primary diagnostics: percentage of warped samples crossing a GT boundary
(evaluation only), posterior consistency after inverse warp, boundary F1,
transport action-consistency, MoF/F1/mIoU and collapse under the strongest
allowed scale.
