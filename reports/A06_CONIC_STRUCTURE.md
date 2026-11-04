# A06 — Is there conic (not merely linear) structure in the action space?

Date: 2026-11-04. Frozen checkpoint `E00_baseline_FSeval_s1`, 12 FSeval videos.
Script `reproductions/tssl/scripts/eval_a06_conic_structure.py`,
artifact `artifacts/diagnostics/a06/FSeval_baseline_s1.json`,
W&B run `A06_FSeval_baseline_s1`.

**Why this packet exists.** A02 reported that matched frame sets are *less
linear* than random and concluded against a ray. That conclusion was drawn too
widely: points on a circle also have low linearity, so a low top-1 PCA variance
fraction is evidence against a *line* and is perfectly compatible with a conic.
`IMPLEMENTATION.md` already states that a conic is a **new hypothesis, not an
automatic fallback**, so it needs its own test. This is that test.

## 1. Method, and why parameter count cannot fake a result

A conic in 2D has 5 free parameters and therefore fits any 5 points exactly, so
a plain in-sample fit is worthless. Three safeguards:

1. **Held-out evaluation.** Each group is split in half; the PCA plane, the
   normalisation and the coefficients are all estimated on the training half,
   and the residual is measured on the held-out half.
2. **Nested models in one algebraic form.** Line (3 dof), circle (4 dof) and
   conic (6 dof) are all fitted as constraints on the coefficient vector of
   `[x², xy, y², x, y, 1]` and scored with the same residual, so they are
   directly comparable.
3. **Matched-covariance null.** Every fit is repeated on Gaussian samples with
   the group's own in-plane covariance and the same point count. The statistic
   is the ratio `real / null`.

A conic is a *planar* object, so a non-planar group cannot be described by one
however good the fit looks. The packet therefore also fits a **quadric
hypersurface** in the top-3 and top-5 PCA dimensions (9 and 20 parameters),
with the same held-out and null discipline — "a conic, but not confined to a
plane".

**The detector is validated against known shapes** before being believed
(`scripts/test_diagnostics.py`, two named tests, 17/17 invariants passing):

| synthetic input | line ratio | circle ratio | conic ratio | quadric3d ratio |
|---|---:|---:|---:|---:|
| circle in 40-d | 1.059 | **0.056** | **0.029** | — |
| ellipse in 40-d | 1.020 | 0.966 | **0.031** | — |
| 3-d sphere shell in 40-d | — | — | — | **0.041** |
| 3-d ellipsoid in 40-d | — | — | — | **0.066** |
| Gaussian noise | 0.498 | 0.990 | 0.599 | 0.601 |

The ellipse row is the decisive one for this packet: an ellipse is invisible to
the circle model (0.966) and caught cleanly by the conic (0.031). The detector
can tell a conic from a circle from noise.

One honest caveat that the validation exposed: for the **line** model the
ratio statistic saturates near 0.5 for both a true line and noise, because a
Gaussian null copying an elongated covariance is itself nearly a line. For the
line model only the absolute residual discriminates. This does not affect the
circle/conic/quadric arms, whose nulls are not degenerate.

## 2. Result, (video, GT action) groups — the object the G02 hypothesis names

300 held-out fits over 60 groups, mean 38 frames per group.

| model | held-out residual | null residual | **ratio** | reference: true shape | reference: noise |
|---|---:|---:|---:|---:|---:|
| line | 0.396 | 0.552 | 0.723 | 0.505 | 0.498 |
| circle | 0.351 | 0.510 | 0.692 | **0.056** | 0.990 |
| conic | 0.254 | 0.388 | **0.672** | **0.031** | 0.599 |
| quadric 3-d | 0.155 | 0.228 | 0.703 | **0.041** | 0.601 |
| quadric 5-d | 0.086 | 0.102 | 0.847 | **0.066** | — |

Every arm lands at the **noise reference**, not the true-shape reference. The
conic ratio of 0.672 sits beside the 0.599 that matched-covariance Gaussian
noise produces, and two orders of magnitude away from the 0.031 a real ellipse
produces. The same holds for whole-video scope (conic 0.716, quadric3d 0.784,
quadric5d 0.866).

Two details reinforce it rather than soften it:

- **The groups are not planar.** The best 2-plane captures only **54%** of a
  group's variance (median 0.542, 10th–90th percentile 0.420–0.654); the
  validated conic shapes were at 0.99+. The top 5 PCA dimensions still capture
  only 70%. A planar conic is the wrong model class for this data regardless of
  residual, which is consistent with A03's two-NN intrinsic dimension of ~10.8.
- **Adding parameters makes the ratio worse**, not better: quadric 3-d 0.703 →
  quadric 5-d 0.847. If there were a genuine quadratic surface, extra
  dimensions would let the fit track it and the ratio would fall. Rising with
  parameter count is the signature of fitting noise.

## 3. The specific circle claim: constant radius around the action prototype

G02 states it precisely — "frames of action k from video v lie near an
action-centered circle with video-specific radius `r_{v,k}`". Both halves fail:

| quantity | value |
|---|---:|
| radius CV around the **prototype** | 0.130 |
| radius CV of the matched-covariance null | 0.113 |
| **ratio real / null** | **1.143** |
| radius CV around the group centroid | 0.231 |
| same, null | 0.260 |
| ratio real / null | 0.894 |

The distance to the action prototype is **less** constant than matched noise
(ratio 1.14). Around the group's own centroid — a strictly easier target that
is not the hypothesis — it is 11% more constant than noise, which is a weak
effect and not what was claimed.

The "video-specific radius" half:

| quantity | value |
|---|---:|
| between-video std of the mean radius, per action | 0.059 |
| within-group std of the radius | 0.144 |

Variation *inside* one group is 2.4× the variation *between* videos, so there
is no video-specific radius to speak of: the videos' radii are closer to each
other than the frames of a single video are to their own mean.

## 4. Verdict and what remains genuinely open

**The conic hypothesis is tested and not supported on this checkpoint**, by a
detector proven able to find conics, quadrics, circles and ellipsoids when they
are present. This now rests on a direct measurement rather than on the
inference from linearity that A02 made, which was the valid objection.

What this does **not** rule out, stated so the result is not over-read:

- **A conic in a learned projection.** G02's hypothesis is about `q = g(z)` for
  a learned head `g`, not about PCA coordinates. PCA is an unsupervised proxy; a
  trained projection could in principle expose structure PCA does not. This was
  already flagged as the open fallback in G01 and remains **NOT RUN**.
- **Conics on cross-video matched sets.** Those sets have 7 points for a 5-dof
  conic, which is underdetermined even before a held-out split, which is why
  within-group sets were used instead. A larger video group would be needed.
- Other checkpoints, datasets, and the raw feature space; only FSeval seed 1 was
  tested.

Limitations: one checkpoint, 12 videos, 60 groups of ~38 frames, GT actions used
only to define groups. Algebraic residuals are not geometric distances; they are
used consistently across models and nulls, and the synthetic validation confirms
the ranking they produce is correct, but absolute values are not metric errors.
