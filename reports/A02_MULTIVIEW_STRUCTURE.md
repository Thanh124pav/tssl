# A02 — What structure do the frames matched to one frame form across n ≥ 3 videos?

Date: 2026-11-04. Frozen checkpoint `E00_baseline_FSeval_s1`, no training.
Script `reproductions/tssl/scripts/eval_a02_multiview_structure.py`,
artifact `artifacts/diagnostics/a02/FSeval_baseline_s1_g8.json`,
W&B run `A02_FSeval_baseline_s1_g8`.

Group of 8 FSeval videos, so each matched set has **7 points, one per other
video**. 4 videos take the reference role, every 8th frame is a reference frame
(32 per video), all 56 ordered pairwise plans solved with the public solver.

## 1. The matched set is tight — tighter than any baseline

Each arm contributes exactly 7 points, one per other video, so the shape
descriptors are comparable across arms.

| descriptor | **matched** | same_time | same_action | random |
|---|---:|---:|---:|---:|
| mean pairwise cosine | **0.820** | 0.700 | 0.730 | 0.589 |
| min pairwise cosine | **0.686** | 0.450 | 0.492 | 0.245 |
| mean radius from centroid | **0.383** | 0.494 | 0.469 | 0.581 |
| mean cosine to the reference frame | **0.855** | 0.698 | 0.729 | 0.592 |
| mean distance to the reference frame | **0.526** | 0.753 | 0.715 | 0.877 |

The transport plan finds sets that are substantially tighter than taking the
same relative time index in each video, and tighter than taking a random frame
of the *correct GT action* in each video. The `min` row matters as much as the
mean: the worst pair inside a matched set (0.686) is still better than the
*average* pair of either baseline. So the tightness is consistent, not an
average over a mix of good and bad matches. **This is a positive result for
VASOT's alignment: it is doing something neither time nor action identity
explains.**

## 2. The set is a tight blob, not a line or a ray

| descriptor | matched | same_time | same_action | random |
|---|---:|---:|---:|---:|
| top-1 PCA variance fraction (linearity) | **0.379** | 0.428 | 0.432 | 0.453 |
| top-2 PCA variance fraction | 0.615 | 0.668 | 0.661 | 0.697 |
| participation ratio (of 7 points) | **4.08** | 3.63 | 3.62 | 3.37 |

The matched set is the **least** linear and the **most** isotropic of the four
arms: its variance is spread over ~4.1 effective directions out of the 6
available to 7 points, and no single direction carries even 38% of it. It is a
compact, full-dimensional cloud.

This is a direct measurement against the geometric hypothesis Phase G was built
on. A prototype-centered ray or circle predicts that corresponding frames line
up along a shared direction — i.e. high linearity. The matched sets are less
linear than *random* frames. Taken with G01 (centered directions carry no phase;
a random or even wrong centre scores the same), the ray/circle picture is
contradicted by two independent measurements rather than one.

## 3. Temporal spread of the matched indices

| quantity | value |
|---|---:|
| mean matched index (reference frames average index 124) | 124.4 |
| **bias** of the matched index vs the reference index `i` | **+0.39 frames** |
| std of the matched index across the 7 videos | **17.5 frames** |
| range (max − min) of the matched index | 51.3 frames |

There is no systematic temporal bias (+0.39 of 256 frames), so the plan is
temporally unbiased on average — but the per-video scatter is large: the seven
videos place the match over a window of ~51 frames, 20% of the video. Matched
frames agree on the GT action only **51.7%** of the time.

So the picture from §1–§3 is: the matched frames are close *in the embedding*,
but spread out *in time* and only half-agree on the action label. The alignment
is finding a consistent appearance cluster, not a consistent moment in the task.

## 4. The plans are far from mutually consistent

Composing plans should commute if correspondence were a well-defined map.

| test | value |
|---|---:|
| X→Z vs X→Y→Z, mean absolute disagreement | **18.5 frames** |
| same, median | 8.2 frames |
| fraction of frames agreeing within 4 frames | **0.390** |
| round trip X→Y→X, mean error | 9.4 frames |
| round trip, fraction within 4 frames | 0.456 |

Only 39% of frames land within 4 frames of each other when the correspondence
is composed through a third video rather than taken directly, and a plain
round trip X→Y→X returns to within 4 frames for only 46% of frames. Over
24 ordered triples and 28 ordered pairs.

**This is the most actionable finding of the packet.** Any multi-video method
that wants to propagate correspondences through a reference — including the G03
star/reference matching idea, which was gated off by G01 — inherits an ~18 frame
composition error per hop. Cycle consistency is not a property this transport
has, so it would have to be *imposed* (as a loss or a constraint), not assumed.
That is a concrete, testable proposal that the Phase G line did not reach, and
it does not depend on the prototype geometry that G01 falsified.

## 5. Limitations

One checkpoint, one dataset, one group of 8 videos, 4 reference videos, every
8th frame. Shape descriptors on 7 points are noisy by construction, which is why
all four arms share the same point count and the comparison is relative rather
than absolute. `same_action` draws a random frame of the correct action, so it is
a weak baseline for a *specific* moment; `matched` beating it establishes
tightness, not correctness. GT actions are used only to define the
`same_action` arm and the agreement figure. There is no ground-truth frame
correspondence for FSeval, so §1–§2 describe the geometry the model produces, not
its accuracy; the N-C1 warp protocol remains the only place in this project where
correspondence accuracy can be measured directly.
