# A03 — Geometry of the frames inside one video

Date: 2026-11-04. Frozen checkpoint `E00_baseline_FSeval_s1`, 12 FSeval videos,
256 sampled frames. Three arms, so "training did this" is separable from "the
features were already like that": the **trained** encoder, an **untrained**
encoder rebuilt from the same hyper-parameters, and the **raw** 64-d features.

Script `reproductions/tssl/scripts/eval_a03_within_video_geometry.py`,
artifact `artifacts/diagnostics/a03/FSeval_baseline_s1.json`,
W&B run `A03_FSeval_baseline_s1`.

## 1. Training compresses global variance but not local dimension

| quantity | trained | untrained | raw features |
|---|---:|---:|---:|
| ambient dimension | 40 | 40 | 64 |
| **two-NN intrinsic dimension** | **10.76** | **10.84** | **10.85** |
| participation ratio | **5.81** | 17.81 | 22.72 |
| dimensions for 90% of variance | 14.4 | 22.4 | 34.7 |
| top-1 variance fraction | 0.337 | 0.121 | 0.125 |

The two numbers disagree on purpose and the disagreement is the finding. The
**global** spread collapses hard — participation ratio 22.7 → 5.8, and a single
direction goes from carrying 12% of the variance to 34% — while the **local**
intrinsic dimension of the point cloud is unchanged to two decimal places
(10.76 vs 10.85). Training rotates and flattens the cloud into a lower-variance
subspace without making the data locally simpler. A method that assumes frames
lie on a low-dimensional curve or manifold has to contend with a local
dimension of ~11, which is why the E14/E15 B-spline fits could not reach
tolerance: the obstruction is local, and global variance compression does not
remove it.

## 2. The trajectory is smooth in correlation, jagged in geometry

| quantity | trained | untrained | raw |
|---|---:|---:|---:|
| mean step ‖z_{i+1} − z_i‖ | 0.545 | 0.940 | 1.116 |
| **median turning angle between consecutive steps** | **116.4°** | 117.2° | 116.3° |
| path length | 138.4 | 239.1 | 283.4 |
| endpoint distance | 0.66 | 0.96 | 1.14 |
| straightness (endpoint / path length) | 0.0047 | 0.0040 | 0.0040 |

Steps shrink by 40% after training, but the **turning angle does not move at
all** — 116° in all three arms, where 90° is what uncorrelated random steps
give and 180° is reversal. The trajectory is a jittery walk that reverses
direction at nearly every frame, before and after training alike. Training
shortens the steps; it does not straighten the path. Path length is 210× the
endpoint distance.

This is the direct geometric counterpart to the curve-fitting failures: there is
no smooth low-order curve to fit because consecutive displacement vectors are
anti-correlated at every scale tested.

## 3. Temporal autocorrelation rises sharply — this is what training does

Mean cosine between frames separated by lag L:

| lag | trained | untrained | raw |
|---|---:|---:|---:|
| 1 | **0.838** | 0.527 | 0.333 |
| 2 | 0.816 | 0.495 | 0.276 |
| 4 | 0.792 | 0.454 | 0.210 |
| 8 | 0.756 | 0.428 | 0.161 |
| 16 | 0.717 | 0.420 | 0.140 |
| 32 | 0.674 | 0.411 | 0.125 |
| 64 | 0.625 | 0.406 | 0.108 |
| 128 | 0.530 | 0.387 | 0.070 |

Training lifts the whole profile and makes it decay slowly and smoothly: frames
128 apart still sit at cosine 0.53, where the raw features are at 0.07. The
decay is close to linear in log-lag with no plateau or shoulder, i.e. **no
characteristic segment length is visible in the geometry** — nothing in the
profile marks where one action ends and the next begins. Combined with §2 (step
and turn unchanged in character), the clearest single description of what
training does here is: *it makes frame embeddings a slowly varying function of
absolute time*, which is exactly what the frame-index temporal prior rewards.
That also explains N-A1, where the frame-index structure beat every
feature-based and dynamics-based alternative.

## 4. Nearest neighbours are often temporally distant

| quantity | trained | untrained | raw |
|---|---:|---:|---:|
| median temporal lag of the nearest neighbour | 12.3 | 25.8 | 15.2 |
| fraction whose NN is more than 20 frames away | **0.417** | 0.539 | 0.456 |
| fraction whose NN is temporally adjacent (≤1) | **0.168** | 0.133 | 0.174 |

Despite cosine 0.838 at lag 1, only 17% of frames have a temporal neighbour as
their nearest neighbour, and 42% have their nearest neighbour more than 20
frames away. High autocorrelation and poor nearest-neighbour locality coexist
because the cloud is locally ~11-dimensional and the per-frame jitter (§2) is
large relative to the drift. This is the same phenomenon N-A1 measured as "only
9.2% of frames have their successor as nearest neighbour".

## 5. GT action structure is present but very weak

| quantity | trained | untrained | raw |
|---|---:|---:|---:|
| within-action scatter | 0.421 | 0.709 | 0.856 |
| between/within separation ratio | **1.461** | 0.647 | 0.715 |
| **silhouette of GT labels (cosine)** | **0.054** | −0.032 | 0.009 |

Training does move the representation in the right direction — the separation
ratio crosses 1 and the silhouette turns positive — but a silhouette of 0.054
is barely distinguishable from no cluster structure at all (0 means points sit
as close to a neighbouring action as to their own). The GT actions are not
geometrically separated clusters in this embedding; they are weakly biased
regions of one continuous, slowly drifting, locally high-dimensional cloud.

## 6. Reading

The within-video geometry is: **a slowly drifting, locally ~11-dimensional,
frame-to-frame jittery cloud with weak action structure and no geometric
signature of segment boundaries.** Three previously separate results fall out of
this single description — the B-spline fit failures (E14/E15: no smooth curve,
local dimension unchanged by training), the dominance of the frame-index
structural cost (N-A1: absolute time is the one thing the embedding encodes
strongly), and the near-chance within-action phase discrimination (G01, N-B1:
no local geometry to tell early from late inside a segment).

Limitations: one checkpoint and one dataset; the two-NN estimator assumes
locally uniform density and is biased on 256 points, so its absolute value
should be read as "about ten", not 10.76; the untrained arm is a single
initialisation seed; lag profiles are averaged over videos of differing true
length after uniform subsampling to 256 frames.
