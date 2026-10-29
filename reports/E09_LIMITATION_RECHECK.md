# E09 — Recheck A/B with the actual alignment plan and predicted boundaries

Date: 2026-10-29. The earlier D04 report measured the segmentation plan
`T_xa` but used that result to discuss alignment `T_xy`. This packet measures
both plans separately at the training sample size and evaluates predicted
boundaries directly.

## A: transport support

FSeval, seed 1, 50 fixed adjacent video pairs, 256 uniformly sampled frames
per video, trained checkpoints. `T_xy` is conditioned on its real target
columns; its mass on those columns is also reported to expose virtual-frame
effects. The public solver's training parameters are used.

| Checkpoint | Plan | Normalized row entropy | Effective support | Mean top-1 mass | Real-target row mass |
|---|---|---:|---:|---:|---:|
| baseline | `T_xa` | 0.109 | 1.50 actions | 0.914 | 1.000 |
| baseline | `T_xy` | 0.584 | 30.87 frames | 0.126 | 0.999 |
| rho=0 | `T_xa` | 0.118 | 1.56 actions | 0.908 | 1.000 |
| rho=0 | `T_xy` | 0.624 | 41.45 frames | 0.114 | 0.999 |

The same 50 pairs evaluated on a newly initialized seed-1 model (recreated
from the checkpoint hyperparameters, without loading its weights) give
`T_xa` entropy 0.404, support 3.47 actions, top-1 0.685. Training sharpens
the segmentation plan markedly to entropy 0.109, support 1.50, top-1 0.914.
For `T_xy`, initialization gives entropy 0.611, support 41.51 frames,
top-1 0.199; the trained values are 0.584, 30.87, and 0.126 respectively.
The entropy/support and top-1 indicators move in different directions for
alignment, so training does not produce a simple top-1 collapse. This is a
single initialization and fixed-pair diagnostic, not a training trajectory.

The contrast is consistent across the 50 baseline pairs: pair-level mean
`T_xy` top-1 mass ranges 0.098–0.157, while `T_xa` ranges 0.879–0.936.
The original A claim that **alignment `T_xy` is nearly one-hot is refuted on
this FSeval protocol**. Its segmentation counterpart is supported. A weaker
version of A remains plausible: 30.87 effective alignment targets out of
256 frames on FSeval is banded support, not a uniform full-matrix assignment.

Two additional baseline checks, 20 fixed adjacent pairs per dataset at 256
sampled frames, show the same qualitative distinction:

| Dataset / state | `T_xa` top-1 | `T_xa` support | `T_xy` top-1 | `T_xy` support |
|---|---:|---:|---:|---:|
| Desktop Assembly / init s0 | 0.568 | 7.69 actions | 0.621 | 7.02 frames |
| Desktop Assembly / trained s0 | 0.969 | 1.11 actions | 0.285 | 7.78 frames |
| YTI coffee / init s0 | 0.892 | 1.72 actions | 0.093 | 24.53 frames |
| YTI coffee / trained s0 | 0.954 | 1.23 actions | 0.112 | 21.45 frames |
| YTI coffee / rho=0 s0 | 0.934 | 1.39 actions | 0.094 | 27.63 frames |

The YTI probe contains padded rows; top-1 and support exclude zero-mass
rows. DA alignment is more concentrated than FSeval/YTI but still not a
top-1 target on average. Segmentation sharpens after training on both DA
and YTI. Alignment does **not** have a uniform training response: DA top-1
falls from 0.621 to 0.285, whereas YTI coffee rises slightly from 0.093 to
0.112. The trained effective support is only 7.78/256 on DA and 21.45/256
on YTI coffee, again indicating a banded rather than top-1 or fully diffuse
plan. Raw results are the corresponding
`transport_{DA,YTI_coffee}_*_256_20.json` diagnostics.

For the seed-1 curve checkpoints on the same 50 pairs, X-space B-spline and
Gaussian make `T_xa` slightly less concentrated but make `T_xy` **more**
concentrated:

| Checkpoint | `T_xa` top-1 | `T_xy` top-1 | `T_xy` effective support |
|---|---:|---:|---:|
| baseline | 0.914 | 0.126 | 30.87 |
| B-spline X | 0.881 | 0.277 | 18.83 |
| Gaussian σ=2 | 0.891 | 0.271 | 18.80 |

The Bézier seed-0 and seed-1 checkpoints have 13,960/13,960 non-finite model
parameters each; their transport plans are all zero in this probe. They are
numerical failures, regardless of the `completed` process status. See the
corrected E05 matrix.

The second A claim, that a peaky target gives no negative signal, does not
follow from the implemented cross-entropy. For a target row `T`, let
`m = Σ_j T_j` over real targets and let `P` be the row softmax. Then
`∂CE/∂logit_j = mP_j − T_j`; an unmatched column with `T_j = 0` receives a
positive (repulsive) gradient whenever `m > 0` and `P_j > 0`. The strength
of that signal remains an empirical question, but it is not absent by
construction. Rows wholly routed to the virtual target have `m = 0` and no
alignment CE signal in this implementation.

These measurements use deterministic adjacent pairs. The FSeval
initialization probe is one seed, not a training trajectory; random training
pairs were not measured.
The D04 wording should be revised accordingly.

### Epsilon control and the E07a interpretation

Holding the baseline seed-1 checkpoint **fixed** and changing only the OT
solver epsilon on the same 50 pairs gives:

| Solver ε | `T_xa` top-1 | `T_xa` effective support | `T_xy` top-1 | `T_xy` effective support |
|---:|---:|---:|---:|---:|
| 0.07 | 0.914 | 1.50 | 0.126 | 30.87 |
| 0.10 | 0.860 | 1.99 | 0.473 | 5.25 |
| 0.15 | 0.696 | 3.62 | 1.000 | 1.00 |
| 0.30 | 1.000 | 1.00 | 1.000 | 1.00 |

Thus increasing ε does **not** soften the alignment plan in this
implementation. Even at ε=0.10, `T_xy` becomes substantially sharper. At
ε=0.15, `T_xa` softens but `T_xy` collapses to one-hot;
at ε=0.30 both collapse. On the retrained ε=0.15 seed-1 checkpoint, only
37.2% of source rows retain mass on real alignment targets; the rest are
routed to the virtual target. This is a solver/model degeneration, so the
E07a performance drop cannot establish that peaky assignments are beneficial.
It establishes only that these ε settings fail under the current solver.

Solver-only raw results: `reproductions/tssl/artifacts/diagnostics/transport_FSeval_fixed_baseline_eps*_s1.json`.
Retrained ε result: `reproductions/tssl/artifacts/diagnostics/transport_FSeval_eps0p15_s1_256_all.json`.

Raw results: `reproductions/tssl/artifacts/diagnostics/transport_FSeval_{baseline,rho0}_s1_256_all.json`.
Initialization result: `reproductions/tssl/artifacts/diagnostics/transport_FSeval_init_s1_256_50.json`.
Code: `reproductions/tssl/scripts/eval_transport_support.py`.

## B: direct boundary quality

Predicted and ground-truth change points are matched one-to-one within a
tolerance on the same deterministic 256-frame sampling grid. These scores
are **boundary F1**, distinct from VASOT's reported segment F1.

| Dataset / seed | Checkpoint | Boundary F1@1 | F1@3 | F1@5 | Predicted/GT boundary count |
|---|---|---:|---:|---:|---:|
| FSeval / 1 | baseline | 0.278 | 0.381 | 0.443 | 1.289 |
| FSeval / 1 | rho=0 | 0.314 | 0.375 | 0.410 | 1.511 |
| FSeval / 1 | B-spline X | 0.239 | 0.300 | 0.344 | 3.589 |
| FSeval / 1 | Gaussian σ=2 | 0.250 | 0.320 | 0.366 | 3.051 |
| DA / 0 | baseline | 0.439 | 0.650 | 0.743 | 0.984 |
| DA / 0 | rho=0 | 0.390 | 0.607 | 0.715 | 1.071 |
| DA / 1 | baseline | 0.447 | 0.641 | 0.743 | 0.974 |
| DA / 1 | rho=0 | 0.428 | 0.649 | 0.739 | 1.010 |

`rho=0` improves FSeval frame MoF (0.494→0.599 on seed 1), but boundary
F1@3 is unchanged and F1@5 decreases. On DA seed 0 both frame MoF and
boundary F1 decrease; on DA seed 1, MoF and F1@3 improve slightly but
F1@1 and F1@5 decline. B-spline and Gaussian produce 3.59× and 3.05× as many
predicted boundaries as GT on FSeval, versus 1.29× for baseline; their
boundary precision and F1 fall despite higher boundary recall. This supports
a narrower B statement: frame-cluster
quality and transition quality can diverge. It does not prove that the
segmentation CE is tautological or establish that a boundary-aware loss fixes
the gap. No ground-truth boundary was used during training.

Code inspection confirms there is no explicit change-point or boundary loss:
training sums CE against stop-gradient `T_xa` and `T_xy` pseudo-labels. Yet
calling the intra-video CE a mathematical tautology is too strong: the OT
target also reflects marginal and temporal/structural constraints, and its
gradient is generally nonzero. The causal claim that this loss *causes*
boundary errors remains untested by a boundary-loss intervention.

Raw results: `reproductions/tssl/artifacts/diagnostics/boundary_{FSeval,DA}_*.json`.
Code: `reproductions/tssl/scripts/eval_boundary.py`.

YTI is omitted from the boundary table. Its GT uses `-1` as an excluded
background/unlabelled class. Counting transitions touching `-1` yielded
459 apparent coffee boundaries in 30 videos, but only 35 remain when both
sides must be labelled. A boundary F1 on those 35 edges would omit most
action onsets/offsets and is not comparable to the FSeval/DA diagnostic.
The initial unfiltered YTI JSONs are retained for audit but must not be used
as evidence of over- or under-segmentation.
