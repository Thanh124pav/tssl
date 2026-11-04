# A08 — Audit of the fine-grained control-clustering proposal

Date: 2026-11-04. Frozen checkpoint `E00_baseline_FSeval_s1`, 8 FSeval videos.
Script `reproductions/tssl/scripts/eval_a08_proposal_audit.py`,
artifact `artifacts/diagnostics/a08/FSeval_baseline_s1.json`.

The proposal is a chain of four links. Each was tested as its own precondition
so that a failure localises. **All four fail at their precondition on this
checkpoint.** Section 5 says which part of the intuition survives and what the
measured version of it would be.

## Link 1 — "corresponding frames lie on the line through the action centre"

Rule as stated: for frame `f` of video X with action centre `a_k`, draw
`L = a_k + t(z_f − a_k)`; the corresponding frame in another video is the
candidate that is both similar (cosine or `T_xy`) and nearest to `L`.

Tested as a retrieval rule on monotone time warps, where the true
correspondence is known exactly. Recall@1 counts a hit within 2 frames.

| rule | R@1 | R@5 | alignment error (frames) |
|---|---:|---:|---:|
| `cosine` (baseline score) | 0.9248 | 0.9741 | 3.63 |
| `line` only | 0.9204 | 0.9717 | 2.97 |
| `ray` only (t > 0) | 0.9204 | 0.9717 | 2.97 |
| `cosine + line` (w = 1) | 0.9263 | 0.9775 | 2.88 |
| **`T_xy` only (the existing solver)** | **0.9800** | **0.9980** | **0.35** |
| `T_xy + line` (w = 2) | 0.9814 | 0.9971 | 0.34 |
| `cosine`, top-5, then nearest to line | 0.9238 | 0.9741 | 2.97 |
| **`T_xy`, top-5, then nearest to line** | **0.9673** | 0.9980 | 0.41 |
| `T_xy`, top-25, then nearest to line | 0.9399 | 0.9858 | 1.19 |

Three readings, in order of importance:

1. **The two-stage form of the rule — the one the proposal actually describes —
   makes things worse.** Restricting to the top candidates by `T_xy` and then
   taking the nearest to the line drops R@1 from 0.980 to 0.967, and to 0.940
   if the candidate pool is widened. The line is pulling the choice away from
   the right frame.
2. The additive form gains +0.14 pp R@1 over `T_xy` alone, which is noise at
   this sample size, and **nothing in it comes from the action prototype**:
   with random centres the same rule scores 0.9834 and with deliberately
   shuffled centres 0.9829, both *above* the prototype's 0.9814. This is the
   same result G01 found for the angular score, now for the retrieval rule.
3. The existing solver already achieves 0.35 frames of error on this task. The
   headroom the proposal targets is about a third of a frame wide.

## Link 2 — "group lines converge at a new action centre"

Correspondence groups were built from the solver's own plans (reference video,
every segment sampled at 6 positions, matched into the other 7 videos), a line
fitted through each group, and the lines of one segment tested for concurrency.

| quantity | value |
|---|---:|
| groups fitted | 73 |
| variance of a group explained by its best-fit line | **0.363** |
| normalised line residual | 0.798 |
| common-point residual (normalised) | 0.789 |
| same, direction-randomised null | — |
| **ratio real / null** | **0.893** |
| distance from the fitted common point to the action prototype | **1.007** |

A line explains 36% of a correspondence group's spread, so the groups are not
lines to begin with — consistent with A02's independent measurement of 0.379.
The lines then concur only 11% better than lines with *random directions*
through the same points, which is not concurrency. And the least-squares
"new centre" sits 1.007 away from the action prototype, while frames themselves
sit at radius ≈ 1.12 from it (A06): the recovered point is roughly as far from
the action centre as an ordinary frame is, so it is not a refinement of that
centre.

## Link 3 — "project to a space where a video is a line, keep positions by cross-ratio"

| quantity | value |
|---|---:|
| variance of a whole video explained by its best-fit line | **0.333** |
| cross-ratio, mean over videos | 4.86 |
| cross-ratio, std over videos | 5.69 |
| **coefficient of variation across videos** | **1.17** |
| cross-ratio of the frame indices themselves | 1.56 |
| mean absolute change under a monotone time warp | 9.37 |
| **mean relative change under a monotone time warp** | **0.789** |

The premise fails first: a line captures a third of a video's variance (A06
independently put the whole-video line fit at ratio 0.953 against its null,
i.e. no better than matched noise). On top of that the cross-ratio varies
across videos by more than its own magnitude (CV 1.17), so it is not a shared
invariant, and it changes by 79% under a monotone time warp.

That last number is not a surprise and is worth stating plainly: the
cross-ratio is invariant under *projective* maps, and a nonlinear time warp is
not projective. `IMPLEMENTATION.md` already warns that "cross-ratio is not
automatically invariant to arbitrary temporal reparameterization". The
measurement confirms it at 79%.

## Link 4 — "contrastive loss to prevent geometry collapse"

No new run needed: the collapse this is designed to prevent does not occur.

| quantity | measured | degenerate value |
|---|---:|---|
| nearest-prototype margin (G00) | 0.718 | 0 |
| largest pairwise prototype cosine (G00) | 0.536 | 1 |
| clusters used, K = 12 (A04) | 12 / 12 | ≪ K |
| clusters used, K = 256 (A05) | 255 / 256 | ≪ K |
| hard-assignment purity (A04) | 0.618 | ≈ 1/K |
| embedding participation ratio (A03) | 5.81 of 40 | 1 |

Prototypes do not merge, every cluster is used even at K = 256, and the frame
representation keeps ~5.8 effective dimensions. A contrastive term may help for
other reasons, but "avoiding geometry collapse" is not a problem this system
currently has.

## 5. What survives, and the measured version of it

The multi-video intuition is right about where to look; the geometric mechanism
is the part that fails. Two directions from the same intuition that rest on
measured defects rather than assumed geometry:

1. **Enforce cycle consistency, not collinearity.** A02 measured that the
   transport is *not* cycle-consistent: composing X→Y→Z disagrees with X→Z by
   18.5 frames on average, and only 39% of frames agree within 4 frames; a
   plain round trip X→Y→X returns within 4 frames for 46%. That is a real,
   large, multi-video defect, it needs ≥ 3 videos to even define, and fixing it
   requires no assumption about lines, centres or projective structure. This is
   the version of "use multiple videos to refine alignment" that the data
   supports.
2. **Fine granularity itself has support; the geometric control does not.**
   A05 found K = 48 gives better purity (0.674 vs 0.586) and a 3.7× better ECE
   than the K = 12 baseline. So "fine-grained clustering" is worth pursuing —
   through K and through calibration of `T_xa`, which A04 showed is badly
   miscalibrated (ECE 0.459, 79% of frames claiming 0.99 and right half the
   time).

What this audit does **not** rule out: the proposal asks for a *learned*
low-dimensional projection, and every measurement here is in the embedding and
its PCA coordinates. A trained head could be optimised to make frames collinear
through a centre. The difficulty is that the proposal uses that geometry to
*generate* the correspondence signal, so the structure has to exist well enough
to bootstrap; here it does not, and the rule built on it is worse than the plan
the solver already produces. A jointly learned version would be a different and
much weaker claim — any head can be trained to produce a geometry, and the open
question would then be whether it helps the task, which A07 shows must be
judged on a representation-side metric rather than MoF.

Limitations: one checkpoint, one dataset, 8 videos, warp severity 0.5, and the
correspondence groups in Link 2 come from the solver's own argmax, so they
inherit its errors. The cross-ratio test uses the principal axis of each video,
which is the most favourable 1-dimensional choice available without training.
