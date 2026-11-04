# G01 — Within-action and cross-video action-centered geometry probe

Date: 2026-11-04. Frozen checkpoints, no training. **Decision: NO-GO for a
prototype-centered relation H. G02/G03/G05 stay gated and were not run.**

Script: `reproductions/tssl/scripts/eval_g01_angular_probe.py`.
Artifacts: `reproductions/tssl/artifacts/diagnostics/g01/{FSeval_baseline_s1,DA_baseline_s0}.json`.
W&B: `vasot-repro`, runs `G01_FSeval_baseline_s1`, `G01_DA_baseline_s0`.

## 1. What was measured

For frame embedding `z` and a centre `c`: `r = ||z - c||`,
`u = (z - c)/(r + 1e-6)`. Scores for a frame pair:

| score | definition | note |
|---|---|---|
| `cosine` | `z_x · z_y` | the VASOT baseline score |
| `centered_dot` | `(z_x − c_x) · (z_y − c_y)` | unnormalised |
| `direction` | `u_x · u_y` | same-ray model |
| `line` | `(u_x · u_y)²` | undirected-line ablation |
| `radial` | `−|r_x − r_y|` | radius only |
| `blend_w` | `cosine + w · direction`, w ∈ {0.25, 0.5, 1.0} | |

Centre modes: `hard` (argmax of the `T_xa` posterior), `soft`
(posterior-weighted mean of prototypes), and two controls — `shuffled` (a fixed
permutation of the prototypes, so each frame gets a wrong but consistent and
still action-indexed centre) and `random` (a fixed set of random unit centres,
assigned by the same argmax). The two controls are what decide this packet.

> **Correction, 2026-11-04 code audit.** The first version of this packet drew
> the `shuffled` permutation from an unseeded `torch.randperm` inside the
> scoring function, so each call produced a *different* permutation and video X
> and video Y silently received different wrong mappings. That made `shuffled`
> an "inconsistent centre" control rather than the "wrong but consistent centre"
> control it was supposed to be, and it reported a spurious collapse
> (direction AUROC 0.609 instead of 0.874 on FSeval P1). The permutation is now
> drawn from a fixed generator and the invariant is locked by
> `scripts/test_diagnostics.py :: "scoring: centre controls deterministic"`.
> All numbers below are from the corrected run. The fix **strengthened** the
> NO-GO conclusion — see §2.

Two protocols, because VA ground truth is unavailable (V03/V04 blocked):

- **P1 synthetic warp** — a monotone piecewise-linear time warp of the *same*
  video (linear interpolation in raw feature space) gives an exact index map.
  Positives: |i − src(j)| ≤ 2 frames. Hard negatives: same GT action and
  |i − src(j)| ≥ 16 frames.
- **P2 cross-video phase** — frames of two different videos with the same GT
  action label; GT within-segment relative phase defines positives
  (|Δphase| ≤ 0.10) and hard negatives (|Δphase| ≥ 0.50). Phase is used for
  evaluation only and never enters a score.

## 2. P1 — synthetic warp, exact correspondence (FSeval, 12 videos)

AUROC / Recall@1 / median alignment error in frames:

| score | hard centres | soft centres | **random centres** | **shuffled centres** |
|---|---|---|---|---|
| `cosine` | 0.796 / 0.977 / 0.25 | 0.796 / 0.977 / 0.25 | 0.796 / 0.977 / 0.25 | 0.796 / 0.977 / 0.25 |
| `centered_dot` | 0.823 / 0.685 / 8.36 | 0.851 / 0.740 / 6.12 | 0.855 / 0.832 / 1.36 | 0.815 / 0.716 / 7.87 |
| `direction` | 0.873 / 0.956 / 0.27 | 0.906 / 0.975 / 0.26 | **0.877 / 0.962 / 0.27** | **0.874 / 0.959 / 0.27** |
| `line` | 0.873 / 0.956 / 0.27 | 0.906 / 0.975 / 0.26 | 0.877 / 0.962 / 0.27 | 0.874 / 0.959 / 0.27 |
| `radial` | 0.686 / 0.575 / 65.6 | 0.673 / 0.561 / 70.4 | 0.663 / 0.614 / 51.8 | 0.798 / 0.654 / 42.1 |
| `blend_w0.5` | 0.866 / 0.967 / 0.26 | 0.883 / 0.981 / 0.26 | 0.871 / 0.966 / 0.26 | 0.874 / 0.966 / 0.26 |

Read the `direction` row across the four columns. The learned prototype centre
gives 0.873; a fixed set of **random** centres gives 0.877; a **deliberately
wrong** action-to-centre mapping gives 0.874. The three are indistinguishable.
So the gain of `direction` over raw cosine (0.873 vs 0.796) comes from
subtracting *some* consistent per-frame centre — removing a shared mean
component — and **nothing in it is attributable to the action prototype, nor
even to the centre being the right one for that frame**. The only requirement
is that the same frame gets the same centre in both videos, which the earlier
buggy control accidentally violated and thereby hid.

On the metrics that matter operationally, `cosine` is still ahead anyway:
Recall@1 0.977 vs 0.956, and the median alignment error is identical at a
quarter of a frame.

Desktop Assembly, 12 videos, same protocol (AUROC / Recall@1): `cosine`
0.983 / 1.000; `direction` with hard centres **0.947 / 0.999**, with soft
centres 0.972 / 1.000, with random centres 0.983 / 1.000, with shuffled centres
0.983 / 0.999. On DA the learned prototype is the *worst* of the four centre
choices and the only one that falls below plain cosine.

## 3. P2 — real cross-video pairs, same action, different phase

AUROC / Recall@1 (FSeval, 12 pairs):

| score | hard | soft | random | shuffled |
|---|---|---|---|---|
| `cosine` | **0.571 / 0.494** | 0.571 / 0.494 | 0.571 / 0.494 | 0.571 / 0.494 |
| `direction` | 0.533 / 0.499 | 0.523 / 0.491 | 0.543 / 0.508 | 0.533 / 0.499 |
| `radial` | 0.536 / 0.455 | 0.505 / 0.434 | 0.529 / 0.440 | 0.533 / 0.449 |
| `blend_w0.5` | 0.560 / 0.507 | 0.559 / 0.507 | 0.559 / 0.511 | 0.554 / 0.503 |

Desktop Assembly: `cosine` 0.806 / 0.755; `direction` hard 0.775 / 0.746, soft
0.800 / 0.757, random 0.792 / 0.756, shuffled 0.792 / 0.750. `blend_w0.5` hard
0.817 / 0.759 (+0.011 over cosine) but shuffled 0.810 and random 0.809 — the
prototype-specific part of that margin is 0.007.

Two things follow. First, **no action-centered score beats cosine on real
cross-video pairs on either dataset.** Second, and independently important:
cosine itself is at 0.571 AUROC on FSeval for this task. VASOT embeddings
barely discriminate within-action phase at all — a result that is fed to N-B1.

## 4. The decisive control: direction does not encode phase

Mean cross-video cosine between group-mean directions `u`, grouped by
(assigned cluster, GT phase quintile), over ≥ 3 videos per group:

| dataset | same cluster, **same** phase bin | same cluster, **different** phase bin | groups |
|---|---:|---:|---:|
| FSeval | 0.928 | **0.932** | 42 / 192 comparisons |
| Desktop Assembly | 0.781 | 0.729 | 60 / 246 comparisons |

On FSeval the cross-phase number is *higher* than the same-phase number: the
centered direction is a function of the action, not of the phase within it. DA
shows a small gap (0.781 vs 0.729) in the predicted direction, but `direction`
still loses to `cosine` there (§3), so the gap is not usable. The core
hypothesis of a prototype-centered H — frames at the same action phase share a
direction relative to the action centre — is **not supported**.

## 5. Sensitivity and degeneracy checks

- Radius: `r` mean 1.03–1.08, 5th percentile 0.72–0.80, and **0.000 of frames
  have r < 0.1** on FSeval. The `||z − a_k|| ≈ 0` degeneracy does not occur
  here, so it is not what suppresses the angular score.
- Prototype perturbation σ ∈ {0.01, 0.05, 0.10}: P2 AUROC for `direction`
  moves 0.5323 → 0.5313 → 0.5307. The angular score is almost completely
  insensitive to the prototype — consistent with §4, where it is an
  action-identity feature, robust but uninformative about phase.

## 6. GO / NO-GO decision and limitations

**NO-GO for the prototype-centered relation H.** Grounds: (i) every apparent
gain of the centered/angular score is reproduced both by a fixed *random* centre
and by a deliberately *wrong* action-to-centre mapping (FSeval P1 direction:
0.873 hard, 0.877 random, 0.874 shuffled), so nothing is attributable to the
action prototype — which agrees with G00's finding that the learned prototypes
stay near their random initialisation;
(ii) on real cross-video pairs no variant beats cosine on either dataset;
(iii) the direction feature is shown directly not to carry phase information on
FSeval. Per the Phase G contract, G02's gate ("does not systematically worsen
within-action matching against cosine on a realistic frozen checkpoint") is not
met, so the training grid, G03 multi-video transfer and the conditional G05
cross-ratio work were **not started**.

Limitations, stated rather than hidden: a single baseline checkpoint per
dataset (FSeval s1, DA s0) and 12 videos each; P1 positives come from a
self-warp, so they share appearance statistics with the query and over-state
all scores; P2 uses a GT-phase proxy for correspondence, not true frame
correspondence, and a linear-interpolation warp cannot create genuinely new
viewpoints. The explicitly separate fallback allowed by the contract — a tiny
*learned* projection head before the angular score — is NOT RUN and remains
open; nothing here shows that no projection of this embedding could carry
phase, only that the raw prototype-centered geometry does not.
