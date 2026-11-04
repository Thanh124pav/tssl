# Synthesis — session of 2026-11-04

Covers every packet run today: the un-gated Phase G and Phase N diagnostics, a
code audit, the five requested action-space diagnostics plus three follow-ups,
and the first training runs of the ray/sphere objectives. This document also
serves the `G06` deliverable (section 5).

Raw artifacts under `reproductions/tssl/artifacts/diagnostics/`, training runs
under `artifacts/runs/`, W&B project `vasot-repro`. 28 invariants across
`scripts/test_diagnostics.py` (17) and `scripts/test_ray_geometry.py` (11).

## 1. Status of every packet

| packet | status | verdict in one line |
|---|---|---|
| G00 | DONE | D01 and E09 measured different matrices; prototypes barely leave their own random init |
| G01 | DONE, NO-GO | every centered-score gain is reproduced by a random *and* by a wrong centre |
| G02–G05 | NOT RUN | gated off by G01 |
| N-A1 | DONE, negative | no structure beats the frame index; dynamics structure equals its shuffled control |
| N-B1 | DONE, inconclusive | no gradient conflict; transitions sharpen the plan and make it worse |
| N-C1 | DONE, failure found | real warp failure, but classic DTW absorbs it |
| N-A2 / N-B2 / N-C2 | NOT RUN | gates not met |
| AUDIT | DONE | 5 defects; one upstream off-by-one in the training loss |
| X01 | DONE | that off-by-one costs nothing resolvable (+0.9 pp, paired std 6.2 pp) |
| A01 | DONE | `temporal` init +4.6 pp MoF (p=0.008) but +0.6 pp purity (p=0.46) |
| A02 | DONE | matched sets tight but isotropic; transport is not cycle-consistent (18.5 frames) |
| A03 | DONE | drift 4.7-dim + white jitter 15.4-dim, SNR 0.70 at action scale |
| A04 | DONE | ECE 0.459; accuracy driven by boundary distance, confidence nearly blind to it |
| A05 | DONE | the MoF collapse at large K is the protocol; purity peaks at K=48 |
| A06 | DONE | no conic or quadric, by a detector validated on circles and ellipsoids |
| A07 | DONE | position-only baseline reaches 82% of VASOT per-video |
| A08 | DONE | the line-retrieval rule as stated *hurts*: 0.980 → 0.967 |
| A09 | DESIGNED | objectives + 11 invariants; the degeneracy is centre escape, not ray collapse |
| A10 | PARTIAL | MoF not established at n=3; seed variance collapses 29× |

## 2. One mechanism, six previously separate results

A03's decomposition is the centre of the picture. The embedding is a
**4.7-dimensional slow drift (lag-1 autocorrelation 0.999) inside an equal
amount of white 15.4-dimensional jitter (lag-1 0.108)**, variance split
46.5 / 48.1. Training compresses global variance (participation ratio 22.7 →
5.8) without changing the local intrinsic dimension (10.85 → 10.76) or the
turning angle (116°). Over one median action the drift moves 0.293 against a
jitter radius of 0.418, so the **signal-to-noise ratio at action scale is 0.70**.

In one sentence: training makes the embedding a slowly varying function of
absolute time, and almost nothing else.

That single description predicts six results that were found independently:

| result | number | why it follows |
|---|---|---|
| frame-index structure beats every alternative (N-A1) | 0.349 vs 1.37 frames | absolute time is the only thing encoded strongly |
| B-spline fits fail (E14/E15, A06) | local dim 10.8 unchanged | the obstruction is local; global compression does not remove it |
| within-action phase is near chance (G01, N-B1) | AUROC 0.571 | SNR 0.70 at that scale |
| a one-frame pseudo-label shift costs nothing (X01) | +0.9 pp, p = 0.82 | adjacent frames differ mostly by white jitter |
| accuracy is set by boundary distance (A04) | 0.355 → 0.673 | the model knows "roughly when", so it errs at transitions |
| no conic or quadric structure (A06) | ratio 0.67 vs 0.03 for a real ellipse | a 4.7-dim non-planar path buried in equal noise |

## 3. Why the reported score is high, and what to measure instead

A07 decomposes it. A **position-only** baseline — cut each video into K equal
blocks, no model, no features — reaches 0.458 per-video against VASOT's 0.560,
i.e. **82%**. The OT decoder with an untrained encoder reaches 0.464 by the same
route. Pooled contributions: 0.225 raw k-means, +0.126 for training the encoder,
+0.105 for the OT decoder, total 0.456 against 0.295 for position alone.

The benchmark rewards precisely the one property the representation has. The
metric also fixes K at the number of GT classes and matches one-to-one; A05
shows that at K = 48 purity is better (0.674 vs 0.586) and ECE is 3.7× better
while MoF *falls* 12 points, with the stranded-frame fraction (0.00 → 0.41 →
0.90) tracking the collapse.

**Consequence, and the single most portable lesson of the session:** on this
setup a MoF gain can be bought by injecting positional prior without improving
the representation. A01 is the worked example — `temporal` prototype init gains
+4.6 pp MoF at p = 0.008 while its many-to-one purity moves +0.6 pp at p = 0.46.
Every future claim should carry a representation-side number (many-to-one
purity, ECE, or a correspondence metric) beside MoF.

## 4. Hypothesis scoreboard

Each row names the control that decided it, because in every case the control
was what mattered.

| hypothesis | verdict | the control that decided it |
|---|---|---|
| alignment plan `T_xy` is near one-hot | refuted (E09, confirmed G00) | measuring the right matrix; `T_xa` is the sharp one |
| prototype-centered rays carry phase | refuted | a **random** centre scores the same (0.877 vs 0.873), and a **wrong** one too (0.874) |
| matched frames are collinear | refuted | matched sets are *less* linear than random (0.379 vs 0.453) |
| matched frames lie on a conic or quadric | refuted | detector validated on ellipses (0.031) puts real data at the noise level (0.67) |
| frames of an action sit at constant radius | refuted | radius CV around the prototype is 1.14× the matched-noise null |
| radius encodes video identity | refuted | between-video std 0.059 vs within-group 0.144 |
| local dynamics beat the frame index | refuted | velocity structure equals its own shuffled control (1.371 vs 1.357) |
| transitions carry phase the state misses | inconclusive | +0.024 AUROC, below the plan's own 2% perturbation sensitivity |
| E13 was weak because of gradient conflict | refuted | cos(seg, align) = **+0.327**, positively aligned |
| geometry collapse needs preventing | refuted | 255/256 clusters used at K = 256; margin 0.718 |
| nonlinear warps break the structural cost | **supported** | oracle cost 0 → 0.089; but DTW absorbs it (1.33 vs 15.9 frames) |
| transport is cycle-consistent | **refuted, and this one is actionable** | X→Y→Z disagrees with X→Z by 18.5 frames; 39% within 4 |

## 5. Phase G decision (the `G06` deliverable)

**Decision: STOP the prototype-centered geometry line. REVISE toward
cycle-consistency.**

Ablation matrix for the Phase G components, all measured:

| component | measured effect | control |
|---|---|---|
| action-centre choice | none attributable | random 0.877, wrong 0.874, learned 0.873 |
| radius | harmful as a score (0.686 AUROC vs 0.796 cosine) | — |
| angular scoring | loses to cosine on real pairs (0.533 vs 0.571) | — |
| multi-video transfer | not reached | gated |
| prototype feedback | harmful when free (centre escape); ablating it is better | A10 `nocentre` ≥ `full` on 3/3 seeds |
| cross-ratio | not an invariant (CV 1.17 across videos; 79% change under warp) | a monotone warp is not projective |
| frame-index prior | load-bearing; `rho=0` at decode costs nothing | the prior is already absorbed into the representation |

The final claim is scoped to: FSeval, one baseline checkpoint per dataset for
the frozen packets, 8–20 videos, 2–3 seeds, and the sampled-frame protocol. No
SOTA claim is made, and DA/YTI were only touched in G00 and G01.

Conic is **not** an automatic fallback and was tested on its own terms (A06); it
is also refuted. The one genuinely untested variant is a **learned low-dimensional
projection** before the angular score, which no measurement here rules out.

## 6. What survives, ranked by evidence

1. **Enforce cycle-consistency across ≥3 videos.** A02 measured the defect
   (18.5 frames composition error, 39% within 4, round trip 9.4 frames). It
   needs ≥3 videos to define, rests on no geometric assumption, and is the
   version of the multi-video intuition the data supports.
2. **Calibrate `T_xa` before using it as a pseudo-label.** A04 measured ECE
   0.459 with the pseudo-label sharper (0.915) than the student it teaches
   (0.674). A05 showed calibration improves 3.7× at K = 48 on its own.
3. **Raise K and report purity, not MoF.** K = 48 is measurably the better
   model on purity and ECE today, at no training cost.
4. **Compare against soft-DTW / Gromov-DTW structure.** N-C1's failure regime is
   real and a classic elastic baseline handles it; VASOT's structural term should
   have to beat that baseline before any new invariance theory is credited.
5. **A09's objectives, with goal (2) off.** A10 interim: freezing the centres
   beats updating them on 3/3 seeds, consistent with the centre-escape
   measurement. The pre-registered purity and retrieval tests still decide.

## 7. A10 interim, stated against the pre-registered criteria

| arm | s0 | s1 | s2 | mean | std | paired Δ | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.4510 | 0.4945 | 0.5266 | 0.4907 | 0.0380 | — | — |
| `sphere_full` | 0.5245 | 0.4949 | 0.5269 | 0.5155 | 0.0178 | +0.0248 | 0.42 |
| `sphere_nocentre` | 0.5274 | 0.5283 | 0.5300 | 0.5286 | **0.0013** | +0.0379 | 0.22 |
| `ray_full` | 0.5245 | — | — | — | — | — | running |

Neither arm establishes a MoF gain at n = 3. The striking observation is the
**seed variance**: 0.0380 → 0.0178 → 0.0013, a 29× reduction, with
`sphere_nocentre` landing inside a 0.26 pp band. With n = 3 a variance claim is
statistically very weak and needs more seeds before it is called a result.

The geometry did get imposed: radius spread across actions went from 0.038 on
the frozen checkpoint to 0.38–0.44, and the nearest centre pair from 0.268 to
0.315–0.340. So this is the case A09 §6 criterion (a) anticipated — the geometry
is imposed and MoF does not move. Purity, ECE and the A08 retrieval test are the
deciding measurements and have not been run on these checkpoints yet.

## 8. Methodology, including this session's own errors

Three near-misses, all caught by controls rather than by judgement:

- **X01 seed 0** showed +6.7 pp MoF and would have been reported as a win had
  the sweep stopped there; seeds 1 and 2 brought the paired mean to +0.9 pp with
  a 6.2 pp paired standard deviation. Same pattern for A01 `orthogonal` and
  `kmeans`, which have the largest means in their table and flip sign.
- **A G01 control was mis-implemented** (an unseeded permutation drawn per call),
  which produced a spurious collapse and a wrong explanation in the first
  version of that report. Fixing it changed the conclusion — in the stronger
  direction, which is luck, not method.
- **A design claim was falsified by its own unit test.** The claim that
  collinearity alone collapses every frame onto one ray is false; the loss is
  separable across groups. The real degeneracy is centre escape, found only
  because the test was written to fail.

Two conclusions were also stated too broadly and later corrected by measurement:
"matched sets are not linear, so the geometric hypothesis fails" (linearity says
nothing about conics — A06 then tested them properly), and "nothing in the
geometry marks where an action ends" (step size predicts a boundary at AUROC
0.709; what does not exist is a characteristic *scale*).

The practice that worked, in every case: name the control before looking at the
result, and prefer a control that reproduces the claimed effect by a mechanism
the hypothesis forbids — a random centre, a shuffled velocity field, a
matched-covariance null, a position-only baseline.

## 9. Limitations that apply to everything above

FSeval is the only dataset used throughout; DA and YTI appear only in G00, G01
and E09. Frozen packets use one baseline checkpoint per dataset and 8–20 videos.
Training packets use 2–3 seeds at 30 epochs, where a single arm spans 0.451–0.527
MoF, so the resolution floor is roughly 5 pp unless the paired standard
deviation is small. The diagnostic protocol samples 256 frames and pools them,
which is not the official full-length test protocol: the same checkpoint scores
0.456 under it and 0.494 officially, so only within-protocol comparisons hold.
No ground-truth frame correspondence exists for FSeval, so correspondence
accuracy is measured only against synthetic monotone warps, which share
appearance statistics with the query and therefore flatter every method.
