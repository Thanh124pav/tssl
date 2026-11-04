# A09 — Ray-geometry objectives: design, constraints, and pre-registered test

Date: 2026-11-04. Design of a training objective that **imposes** the user's
goals (1) and (2) on the representation, as opposed to A08, which tested whether
they already hold. Module `repos/VASOT/src/ray_geometry.py`, invariants
`scripts/test_ray_geometry.py` (9/9 passing). Nothing is trained yet; §6 is the
pre-registered protocol.

## 1. The goals, restated as objectives

**(1)** Frames that align across videos should share a ray from the action
centre, so that the ray generalises correspondence to a video not in the pair.
**(2)** The lines of the correspondence groups inside one segment should be
concurrent, and that concurrency point should update the action centre.

## 2. Two design decisions forced by measurements

### 2.1 The codebook cannot also be the geometric centre

The segmentation cost is `1 − z a_kᵀ`, so the codebook rows want unit norm; a
centre that frames radiate from must sit inside the cloud. A06 measured the
conflict directly: the radius around the prototype is **less** constant than
matched-covariance noise (ratio 1.14), while around the group centroid it is
0.894. The module therefore adds a separate, unconstrained centre `c_k`,
initialised at the action's frame centroid, and leaves `clusters` to the
segmentation cost.

### 2.2 Directions must be computed from the drift, not the raw frame

A03 decomposed the trajectory into a 4.7-dimensional slow drift and an equal
amount of 15.4-dimensional white jitter, and measured that over one median
action (11 frames) the drift moves 0.293 while the jitter radius is 0.418
(SNR 0.70). A direction taken from a raw frame embedding is therefore
noise-dominated at exactly the scale the objective operates on. `smooth_drift`
takes the moving-average component first; the unit test confirms it lifts
lag-1 autocorrelation from −0.01 to 0.94 on synthetic input.

## 3. The degeneracy, measured — and a corrected claim

The first version of this design asserted that goal (1) alone collapses every
frame of an action onto a single ray. **The unit test falsified that.** The ray
loss is separable across groups, so each group independently aligns to whichever
ray is nearest its initialisation; between-group cosine stays at 0.027.

The real degeneracy is **centre escape**, and goal (2) is what triggers it.
Free points, 8 groups of 5, 600 Adam steps:

| configuration | loss | ‖c‖ | mean radius | between-group cosine |
|---|---:|---:|---:|---:|
| centre fixed | 2e-8 | 0.00 | 3.60 | 0.027 |
| **centre free (goal 2)** | 3e-8 | **4.27** | **6.32** | **0.866** |
| centre free + radius pinned | 2e-8 | 0.22 | 1.00 | 0.018 |

With the centre free, the optimiser pushes it out of the cloud. At a distance
every direction converges on the direction from the centre to the cloud, so
every pair looks collinear and the loss reaches zero **carrying no information**.
Goal (1) is then perfectly satisfied and perfectly vacuous.

What blocks it is a **radius constraint**, not a contrastive term. This reverses
the role the proposal assigned to idea (4): the anti-collapse mechanism that is
structurally required is a radius regulariser plus centre containment; the
contrastive term is a useful addition, not the guard.

## 4. The objective

Per batch of V ≥ 3 videos, with a reference video r. All transport is detached,
exactly as the existing pseudo-labels are.

**Groups.** `build_groups` anchors a group on each reference frame (stride s)
and takes each other video's argmax match. Every member must survive the round
trip within `cycle_tol` frames, and a group is kept only if at least
`min_consistent_frac` of its members do. This filter is not optional: A02
measured that a plain round trip X→Y→X returns within 4 frames for only 46% of
frames, so unfiltered groups would be built from roughly half-wrong members. The
surviving yield is logged, not assumed.

**Losses.**

```
L = L_VASOT
  + λ_ray     · mean_groups mean_pairs ( 1 − u_a · u_b )        goal (1)
  + λ_radius  · mean ( ‖z − c_k‖ − ρ_k )²                       blocks §3
  + λ_group   · [ InfoNCE(group id | directions) + hinge ]      keeps groups apart
  + λ_csep    · mean_{k≠l} relu( m − ‖c_k − c_l‖ )²             guards the new parameter
```

with `u = (s_i − c_k)/‖s_i − c_k‖`, `s` the smoothed drift, `ρ_k` the action's
measured mean radius, and pair weights `T_xy[i,j] · q_k(i) · q_k(j)`.

**Centre update (goal 2).** Not a gradient. `concurrency_centre_update` fits a
line per group, solves for the least-squares common point in closed form, and
moves `c_k` part of the way, under three guards: skip when the normal matrix's
smallest eigenvalue is below `min_conditioning` (near-parallel lines leave the
intersection undefined), cap the step length, and clamp the result to within
`max_anchor_offset` of the action's frame centroid.

A closed-form step rather than a gradient is a deliberate choice backed by G00:
gradient-trained prototypes stay at cosine 0.872 from their **own** random
initialisation, while two random codebooks sit at 0.24–0.27 from each other —
SGD does not move this parameter. A01 then showed the initialisation materially
changes the result (`temporal` init: +0.0455 MoF, t = 10.91, p = 0.008), so the
centre is a real lever that gradient descent is failing to use.

## 5. What the invariants already establish

`scripts/test_ray_geometry.py`, 9/9:

- the ray loss is 0 exactly on rays and rises on perturbed groups;
- minimising it drives within-group direction cosine to 1.0000;
- the separation loss discriminates distinct from shared directions
  (NCE 0.044 vs 1.792);
- centre escape reproduces and the radius pin blocks it (§3);
- the closed-form common point recovers a known centre to 1.9e-4 and reports
  near-parallel lines as ill-conditioned (0.008 vs 4.708);
- the centre update moves toward a known centre (1.434 → 0.385) and the anchor
  clamp holds at the limit;
- the group builder drops cycle-inconsistent members.

## 6. Pre-registered evaluation, written before any training run

**Required reporting.** A07 showed that MoF on this benchmark can be bought by
injecting positional prior without improving the representation: A01's
`temporal` init gains +4.6 pp MoF (p = 0.008) while its many-to-one purity moves
+0.6 pp (p = 0.46). Every arm here therefore reports **MoF, many-to-one purity,
and ECE** together, on ≥ 3 paired seeds. X01 fixes the resolution floor: a
single arm spans 0.451–0.527 MoF across seeds, so effects below ≈ 5 pp need
either many seeds or a paired std like A01's 0.0072 to be credible.

**Arms, one factor at a time.** baseline; +L_ray only; +L_ray +radius;
+L_ray +radius +group; full with the centre update; full with the centre update
and the cycle filter disabled (to price the filter).

**Mechanism checks that must move, or the objective is cosmetic.** Imposing the
geometry must show up in the geometry: within-group direction cosine up,
between-group cosine *not* up, radius CV down toward the A06 null, centre
displacement non-zero and the conditioning guard not firing constantly, and the
group yield after the cycle filter above a stated floor.

**Falsification, agreed in advance.** The objective is judged a failure if
`L_ray` falls and **either**
(a) many-to-one purity does not improve beyond seed noise, **or**
(b) A08's line-retrieval test still does not beat `T_xy` alone.

Condition (b) is the sharp one. A08 measured that today the line rule *hurts*:
`T_xy` alone reaches R@1 0.9800 and error 0.35 frames, while "top-5 by `T_xy`,
then nearest to the line" drops to 0.9673, and the additive form's +0.14 pp is
matched or beaten by random (0.9834) and shuffled (0.9829) centres. The whole
point of imposing the geometry is to make that rule work. If after training the
rule still loses to `T_xy` alone, the geometry was imposed and is useless, and
the line of work should stop there rather than continue to (3) and (4).

**Not yet implemented:** the batch must carry V ≥ 3 videos, and
`VideoDataset.__getitem__` returns a pair. A multi-video collate wrapper is the
remaining piece before the first run; it changes the data path, so it goes in
behind a flag with the pair path unchanged by default.

## 7. Status

Module and invariants: done. Training integration, the multi-video batch, and
the sweep in §6: not started. Flags default to off, so no existing baseline or
checkpoint is affected.
