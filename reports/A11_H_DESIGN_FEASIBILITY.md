# A11 — What H can be, given the four conditions and what the space contains

Date: 2026-11-04. Written in response to the four conditions placed on the
relation H: generalization, clustering, interactive, nontrivial. Measurements on
frozen checkpoints; no training.

## 1. The feasibility ceiling, measured first

Before proposing any H: is the information H is meant to expose actually in the
representation? A probe is trained on some videos to predict within-action
relative phase and tested on held-out videos. The probe is **given the action
label**, so only phase has to be explained.

| space | ridge R² | forest R² | shuffled-label null |
|---|---:|---:|---:|
| trained embedding | 0.023 | 0.040 | −0.012 |
| **drift component** | 0.023 | **0.104** | −0.004 |
| **raw features** | **0.057** | 0.058 | −0.018 |
| untrained encoder | 0.025 | 0.040 | −0.010 |
| action label alone | 0.000 | — | — |

Two readings, the second more actionable than the first:

1. **The ceiling for any H is about 10% of within-action phase variance**, and
   only with a nonlinear readout on the drift component. H cannot expose what is
   not there. Every proposal below is bounded by this.
2. **Training destroys phase information.** The raw features are more than twice
   as linearly phase-decodable as the trained embedding (0.057 vs 0.023), and the
   untrained encoder (0.025) is no worse than the trained one. The current
   objective is removing the very signal the proposal wants. The highest-value
   intervention is therefore not to add a constraint but to stop destroying,
   which is consistent with A03: training turns the embedding into a slowly
   varying function of absolute time and compresses the rest away.

## 2. The four conditions, read as mathematics

**(1) generalization + (3) interactive are one requirement: factor the pairwise
relation through a shared coordinate.** If every video `v` has a map
`φ_v : frames → Z` into a common space, then the alignment between any two
videos is `φ_w⁻¹ ∘ φ_v`, and multi-video consistency is a *consequence* rather
than something to impose: compositions commute by construction. Concurrency and
spheres are one very specific way to obtain that property, and A02/A06/A08
measured that this particular way is not supported by the data. The requirement
itself is standard and has a standard answer — a template, a barycenter, a
canonical coordinate.

This matters because cycle-inconsistency is the one multi-video defect actually
measured: composing X→Y→Z disagrees with X→Z by 18.5 frames, and only 39% of
frames agree within 4 (A02). A shared coordinate removes it by construction; a
geometric constraint can only penalise it.

**(2) clustering** then becomes a factorisation of that coordinate:
`Z ≅ action × phase`. "Alignment lines that are similar share something
characteristic" = they share the action component of the code. Nothing needs to
be collinear or concurrent for this.

**(4) nontrivial** becomes a nondegeneracy condition on the codebook, and this is
where a better object than contrastive repulsion exists — see §3.

## 3. Candidate H, in order of how well the data supports it

### H1 — Ordered sub-action codebook (a canonical phase axis per action)

Replace one prototype per action with **M ordered sub-prototypes**, and *define*
alignment as composition through the sub-index.

- **(1)** automatic and exactly cycle-consistent: alignment is composition
  through a shared index, so no concurrency constraint is needed.
- **(2)** the shared characteristic is the action's sub-codebook block.
- **(3)** sub-prototypes are re-estimated from every video's assignments while
  each frame is pulled toward its assigned sub-prototype: mutual update by
  construction, which is what the existing OT/EM loop already does.
- **(4)** collapse is prevented by a **monotonicity constraint** — the
  sub-index must increase with time inside a segment — plus a log-det volume
  term on the sub-codebook. Ordering is a stronger and much better-conditioned
  anti-collapse device than contrastive repulsion: it forbids merging *and*
  removes the labelling ambiguity, with no temperature or margin to tune.

Supporting measurements, all already in hand:

| evidence | number |
|---|---|
| A05: raising K already improves the representation | purity 0.586 → **0.674** at K=48, ECE 0.490 → **0.134** |
| a shared phase index already exists, weakly | sub-cluster phase ordering across videos, Kendall τ **+0.078** at K=48 vs null +0.008 (at K=12, +0.213 vs +0.065) |
| it needs no geometric assumption | A06 found no line, circle, conic or quadric to build on |
| it targets the measured defect | A02's 18.5-frame composition error |

The honest caveat: K=48 improves purity but MoF *falls* 12 points under the
one-to-one protocol (A05), so this direction only looks good if the evaluation
reports purity and calibration, which A07 argues it must anyway.

### H2 — Simplex / archetypal structure, with log-det volume as nontriviality

If a geometric object is wanted, a **simplex is the natural shape for a point
cloud and a sphere is not.** Each action's frames are convex combinations of M
archetypes; barycentric coordinates are the shared coordinate across videos.

- **(1)** corresponding frames share barycentric coordinates — a shared
  coordinate again, so multi-video consistency is free.
- **(2)** the shared characteristic is the action's archetype set.
- **(3)** archetypes are estimated from all videos; frames are reconstructed
  from them.
- **(4)** nontriviality is **nonzero simplex volume**, a log-det term. This is
  the principled version of "do not collapse", and unlike the sphere it is
  **affine-invariant**, so there is no radius to pin and no centre to escape —
  the exact degeneracy that broke the sphere formulation (A09 §3: with a free
  centre the loss reaches zero carrying no information, between-group cosine
  0.866) cannot occur.

This is untested. Its precondition — are action clouds better described by a
simplex than by a Gaussian of the same covariance — is measurable with the A06
machinery (held-out fit against a matched-complexity null) and has not been run.

### What I would not build

A geometric relation whose centre is a free parameter. A09's own unit test
showed the failure mode is centre escape, not ray collapse, and the guard that
works is a constraint on the radius — which is to say, a constraint that removes
the freedom the proposal introduced.

## 4. Ranked recommendation

1. **Stop destroying phase information** (§1, finding 2): the raw features carry
   2.5× more linearly decodable phase than the trained embedding. An auxiliary
   term that preserves within-action neighbourhood structure from the raw
   features costs nothing geometric and attacks the measured loss directly.
2. **H1 with ordering**, evaluated on purity/ECE rather than MoF. A05 already
   shows the K direction helps the representation; the ordering constraint is
   the cheap addition that turns it into a shared phase index.
3. **Enforce cycle-consistency directly** as a loss, which is the version of
   (1)+(3) that needs no new structure at all.
4. **H2's precondition test** before any H2 implementation.

Expectations should be set by §1: with a 10% R² ceiling on within-action phase,
none of these can produce a large correspondence gain on this checkpoint family.
The realistic target is a better-calibrated, more consistent, finer-grained
segmentation, not a solved frame correspondence.

## 5. Limitations

One checkpoint family (FSeval), 16–20 videos, GT phase from eval-granularity
segments. The probe in §1 bounds what a *decoder* can read linearly or with a
small forest; it does not bound what a different training objective could create,
only what is currently present. The Kendall τ in §3 uses the solver's own hard
assignments, so it inherits their errors.

---

## 6. Addendum — what if H lives in a different latent space?

Date: 2026-11-04, same session. The question is whether H should be imposed on a
dedicated latent space rather than on the action/frame embedding.

### 6.1 The structural argument, both ways

In favour: the current 40-d embedding is **overloaded**. It serves the
segmentation cosine cost, the alignment cosine cost and the GW temporal
structure at once, so any geometric constraint added to it fights all three.
A06 measured that conflict directly — the codebook wants unit-norm rows while a
geometric centre must sit inside the cloud.

Against: in the shared space the geometry at least has to coexist with the task
costs. In a dedicated space **nothing stops it from being satisfied vacuously**
— the general form of the A09 degeneracy, since any head can be trained to
produce any geometry. A dedicated latent space is therefore *easier to satisfy
and harder to make meaningful*, and it is only accountable if the correspondence
is actually computed in it, so that the alignment loss flows through it.

### 6.2 The precedent in this repository

VASOT already has a dedicated learnable latent bottleneck
(`bspline_space='latent'`, `latent_basis`), used by E10b. All eight
configurations landed at or below the baseline:

| run | MoF |
|---|---:|
| E00 baseline s1 | 0.4945 |
| E10b latent dz12 b0.3 s0 / s1 | 0.4998 / 0.4928 |
| E10b latent dz12 b1.0 s0 / s1 | 0.4307 / 0.4407 |
| E10b latent dz24 b0.3 s0 / s1 | 0.4239 / 0.4812 |
| E10b latent dz24 b1.0 s0 / s1 | 0.3709 / 0.4639 |

That is explained rather than merely noted: the bottleneck was fed from the
embedding, whose phase ceiling is 0.055 (§6.3), and it was asked to support the
curve hypothesis that A06 later refuted.

### 6.3 What a latent head could read, by input

Phase R² on held-out videos, action label given to the probe:

| input to the head | linear | 2-layer MLP |
|---|---:|---:|
| raw features | 0.057 | **0.103** |
| trained embedding | 0.023 | 0.055 |
| drift component | 0.023 | overfits |

So a latent head **fed from the raw features roughly doubles** what one fed from
the embedding can expose. The first design conclusion is therefore concrete: if
such a head is built, it must branch off the raw features, not off `z`.

### 6.4 The finding that reframes the question

The probe was then given a temporal **window** of raw features instead of a
single frame:

| window | input dims | linear | 2-layer MLP |
|---|---:|---:|---:|
| ±0 (one frame) | 64 | 0.057 | 0.103 |
| ±2 | 320 | 0.166 | 0.228 |
| **±4** | 576 | **0.177** | **0.314** |
| ±8 | 1088 | 0.092 | 0.286 |
| ±16 | 2112 | −0.313 | 0.234 |

**A ±4 frame window triples the phase ceiling, 0.103 → 0.314.** No new geometry,
no new latent space, no new loss — only a wider input.

The optimum is also interpretable: ±4 frames is a 9-frame window against a
median FSeval action of 11 frames. The phase information sits at exactly the
action scale, which is precisely the scale at which A03 measured a
signal-to-noise ratio of 0.70 for a *single* frame. The model reads one frame at
a time, so the only thing it can learn is roughly where in the video it is —
which a single frame does convey through the slow drift — while phase within an
action requires comparing neighbours, which the architecture never does.

### 6.5 Answer

The leverage is **not the space, it is the input**. A different latent space
helps only through what it reads, and the measurement says the input is the
binding constraint: one frame 0.103, a ±4 window 0.314.

Revised ranking, replacing §4:

1. **Give the encoder a temporal window** (±4 frames). It triples the phase
   ceiling and is the smallest change with the largest measured effect. Every H
   built after this has three times the budget.
2. **Stop destroying what is there** — the raw features are still 2.5× more
   linearly phase-decodable than the trained embedding.
3. **Then** impose H, in a space fed from the windowed features, and only if the
   correspondence is computed in that space so the geometry stays accountable.
4. H1 (ordered sub-action codebook) before H2 (simplex), on the evidence in §3.

This also predicts why every geometric proposal tested today failed at its
precondition: they all operate on a representation whose per-frame phase content
is near zero, so there is no structure for them to find or to impose
meaningfully. The order of operations was wrong, not only the shape.

### 6.6 Limitations

One checkpoint family, FSeval, 20 videos, 4-fold leave-videos-out. The windowed
probe sees up to 2112 input dimensions against roughly 4000 training frames, so
the linear degradation at ±8 and ±16 is partly overfitting rather than a real
decline; the MLP curve, which peaks at ±4 and declines gently, is the more
trustworthy shape. An R² ceiling measured by a probe bounds what a decoder can
read from the *current* features; it does not bound what a different feature
extractor could provide.
