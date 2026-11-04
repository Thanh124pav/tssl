# A14 — Template-factored alignment: works, for the wrong reason

Date: 2026-11-04. FSeval, 3 arms × 3 seeds. Script
`reproductions/tssl/scripts/run_a14_template_alignment.sh`, W&B tag `A14`.
Baseline arm `A01_random_FSeval_s{0,1,2}`.

**Two verdicts, and they disagree.** The intervention gives +5.5 pp MoF at
p = 0.013 with purity and calibration also improving. And it **fails its own
primary pre-registered criterion**: eval-time cycle error roughly doubles. It
also fails the pre-registered granularity control. So the method works and the
reason it was built does not hold.

## 1. MoF

| arm | s0 | s1 | s2 | mean | paired Δ | t | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.4510 | 0.4945 | 0.5266 | 0.4907 | — | — | — |
| **tmpl64** | 0.5153 | 0.5376 | 0.5853 | **0.5461** | **+0.0554** | 8.73 | **0.013** |
| tmpl12 (control) | 0.5368 | 0.5286 | 0.5953 | 0.5536 | +0.0629 | 4.14 | 0.054 |
| tmpl64 + K=48 | 0.3430 | 0.3189 | 0.3532 | 0.3384 | — | — | — |

The K=48 arm is read against the K=48 baseline (0.3526), not the K=12 one, so
it is roughly flat there rather than down 15 points.

## 2. Pre-registered falsification: the granularity hypothesis is refuted

The run script said, before any run: *"L=12 is expected to behave like E13
(little change). If L=64 and L=12 are the same, granularity is not the active
ingredient and the design is wrong."*

Measured: **tmpl12 − tmpl64 = +0.0075, per-seed +0.022 / −0.009 / +0.010,
p = 0.487.** They are the same, and the coarse arm is nominally *ahead*.

So the argument that distinguished A14 from E13 — that 12 slots are too coarse
to carry frame correspondence — does not hold. Twelve slots work exactly as
well as sixty-four.

## 3. Primary criterion: eval-time cycle error got WORSE

The template makes the training **target** exactly cycle-consistent (verified to
3.0e-08). Whether the learned model then produces consistent pairwise plans was
stated in advance as empirical, not a consequence. It does not.

| checkpoint | cycle err (frames) | within 4 fr | round trip | purity | ECE | MoF |
|---|---:|---:|---:|---:|---:|---:|
| baseline K=12 (s0) | **15.62** | 0.396 | 8.69 | 0.5857 | 0.4897 | 0.4510 |
| tmpl64 s0 | **30.91** | 0.310 | 13.49 | 0.6523 | 0.4198 | 0.5153 |
| tmpl64 s1 | 29.99 | 0.297 | 11.42 | 0.6465 | 0.3930 | 0.5376 |
| tmpl12 s0 | 20.24 | 0.358 | 11.36 | 0.6441 | 0.4020 | 0.5368 |
| tmpl12 s1 | 13.96 | 0.433 | 7.02 | 0.6346 | 0.4071 | 0.5286 |

Cycle error nearly doubles at L=64 (15.6 → 30.9) and the fraction agreeing
within 4 frames falls. **The packet was built to fix cycle-consistency and made
it worse.** This is the A13 pattern a second time: a mechanism that is exact in
the target does not transfer to the learned model.

Purity and calibration did improve: purity 0.586 → 0.652 and 0.618 → 0.647,
ECE 0.490 → 0.420 and 0.459 → 0.393.

## 4. What is actually happening — hypothesis, under test

The hard induced plan is **block-structured**: every X frame in slot `l` is
mapped uniformly across all Y frames in slot `l`. That is a strong smoothing of
the alignment target, which would explain every number above at once —
segmentation metrics up because the target is less noisy, cycle error up because
the model is taught a block structure rather than a consistent map, and
granularity irrelevant because 12 and 64 slots both produce blocks.

If that is right, the working ingredient is **pseudo-label regularisation**, not
shared-coordinate factoring, and the design rationale should be rewritten rather
than defended.

A15 tests it with three one-factor ablations, running now:

| arm | what it decides |
|---|---|
| `soft` | composition no longer exact (0.006 vs 3e-08). If the gain survives, exactness is not the ingredient |
| **`frozen`** | the template is never learned. If a random fixed template works as well, the factoring is the ingredient and the template's content is not — the control logic that decided G01 |
| `blend05` | dose-response at half weight |

## 5. Honest status

Keep the result, drop the story. +5.5 pp MoF at p = 0.013 over three paired
seeds, with purity and ECE moving the same way, is the largest and
best-supported training gain found today. But it is not evidence for
template-factored multi-video alignment, because the multi-video property it
was built to deliver is measurably absent, and the granularity control that was
supposed to separate it from E13 came back null.

## 6. Limitations

FSeval, 3 seeds, 30 epochs. Cycle error and the metric triple are computed on
seed 0 and 1 checkpoints only, 8 videos, stride 8. The K=48 arm has no paired
third seed for its own baseline. The block-structure hypothesis in §4 is a
hypothesis; A15 is what decides it.

---

## 7. Addendum — the mechanism, measured directly

Same session, on CPU while A15 trained. Three measurements on
`A14_tmpl64_FSeval_s0`.

### 7.1 The block-structure hypothesis of §4 is confirmed exactly

| quantity | induced plan | pairwise plan |
|---|---:|---:|
| variance explained by slot-pair blocks | **1.0000** | 0.1969 |
| effective support | 12.43 frames | 30.01 frames |
| top-1 mass | 0.197 | 0.198 |

The induced plan *is* a block matrix, by construction. Its effective support of
12.4 frames is about one median FSeval action (11 frames), so it localises to
action precision, not frame precision.

### 7.2 A trap I walked into, and the measurement that got me out

The pre-registered criterion was eval-time cycle error, and §3 measured it on the
model's **pairwise** plan, where it got worse. It is tempting to answer that the
method's deliverable is the *induced* plan, and to measure that instead:

| plan, at inference | cycle error | within 4 frames |
|---|---:|---:|
| induced | **2.49 frames** | **0.969** |
| pairwise | 32.75 frames | 0.298 |
| argmax agreement between the two | 0.059 | |

A 13× improvement on the A02 defect. **This is not a rescue, and reporting it as
one would be moving the goalposts** — it is a different object, measured after
seeing the result. Worse, it is almost circular: a plan composed through a shared
coordinate is cycle-consistent *as arithmetic*. Consistency is free here.

What is not free is accuracy, so that was measured next, on monotone warps where
the true index map is exact:

| checkpoint | plan | mean error (frames) | R@1 (≤2 fr) |
|---|---|---:|---:|
| A14 tmpl64 | pairwise | **0.381** | **0.972** |
| A14 tmpl64 | **induced** | **59.3** | **0.136** |
| A14 tmpl12 | induced | 65.0 | 0.063 |
| baseline | pairwise | 0.405 | 0.969 |

The induced correspondence is consistent and **almost entirely wrong**: 59 frames
of error against 0.38, and R@1 0.136 against 0.972. It agrees with the pairwise
plan on 5.9% of frames. So the template does **not** deliver a usable multi-video
correspondence, and §3's verdict stands.

### 7.3 What the gain actually is, and why it is still interesting

Training on the induced target improves segmentation (+5.5 pp MoF, purity
+0.066, ECE −0.070) while that target is 94% different from the pairwise plan and
150× less accurate. The learned pairwise plan afterwards is *sharper* than
baseline (support 5.04 vs 12.60) and marginally more accurate (0.381 vs 0.405).

So the honest statement is: **training on a deliberately coarse, mostly-wrong but
smooth alignment target improves segmentation.** Which means the alignment
pseudo-label's frame-level content was not merely useless but actively harmful to
the segmentation branch.

That coheres with everything else measured today rather than contradicting it:
within-action phase is near zero in the representation (A11, R² ≈ 0.10 ceiling),
the embedding encodes absolute position (A03), the benchmark rewards position
(A07), and a one-frame shift in the pseudo-label costs nothing (X01). The fine
structure of the alignment target was noise, and replacing it with action-scale
blocks removed that noise.

This also predicts A15's `frozen` arm: if the active ingredient is block
structure rather than a learned shared coordinate, a never-learned random
template should work just as well.

---

## 8. Correction — the template was never learned, and A15's control was vacuous

Found while computing purity for A15: `A14_tmpl64_FSeval_s0` and
`A15_frozen_FSeval_s0` agree on MoF, purity and ECE to four decimals
(0.5153 / 0.6523 / 0.4198). Checked directly: the two checkpoints are
**bit-identical across every weight** (max abs difference 0.000e+00), and the
template equals its seeded initialisation while `clusters` did move.

Root cause: `_template_plan` is called inside the `with torch.no_grad()` block
where the pseudo-label is built, so the template parameter never receives a
gradient. `--template-frozen` is a no-op.

This cuts both ways and both must be stated:

- **A15's `frozen` arm is vacuous as a control.** It is a duplicate run, not an
  ablation. The claim "a never-learned random template works as well, therefore
  the template's content is not the active ingredient" is **withdrawn as an
  inference from that arm**, because the arm tested nothing.
- **A14's own description was wrong.** The template was a *random, fixed*
  projection in every arm, including the ones described as learned. So the
  "learned shared coordinate" framing never applied, and the +5.5 pp MoF was
  obtained with a random never-learned template all along.

The empirical content I wanted from the control therefore survives, by accident
rather than by design: a random fixed template gives +5.5 pp. But it is not
evidence from a comparison, and the honest form is weaker — the slots are a
fixed random partition that the *encoder* organises itself around, so "the
template learned nothing" is not the same as "a partition is unnecessary".

What is now genuinely untested is the opposite question: **does a template that
actually adapts do better?** A18 answers it with an EMA update of each slot
toward the mean embedding of the frames assigned to it, which is how the repo
learns its other codebook-like objects and what A09 did for the geometric
centres. Queued.
