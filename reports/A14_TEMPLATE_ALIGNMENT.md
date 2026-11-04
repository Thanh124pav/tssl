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
