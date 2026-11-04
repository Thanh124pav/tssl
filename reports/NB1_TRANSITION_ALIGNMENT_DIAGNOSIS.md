# N-B1 — Transition-aware matching and gradient conflict: diagnosis

Date: 2026-11-04. Frozen checkpoint, no training.
**Verdict: INCONCLUSIVE, leaning negative. The N-B2 gate is NOT passed and
transition-commutative transport was not started.**

Script: `reproductions/tssl/scripts/eval_nb1_transition.py`.
Artifact: `reproductions/tssl/artifacts/diagnostics/nb1/FSeval_baseline_s1.json`.
W&B: `vasot-repro`, run `NB1_FSeval_baseline_s1`. FSeval baseline s1, 8 videos,
256 frames. GT action labels and GT within-segment relative phase are used for
evaluation only.

> **Padding note (2026-11-04 audit).** Every `T_xy` in this report is solved
> with the virtual row **and** column prepended, so the index `align_ot` drops
> is the virtual one. The published `train.py` appends the virtual column
> instead, which shifts its alignment pseudo-label by one frame (see
> `reports/X01_VIRTUAL_COLUMN_AUDIT.md`). The checkpoints audited here were
> therefore *trained* under the published padding and *measured* under the
> corrected one; this matches the convention used by E09 and by
> `scripts/eval_transport_support.py`, so the numbers remain comparable with
> earlier packets.

## A. Transport keeps action identity and loses phase — but only partly

Row-normalised `T_xy` on 8 real adjacent pairs:

| quantity | value |
|---|---:|
| mass on the correct GT action | 0.510 |
| mass on the correct phase (same action, \|Δphase\| ≤ 0.10) | 0.118 |
| expected phase mass if mass were uniform **inside** the correct action | 0.098 |
| lift over that expectation | **1.21×** |
| mass-weighted mean \|Δphase\| inside the correct action | 0.324 |
| same, if uniform inside the correct action | 0.361 |

This is the quantitative form of the packet's question. Alignment puts about
half its mass on the correct action, and *within* that action it is only 1.28×
better than guessing uniformly — it shrinks the mean phase error from 0.361 to
0.324, i.e. by 10%. So the hypothesis "segmentation shares action identity but
alignment fails to preserve within-action phase" is **largely confirmed**, with
the correction that phase information is not zero, just small. This matches
G01, where cosine scored 0.571 AUROC separating same-phase from distant-phase
frames of the same action.

## B. Transition features carry slightly more phase signal than state

AUROC separating same-phase (|Δphase| ≤ 0.10) from distant-phase (≥ 0.50)
frames of the same action, across 8 real pairs:

| feature | directed `v_i = z_i − z_{i−w}` | symmetric (backward ⊕ forward) |
|---|---:|---:|
| state cosine (baseline) | 0.557 | 0.557 |
| window 1 | 0.509 | 0.511 |
| window 2 | 0.511 | 0.522 |
| window 4 | 0.525 | 0.555 |
| window 8 | 0.531 | **0.581** |
| window 16 | 0.538 | 0.573 |
| window 32 | 0.521 | 0.531 |

The best transition variant (symmetric, window 8) beats state cosine by 0.024
AUROC, with a clear interior optimum in the window — short windows are pure
noise, very long windows wash out. Symmetric beats directed at every window.
This is a **repeatable but very small** mismatch between state matching and
transition information: both scores are close to chance for this task.

## C. The transport plan is not stable enough to carry a 0.024 AUROC effect

Embeddings perturbed by Gaussian noise of scale σ, then re-solved:

| σ | top-1 agreement with the unperturbed plan | row L1 drift | mean argmax shift |
|---|---:|---:|---:|
| 0.01 | 0.848 | 0.082 | 1.28 frames |
| 0.02 | 0.712 | 0.189 | 2.67 frames |
| 0.05 | 0.444 | 0.403 | 5.36 frames |

A 2% perturbation of the embedding already moves 29% of the top-1 assignments
by a mean of 2.7 frames. The effect size in §B is far smaller than this
sensitivity, which is the main reason this packet is called inconclusive rather
than positive.

## D. There is no gradient conflict to explain E13

Gradients of each loss with respect to the shared encoder (`model.mlp`), on 4
real pairs at the frozen checkpoint:

| quantity | value |
|---|---:|
| segmentation CE | 1.692 |
| alignment CE | 4.597 |
| action-bridge CE (E13) | 5.892 |
| ‖grad‖ segmentation | 2.310 |
| ‖grad‖ alignment | 1.179 |
| ‖grad‖ action bridge | 6.617 |
| **cos(grad seg, grad align)** | **+0.327** |
| cos(grad seg, grad bridge) | +0.248 |
| cos(grad align, grad bridge) | +0.303 |

All three pairs of objectives have *positively* aligned gradients.
**Gradient conflict is ruled out as the explanation for why the E13 action
bridge barely helped alignment.** Note also that the bridge gradient norm is
5.6× the alignment gradient norm, so with equal loss weights the bridge term
dominates the encoder update — a scale issue, not a direction conflict. That is
a more plausible E13 mechanism and is worth one controlled weight sweep if E13
is revisited.

## E. Transition cost does not improve correspondence at equal budget

Same solver, same budget, same priors and marginals; only the ground cost
changes. Alignment error against monotone piecewise warps (severity 0.5) with a
known index map:

| cost | align err mean | align err median | effective support | top-1 mass |
|---|---:|---:|---:|---:|
| `state_only` (baseline) | **0.349** | 0.258 | 11.41 | 0.323 |
| `state + action_bridge` (E13) | 0.394 | 0.261 | 9.88 | 0.353 |
| `state + transition` (0.5/0.5) | 0.616 | 0.369 | 4.00 | 0.610 |
| `transition_only` | 0.816 | 0.467 | 2.85 | 0.714 |

Transition cost makes the plan dramatically sharper (support 11.4 → 2.9, top-1
mass 0.32 → 0.71) while making it **less** correct. This is the specific trap
the packet warned about: the apparent improvement in sharpness is not a gain in
phase correspondence. The E13 bridge behaves the same way in miniature
(sharper, slightly worse).

## Exit criterion and gate

The exit criterion asks to "isolate at least one repeatable mismatch between
state/action matching and transition correspondence, or document negative
evidence and stop B2 escalation". Both halves partly apply:

- Repeatable mismatch found: §A (phase lift only 1.21× over uniform-in-action)
  and §B (transition features beat state cosine by 0.024 AUROC, with a
  consistent interior window optimum).
- Negative evidence found: §E (transitions in the cost hurt correspondence),
  §C (plan instability exceeds the effect size), §D (no gradient conflict).

The N-B2 gate requires that N-B1 "demonstrates information in transitions that
existing action/state matching misses". A 0.024 AUROC margin, measured on one
checkpoint, below the plan's own perturbation sensitivity, and which reverses
when actually used in the cost, does not meet that bar. **B2 was not started.**

Limitations: one checkpoint, 8 videos, hand-designed velocity features rather
than a learned transition encoder (the controlled local transition encoder named
in the packet is **NOT RUN**), phase labels from GT segment position rather than
true frame correspondence, and warp-pair correspondence that shares appearance
statistics with the query. Nothing here equates directional temporal prediction
with intervention-defined causality, and no causal claim is made.
