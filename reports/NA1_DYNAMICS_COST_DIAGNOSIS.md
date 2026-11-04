# N-A1 — Dynamics-aware structural cost: diagnosis

Date: 2026-11-04. Frozen checkpoint, no training.
**Verdict: the hypothesis is not supported. The N-A2 gate is NOT passed and
distributional relational transport was not started.**

Scripts: `reproductions/tssl/scripts/eval_na1_dynamics_cost.py`,
`reproductions/tssl/scripts/fgw_dense.py`.
Artifact: `reproductions/tssl/artifacts/diagnostics/na1/FSeval_baseline_s1.json`.
W&B: `vasot-repro`, run `NA1_FSeval_baseline_s1`.

> **Padding note (2026-11-04 audit).** Every `T_xy` in this report is solved
> with the virtual row **and** column prepended, so the index `align_ot` drops
> is the virtual one. The published `train.py` appends the virtual column
> instead, which shifts its alignment pseudo-label by one frame (see
> `reports/X01_VIRTUAL_COLUMN_AUDIT.md`). The checkpoints audited here were
> therefore *trained* under the published padding and *measured* under the
> corrected one; this matches the convention used by E09 and by
> `scripts/eval_transport_support.py`, so the numbers remain comparable with
> earlier packets.

## 1. The solver was not changed — this is verified, not asserted

`alignment_ot.align_ot` hard-codes `C^X`, `C^Y` as conv1d filters, so swapping
the structure needs a dense restatement. `fgw_dense.align_ot_dense` reimplements
the same mirror-descent loop with explicit matrices, and
`fgw_dense.index_structure` rebuilds exactly the matrices those filters
represent:

```
C^X[i,k] = 1/radius  if 1 <= |i-k| <= radius*N  else 0
C^Y[j,l] = 0         if 1 <= |j-l| <= radius*M  else 1
```

`fgw_dense.check_equivalence` at the checkpoint's own solver settings:

| quantity | value |
|---|---:|
| max abs difference in the transport plan | 1.34e-07 |
| max rel difference in the transport plan | 1.68e-06 |
| max abs difference in the objective trace | 8.94e-08 |

Any difference below is therefore attributable to the structure, not to the
optimiser, the cross-video cost, the priors, the marginals, the iteration
budget or the seeds, all of which are held fixed.

## 2. Diagnosis: is frame index an inadequate proxy for structure?

FSeval baseline s1, 8 videos, 256 frames, GT used only to locate boundaries:

| quantity | value |
|---|---:|
| step size `||z_{i+1} − z_i||` within an action | 0.537 |
| step size at a GT boundary | 0.661 |
| ratio boundary / within | **1.233** |
| transition similarity `cos(v_i, v_{i−1})` within an action | 0.189 |
| transition similarity at a boundary | 0.163 |
| rank of the temporal successor under the feature metric, mean | 38.8 / 256 |
| same, median | 21.1 / 256 |
| fraction of frames whose nearest feature neighbour is the next frame | **0.092** |
| step size, short segments (< 16 frames) | 0.576 |
| step size, mid segments (16–48 frames) | 0.509 |

The first half of the hypothesis is confirmed: the feature metric does **not**
encode temporal adjacency — only 9.2% of frames have their own successor as
nearest neighbour, and the median successor rank is 21. The frame-index
structure is thus genuinely carrying information that feature distance does
not. But the dynamics signal itself is weak: a boundary raises the step size by
only 23% and *lowers* transition similarity by only 0.026, so local dynamics
are barely different at boundaries from inside an action.

## 3. Controlled structure swap

One factor changed at a time: the structure matrices, nothing else. Correspondence
error is measured against monotone piecewise warps (severity 0.5, mean max/min
speed ratio 2.12) with an exact known index map; the action-agreement column is
a labelled **proxy** on real adjacent video pairs, not a correspondence metric.
Non-index structures are scale-matched to the index structure's mean entry
(1.904).

| structure | align err (frames) | effective support | runtime (ms) | action-agreement proxy |
|---|---:|---:|---:|---:|
| `index` (baseline) | **0.349** | 11.4 | 10.1 | 0.567 |
| `feature` (dense) | 16.271 | 46.6 | 10.8 | 0.460 |
| `velocity` (dense) | 1.371 | 105.4 | 9.9 | 0.495 |
| `shuffled` velocity (control) | 1.357 | 105.9 | 10.0 | 0.496 |
| `random` (control) | 1.413 | 105.9 | 10.6 | 0.501 |
| `none`, alpha = 0 (control) | 1.373 | 68.7 | 9.6 | 0.495 |
| `band_feature` | 0.372 | 13.3 | 10.6 | 0.582 |
| `band_feature_shuffled` (control) | 0.401 | 16.0 | — | 0.572 |
| `band_velocity` | 0.461 | 58.4 | 10.3 | 0.558 |
| `band_shuffled` velocity (control) | 0.472 | 57.3 | 10.7 | 0.565 |

The `band_*` variants keep the index band exactly — so sparsity matches the
baseline — and only reweight entries inside it by feature or transition
similarity. This is the fair form of the proposed change; the dense variants
are reported because the packet asked for them.

Results, in order of importance:

1. **No alternative structure beats the frame-index structure.** The baseline
   is best on alignment error (0.349) and the only variant within noise of it
   is `band_feature` (0.372), which is the feature, not the dynamics, arm.
2. **Dense dynamics structure is indistinguishable from noise.** `velocity`
   1.371 vs `shuffled` 1.357 vs `random` 1.413 vs no structure at all 1.373.
   Band-limited dynamics behaves the same way: `band_velocity` 0.461 vs its own
   shuffled control 0.472.
3. The one nominal improvement, `band_feature`'s action-agreement proxy 0.582
   vs the index baseline's 0.567, does not survive its own control:
   `band_feature_shuffled` scores 0.572, so at most 0.010 of the 0.015 gap is
   attributable to the structure's content.
4. Removing the structural term entirely (`none`) costs 1.02 frames of
   alignment error and 0.072 of action agreement, so the index structure is
   doing real work — the problem is that nothing proposed here does it better.
5. Feature-jitter sensitivity σ ∈ {0, 0.02, 0.05} changes the index result not
   at all (0.349 / 0.349 / 0.349) and the dynamics results by < 0.03 frames.
   The structures are robust and uninformative, not fragile.
6. Runtime is identical (9.6–10.8 ms per solve) and transport support, not
   runtime, is what separates the variants: every variant that loses also
   produces a far more diffuse plan (up to 105 of 256 effective targets).

## 4. Distinguishing the improvement from the alternatives the contract names

- Not a `rho_seg` effect: `rho_seg` was never touched here; the segmentation
  prior is absent from this experiment, and the alignment prior `rho_align` is
  identical across all rows.
- Not an optimisation-difficulty effect: identical solver, identical budget,
  identical step-size calibration, runtimes within 12% of each other, and the
  equivalence check in §1 bounds the solver difference at 1e-7.

## 5. Falsifiable failure case

`band_velocity` vs `band_shuffled`: if local transition structure captured a
within-video relationship that the temporal index misses, permuting the
velocity field across frames should destroy it. It does not — the permuted
control is within 0.011 frames and 0.007 action agreement of the real one. The
prediction that distinguishes the hypothesis from its null is made and fails.

## 6. Exit criterion and gate

Exit criterion met with a **negative** result: the paired comparison is in §3,
the mechanistic evidence in §2 (the index structure is not redundant with
feature distance, but local dynamics carry almost no boundary or phase signal),
and the falsifiable failure case in §5. Per the Phase N contract — "a paired-seed
table and mechanistic evidence showing whether dynamics-aware cost captures a
relationship missed by temporal index; otherwise stop A2 scaling" — **A2 is
stopped.** Relational distributional transport was not started.

Limitations: one checkpoint (FSeval baseline s1), 8 videos, one warp severity,
and hand-designed rather than learned dynamics features. The learned dynamics
encoder named in the packet is **NOT RUN** — it was gated behind the cheaper
arms showing any signal, and none did; it remains the only untested path to
this hypothesis. Scale matching uses the mean entry, so a different
normalisation of the dense structures would change their magnitudes, but it
cannot explain why the real dynamics structures match their own shuffled
controls.
