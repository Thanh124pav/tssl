# Code audit — 2026-11-04

Audit of the Phase G / Phase N diagnostic code written on 2026-11-04, plus the
VASOT paths it calls. Five defects found: four in the new diagnostic code, one
upstream in `repos/VASOT/src/train.py`. One of the four changed a reported
conclusion, and the upstream one is a genuine training bug.

Verification harness: `reproductions/tssl/scripts/test_diagnostics.py`,
15 invariants, all passing. Each fix below is locked by a named test so the
same defect cannot return silently.

```
$ python reproductions/tssl/scripts/test_diagnostics.py
...
15/15 invariants hold
```

## D1 — `shuffled` centre control was redrawn on every call (G01)

**Defect.** `eval_g01_angular_probe.py :: centers()` built the `shuffled`
control with an unseeded `torch.randperm`, inside a function called once per
video. Video X and video Y therefore received *different* wrong action-to-centre
mappings, and the control was non-reproducible between runs.

**Why it mattered.** The control was specified as "wrong but consistent
mapping". As implemented it was "inconsistent mapping". It reported a collapse
(direction AUROC 0.609 on FSeval P1) that was an artifact of the inconsistency,
and the first version of the G01 report used that collapse to argue the centre
merely had to be *consistent*.

**Fix.** Fixed generator seed, so the permutation is the same wrong mapping
everywhere. Test: `scoring: centre controls deterministic`.

**Effect on conclusions.** The corrected control scores 0.874 — statistically
indistinguishable from the learned prototype (0.873) and from random centres
(0.877). This **strengthened** the G01 NO-GO: not only is the prototype
replaceable by a random centre, it is replaceable by a *deliberately wrong*
centre. The G01 report has been corrected and carries the correction notice.

## D2 — solver plan costed at the wrong scale (N-C1)

**Defect.** `align_ot` rescales its output so each row sums to 1 (total mass
N), while the oracle plan built for comparison carries total mass 1. Feeding
both to the same cost decomposition compared quantities a factor of 256 apart,
producing the nonsense "oracle minus solver objective" of −3443 to −8023.

**Fix.** The solver plan is normalised to total mass 1 before costing. The
corrected gap is a near-constant −0.079 to −0.088, which is the expected
signature of an entropy-regularised objective preferring a diffuse plan, and is
now reported as such. Test: `N-C1: zero cost at identity` pins the zero point
of the decomposition; `N-C1: oracle plan and DTW sanity` pins the oracle's unit
mass.

**Effect on conclusions.** None of the N-C1 headline numbers (oracle structural
cost, solver alignment error, DTW comparison) depended on this quantity; the
bug was caught before the report quoted it.

## D3 — expectation computed as a product of means (N-B1 §A)

**Defect.** The "expected phase mass if uniform inside the correct action"
baseline multiplied the mean over rows of one factor by the mean over rows of
the other, instead of averaging the per-row product.

**Fix.** Per-row product, then mean. **Effect:** the expected mass moves 0.092
→ 0.098 and the reported lift 1.28× → 1.21×. The direction of the conclusion is
unchanged (alignment barely beats uniform-inside-the-action), but the report
now states the correct figure.

## D4 — silent drops in two aggregations

`eval_g00_geometry_audit.py` discarded frames from videos with no GT boundary
(their distance is `inf`, which fell between the bins) without recording it;
`eval_nc1_warp_stress.py` would have produced garbage indices had the DTW path
failed to cover a row. Both now count and report the affected rows rather than
dropping them. No reported number changed.

## D5 — upstream: the alignment pseudo-label is shifted one frame

**Defect, in `repos/VASOT/src/train.py`.** The virtual frame is added to the
alignment cost matrix like this:

```python
top_row = torch.ones(B, 1, K) * 0.5
cost_matrix = torch.cat((top_row, cost_matrix), dim=1)      # row PREPENDED
left_column = torch.ones(B, N + 1, 1) * 0.5
cost_matrix = torch.cat((cost_matrix, left_column), dim=2)  # column APPENDED
```

while `alignment_ot.align_ot` ends with `return T[:, 1:, 1:]` — it drops index 0
on **both** axes. The row that gets dropped is the virtual row, correctly. The
column that gets dropped is **column 0, a real frame of video Y**, and the
virtual column survives at the far end of the pseudo-label. Column `c` of
`opt_codes` is therefore real Y frame `c+1`, while column `c` of the prediction
`codes` is real Y frame `c`: the cross-entropy in Eq. (7) is computed between
two tensors offset by one frame.

**Demonstration, not inference.** Align a video with *itself*, where the correct
plan is exactly the identity:

| padding | mean argmax offset | rows exactly on the diagonal |
|---|---:|---:|
| `train.py` (row prepended, column appended) | **−0.953 frames** | 0.000 |
| virtual column prepended as well | 0.000 frames | 1.000 |

Test: `solver: virtual column placement`.

**Scope.** `training_step` is the only place that builds this matrix;
`validation_step` and `test_step` do not call `align_ot`, so reported
segmentation metrics are not directly affected — the damage is to the training
signal. The diagnostics in this project (`eval_transport_support.py` and the new
`diag_common.align_plan`) prepend both, so every `T_xy` measurement in E09 and
in the 2026-11-04 packets is on the corrected convention. The audited
checkpoints were thus *trained* with the shifted pseudo-label and *measured*
without it; each affected report now carries a padding note.

**Fix.** `--fix-virtual-column` (constructor `fix_virtual_column`), **default
OFF** so every frozen baseline and checkpoint stays bit-identical, per the
standing rule that new feature flags are off by default. Old checkpoints still
load (the kwarg defaults).

**Measurement.** X01 (`scripts/run_x01_virtual_column_fix.sh`) runs both arms
fresh over 3 paired seeds on FSeval with one factor changed. Results in
`reports/X01_VIRTUAL_COLUMN_AUDIT.md`. The frozen E00 baseline is deliberately
*not* reused as the control arm, so the comparison is paired rather than
historical.

## What the audit says about the earlier conclusions

| packet | affected | conclusion after audit |
|---|---|---|
| G00 | padding note only | unchanged |
| G01 | **yes, D1** | NO-GO, now on stronger evidence |
| N-A1 | no (field renamed for clarity) | unchanged |
| N-B1 | yes, D3 | unchanged in direction, figure corrected |
| N-C1 | yes, D2 | headline results unchanged |

All five packets were re-run end to end after the fixes; the JSON artifacts and
W&B runs in `vasot-repro` are the post-fix versions.
