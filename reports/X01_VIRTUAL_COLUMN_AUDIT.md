# X01 — Paired impact of the virtual-column off-by-one

Date: 2026-11-04. Follows the defect D5 in `reports/CODE_AUDIT_2026-11-04.md`.
**Verdict: the defect is real and proven, but it has no reliable effect on
FSeval MoF at 3 paired seeds. Mean paired delta +0.9 pp MoF with a per-seed
spread of ±6 pp in both directions.** The interesting result is *why* that is.

Script: `reproductions/tssl/scripts/run_x01_virtual_column_fix.sh`.
Runs: `X01a_vcol_published_FSeval_s{0,1,2}` (control),
`X01b_vcol_fixed_FSeval_s{0,1,2}` (treatment). W&B project `vasot-repro`,
tag `X01`.

## 1. The defect

`train.py` prepends the virtual alignment row but appends the virtual column,
while `align_ot` returns `T[:, 1:, 1:]`. The dropped column is a real frame of
video Y and the virtual column survives in the pseudo-label, so column `c` of
`opt_codes` is Y frame `c+1` while column `c` of the prediction `codes` is Y
frame `c`. Aligning a video with itself, where the answer is exactly the
identity, shows it directly:

| padding | mean argmax offset | rows on the diagonal |
|---|---:|---:|
| published (`train.py`) | **−0.953 frames** | 0.000 |
| virtual column prepended | 0.000 frames | 1.000 |

Locked by `scripts/test_diagnostics.py :: "solver: virtual column placement"`.

## 2. Paired training result

Both arms trained fresh, same config as `E00_baseline_FSeval_s1`, one factor
changed. The control arm at seed 1 reproduces the frozen baseline exactly
(0.4945 vs E00's 0.49446), which confirms the flag's default path is unchanged.

| seed | MoF published | MoF fixed | ΔMoF | ΔF1 | ΔmIoU |
|---|---:|---:|---:|---:|---:|
| 0 | 0.4510 | 0.5183 | **+0.0673** | +0.0709 | +0.0700 |
| 1 | 0.4945 | 0.5095 | +0.0151 | +0.0085 | +0.0168 |
| 2 | 0.5266 | 0.4700 | **−0.0566** | −0.0835 | −0.0622 |
| **mean** | 0.4907 | 0.4993 | **+0.0086** | −0.0014 | +0.0082 |
| std (arm) | 0.0380 | 0.0257 | paired std 0.0622 | 0.0777 | 0.0665 |

The sign flips across seeds. The paired mean (+0.86 pp MoF) is an order of
magnitude smaller than the paired standard deviation (6.2 pp), and the F1 mean
is slightly *negative*. Validation MoF goes the other way (0.5793 published vs
0.5737 fixed). **No improvement is established.** Seed 0 alone looked like a
+6.7 pp win and would have been a false positive had the sweep stopped there —
which is why all three seeds were run before any claim.

## 3. The informative part: why a one-frame shift does not matter

A one-frame offset out of 256 sampled frames is a 0.4% temporal error in the
alignment pseudo-label. Three independent measurements from the same session say
the model never uses frame-level precision in that signal:

- **G01:** separating same-phase from distant-phase frames of the same action is
  near chance for the embedding — AUROC 0.571 on FSeval.
- **N-B1 §A:** transport mass inside the correct action is only 1.21× better
  than uniform within that action; mean |Δphase| 0.324 vs 0.361 for uniform.
- **N-B1 §C:** a 2% embedding perturbation already moves 29% of top-1
  assignments by a mean of 2.7 frames — far more than the 1 frame at issue.

So the defect is swamped by the model's own insensitivity. That is a statement
about VASOT, not an excuse for the bug: **the alignment branch is operating at
a temporal resolution where ±1 frame is noise**, which bounds how much any
method that refines frame-level correspondence can gain on this setup.

## 4. What was done with the fix

Kept, behind `--fix-virtual-column`, **default OFF**. Reasons: the published
padding is what every frozen checkpoint and every E/G/N-series comparison was
produced with, and turning the flag on by default would silently invalidate
them. The flag is correct code available for any future run that wants it;
nothing in this session's evidence justifies making it the default.

Limitations: 3 seeds, one dataset (FSeval), one config, 30 epochs. The spread
within a single arm (0.451–0.527 MoF) means this protocol cannot resolve effects
smaller than roughly 5 pp; a claim either way would need ~10+ seeds. DA and YTI
were not tested. It remains possible that the fix matters on a dataset or
configuration where alignment precision carries more weight.
