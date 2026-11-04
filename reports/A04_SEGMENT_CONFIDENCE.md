# A04 — Confidence and accuracy of the frame → segment assignment

Date: 2026-11-04. Frozen checkpoint `E00_baseline_FSeval_s1`, 20 FSeval videos,
5120 unpadded frames, K = 12 clusters against 12 GT eval actions.

Script `reproductions/tssl/scripts/eval_a04_segment_confidence.py`,
artifact `artifacts/diagnostics/a04/FSeval_baseline_s1.json`,
W&B run `A04_FSeval_baseline_s1`.

## 1. Two protocols, two accuracies

| protocol | accuracy |
|---|---:|
| one-to-one Hungarian (`metrics.pred_to_gt_match`, what `eval_mof` uses) | **0.456** |
| many-to-one majority (purity) | **0.618** |

All 12 clusters are used and all 12 are matched, so no frames are lost to
unmatched clusters here (that becomes the issue in A05, at larger K). The 16-pp
gap between the protocols is pure *assignment* loss: several clusters want the
same GT action and the one-to-one constraint forces the losers onto a wrong
action. Both numbers are reported below; the one-to-one figure is the one
comparable with published MoF.

## 2. The OT posterior is severely overconfident

| confidence source | accuracy | mean confidence | overconfidence | ECE | MCE | AUROC(conf → correct) |
|---|---:|---:|---:|---:|---:|---:|
| `T_xa` posterior (drives the pseudo-label) | 0.456 | **0.915** | **+0.459** | **0.459** | 0.519 | 0.627 |
| softmax codes (what the CE predicts) | 0.456 | 0.674 | +0.218 | 0.226 | 0.433 | 0.558 |
| `T_xa` posterior vs many-to-one accuracy | 0.618 | 0.915 | +0.297 | 0.298 | 0.398 | 0.643 |

Reliability of the `T_xa` posterior:

| confidence bin | n | accuracy | mean confidence | gap |
|---|---:|---:|---:|---:|
| 0.2–0.3 | 42 | 0.095 | 0.269 | +0.174 |
| 0.3–0.4 | 101 | 0.158 | 0.359 | +0.200 |
| 0.4–0.5 | 134 | 0.261 | 0.449 | +0.188 |
| 0.5–0.6 | 184 | 0.310 | 0.545 | +0.235 |
| 0.6–0.7 | 127 | 0.323 | 0.651 | +0.328 |
| 0.7–0.8 | 179 | 0.313 | 0.754 | +0.441 |
| 0.8–0.9 | 287 | 0.338 | 0.857 | +0.519 |
| **0.9–1.0** | **4066** | **0.499** | **0.987** | **+0.488** |

**79% of all frames sit in the top confidence bin, where the posterior claims
0.987 and is right 0.499 of the time.** The expected calibration error of 0.459
is not a tail effect — it is the modal behaviour. This is the segmentation plan
that D01 called "peaky" and that G00 measured at top-1 mass 0.915 and effective
support 1.50 actions: that sharpness is now shown to be **unjustified**, not a
sign of a confident correct assignment.

The ranking information is weak but not absent: AUROC 0.627 for confidence
predicting its own correctness, with the most-confident decile at 0.617 accuracy
and the least-confident decile at 0.246. So confidence is usable for triage and
useless as a probability.

Note that the two confidences disagree in character: the OT posterior is far
sharper than the softmax codes it supervises (0.915 vs 0.674 mean). The
pseudo-label is more certain than the model it teaches, which is the wrong way
round for a self-training scheme and is a concrete, cheap target — temperature
or entropy control on `T_xa` before it is used as a target.

## 3. What actually predicts correctness: distance to a boundary

| |dist to GT boundary| | n | accuracy (1-to-1) | accuracy (many-to-1) | mean confidence | overconfidence |
|---|---:|---:|---:|---:|---:|
| 0 | 363 | 0.355 | 0.386 | 0.854 | +0.499 |
| 1–2 | 1356 | 0.350 | 0.429 | 0.871 | +0.521 |
| 3–5 | 1393 | 0.416 | 0.584 | 0.915 | +0.498 |
| 6–10 | 1154 | 0.512 | 0.782 | 0.951 | +0.439 |
| 11–20 | 609 | 0.647 | 0.844 | 0.971 | +0.324 |
| ≥21 | 245 | 0.673 | 0.865 | 0.943 | +0.270 |

Accuracy nearly doubles from a boundary frame (0.355) to a segment interior
(0.673), and more than doubles under the many-to-one protocol (0.386 → 0.865).
Confidence moves over that range by 0.09 — from 0.854 to 0.943 — i.e. **the
posterior barely tracks the single strongest predictor of its own correctness.**
Overconfidence is therefore worst exactly where errors concentrate: +0.52 near
boundaries against +0.27 in segment interiors.

G00 found the boundary effect in the *entropy* and called it "real but modest".
A04 shows why modest entropy still matters: a 0.09 confidence swing is
accompanied by a 0.48 accuracy swing.

By segment duration:

| duration | n | accuracy | many-to-one | mean confidence |
|---|---:|---:|---:|---:|
| < 16 frames | 2277 | 0.419 | 0.438 | 0.904 |
| 16–47 | 2385 | 0.471 | 0.753 | 0.924 |
| ≥ 48 | 458 | 0.563 | 0.808 | 0.927 |

Short segments are both harder and, under the many-to-one protocol, dramatically
harder (0.438 vs 0.808) — short segments are the ones whose frames get absorbed
into a neighbouring action's cluster.

## 4. Reading

1. The sharpness of `T_xa` is miscalibration, not confidence: 79% of frames at
   claimed 0.99, actual 0.50.
2. The pseudo-label is sharper than the student it supervises, which bakes that
   miscalibration into training.
3. Correctness is governed by boundary distance and segment duration, and
   confidence is nearly blind to both.
4. A cheap, well-targeted intervention follows directly: calibrate or
   temperature-control `T_xa` before using it as a target, and/or down-weight
   frames near predicted boundaries. This does not require any new theory, and
   unlike the Phase G/N proposals it addresses a defect that is measured rather
   than hypothesised. It has **not** been run — it is a proposal, and it would
   need the same paired-seed discipline X01 used, where a 5 pp effect is the
   resolution floor of this protocol.

Limitations: one checkpoint, one dataset, K = 12 with no unmatched clusters, and
GT boundaries from the eval granularity. The one-to-one protocol's 0.456 is a
frame accuracy pooled over videos, not the published MoF figure for this
checkpoint (0.494 on the full-length test split with its own sampling), so the
two should not be compared directly; only the internal comparisons in this
report are protocol-consistent.
