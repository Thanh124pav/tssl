# A07 — How much of VASOT's segmentation score needs VASOT?

Date: 2026-11-04. Frozen checkpoint `E00_baseline_FSeval_s1`, 20 FSeval videos.
Script `reproductions/tssl/scripts/eval_a07_trivial_baselines.py`,
artifact `artifacts/diagnostics/a07/FSeval_baseline_s1.json`,
W&B run `A07_FSeval_baseline_s1`.

**Question.** A02–A06 say the representation is weak: within-action phase is
near chance, GT actions overlap (nearest-centroid / within-scatter = 0.42), no
conic or quadric structure, and the one thing training encodes strongly is
absolute time. The published scores are nonetheless competitive. Which
component produces the score?

## 1. Protocol

Every arm is scored identically: frames pooled over 20 videos at 256 uniformly
sampled frames each, a one-to-one Hungarian cluster→GT assignment
(`metrics.pred_to_gt_match`, exactly what `eval_mof` uses), plus a many-to-one
per-cluster-majority reading. Per-video numbers fit the assignment separately
per video, which is more generous to every arm.

This is **not** the official full-length test protocol: the same checkpoint
scores 0.494 there and 0.456 here. Only the comparison between arms is
meaningful, and all arms share the protocol.

## 2. Result

| arm | what it may use | 1-to-1 | many-to-1 | per-video 1-to-1 |
|---|---|---:|---:|---:|
| `uniform` | position only: frame `i` → cluster `⌊i/N·K⌋`. No model, no features, no training | 0.295 | 0.482 | **0.458** |
| `kmeans_raw` | raw features, no time, no training | 0.225 | 0.393 | 0.330 |
| `kmeans_trained` | trained embeddings, no OT, no prior | 0.351 | 0.563 | 0.435 |
| `untrained_ot` | full OT decoder + prior, untrained encoder | 0.284 | 0.412 | **0.464** |
| `vasot_rho0` | full pipeline, prior off at decode | 0.463 | 0.625 | 0.549 |
| **`vasot`** | full pipeline | **0.456** | **0.618** | **0.560** |

### 2.1 A position-only baseline reaches 82% of the score

`uniform` uses no features at all — it cuts every video into K equal temporal
blocks in order. Per-video it scores 0.458 against VASOT's 0.560. The decoder
with an untrained encoder scores 0.464, which is the same thing by another
route: the temporal prior inside the OT decoder is itself a position baseline.

This is the main answer. The one property the representation has (A03: training
makes the embedding a slowly drifting function of absolute time) is exactly the
property this benchmark rewards. The model is well matched to the metric, which
is not the same as the model understanding actions.

### 2.2 Contribution breakdown

```
0.225   k-means on raw features        (appearance alone)
+0.126  training the encoder
+0.105  the OT decoder and its balanced assignment
──────
0.456   VASOT                          (position alone: 0.295)
```

Training and the decoder each add real, comparable value. VASOT beats the
position-only baseline by +0.161 pooled and +0.102 per-video. The method is not
vacuous — but the headroom implied by "≈0.5 MoF" is much smaller than the number
suggests, because a trivial baseline already holds most of it.

### 2.3 The temporal prior is not needed at decode time

`vasot_rho0` (0.463) matches `vasot` (0.456). Switching off `rho_seg` when
decoding costs nothing, because the prior's effect has already been absorbed
into the representation during training. This is the decode-time counterpart of
A03's lag profile, and it is consistent with E07b, where training with `rho=0`
also did not collapse.

## 3. The metric's granularity is fixed in the model's favour

The baseline sets K = 12 = the number of FSeval eval actions, so the model never
has to choose a granularity, and the one-to-one assignment can in principle use
every cluster. A05 shows what happens when that gift is withdrawn — see
`reports/A05_GRANULARITY.md`: at K = 48 the representation is measurably
*better* (many-to-one 0.674 vs 0.586) and far better calibrated (ECE 0.134 vs
0.490), while MoF *falls* from 0.473 to 0.353, entirely because 41% of frames
land in clusters the one-to-one matching cannot use.

So the reported metric rewards a particular K, not representation quality.

## 4. Reading

The score is high for three reasons, in order of size: the benchmark is largely
solvable from temporal position, which is the one thing the representation
encodes; the metric fixes K at the number of GT classes and matches one-to-one;
and VASOT does add a real but moderate amount on top of both.

A practical consequence for the project: improvements measured only as MoF on
this setup can be bought by injecting more positional prior without improving
the representation at all. A01 is a worked example — temporal prototype
initialisation gains +4.6 pp MoF (p = 0.008) while its many-to-one accuracy
moves +0.6 pp (p = 0.46). Any future claim should report the many-to-one or
another representation-side number next to MoF.

Limitations: FSeval only, one checkpoint for the VASOT arms, 20 videos, and a
sampled-frame protocol rather than the official full-length one. DA and YTI are
untested and cheap to add. `kmeans_raw` and `kmeans_trained` are fitted on the
pooled evaluation frames, which is generous to them, and they still lose.
