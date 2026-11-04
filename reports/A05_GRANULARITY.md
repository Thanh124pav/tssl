# A05 — How does segmentation granularity K affect training?

Date: 2026-11-04. FSeval (12 GT eval actions, 256 sampled frames per video),
K ∈ {12, 24, 48, 128, 256, 512} × 2 seeds, 12 training runs.
Script `reproductions/tssl/scripts/run_a05_k_sweep.sh`, W&B tag `A05`.
Per-checkpoint analysis by `eval_a04_segment_confidence.py` (seed 0).

**Verdict: MoF collapses as K grows, but almost all of the collapse is the
one-to-one evaluation protocol, not the representation. At K = 48 the
representation is better and the calibration is far better, while MoF is worse.**

## 1. The protocol warning, measured rather than assumed

`metrics.eval_mof` assigns clusters to GT classes **one-to-one**
(`scipy.linear_sum_assignment` over the contingency table). With K > #GT, at
most #GT clusters can ever be matched and every frame in an unmatched cluster
counts as an error. The sweep therefore reports MoF next to a many-to-one
(purity) accuracy and the fraction of frames stranded in unmatched clusters.

## 2. Result

| K | MoF s0 | MoF s1 | mean MoF | 1-to-1 acc | **many-to-1 acc** | ECE | mean conf | frames in unmatched clusters |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.4510 | 0.4945 | **0.4727** | 0.428 | 0.586 | 0.490 | 0.918 | 0.000 |
| 24 | 0.3897 | 0.3919 | 0.3908 | 0.381 | **0.672** | 0.382 | 0.763 | 0.246 |
| 48 | 0.3664 | 0.3387 | 0.3526 | 0.358 | **0.674** | **0.134** | 0.491 | 0.411 |
| 128 | 0.0894 | 0.0971 | 0.0933 | 0.088 | 0.582 | 0.027 | 0.061 | 0.828 |
| 256 | 0.0553 | 0.0632 | 0.0592 | 0.057 | 0.581 | 0.028 | 0.030 | 0.901 |
| 512 | 0.0426 | 0.0438 | 0.0432 | — | — | — | — | — |

Three things move in different directions, which is the whole point:

- **MoF falls monotonically**, by 10× from K = 12 to K = 512.
- **Many-to-one accuracy rises then plateaus**: 0.586 → 0.672 → 0.674, then
  settles at 0.58 for K ≥ 128. Finer clusters describe the data *better* up to
  K ≈ 48 and never get much worse.
- **The stranded-frame fraction tracks the MoF collapse almost exactly**: 0.00,
  0.25, 0.41, 0.83, 0.90. Every cluster is used at every K (255/256 at K = 256),
  so the clusters are not dying; the matching simply cannot spend them.

## 3. Calibration improves sharply with K

Mean `T_xa` top-1 confidence falls 0.918 → 0.491 → 0.061 and ECE falls
0.490 → 0.134 → 0.027. The severe overconfidence A04 found at K = 12 — 79% of
frames claiming 0.99 and being right half the time — is a property of a coarse
codebook, not of the OT solver. With more clusters the posterior spreads over
genuinely competing neighbours and becomes close to honest.

K = 48 is the interesting operating point: the best purity in the sweep, a 3.7×
better ECE than the baseline, and a MoF that looks 12 pp worse.

## 4. Reading

1. Increasing granularity does not break training. It improves the
   representation modestly and the calibration substantially.
2. The MoF drop at large K is mostly an artefact of one-to-one matching. Any
   paper-style comparison that varies K without reporting a many-to-one number
   or the stranded-frame fraction is measuring the protocol.
3. Combined with A07, this bounds what MoF on this benchmark can tell us: it is
   maximised near K = #GT with a representation that encodes position, which is
   not the same as the setting that represents actions best.
4. If a downstream use needs calibrated frame posteriors rather than a
   MoF number, K ≈ 48 with the same training is measurably the better model.

Limitations: FSeval only, 2 seeds per K, 30 epochs, K = 512 analysed for MoF
only. The analysis arm uses seed 0 checkpoints and the sampled-frame protocol of
A04, not the official full-length test protocol, so the absolute accuracies are
not directly comparable with published MoF; the trends across K are.
