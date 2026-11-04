# A01 — Does the action-prototype initialisation change what is learned?

Date: 2026-11-04. FSeval, 3 paired seeds per arm, 15 training runs.
Script `reproductions/tssl/scripts/run_a01_cluster_init.sh`, flag
`--cluster-init` (default `random`, so every frozen baseline is untouched).
W&B tag `A01`.

**Verdict: yes, and one arm survives the paired-seed test cleanly — but it
improves the cluster-to-action assignment, not the representation.**

## 1. Motivation

G00 measured that the learned prototypes stay at cosine 0.872 (FSeval) from
their own random initialisation, while two different random codebooks sit at
0.24–0.27 from each other. If the codebook barely moves, the initialisation
largely *is* the final codebook, and changing it should change the outcome.
Arms (none uses labels): `random` (published), `orthogonal`, `data` (sampled
frame embeddings), `temporal` (mean embedding of each of K equal relative-time
bins), `kmeans`.

## 2. Result

| init | s0 | s1 | s2 | mean MoF | paired Δ | per-seed Δ | t | p |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `random` | 0.4510 | 0.4945 | 0.5266 | 0.4907 | — | — | — | — |
| `orthogonal` | 0.5718 | 0.5080 | 0.5114 | 0.5304 | +0.0397 | +0.121, +0.014, −0.015 | 0.96 | 0.44 |
| `data` | 0.4892 | 0.5456 | 0.5311 | 0.5219 | +0.0313 | +0.038, +0.051, +0.005 | 2.25 | 0.15 |
| **`temporal`** | 0.4903 | 0.5381 | 0.5801 | **0.5362** | **+0.0455** | **+0.039, +0.044, +0.054** | **10.91** | **0.008** |
| `kmeans` | 0.5868 | 0.5331 | 0.4835 | 0.5345 | +0.0438 | +0.136, +0.039, −0.043 | 0.85 | 0.49 |

`orthogonal` and `kmeans` have the eye-catching means and fail the test: their
per-seed deltas change sign, exactly the trap X01 fell into at seed 0. Only
`temporal` is consistent — three of three seeds positive, paired std 0.0072,
which is 8.6× tighter than X01's 0.0622. F1 and mIoU agree (`temporal` 0.547 /
0.311 vs `random` 0.514 / 0.286).

## 3. What the gain actually is

A04 re-run on the `random` and `temporal` checkpoints, 3 seeds each:

| | one-to-one accuracy | many-to-one accuracy | ECE |
|---|---:|---:|---:|
| `random` | 0.4681 | 0.6155 | 0.448 |
| `temporal` | 0.5243 | 0.6215 | 0.386 |
| **paired Δ** | **+0.0563** (t = 10.51, p = 0.009) | **+0.0060** (t = 0.91, p = 0.46) | −0.062 |

The one-to-one number moves and the many-to-one number does not. Purity — how
well the clusters actually separate GT actions — is statistically unchanged.
**Temporal initialisation does not learn a better representation; it arranges
the codebook so the one-to-one Hungarian assignment lands better.**

That is consistent with the mechanism the other packets established: A03 shows
the embedding is mainly a slow function of absolute time, A07 shows a
position-only baseline reaches 82% of VASOT's per-video score, and `temporal`
init seeds the prototypes along exactly that coordinate. It injects more of the
positional prior the benchmark already rewards.

## 4. Reading

Three findings, in order of confidence:

1. Initialisation matters, confirming G00's prediction: the codebook does not
   travel far enough for the optimiser to undo a bad start.
2. `temporal` init is a real, reproducible +4.6 pp MoF for one line of code.
3. It is not a representation improvement, so it should be reported as what it
   is. Any MoF gain on this benchmark needs a representation-side number beside
   it — A07 §4.

Limitations: 3 seeds, FSeval, K = 12, 30 epochs, one config; with a paired std
of 0.0072 the `temporal` effect is resolvable at n = 3, but `data` (p = 0.15)
would need roughly 10 seeds to settle. The arms differ in how much prior
knowledge they inject, not only in geometry, so this is not a clean test of
"better geometry"; `temporal` in particular encodes the same ordered-and-evenly
spread assumption as the temporal prior itself. DA and YTI are untested.
