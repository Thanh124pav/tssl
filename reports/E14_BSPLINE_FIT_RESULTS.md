# E14 — B-spline fit-quality and sensitivity results

Date: 2026-10-29. Fixed FSeval seed-1 baseline checkpoint, 20 deterministic
videos of 256 sampled frames. These are fitting diagnostics, not training
results.

## Capacity summary (cubic, tol=0.05)

| Space | kmax | Normalized RMSE | Mean max error | Hit kmax | Meet tol |
|---|---:|---:|---:|---:|---:|
| frame X | 32 | 0.399 | 0.732 | 100% | 0% |
| frame X | 64 | 0.345 | 0.633 | 100% | 0% |
| frame X | 128 | 0.253 | 0.505 | 100% | 0% |
| frame X | 192 | 0.150 | 0.378 | 100% | 0% |
| frame X | 256 | ~0.000 | ~0.000 | 100% | 100% |
| action logits | 32 | 0.508 | 0.862 | 100% | 0% |
| action logits | 64 | 0.426 | 0.733 | 100% | 0% |
| action logits | 192 | 0.165 | 0.343 | 100% | 0% |
| action logits | 256 | 0.0002 | 0.0017 | 95% | 100% |
| action posterior | 32 | 0.448 | 0.930 | 100% | 0% |
| action posterior | 64 | 0.369 | 0.871 | 100% | 0% |
| action posterior | 192 | 0.099 | 0.318 | 100% | 0% |
| action posterior | 256 | 0.020 | 0.081 | 0% | 15% |

Degree 1/2/3 at kmax 32 are all invalid. Changing tol among
0.025/0.05/0.10 does not change the cubic kmax-32 solution because capacity,
not the stopping threshold, is active. No tested configuration passes the
pre-registered valid-fit gate.

## Interpretation

The E01/E10a global curves were underfit: they used kmax 32, saturated on
every video and reconstructed neither X nor action logits accurately. Their
negative downstream metrics are valid smoothing outcomes, but do **not**
establish that a well-fitted B-spline representation fails.

X/logits meet tol only at roughly one control point per frame, where the
curve becomes an expensive near-identity interpolant and loses the intended
compression/smoothing. Action posterior is more compressible, but the current
simple-knot fitter often stops around 237 controls without meeting max-error
tol. Cubic simple knots impose C2 continuity; sharp action transitions need
repeated knots (lower continuity) or independent segment curves. The fitter's
minimum-gap rule explicitly prevents such repeated knots.

Knots do concentrate near boundaries enough to give high boundary coverage
as capacity grows, but precision falls (approximately 0.58 at X/k32 to 0.42
at X/k256): adding enough knots covers every boundary largely because it
covers nearly every frame. Coverage alone is not semantic discovery.

**E14 task training: none graduates.** E15 will first test whether
boundary-aware repeated/piecewise knots achieve a valid fit with substantially
fewer than 256 controls. Only then may a downstream curve comparison run.

Raw grid: `reproductions/tssl/artifacts/diagnostics/e14_bspline_fit/*.json`.
Protocol: `reports/E14_BSPLINE_FIT_PROTOCOL.md`.
