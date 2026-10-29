# E15a — Boundary-aware B-spline fit ceiling

Date: 2026-10-29. Fixed FSeval seed-1 checkpoint, 20 videos. This packet
tests independent cubic splines inside action segments before any model
training. Independent segments are at least as flexible at a transition as
repeating a cubic knot to remove cross-boundary smoothness.

| Curve space | Boundary source | Mean segments | Controls (ratio 0.25 / 0.5) | Normalized RMSE | Meet tol |
|---|---|---:|---:|---:|---:|
| X | GT oracle | 18.1 | 88 / 137 | 0.326 / 0.249 | 0% / 0% |
| X | ASOT | 11.8 | 78 / 136 | 0.332 / 0.248 | 0% / 0% |
| X | posterior changes | 36.9 | 152 / 175 | 0.242 / 0.201 | 0% / 0% |
| action posterior | GT oracle | 18.1 | 88 / 137 | 0.361 / 0.256 | 0% / 0% |
| action posterior | ASOT | 11.8 | 78 / 136 | 0.358 / 0.239 | 0% / 0% |
| action posterior | posterior changes | 36.9 | 152 / 175 | 0.217 / 0.169 | 0% / 0% |

ASOT pseudo-boundaries have mean precision 0.71 but cover only 0.43 of GT
boundaries within three sampled frames: they under-segment. Posterior argmax
changes cover 0.84 but have precision 0.50 and produce about twice the GT
segment count: they over-segment. Neither source passes fit validity.

Most importantly, even the GT-oracle segmentation fails. Removing continuity
across true boundaries is insufficient because the raw frame/action
trajectories remain highly variable inside actions. Therefore the current
least-squares/interpolation-tolerance fitter has no valid boundary-aware
configuration to train. **Training gate: STOP**, rather than reporting a task
failure from another underfit curve.

The next mathematically sound fitter should optimize a reconstruction–roughness
Pareto objective (smoothing spline) or fit phase-normalized within-segment
residuals, and validate on held-out frames. Exact max-error fitting drives the
current model toward one control per frame and defeats smoothing. E16 should
start with linear interpolation as a control; B-spline resampling graduates
only after it beats that control on held-out reconstruction at comparable
curvature/capacity.

Raw diagnostics:
`reproductions/tssl/artifacts/diagnostics/e15_boundary_fit/*.json`.
Implementation: `eval_boundary_aware_bspline_fit.py` and
`run_e15_boundary_fit_ceiling.sh`.
