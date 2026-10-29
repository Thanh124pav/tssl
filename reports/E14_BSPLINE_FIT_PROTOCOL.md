# E14 — B-spline fit-quality and sensitivity protocol

Date: 2026-10-29. Downstream degradation alone is not sufficient to call a
B-spline representation a failure. E14 first verifies that the fitted curve
actually approximates the trajectory under its declared tolerance.

## Fixed fit audit

Use 20 deterministic FSeval videos and a fixed trained baseline checkpoint.
Audit normalized frame embedding space (`X`), action-logit space (`X A^T`),
and the action-posterior trajectory `softmax(X A^T / temp)`. The posterior
extension is registered before its results because the user hypothesis is a
curve in action space, whereas E10a fitted unnormalized logits. Ground-truth
boundaries are evaluation-only and never affect knot placement.

One-factor-at-a-time grid around cubic/tol=0.05/kmax=32:

- degree: 1, 2, 3;
- tolerance: 0.025, 0.05, 0.10;
- maximum control points: 16, 32, 64.

This is seven unique configurations per representation space. Blend is not a
fit-quality parameter; it is audited separately as the downstream injection
strength. Existing blend 0.3/1.0 results are retained, and at most one weaker
blend (0.1) may graduate after fit validity is established.

**Capacity-recovery rule (registered after the first grid, before recovery
results):** if every `kmax<=64` configuration saturates and zero videos meet
tolerance, hold degree=3/tol=0.05 fixed and test `kmax={96,128,192,256}`. This
is a diagnostic escalation, not a task-metric hyperparameter search. It asks
whether the current adaptive fitter can ever satisfy its declared tolerance
and how close its required capacity is to the original 256 discrete frames.

For action posterior, test cubic/tol=0.05 at
`kmax={32,64,96,128,192,256}`, plus degree 1 at kmax 64. This distinguishes
piecewise-probability structure from the noisier unnormalized logits.

## Valid-fit gate

A configuration is a valid curve fit only if all hold:

1. finite-fit fraction = 1.0 and endpoint max error <= 1e-4;
2. at least 90% of videos meet the requested maximum-error tolerance;
3. `kmax` saturation fraction <= 25% (capacity margin remains);
4. normalized RMSE, q90/max error, control count and runtime are reported.

Knot-boundary precision/coverage and curvature are mechanism diagnostics, not
fit-validity criteria. A curve may fit accurately without discovering semantic
boundaries. Conversely, boundary concentration cannot excuse a poor fit.

## Downstream decision

Only valid-fit configurations can enter paired training. At most two enter:
the best valid action-space fit and one capacity/smoothness contrast. A finite,
valid curve whose paired downstream metrics fail is a representation/task
failure. Failure to meet the gate is recorded as a fitting/capacity failure
and does not support a general claim against B-splines.

Diagnostic implementation:
`reproductions/tssl/scripts/eval_bspline_fit_quality.py`.
