# TSSL: reproduction, limitations, and continuous-curve experiments

**Audit date:** 2026-10-27  
**Branch:** `tssl`  
**Target paper:** Ali et al., *VAOT/VASOT*, ICCV 2025 (arXiv:2503.16832)  
**Hardware observed in completed logs:** NVIDIA H100 80 GB

## Executive summary

The paper reproduction is substantial but incomplete. The paper-hyperparameter
reproduction covers 50 Salads Mid, 50 Salads Eval, and all ten Breakfast
activities. 50 Salads Mid is reproduced closely; FSeval reproduces F1/mIoU but
not the paper's MoF; Breakfast exposes a conflict between Table 5 (`r=0.02`) and
the authors' public script (`r=0.04`). YTI, Desktop Assembly, and the VAOT
alignment tables remain unverified because their required data/code path was not
available in this workspace.

Three limitations were instrumented: the segmentation transport plan is nearly
one-hot, learned embeddings are boundary-blind, and frame-index distance is only
moderately correlated with semantic distance. An arc-length continuous
reparameterisation (E1) was coded and run. A cubic adaptive B-spline retrofit,
Gaussian smoothing controls, and a combined arc+B-spline variant were coded,
but their planned sweep did not finish because the compute allocation ended.

The overnight status is therefore **20 completed runs out of 80 planned**, plus
four B-spline runs that started but were interrupted before producing test
metrics. Conclusions below distinguish completed evidence from proposals.

## 1. What was reproduced

Paper Table 5 hyperparameters were used (`r=0.02`; seed 0 for this table):

| Dataset | Paper MoF/F1/mIoU | Reproduction MoF/F1/mIoU | Assessment |
|---|---:|---:|---|
| 50 Salads Mid, K=19, 100 ep | 47.2 / 41.3 / 26.1 | 46.73 / 44.75 / 30.05 | reproduced; F1/mIoU exceed paper |
| 50 Salads Eval, K=12, 30 ep | 60.6 / 57.4 / 34.5 | 48.83 / 55.32 / 33.17 | F1/mIoU close; MoF gap −11.77 pp |
| Breakfast, 10 activities, 50 ep mean | 57.5 / 39.0 / 18.8 | 47.69 / 36.60 / 19.26 | mixed; mIoU matches, MoF does not |

Breakfast's public `run_bf.sh` uses `r=0.04`, contrary to Table 5's `r=0.02`.
The public-script setting gives a higher MoF and is closer to the published
aggregate. Both configurations should be reported rather than selecting the
more favorable one after the fact.

Not completed: YTI, Desktop Assembly, and VAOT alignment. The detailed data and
code blockers are recorded in `reproductions/tssl/REPRODUCTION_STATUS.md`.

## 2. Measured limitations

| Limitation | Probe | FS Mid | FSeval | Interpretation |
|---|---|---:|---:|---|
| A. Peaky transport supervision | normalized row entropy of T* | 0.077 | 0.055 | only 5–8% of uniform entropy; effectively top-1 supervision |
| B. Boundary-blind embeddings | boundary z-score of `||ΔX||` | 0.026 | 0.047 | GT boundaries are almost indistinguishable from interiors |
| C. Index prior is semantically weak | corr(index distance, different GT label) | 0.375 | 0.384 | temporal order helps, but sampled index is a loose semantic proxy |

These measurements support the motivation for changing the representation, but
they do not prove that a B-spline is the unique or best remedy. In particular,
smoothing can erase boundary sharpness and make cosine costs degenerate. The
implementation therefore uses a residual blend and includes Gaussian smoothing
as a control.

## 3. Continuous-curve code and completed experiments

Implemented variants:

- `E1_arc`: parameterise each embedding sequence by cumulative arc length,
  resample uniformly in that coordinate, and inverse-warp predictions for
  evaluation.
- `E2_bsx_*`: adaptive cubic B-spline fitted in embedding space with residual
  blends 0.1 or 0.3.
- `E2_bsl_b03`: adaptive cubic B-spline in action-logit space.
- `E2d_gauss{2,5}`: Gaussian temporal smoothing controls.
- `E3_arc_bsx`: arc-length reparameterisation plus B-spline smoothing.

The complete two-seed evidence currently available is baseline versus E1:

| Dataset | Baseline MoF/F1/mIoU | E1 arc MoF/F1/mIoU | E1 change |
|---|---:|---:|---:|
| FSeval | 49.00 / 56.07 / 33.81 | 47.07 / 52.27 / 32.12 | −1.93 / −3.80 / −1.69 pp |
| FS Mid | 43.72 / 43.18 / 27.88 | 45.23 / 44.12 / 28.58 | +1.51 / +0.94 / +0.70 pp |
| Breakfast cereals | 44.65 / 39.07 / 25.77 | 43.99 / 37.51 / 24.53 | −0.66 / −1.56 / −1.24 pp |
| Breakfast coffee | 28.84 / 16.19 / 14.96 | 27.17 / 14.50 / 13.94 | −1.67 / −1.69 / −1.02 pp |
| Breakfast scrambledegg | 47.05 / 30.10 / 12.24 | 46.96 / 30.32 / 12.72 | −0.09 / +0.22 / +0.48 pp |

Values are means over seeds 0 and 1. Seed ranges are large in several cells
(for example baseline FS Mid MoF range 5.07 pp and E1 FSeval MoF range 5.62 pp),
so one-seed gains reported earlier were not robust. E1 helps FS Mid but does not
generalise across the five tested datasets. It should not replace the baseline
without more seeds or a better gating criterion.

## 4. Sweep audit

- Planned: 8 variants × 5 datasets × 2 seeds = 80 runs.
- Completed: baseline and E1 for all five datasets and two seeds = 20 runs.
- Interrupted: the four first `E2_bsx_b03` jobs (FSeval/FSmid, seeds 0/1).
  Their logs end during training and contain no `test_*_full` metrics.
- Never started: the remaining 56 runs.
- Post-training diagnostics for the 80-run matrix were not generated because
  the sweep did not reach finalisation.

Raw completed metrics and run IDs are in
`reproductions/tssl/artifacts/overnight_partial_2026-10-27.tsv`. The source logs
remain local because the repository intentionally ignores `logs/`, checkpoints,
datasets, and W&B caches.

## 5. What the evidence supports

1. The public implementation can reproduce FS Mid and much of FSeval, but not
   the published FSeval MoF or Breakfast MoF under the paper's stated settings.
2. The three diagnostic limitations are real in the measured checkpoints.
3. Arc-length reparameterisation is dataset-dependent: positive on FS Mid,
   negative on FSeval and two of three Breakfast activities.
4. No empirical claim about B-spline superiority is justified yet. Its code
   path exists and passes CPU smoke/gradient tests, but its GPU experiments were
   interrupted.

## 6. Reproduction and continuation

From `reproductions/tssl/repos/VASOT` after preparing the upstream datasets and
environment:

```bash
# CPU verification of the continuous-curve primitives
pytest -q tests/test_continuous_curve.py

# Full planned sweep (requires an active GPU allocation)
MAX_PAR=4 bash overnight_sweep.sh

# Aggregate whatever completed successfully
python src/aggregate_results.py \
  /mnt/data/thanhpv/acvcl/reproductions/tssl/logs/overnight/_status.tsv
```

The dispatcher is resumable: it preserves `_status.tsv`, skips keys with a
successful (`exit=0`) row, and retries interrupted or failed cells.
