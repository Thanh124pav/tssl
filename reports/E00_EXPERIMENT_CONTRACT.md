# E00 — Frozen Experimental Contract for the E/Mechanism Phase

**Date:** 2026-10-29 (backfilled: this contract was followed implicitly by E01–E05;
this document is the written form so the record is reconstructible without
grepping train.py args.)

## 1. Development ladder

Experiments climb the ladder only if the lower rung passes the scale gate.

| Rung | Dataset | Purpose |
|---|---|---|
| 0 | FSeval (50 Salads Eval) | Screening — fastest, 2 min baseline on 1× H100 |
| 1 | FS Mid | Variance under different split structure |
| 2 | Desktop Assembly | Dense label budget, long-tail actions |
| 3 | YTI (5 activities, aggregated) | Noisy frames + missing-video regime |

Scaling to DA/YTI without passing FSeval first is explicitly prohibited: these
datasets dominate the compute budget and are the research deliverables, not
exploration vehicles.

## 2. Seed policy

- Screening (any variant on FSeval): **paired seeds {0, 1}**. Both must finish.
- Scale gate decision: mean MoF over the 2 seeds, with the per-seed delta
  reported alongside so single-seed outliers are visible.
- Only variants that pass the scale gate graduate to a 3rd seed (seed 2) on
  FSeval, and only on the rung they currently occupy.
- Final claim (any dataset): mean ± std over 3 seeds, same seed set as the
  baseline it is compared against.
- Never report mean-of-2 as if it were mean-of-3; the N is a column in every
  summary table.

## 3. Scale gate

A variant advances to the next rung iff **both** hold:

1. Mean MoF over the paired seed set is at least **1.0 percentage point**
   above the baseline mean on the same rung, **or** a diagnostic metric
   (T_entropy_norm, boundary_z, idx_vs_sem_corr) moves by at least one
   pre-registered threshold and task MoF loss is below 1.0 pp.
2. No seed collapses more than 2 pp below baseline. A variant that improves
   mean but kills one seed is a bad bet to scale.

Numbers are frozen before results are read. No post-hoc gate relaxation; a
missed gate is a documented negative result, not a trigger to retune.

## 4. Baseline config (frozen)

Baseline for FSeval is `reproductions/tssl/configs/vasot/baselines/fs_eval.yaml`,
hashed below. All E runs inherit this baseline and override only the single
mechanism under test.

Hash:

```
$ sha256sum reproductions/tssl/configs/vasot/baselines/fs_eval.yaml
```

(see `reports/E00_contract.lock` for the pinned hash and git commit.)

## 5. Compute budget

Per packet (E01, E02, …):

- At most **4 training runs** per interactive session (matches S00/S03 run cap
  in `IMPLEMENTATION.md`).
- No single run allowed to exceed **30 min** on FSeval without justification.
- No background training launched from a session Claude cannot see finish
  without a status file on disk; use `nohup` + status.json or a Slurm sbatch,
  not a shell `&`.
- Wandb logging mandatory for every training run (project `vasot-repro`),
  including seed and variant tags.

## 6. Stopping rule

A variant family (B-spline, Gaussian, Bézier, ε-knob, …) is retired from
further exploration once **either**:

- It fails the scale gate on FSeval with ≥ 2 seeds, **and** the diagnostic
  gate also fails (no interesting A/B/C movement), **or**
- Two consecutive hyperparameter sweeps inside the family both fail the gate.

Retirement is documented in the matching `reports/E0*_*.md` with per-seed
numbers and the explicit "scale: SKIP" line, as E05 does for B-spline,
Gaussian, Bézier and arc-length.

## 7. What this contract is *not*

- Not a performance target. The scale gate is a go/no-go on *expected improvement*,
  not a floor; a variant can fail the gate and still be a useful null.
- Not a license to delete negative results. Everything that was run gets a row
  in the E05 (and later E07) matrix, including failures and NaN runs.
- Not retroactive protocol tightening. Numbers already produced under the
  frozen baseline config are kept; config changes require a new run.
