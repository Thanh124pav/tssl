# N-C1 — Nonlinear temporal-warp stress test

Date: 2026-11-04. Frozen checkpoint, no training.
**Verdict: a reproducible warp failure regime exists and is attributable to the
GW structure, not the diagonal prior — but a classic DTW baseline on the same
cost matrix absorbs it, which weakens rather than supports the novelty claim of
N-C2.**

Script: `reproductions/tssl/scripts/eval_nc1_warp_stress.py`.
Artifact: `reproductions/tssl/artifacts/diagnostics/nc1/FSeval_baseline_s1.json`.
W&B: `vasot-repro`, run `NC1_FSeval_baseline_s1`. FSeval baseline s1, 8 videos,
256 frames, alpha_train 0.3, radius_gw 0.04, rho_align 0.1.

> **Padding note (2026-11-04 audit).** Every `T_xy` in this report is solved
> with the virtual row **and** column prepended, so the index `align_ot` drops
> is the virtual one. The published `train.py` appends the virtual column
> instead, which shifts its alignment pseudo-label by one frame (see
> `reports/X01_VIRTUAL_COLUMN_AUDIT.md`). The checkpoints audited here were
> therefore *trained* under the published padding and *measured* under the
> corrected one; this matches the convention used by E09 and by
> `scripts/eval_transport_support.py`, so the numbers remain comparable with
> earlier packets.

## 1. Protocol

A video is warped in time by a monotone map; the correspondence is therefore
known exactly and action order is preserved by construction.

- **Interpolation policy:** linear interpolation of the *raw* features at float
  source positions, then the frozen MLP. Stated explicitly because it is the
  main confound, and controlled for in §3.
- **Warp families:** `piecewise` (5 knots, per-segment speed `exp(U(−s, s))`),
  `global` (a smooth power-law reparameterisation), `identity`, and
  `resample_only` (a half-frame shift — pure interpolation, no warp).
- **Severity → realised max/min speed ratio:** s = 0.5 → 2.12, s = 1.0 → 4.54,
  s = 2.0 → 21.13 (mean over the 8 videos).
- **Oracle plan:** row `i` of X one-hot onto the warped frame whose source
  position is closest to `i`, total mass 1. The solver's plan is rescaled to
  total mass 1 before being costed, since `align_ot` returns rows summing to 1.

The objective is decomposed into exactly the terms the repo uses:
`0.5·alpha·⟨C^X T C^Y, T⟩` (structural), `(1−alpha)·⟨1 − z_x z_yᵀ, T⟩`
(feature) and `(1−alpha)·⟨rho·|i/N − j/M|, T⟩` (diagonal prior), with the
repo's own conv filters.

## 2. The correct correspondence is charged for being warped

Cost of the **oracle** correspondence (lower is better; identity is the
reference point):

| variant | structural | feature | index prior | total |
|---|---:|---:|---:|---:|
| `identity` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| `resample_only` | 0.00011 | 0.04378 | 0.00041 | 0.04430 |
| `global` s0.25 | 0.01030 | 0.01410 | 0.00307 | 0.02747 |
| `global` s0.5 | 0.01728 | 0.01591 | 0.00580 | 0.03898 |
| `global` s1.0 | 0.02483 | 0.01977 | 0.01042 | 0.05503 |
| `global` s2.0 | 0.03078 | 0.02663 | 0.01735 | 0.07476 |
| `piecewise` s0.25 | 0.01754 | 0.01473 | 0.00220 | 0.03447 |
| `piecewise` s0.5 | 0.03134 | 0.01808 | 0.00439 | 0.05381 |
| `piecewise` s1.0 | 0.05087 | 0.02694 | 0.00852 | 0.08634 |
| `piecewise` s2.0 | **0.08914** | 0.05131 | 0.01507 | 0.15551 |

Three readings:

1. The structural term on the *correct* correspondence rises from 0 to 0.089 as
   the warp strengthens. This penalty is avoidable in principle — the
   correspondence is right — so it is a property of the cost, not of the data.
2. At matched severity the **nonlinear** warp is penalised 2.9× more than the
   global speed change (0.089 vs 0.031 at s = 2.0), while their feature terms
   differ by less than 2×. The penalty is specific to *local* speed variation.
3. The diagonal index prior is the smallest term throughout (≤ 0.015, 10% of
   the total at s = 2.0). **The warp penalty lives in the GW structure, not in
   `rho_align`.**

## 3. It is not an interpolation artifact

`resample_only` is the control: a half-frame shift resampled through the same
interpolator, no warp at all. It has the **largest feature term of any variant**
(0.0438) and a structural term of 0.0001 — three orders of magnitude below the
piecewise warps. Interpolation damages the features and leaves the structure
untouched; the structural penalty in §2 is therefore not an interpolation
artifact.

## 4. Where the solver actually breaks, and the baseline that does not

Mean alignment error in frames against the known map, plus the plan's effective
support:

| variant | solver, rho=0.1 | solver, rho=0 | **DTW** on the same cost | support |
|---|---:|---:|---:|---:|
| `identity` | 0.000 | 0.000 | 0.000 | 6.7 |
| `resample_only` | 0.501 | 0.502 | 0.500 | 9.8 |
| `global` s0.5 | 0.275 | 0.277 | 0.270 | 9.7 |
| `global` s1.0 | 0.337 | 0.340 | 0.306 | 11.4 |
| `global` s2.0 | 0.549 | 3.119 | 0.357 | 14.4 |
| `piecewise` s0.5 | 0.349 | 0.339 | 0.289 | 11.4 |
| `piecewise` s1.0 | **1.640** | 1.233 | 0.403 | 16.7 |
| `piecewise` s2.0 | **15.855** | 16.273 | 1.328 | 33.6 |

- **Failure regime identified:** piecewise warps with max/min speed ratio ≳ 4.5
  (s = 1.0). Error jumps 0.35 → 1.64 → 15.86 frames while the plan's support
  triples (11.4 → 33.6), i.e. the solver does not just mis-locate, it gives up
  on concentrating.
- **Removing the index prior does not fix it.** At `piecewise` s2.0, rho = 0
  gives 16.27 frames versus 16.27... versus 15.86 with the prior — marginally
  *worse*. (On `global` s2.0 removing the prior is much worse, 3.12 vs 0.55, so
  the prior is load-bearing elsewhere.) This independently confirms §2's
  decomposition: the failure is the GW structure's.
- **A classic DTW on the identical cost matrix degrades 12× less** (1.33 vs
  15.86 frames at s = 2.0) and never exceeds 1.33 frames anywhere in the grid.

A note on the oracle-vs-solver comparison: the oracle correspondence has lower
raw weighted FGW cost than the solver's own output in *every* variant, by a
near-constant 0.079–0.088. This is expected for an entropy-regularised
unbalanced objective — the diffuse plan buys entropy — and is reported so that
"the oracle is cheaper" is not mistaken for "the solver failed to optimise".

## 5. Exit criterion and the N-C2 gate

Exit criterion met: a reproducible warp regime where the correct correspondence
incurs avoidable structural distortion (§2), with the effect established as
coming from the GW structure rather than the diagonal prior (§2 decomposition
plus the rho = 0 ablation in §4), and separated from interpolation artifacts
(§3).

The N-C2 gate reads: "N-C1 establishes a structural warp failure not solved by
a simple existing alignment baseline, **or** provides a theory-motivated case
worth testing." The first clause **fails** — DTW, which is about as simple a
baseline as exists, solves it. Honest reading: the measured defect is real and
localised in VASOT's index-based GW structure, but it is the kind of defect
elastic alignment has handled since the 1970s. The actionable consequence is
*not* "build quotient-space reparameterised transport"; it is "VASOT's
structural term should be compared against, and must beat, standard
elastic/DTW-style alignment before any new invariance theory is credited".
Starting N-C2 on this evidence would risk reinventing Gromov-DTW and SRVF
elastic matching. **N-C2 was not started**; the recommended next step is a
cheap controlled comparison of VASOT against soft-DTW / Gromov-DTW structure on
real pairs, not a new formalism.

Limitations: one checkpoint (FSeval baseline s1), 8 videos, synthetic warps of
the same video rather than genuinely differently-paced recordings, linear
interpolation in feature space, and DTW evaluated as a path over the same cost
matrix without the unbalanced/virtual-frame machinery VASOT needs for
partial overlap — so the DTW comparison is a lower bound on difficulty, not a
drop-in replacement claim.
