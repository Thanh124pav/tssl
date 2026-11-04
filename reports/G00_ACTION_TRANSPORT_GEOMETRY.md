# G00 — Transport and action-prototype geometry audit

Date: 2026-11-04. Frozen checkpoints only, no training, no GPU-hours of
optimisation. GT labels are used for evaluation-side stratification only.

Script: `reproductions/tssl/scripts/eval_g00_geometry_audit.py`
(shared loaders in `reproductions/tssl/scripts/diag_common.py`).
Raw artifacts: `reproductions/tssl/artifacts/diagnostics/g00/*.json`,
W&B project `vasot-repro`, runs `G00_{FSeval_baseline_s1,DA_baseline_s0,YTI_coffee_s0}`.

## 0. Matrix definitions and normalisation (required by the exit criterion)

| symbol | shape | built from | normalisation used here |
|---|---|---|---|
| `T_xa` | 256 × K actions | `segmentation_ot.segment_ot` on `1 - z a^T + rho_seg * P_seg` | rows renormalised over the K actions; entropy in nats; `row_entropy_norm = H/log K`; `effective_support = exp(H)` |
| `T_xy` | 256 × 256 frames | `alignment_ot.align_ot` on `1 - z_x z_y^T + rho_align * P_align`, virtual row/column added then dropped by the solver | rows renormalised over the real Y columns; `row_entropy_norm = H/log N_y`; pre-normalisation row mass reported separately |

> **Padding note (2026-11-04 audit).** Every `T_xy` in this report is solved
> with the virtual row **and** column prepended, so the index `align_ot` drops
> is the virtual one. The published `train.py` appends the virtual column
> instead, which shifts its alignment pseudo-label by one frame (see
> `reports/X01_VIRTUAL_COLUMN_AUDIT.md`). The checkpoints audited here were
> therefore *trained* under the published padding and *measured* under the
> corrected one; this matches the convention used by E09 and by
> `scripts/eval_transport_support.py`, so the numbers remain comparable with
> earlier packets.

Solver parameters are the checkpoint's own training values (`eps`, `alpha`,
`radius_gw`, `lambda_*`, `n_ot_train`, `ub_frames=False`, `ub_actions=True`).
Both plans come from the public solver; nothing was reimplemented.

## 1. The D01 / E09 discrepancy is a difference of matrix, not of measurement

20 videos, 20 fixed adjacent pairs, 256 uniformly sampled frames.

| dataset / checkpoint | plan | entropy norm | effective support | top-1 mass | real-target row mass |
|---|---|---:|---:|---:|---:|
| FSeval baseline s1 | `T_xa` | 0.110 | 1.50 / 12 actions | 0.915 | 1.000 |
| FSeval baseline s1 | `T_xy` | 0.589 | 32.45 / 256 frames | 0.127 | 0.999 |
| DA baseline s0 | `T_xa` | 0.027 | 1.11 / 22 actions | 0.969 | 1.000 |
| DA baseline s0 | `T_xy` | 0.358 | 7.75 / 256 frames | 0.285 | 0.999 |
| YTI coffee s0 | `T_xa` | 0.065 | 1.23 / 10 actions | 0.954 | 0.963 |
| YTI coffee s0 | `T_xy` | 0.553 | 22.23 / 256 frames | 0.108 | 0.957 |

These reproduce E09 (FSeval 0.584 / 30.87 / 0.126; DA 0.285 / 7.78; YTI coffee
0.112 / 21.45) within sampling noise of the pair set. **Verdict: D01's "peaky"
claim and E09's "not peaky" claim are both correct and not in conflict — D01
measured the segmentation plan `T_xa`, E09 measured the alignment plan `T_xy`.**
Nothing in the old verdicts needs rewriting; what was missing was the statement
of which matrix each number belongs to. Both are now defined above.

Untrained control (model rebuilt from the same hyper-parameters, weights not
loaded, same videos and pairs):

| dataset | `T_xa` top-1 init → trained | `T_xy` top-1 init → trained | `T_xy` support init → trained |
|---|---|---|---|
| FSeval | 0.687 → 0.915 | 0.193 → 0.127 | 43.1 → 32.4 |
| DA | 0.568 → 0.969 | 0.622 → 0.285 | 7.06 → 7.75 |
| YTI coffee | 0.892 → 0.954 | 0.090 → 0.108 | 25.5 → 22.2 |

Training sharpens segmentation on all three datasets. Alignment has no uniform
training response (DA top-1 falls by half, YTI rises slightly), confirming E09.

## 2. Prototype separation — and the finding that prototypes barely move

FSeval baseline s1, K = 12, d = 40:

| quantity | trained | untrained (same seed) |
|---|---:|---:|
| prototype norm | 1.0005 ± 0.0004 | 1.0000 |
| mean off-diagonal cosine | −0.051 | −0.023 |
| max off-diagonal cosine | 0.536 | 0.369 |
| mean nearest-prototype cosine | 0.282 | 0.270 |
| mean nearest-prototype margin (1 − cos) | 0.718 | 0.730 |

Prototypes are well separated in the weak sense that no two collapse, but
**training does not make them more separated than 12 random unit vectors in 40
dimensions** — the nearest-prototype margin is marginally *worse* after
training, and the largest pairwise cosine grows from 0.37 to 0.54.

The drift measurement, with the control the claim needs
(`diagnostics/g00/prototype_drift_controls.json`, Hungarian-matched cosine):

| comparison | matched cosine | min matched cosine |
|---|---:|---:|
| trained s1 vs **its own** random init (seed 1) | **0.872** | 0.718 |
| trained s1 vs a *different* random init (seed 2) | 0.253 | 0.016 |
| random init 1 vs random init 2 | 0.239 | 0.003 |
| random init 1 vs random init 3 | 0.267 | 0.173 |
| trained FSeval s1 vs trained FSeval s2 | 0.251 | 0.010 |
| trained DA s0 vs **its own** init | 0.677 | — |
| trained YTI coffee s0 vs **its own** init | 0.984 | — |

Random-vs-random matching sits at 0.24–0.27, so 0.872 (FSeval), 0.677 (DA) and
0.984 (YTI coffee) are far outside the control. The Hungarian assignment also
recovers the identity permutation, so no re-indexing is involved.
**Learned action prototypes stay close to their random initialisation; across
seeds they are as different from each other as two random codebooks are.** The
action codebook is therefore much closer to a fixed random reference frame than
to a learned, dataset-determined set of action directions. This is a direct,
measured constraint on G04 (prototype feedback) and on any method that treats
the prototype as a meaningful geometric anchor — see G01, where a *random*
centre matches the learned prototype on every score.

Other geometry: hard-assignment purity against GT actions 0.618 (FSeval, 12/12
clusters used), 0.791 (DA, 22/22), 0.692 (YTI coffee, 10/10); mean within-cluster
L2 scatter to the assigned prototype 1.077 (FSeval), 0.644 (DA), 0.955 (YTI).

## 3. Boundary behaviour of `T_xa`

Frames stratified by |signed distance| to the nearest GT boundary; the
"neighbour mass" column is the posterior mass on the dominant clusters of the
adjacent GT segments, with the own-segment cluster excluded.

FSeval baseline s1:

| |dist| (frames) | n | entropy norm | top-1 mass | neighbour mass |
|---|---:|---:|---:|---:|
| 0 | 363 | 0.176 | 0.854 | 0.119 |
| 1–2 | 1356 | 0.162 | 0.871 | 0.093 |
| 3–5 | 1393 | 0.110 | 0.915 | 0.063 |
| 6–10 | 1154 | 0.068 | 0.951 | 0.037 |
| 11–20 | 609 | 0.043 | 0.971 | 0.015 |
| ≥21 | 245 | 0.074 | 0.943 | 0.004 |

DA s0: 0.046 → 0.018 entropy norm, neighbour mass 0.418 at a boundary falling
to 0.003 far from one. YTI coffee s0 is nearly flat (0.083 → 0.054, neighbour
mass 0.128 → 0.083), i.e. the boundary effect is dataset dependent.

**Verdict:** uncertainty in `T_xa` is genuinely concentrated at real
transitions on FSeval and DA (monotone in boundary distance, and the extra mass
goes to the neighbouring action rather than spreading at random), but it is a
modest effect — even on a boundary frame the posterior keeps 0.854 (FSeval) /
0.944 (DA) on its top action. `T_xa` is not diffuse at boundaries; it is
slightly less sharp. The non-monotone last row on FSeval (≥21 frames) comes from
the few very long segments and is not evidence of a second mechanism.

## 4. Jitter: posterior vs checkpoint

These are different quantities and are reported separately, as required.
Per-video action-posterior temporal jitter, measured as the argmax flip rate
between consecutive frames *inside* GT segments: **0.079** (FSeval), 0.069
(DA), 0.032 (YTI coffee). Checkpoint drift of the global prototypes is the
table in §2 and is not a per-video quantity at all.

## 5. Evidence that contradicts the proposed mechanisms

- Against "prototype collapse": no two prototypes merge, all clusters are used
  on all three datasets, and hard-assignment purity is 0.62–0.79.
- Against "T_xa is diffuse, so the action space is weak": `T_xa` is the
  sharpest object in the system (support 1.1–1.5 actions).
- Against "training organises the action codebook": §2 — prototypes stay near
  their random init, and two seeds give unrelated codebooks.
- Against "boundary uncertainty explains segmentation error": the boundary
  effect is real but small on FSeval/DA and almost absent on YTI coffee, which
  nonetheless has the lowest purity of the three.

## 6. Exit criterion

Met. Every metric has its matrix and normalisation stated (§0); the
D01–E09 discrepancy is resolved as a matrix mix-up, not a measurement error
(§1); prototype separation is quantified against a random control and found to
be no better than random, with prototypes remaining close to their own
initialisation (§2); boundary uncertainty is quantified and found real but
modest (§3). No GPU training was needed. G01 proceeds.
