# F00 — Final reproducibility audit and project report

> **Historical snapshot, not the current final verdict.** This document
> predates E08–E12 and paired seed-1 controls. Its A/C causal verdicts,
> Bézier task scores, VAOT port status, and “no curve winner” statement are
> superseded by `reports/E08_RHO0_TRANSFER.md`,
> `reports/E09_LIMITATION_RECHECK.md`, `reports/E10_CURVE_SPACE_COMPARISON.md`,
> `reports/E11_RHO_SPLIT.md`, `reports/E12_FS_MID_TRANSFER.md`, and the corrected E05 matrix. See the live
> ledger in `IMPLEMENTATION.md` for current run states; do not cite the
> body below as the final project conclusion.

**Date:** 2026-10-29
**Branch:** `main` (uncommitted work: `V00 PROVENANCE.md`, `collect_e_results.py`,
E05/E00/E07a/V01 reports, prepare script, run script).
**Baseline config lock:** `reports/E00_contract.lock` — sha256
`63148876ba0ddfe58332b066f1211349f04b16b39f9f33b6b57feb7446ea9f99`,
git commit `92aef7f`.

This document is the single traceable surface for the project. Every number
below cites a run manifest, log file, or report that can be read from this
repository. Nothing is sourced from terminal scrollback.

## 1. Status by packet

See `IMPLEMENTATION.md` §7 for the live ledger. Snapshot at the time of this
audit:

| Phase | Packet | State | Deliverable |
|---|---|---|---|
| S | S00–S03 | DONE | Infrastructure + protocol freeze + runner + preflight |
| R | R00, R01 | DONE | Canonical DA + YTI aggregate |
| R | R02 | CONDITIONAL / not needed | Gate inactive; no variance expansion required |
| D | D00–D04 | DONE | Diagnostics schema + A/B/C hypothesis report |
| V | V00 | DONE | Source import + provenance |
| V | V01 | DONE | Data audit (this session) — Pouring labels degenerate; Penn/IKEA missing |
| V | V02 | **BLOCKED (data)** | Frames extracted; smoke deferred, labels degenerate for phase eval |
| V | V03, V04 | **BLOCKED (data)** | No labelled VAOT dataset on cluster |
| E | E00 | DONE | Experimental contract backfilled (this session) |
| E | E01, E02 | DONE | B-spline paired seeds; FAIL scale gate (−11.95 pp) |
| E | E03 | DONE | Gaussian σ=2 paired seeds; FAIL (−10.55 pp) |
| E | E04 | DONE | Bézier paired seeds; FAIL (−11.89 pp, mIoU collapsed) |
| E | E05 | DONE | Comparison matrix; no variant passes gate |
| E | E06 | SKIP | Gate not passed — no winner to scale |
| E | E07a | DONE | ε direct intervention (this session); refutes A as actionable |
| F | F00 | DONE | This document |

## 2. VASOT reproduction (track 1 and 2)

### 2.1 Desktop Assembly — R00

Canonical run: `R00_DA_baseline_s0`, 30 epochs, seed 0, config from
`configs/vasot/baselines/desktop_assembly.yaml` (frozen in S01).

| Metric | Local | Paper |
|---|---:|---:|
| MoF | 0.5986 | (per paper table) |
| F1 | 0.6380 | — |
| mIoU | 0.4211 | — |

- Manifest: `reproductions/tssl/artifacts/runs/R00_DA_baseline_s0/manifest.json`
- Wandb: `ge8a3wu7`
- See reports/S00_INVENTORY.md §2 for the pre-S00 historical results on the
  same seed. The S02 isolated-runner reproduction here matches that history.

### 2.2 YTI aggregate — R01

Per-activity, seed 0, 30 epochs, config from `configs/vasot/baselines/yti.yaml`:

| Activity | MoF | F1 | mIoU | Wandb |
|---|---:|---:|---:|---|
| changing_tire | 0.4483 | 0.2956 | 0.2040 | tihe3mu9 |
| coffee | 0.6847 | 0.4720 | 0.3249 | 6jaj64ev |
| cpr | 0.5590 | 0.4924 | 0.2727 | 5cg9rker |
| jump_car | 0.3392 | 0.2604 | 0.1744 | — |
| repot | 0.3655 | 0.2261 | 0.1535 | — |
| **Mean (unweighted over activities)** | **0.4793** | **0.3493** | **0.2259** | |

Aggregation rule (frozen S01): per-activity training, per-activity test, then
unweighted mean across the 5 activities. Report both per-activity and aggregate.

### 2.3 Reproduction verdict

VASOT reproduction on DA and YTI (seed 0) is **closed** at the paper-protocol
config frozen in S01. No contradictions with the historical pre-S00 numbers
(reports/S00_INVENTORY.md §2–§3) within rounding. Variance (seeds > 0) is
gated to R02 and intentionally not run — nothing downstream here required it.

## 3. VAOT (track 3)

Source imported from `trquhuytin/VAOT-ICCV25`,
commit `daec4a6d05fef827e19a69ce505423ee4a1ad0e8` (V00
`reproductions/tssl/repos/VAOT/PROVENANCE.md`).

**No canonical baseline exists.** V01 (this session, see
reports/V01_VAOT_DATA_AUDIT.md) determined:

- Pouring is the only dataset with raw .mov files on the cluster.
- Frames were re-extracted (2 GB, 6354 jpg files, per-video subdirs).
- The sermanet pkl's `frame_label` is identically 0 across every video and
  split; VAOT phase classification / progression / AP@k are therefore not
  computable without labels we do not have.
- Penn Action and IKEA ASM raw data are not present on the cluster.

**Status reported honestly:** VAOT track is blocked on data, not on code or
protocol. The code port (PL 2.x) remains open; even a successful port would
produce only a pipeline-smoke run on Pouring, not a canonical paper-comparable
result.

## 4. Limitation diagnosis (D-phase + E07a)

The D phase (D01, D02, D03, D04) ran diagnostics across 13 dataset/activity
combinations. E07a (this session) added the first direct-intervention
counter-check for Hypothesis A.

| Hypothesis | D-phase verdict | Direct intervention | Final reading |
|---|---|---|---|
| A — T* is peaky, supervision near top-1 | SUPPORTED (D01: T_ent 0.024–0.077, exp(H)≈1) | E07a: ε ↑ monotonically harms MoF/mIoU; E01/E03/E04: softening at representation level also harms | **Peaky T\* is observed but is a working-mode, not a lever.** Correctly diagnosed; as an actionable cause of VASOT's limitations it is **refuted** by the controlled interventions. |
| B — tautological intra-video CE, boundary blindness | PARTIAL (D02: boundary z=+0.145 mean, negative on Breakfast) | Not yet run as a direct intervention | **Open.** The transport-plan boundary z-score is a proxy, not a boundary P/R/F1. The direct test (boundary-aware auxiliary head) is specified in E07bc §Follow-ups. |
| C — frame-index prior | SUPPORTED (D03: idx_sem_corr 0.063–0.516, weakest on YTI 0.063–0.202) | E07b: **rho=0 improves MoF by +7.95 pp on FSeval, both seeds**; rho=0.05 also PASSES (+4.10 pp). E07c: radius_gw widening FAILS. | **CONFIRMED actionable, via the temporal prior `R`.** The structural prior `C^x` is a different mechanism (not improvable by widening). First variant in the project to pass the scale gate. |

Full diagnostic numbers: `reports/D04_DIAGNOSIS_AGGREGATE.md`.
ε direct intervention: `reports/E07a_EPS_ABLATION.md`.

## 5. Continuous-curve proposal (E-phase)

**Headline:** on FSeval, no continuous-curve variant (arc-length, B-spline,
Gaussian σ=2, piecewise-cubic Bézier) passes the ≥ 1 pp scale gate. Every
smoothing variant degrades the task metrics by ~10–12 pp MoF; arc-length is
roughly neutral in expectation (−0.78 pp) with a bad per-seed variance
(+3.4 pp on s0, −5.0 pp on s1).

| Variant | Mean MoF | Δ MoF (pp) | Mean mIoU | Scale gate |
|---|---:|---:|---:|---|
| Baseline (rho=0.15, eps=0.07, radius=0.04) | 0.4727 | — | 0.2697 | — |
| Arc-length | 0.4649 | −0.78 | — | FAIL |
| B-spline (E01) | 0.3532 | −11.95 | 0.2100 | FAIL |
| Gaussian σ=2 (E03) | 0.3673 | −10.55 | 0.2230 | FAIL |
| Bézier 4-seg (E04) | 0.3538 | −11.89 | 0.0295 | FAIL (mIoU collapse) |
| ε=0.15 (E07a — not a curve) | 0.3237 | −14.90 | 0.0330 | FAIL |
| ε=0.30 (E07a — not a curve) | 0.2140 | −25.87 | 0.0608 | FAIL |
| **rho=0.05 (E07b)** | **0.5138** | **+4.10** | 0.3025 | **PASS** |
| **rho=0.00 (E07b)** | **0.5522** | **+7.95** | 0.3116 | **PASS** |
| radius_gw=0.12 (E07c) | 0.4442 | −2.85 | 0.1831 | FAIL |
| radius_gw=0.50 (E07c) | 0.3207 | −15.20 | 0.2040 | FAIL |

Full matrix: `reports/E05_COMPARISON_MATRIX.md` (curves),
`reports/E07a_EPS_ABLATION.md` (ε knob),
`reports/E07bc_PRIOR_ABLATION.md` (rho / radius_gw).

**Conclusion:** the project's original working proposal — "smoothen discrete
representations via B-spline / Bézier / Gaussian curves to fix VASOT's
limitations" — is **refuted** on FSeval. The right lever is Hypothesis C,
specifically the temporal prior `R` controlled by `rho`. Setting `rho = 0`
(removing R entirely) improves FSeval MoF by +7.95 pp on both seeds, the
first and only variant in the project to pass the scale gate. Widening the
companion structural prior `C^x` via `radius_gw` fails, so Hypothesis C
should be sharpened to "the *temporal* frame-index prior is harmful",
distinct from the structural diagonal prior which is load-bearing.

## 6. Minimal reproducibility recipes

### Canonical VASOT DA (R00)

```bash
cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
source /mnt/data/thanhpv/tssl/reproductions/env.sh
conda activate acvcl
python3 train.py -d desktop_assembly -ac all -c 22 -ne 30 \
    --rho 0.25 -lat 0.16 -r 0.02 -ae 0.6 -at 0.3 \
    -lr 1e-3 -wd 1e-4 -vf 5 -ua -g 0 -ls 512 128 40 --seed 0 \
    --wandb --wandb-project vasot-repro --run-id R00_DA_baseline_s0
```

### Canonical VASOT YTI (one activity, R01)

```bash
cd /mnt/data/thanhpv/tssl/reproductions/tssl/repos/VASOT/src
source /mnt/data/thanhpv/tssl/reproductions/env.sh
conda activate acvcl
python3 train.py -d YTI -ac coffee -c 10 -ne 30 \
    --rho 0.15 -lat 0.12 -r 0.04 -ae 0.6 -at 0.3 \
    -lr 1e-4 -wd 1e-5 -vf 5 -g 0 -ls 3000 32 32 --seed 0 -x -1 \
    --wandb --wandb-project vasot-repro --run-id R01_YTI_coffee_s0
```

Repeat for the other four activities (changing_tire K=11, cpr K=7,
jump_car K=12, repot K=8) with each activity's `-c` value. Aggregate = unweighted mean
over the 5 activities.

### Continuous-curve experiments (E01/E03/E04 pipeline)

```bash
cd /mnt/data/thanhpv/tssl
bash reproductions/tssl/scripts/run_e_pipeline.sh
```

### Hypothesis A direct intervention (E07a)

```bash
cd /mnt/data/thanhpv/tssl
bash reproductions/tssl/scripts/run_e07a_eps_ablation.sh
```

### VAOT Pouring data preparation (V01 preprocess)

```bash
cd /mnt/data/thanhpv/tssl
source reproductions/env.sh && conda activate acvcl
python3 reproductions/tssl/scripts/prepare_pouring_for_vaot.py
# Note: labels in sermanet_pouring are all 0; phase metrics not reproducible
# until a labelled source is obtained.
```

## 7. What is intentionally NOT in this report

- Mean ± std over 3 seeds: not computed, because R02 (variance) was gated and
  not triggered by R00/R01, and nothing downstream required it.
- A single "best proposal" row: none of the proposals passed the scale gate.
  Reporting a least-bad variant would misrepresent the result.
- VAOT numbers: the track is blocked on data; a smoke-only number from an
  unlabelled dataset would not reproduce the paper.
- Memory ablation / lambda_actions / rho sweeps on DA: outside this project's
  frozen protocol (S01).

## 8. Reviewer checklist

A reviewer wanting to spot-check this report can:

1. Pick any run_id in a table and read
   `reproductions/tssl/artifacts/runs/<run_id>/manifest.json`.
2. Confirm the baseline config hash in `reports/E00_contract.lock` matches
   the YAML it refers to.
3. Follow §6 to reproduce any of the canonical commands; wandb runs are tagged
   with the matching packet IDs (`R00`, `R01`, `E01`, `E03`, `E04`, `E07a`).
4. Cross-check hypothesis verdicts in §4 against the raw diagnostics in
   `reports/D04_DIAGNOSIS_AGGREGATE.md` and the ε ablation table in
   `reports/E07a_EPS_ABLATION.md`.
