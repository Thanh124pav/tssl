# Method development on VASOT: status and current proposal

**Date:** 2026-11-11. Workstation runs (GTX 1650, conda env `deeplearning`).
**One line:** two ideas are alive — a prototype **hub** for alignment (helps segmentation on FSeval and YTI, hurts on Desktop Assembly, does not help alignment metrics) and an intra-action **local** loss (one run: better alignment metrics, segmentation unchanged). Their combination is running and has **no results yet**. Decentered cost was dropped.

## 1. Working hypothesis

Local Geometric Concentration: under pseudo-label training, frame embeddings concentrate in shared regions, shrinking margins and producing overlapping action clusters, ambiguous frame correspondences and inconsistent multi-video alignment. Earlier pilots in this repo that added repulsion-style losses (VICReg-like geometry, SpecT-OT TAR, soft consensus) did not beat the control (`FULL_METHOD_MATRIX_2026-11-10.md`), so the arms below act on the transport targets instead.

## 2. Arms

All are flags of `repos/VASOT/src/train.py`; with every flag off the code is the control.

| Arm | Flags | What it does |
|---|---|---|
| `baseline` | — | VASOT, published recipe per benchmark |
| `dc` | `--decenter 1.0` | subtract the EMA embedding mean before every cosine cost/logit. **Dropped.** |
| `tmpl` | `--template-slots K` | alignment target composed through K random fixed vectors (A14) |
| `hub` | `--template-slots K --align-hub clusters` | alignment target composed through the K action prototypes: `T_ij = 1/N_Y(a(i))` if `a(j)=a(i)`, else 0 (hard assignment from the two segmentation plans) |
| `local` | `--local-weight 1.0` | after 5 warm-up epochs: subtract each frame's assigned prototype, contrast the normalised residuals across the two videos among same-pseudo-action frames, target = frame-level OT plan restricted to those pairs |
| `local_raw` | `… --local-raw` | same, residuals not normalised |
| `hub_local`, `hub_local_raw` | hub + local | the current proposal |

Also tried and dropped: Gaussian sampling before OT (`--ot-noise 0.3`, PEOT-inspired): −3.6 MoF alone and −12.6 with `dc` on FSeval (10 epochs, seed 0). Smaller σ was not tried.

## 3. Segmentation results (MoF / F1 / mIoU, full-video test, published epochs)

**FSeval, 30 epochs**

| Arm | s0 | s1 | s2 | mean MoF |
|---|---|---|---|---:|
| baseline | 45.1 / 48.2 / 25.5 | 49.3 / 51.2 / 28.4 | 52.8 / 54.2 / 31.8 | 49.1 |
| hub | 53.3 / 54.9 / 29.8 | 51.3 / 54.5 / 30.4 | 59.2 / 58.7 / 32.3 | 54.6 |
| tmpl | 53.3 / 54.5 / 29.0 | 52.9 / 56.6 / 30.5 | 59.7 / 59.8 / 33.4 | 55.3 |
| dc | 49.8 / 52.8 / 30.9 | — | — | — |

**Desktop Assembly, 30 epochs**

| Arm | s0 | s1 | s2 |
|---|---|---|---|
| baseline | 62.2 / 65.8 / 43.4 | 65.0 / 67.3 / 46.2 | 62.8 / 66.5 / 43.9 |
| hub | 56.6 / 60.8 / 36.9 | 58.9 / 56.6 / 39.4 | 63.4 / 66.7 / 43.8 |
| tmpl | 55.4 / 62.0 / 35.9 | 60.7 / 60.3 / 40.8 | 62.9 / 67.2 / 43.0 |
| dc | 62.0 / 66.4 / 43.5 | — | — |

**YTI, 30 epochs, seed 0**

| Activity | baseline | hub | tmpl | dc | local |
|---|---|---|---|---|---|
| changing_tire | 44.8 / 29.6 / 20.4 | 51.1 / 36.4 / 23.9 | 51.9 / 34.9 / 24.0 | 32.3 / 24.7 / 15.9 | 44.4 / 29.1 / 18.9 |
| coffee | 68.5 / 47.2 / 32.5 | 64.6 / 44.6 / 30.2 | 63.0 / 45.7 / 28.7 | 58.7 / 33.5 / 27.2 | — |
| cpr | 55.9 / 49.2 / 27.3 | 59.2 / 52.2 / 27.1 | 61.7 / 55.6 / 27.1 | 41.2 / 33.7 / 18.8 | — |
| jump_car | 33.9 / 26.0 / 17.4 | 35.2 / 27.0 / 17.5 | 35.1 / 26.8 / 16.1 | 34.8 / 28.1 / 20.5 | — |
| repot | 36.6 / 22.6 / 15.4 | 40.1 / 29.0 / 16.2 | 39.9 / 24.3 / 16.7 | 27.9 / 17.3 / 14.7 | — |
| mean MoF | 47.9 | 50.0 | 50.3 | 39.0 | — |

Reading:

- `hub` beats the baseline on all 3 FSeval seeds (+5.5 MoF mean) and on MoF in 4/5 YTI activities, but loses about 6 MoF on Desktop Assembly in 2/3 seeds.
- `hub` ≈ `tmpl` everywhere: the gain comes from block-structuring the target by action, not from the learned prototypes.
- `dc` wins on FSeval seed 0, ties on Desktop Assembly, loses 9–15 MoF on 4/5 YTI activities. Untested guess: on YTI the mean is dominated by background frames.

## 4. Video-alignment metrics (`scripts/eval_va.py`)

VAOT metric code on frozen embeddings, action label as phase label, 256 evenly spaced frames per video. Acc: SVM fitted on even-indexed videos, scored on odd. τ: Kendall's tau over all video pairs. AP@K: leave-one-video-out frame retrieval. Seed 0 only.

| Benchmark | Arm | Acc | τ | AP@5 | AP@10 | AP@15 |
|---|---|---:|---:|---:|---:|---:|
| FSeval | raw features | 63.41 | 7.78 | 49.85 | 47.75 | 46.57 |
| FSeval | baseline | 64.45 | 31.91 | 55.05 | 54.11 | 53.60 |
| FSeval | tmpl | 66.67 | 28.25 | 54.80 | 54.27 | 53.88 |
| FSeval | hub | 65.05 | 29.29 | 54.39 | 53.89 | 53.57 |
| YTI changing_tire | raw features | 44.44 | 13.07 | 36.27 | 34.97 | 33.91 |
| YTI changing_tire | baseline | 52.24 | 61.90 | 46.77 | 46.07 | 45.79 |
| YTI changing_tire | local | 53.58 | 62.95 | 49.70 | 48.70 | 48.17 |

- `hub`/`tmpl` raise segmentation by 8 MoF on this seed but not alignment: AP@K flat, τ down about 3 points.
- `local` leaves segmentation flat but raises every alignment metric (+2.4 to +2.9 AP@K).

Caveats: this is a proxy on feature benchmarks, not the paper's VA protocol (Pouring / Penn Action / IKEA ASM). τ assumes a shared action order, which 50 Salads does not have. On YTI frames with negative labels are dropped, but whether background is encoded that way was not checked.

## 5. Current proposal

Two levels, matching the two consequences of the hypothesis:

- **Between actions — hub.** Organises concentration by action and makes multi-video correspondence consistent by construction, at linear cost in the number of videos. It does not separate frames inside an action.
- **Inside an action — local.** Removes the prototype and learns where a frame sits within its action, the level the hub leaves empty.

`hub_local` is the direct test: it should keep the hub's segmentation gain and add the local loss's alignment gain. **No `hub_local` number exists yet.**

## 6. Running now / not done

- Running on the workstation: `local`, `hub_local` (seed 0 on FSeval, 5 YTI activities, Desktop Assembly; then seeds 1–2 on FSeval and Desktop Assembly) and `local_raw`, `hub_local_raw` (seed 0). Results append to `reproductions/tssl/artifacts/runs/current/mrt_matrix.tsv`, which is git-ignored along with checkpoints.
- **Breakfast is unresolved.** The baseline on `coffee` gives 30.8 MoF with `run_mrt_matrix.sh`, against 51.98 in `BREAKFAST_VASOT_PROGRESS.md` for the same activity, seed and epochs. The recipe or scoring differs somewhere; the three `bf:coffee` rows should not be used and the Breakfast queue was stopped.
- Not done: VA metrics for every arm beyond the rows above; geometry measurements (effective dimension, lag cosine, silhouette) for hub/local checkpoints; YTI seeds 1–2; why the hub fails on Desktop Assembly; mechanism analysis in general.
- Everything except FSeval/Desktop Assembly `baseline`/`hub`/`tmpl` is single-seed.

## 7. Running on the server

```bash
JOBS=<cpus> bash reproductions/tssl/scripts/run_mrt_queue.sh        # baseline, hub, tmpl on all benchmarks
bash reproductions/tssl/scripts/run_mrt_matrix.sh <bench> <arm> <seed>   # one run
python reproductions/tssl/scripts/eval_va.py --dataset FSeval --ckpts name=<ckpt> ...
```

- `bench`: `fseval`, `da`, `yti:<activity>`, `bf:<activity>`; arms as in §2.
- Each run needs about 1 CPU core, 1.6 GB RAM and under 1 GB VRAM; the bottleneck is CPU, so throughput scales with cores. `run_local_queue.sh` and `run_local_raw_queue.sh` hard-code 2 and 1 workers.
- `data/` is git-ignored: the four feature benchmarks (about 14 GB under `reproductions/tssl/data/final`) must be copied or re-downloaded.
- `TSSL_PYTHON` overrides the interpreter path, which is otherwise the workstation's `deeplearning` env.
- Finished runs are skipped by reading `mrt_matrix.tsv`; do not edit `run_mrt_matrix.sh` while a queue is running (two rows were lost that way and recovered from logs).
