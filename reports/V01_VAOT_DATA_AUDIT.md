# V01 — VAOT data and evaluation protocol audit

**Date:** 2026-10-29
**Source tree:** `reproductions/tssl/repos/VAOT` (clean import — see V00 `PROVENANCE.md`)

## 1. Data matrix: available / missing / invalid

| Dataset | Raw videos | Pre-extracted frames | Phase labels | Verdict |
|---|---|---|---|---|
| **Pouring (sermanet)** | **YES** — 11 train `.mov`, 10 test `.mov` under `/mnt/data/thanhpv/acvcl/data/pouring/sermanet_pouring/videos/{train,test}/` | **PARTIAL on disk, re-extracted in V01** — now `data/vaot/pouring/frames/{train,test}/<name>/<idx:04d>.jpg`, matches each pkl `seq_len` exactly (2330 train / 2012 val / 2012 test frames) | **PRESENT but DEGENERATE** — `train.pkl`/`val.pkl`/`test.pkl` carry per-frame `frame_label`, length matches `seq_len`, **but every value is 0 across every video and every split**. This is the TCC/weak-label-only version of Sermanet Pouring; no phase annotations are provided here. | **BLOCKED for VAOT phase metrics** |
| **Penn Action** | NOT FOUND on cluster | — | — | **MISSING** — need upstream download (~3.5 GB) |
| **IKEA ASM** | NOT FOUND on cluster | — | — | **MISSING** — need upstream download (large) |

Related NLP dataset "Penn" (`/mnt/data/thanhpv/acvcl/third_party/.../Penn_Action_SmoothDTW_*`) is a *trained-checkpoint* directory, not raw Penn Action data; it does not help here.

## 2. Pouring: per-video verification

Using the pkl `seq_len` as ground truth video length:

| video | pkl seq_len | on-disk frame files | ratio | .mov available |
|---|---:|---:|---:|---|
| pouring_002 | 228 | 41 | 0.180 | yes |
| pouring_003 | 283 | 50 | 0.177 | yes |
| pouring_005 | 234 | 42 | 0.179 | yes |
| pouring_006 | 195 | 35 | 0.179 | yes |
| pouring_008 | 166 | 31 | 0.187 | yes |
| pouring_010 | 220 | 40 | 0.182 | yes |
| pouring_011 | 207 | 37 | 0.179 | yes |
| pouring_013 | 221 | 40 | 0.181 | yes |
| pouring_014 | 178 | 33 | 0.185 | yes |
| pouring_015 | 237 | 42 | 0.177 | yes |
| pouring_017 | 161 | 30 | 0.186 | yes |

The ~0.18 ratio across all videos is a **uniform ~5× temporal subsampling** of the
disk frames vs the original video. Confirmed on `pouring_002`: `read_video` returns
228 frames; pkl says `seq_len=228`.

**Decision:** do not use the flat subsampled frames for VAOT. Re-extract full
frames from `.mov` into per-video subdirs. This matches what VAOT's
`align_dataset.py` expects (`glob(path/*)` → one subdir per video, then
`glob(subdir/*)` for frames) and what `video_to_frames.py` was written to produce.

Projected extraction cost on worker-0:

- Train: 11 × ~210 frames avg × ~100 KB/jpg ≈ **230 MB**
- Test: 10 × ~200 frames avg × ~100 KB/jpg ≈ **200 MB**
- Target dir: `/mnt/data/thanhpv/tssl/reproductions/tssl/data/vaot/pouring/frames/{train,test}/<name>/<idx:04d>.jpg`

## 3. VAOT evaluation protocol (frozen)

From `evaluations.py:134` and `evals/`:

- `npyrecords/<dataset>_val.npy`: Python dict pickled as a 0-d ndarray, keyed
  by video name, each value `{'labels': 1-D int array per frame}`. For Pouring:
  `lab_name = "pouring_val"` is hardcoded; evaluation uses `val` split.
- Metrics (not shared with VASOT):
  - **Phase Classification (PC)** at train fractions 0.1 / 0.5 / 1.0 via SVM
    (one-vs-one) over per-frame embeddings (`evals/phase_classification.py`).
  - **Kendall's Tau** between predicted phase progression and ground truth
    (sqeuclidean).
  - **Phase Progression** regression score.
  - **AP@5 / AP@10 / AP@15** for retrieval-style matching.

VASOT metrics (MoF / F1 / mIoU) must not be reused as VAOT numbers.

## 4. Code blockers (from V00 PROVENANCE.md, still open)

Required before V02 smoke:

1. `pytorch_lightning.core.lightning` import → `from pytorch_lightning import LightningModule` (train.py, evaluations.py)
2. `Trainer(gpus=[...])` → `Trainer(devices=[...], accelerator='gpu')`
3. `resume_from_checkpoint=` kwarg → `ckpt_path` on `fit`/`validate`
4. `on_batch_end` → `on_train_batch_end` (callback in utils.py)
5. `models.resnet50(pretrained=True)` → `models.resnet50(weights='IMAGENET1K_V1')`
6. `np.int` → `int` (align_dataset.py:56)
7. `utils.py` content is duplicated top-to-bottom; de-dup before patching, otherwise
   patching the first copy leaves the second copy stale.
8. `evaluations.py` has hardcoded `./Data_Test/` and relative `./npyrecords/`; wire
   those to explicit CLI args.

## 5. One usable dataset config, for V02

Pouring will be the first dataset. Config (not yet persisted as yaml):

```yaml
dataset: pouring
train_frames_root: /mnt/data/thanhpv/tssl/reproductions/tssl/data/vaot/pouring/frames/train
test_frames_root:  /mnt/data/thanhpv/tssl/reproductions/tssl/data/vaot/pouring/frames/test
label_pkl_root:    /mnt/data/thanhpv/acvcl/data/pouring/sermanet_pouring
npyrecords_dir:    /mnt/data/thanhpv/tssl/reproductions/tssl/data/vaot/pouring/npyrecords
splits: {train: train, val: val, test: test}
n_train_videos: 11
n_val_videos: 10
n_test_videos: 10
seq_len_min: 161
seq_len_max: 283
```

Preprocessing script (planned, name: `scripts/prepare_pouring_for_vaot.py`):

1. Reads `videos/{split}/*.mov` with cv2; writes
   `frames/{split}/<name>/<idx:04d>.jpg`.
2. Reads `{train,val,test}.pkl`; writes `npyrecords/pouring_{train,val,test}.npy`
   as `{<name>: {'labels': frame_label.numpy().astype(int)}}` for each split.
3. Dry-run reports expected sizes before writing.

## 6. Decision after audit

**V02 and V03 are blocked for Pouring on this cluster**, not for engineering
reasons but for a data-content reason: the sermanet_pouring pkl's `frame_label`
is identically zero across all videos and splits (verified below). VAOT's
headline metrics — Phase Classification, Phase Progression, AP@k — are
computed against those labels and would be trivial / meaningless with them.

A smoke run (training loop + Kendall's Tau on embeddings) would still exercise
the pipeline, but it would not reproduce the paper's numbers because the
numbers depend on labels we do not have.

```python
>>> import numpy as np
>>> np.unique(np.concatenate([v['labels'] for v in np.load(".../pouring_val.npy", allow_pickle=True).item().values()]))
array([0])  # same for pouring_train.npy and pouring_test.npy
```

Penn Action and IKEA ASM are missing on the cluster, so V04's remaining
datasets are blocked for the same reason (absent data, not absent code).

## 7. V02 scope revised

V02 is retained only as a **pipeline-smoke** scope: port PL 2.x, run one batch
on the extracted frames, verify forward/backward/eval loop. No canonical
numbers; no phase evaluation. V03 is marked **blocked on data** until a labelled
VAOT dataset is obtained.

## 8. What V01 does NOT claim

- Penn Action and IKEA ASM are **missing on cluster**. V04 cannot cover them
  without a download step the user has not authorised. Documenting as blocked.
- The `.mov → frames` reshape **has been** executed (deliverable:
  `data/vaot/pouring/frames/{train,test}/<name>/<idx:04d>.jpg`, 6.3K files,
  2.0 GB — see §5), together with the npyrecords dict. This produced the
  degenerate-label finding above and is kept because V02's pipeline smoke
  still needs frames.
