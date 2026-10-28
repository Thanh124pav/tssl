# SETUP_STATUS

Snapshot of the testbed state after initial setup.

## Compute (self-checked)

- GPU: NVIDIA H100 80GB HBM3, exposed as a **MIG 3g.40gb** slice → 42.4 GB
  total / 42.1 GB free. Not the full 80 GB card. All experiments must stay
  under this budget.
- Driver 570.133.20, CUDA runtime 12.8 (toolkit 12.2 on-host).
- PyTorch 2.4.1 + cu121 wheel. `torch.cuda.is_bf16_supported() == True`.
- CPU: 4 cores. RAM: 2 TiB (not the bottleneck).
- Disk: nfs mount at `/`, 885 G free (92 % used) — budget-sensitive.

## Environment

- Python 3.10.12 in `.venv/`.
- `requirements.txt` installed; `scripts/env_check.py` is all green.
- `transformers` pinned `<5` because v5 requires torch ≥ 2.5.
- `ffmpeg` not system-installed, but the bundled `imageio-ffmpeg` binary at
  `.venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`
  is used by `imageio` / `decord` / PyAV paths.

## Network (self-checked)

Outbound is allow-listed. Reachable: `github.com`, `raw.githubusercontent.com`,
`api.github.com`, `codeload.github.com`, `huggingface.co`, `pypi.org`,
`zenodo.org`, `drive.google.com`, `storage.googleapis.com`.
**Unreachable** (HTTP 000 on connect, DNS fine): `*.ens.fr`, `*.inria.fr`,
`vision.eecs.yorku.ca`, `youcook2.eecs.umich.edu`, `coin-dataset.github.io`
and all other `*.github.io`, `web.archive.org`.

Impact: datasets served only from the ENS / INRIA / Yorku mirrors cannot be
pulled; HuggingFace and GitHub mirrors are the only path.

## Datasets

| dataset   | state                              | notes                                                                      |
|-----------|------------------------------------|----------------------------------------------------------------------------|
| COIN      | **annotations downloaded**         | `data/coin/COIN.json` (11 827 videos, 180 tasks, train 9 030 / test 2 797). |
| YouCook2  | **splits + duration CSVs only**    | from `merve/YouCook2` on HF. Full annotations JSON hosted at YorkU — blocked. |
| HowTo100M | **loader script + metadata only**  | `HuggingFaceM4/howto100m` (dataset_infos.json, extensions.json). No clips. |
| CrossTask | **not downloaded (blocked hosts)** | `crosstask_release.zip` / `crosstask_features.zip` live on `www.di.ens.fr` and `www.rocq.inria.fr`, both unreachable. No HF / Zenodo / GitHub mirror found. Code is in `third_party/CrossTask`. |

## Baselines

| repo                                | path                           | notes                                       |
|-------------------------------------|--------------------------------|---------------------------------------------|
| DmZhukov/CrossTask                  | `third_party/CrossTask`        | code only; data unavailable (see above).    |
| hadjisma/VideoAlignment (LAV)       | `third_party/lav`              | PyTorch; original data pointers to Penn Action etc. |
| google-research/tcc (sparse)        | `third_party/tcc_ref/tcc`      | reference implementation used to port a minimal cycle-regression loss into `src/losses/tcc.py`. |
| VideoSetDiff (ICCV 2025)            | —                              | no public code found.                       |
| SYNCR (2026)                        | —                              | no public code found.                       |

## Codebase skeleton

```
src/
  datasets/coin.py          CoinIndex loader + same-task pair sampler
  selectors/
    base.py                 PairSelector protocol (model, dataset, num_pairs)
    random.py               RandomSameTaskSelector
    similarity.py           SimilaritySelector (cosine NN within task)
    hard_pair.py            HardPairSelector (high feat sim − high pred sim)
  losses/tcc.py             TCC cycle-back-regression loss
scripts/
  env_check.py              torch / cuda / bf16 / decord / pyav / DINOv2
  inspect_dataset.py        per-dataset sanity report
  smoke_test_pairs.py       Tests A + C: select → temporal head → TCC → backward
third_party/                reference baselines (unmodified clones)
```

## Smoke tests

### env_check.py
```
torch 2.4.1+cu121 / cuda True / H100 80GB HBM3 MIG 3g.40gb / bf16 True
decord 16 frames  pyav 16 frames  dinov2-small fwd (1, 257, 384)
```

### smoke_test_pairs.py (on COIN, 200 steps, 4 pairs/step)

TCC cycle-regression loss on surrogate temporal features (random vectors +
per-task prototype + linear drift, since CrossTask features are blocked):

| selector    | loss @ start | loss @ end | drop   |
|-------------|-------------:|-----------:|-------:|
| random      | 3.734        | 3.365      | +0.369 |
| similarity  | 3.905        | 3.367      | +0.538 |
| hard_pair   | 3.847        | 3.316      | +0.531 |

Loss decreases for all three selectors → selector API, dataset indexing, TCC
loss, and GPU autograd path are wired. **This is not a real learning signal
on video features** — the inputs are synthetic; it only validates the
pipeline. True Test D (loss on real features) is blocked on CrossTask.

## Checklist (per SETUP.md §11)

```
GPU:        H100 MIG 3g.40gb (42.4 GB)
CUDA:       12.8 driver / cu121 wheel
PyTorch:    2.4.1

Datasets:
[ ] CrossTask           — upstream hosts blocked
[x] COIN                — annotations
[~] YouCook2            — splits only; annotations blocked
[x] HowTo100M metadata  — HF loader

Baselines:
[x] TCC reference       — google-research/tcc (sparse checkout)
[x] LAV reference       — hadjisma/VideoAlignment
[ ] VideoSetDiff        — no public code
[ ] SYNCR               — no public code

Smoke tests:
[x] CUDA
[x] video decoding (synthetic mp4)
[ ] CrossTask loading     — blocked
[x] random pairing        — COIN
[x] similarity pairing    — COIN (surrogate features)
[x] alignment loss backward — TCC regression on COIN pairs
[ ] feature extraction (Test E) — needs raw videos; deferred (4-core CPU budget)
```

## Open issues for next stage

1.  **Unlock CrossTask features.** Options: ask the user to upload the
    `crosstask_release.zip` and `crosstask_s3d.zip` manually; or route
    traffic through an allow-listed proxy; or rebuild features by
    DINOv2-extracting from the raw CrossTask YouTube IDs (bandwidth-heavy).
2.  **YouCook2 annotations JSON** (`youcookii_annotations_trainval.json`)
    is behind the same blocked host — check for a HF mirror or ask for a
    manual upload.
3.  **Replace surrogate features in `smoke_test_pairs.py`** with real
    pooled DINOv2 / CLIP features over a few hundred YouTube clips (needs
    network to YouTube — currently untested).
4.  **VideoSetDiff / SYNCR**: watch for release of official code; port only
    the selection logic.
5.  **COIN raw-video pipeline (Test E)**: with only 4 CPU cores and 885 GB
    free, cap at ~100 videos for the initial feature-extraction benchmark.
