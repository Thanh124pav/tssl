# ACVCL — Active Cross-Video Consensus Learning (testbed)

Experimental testbed for cross-video representation learning. The current
goal is **not** to implement the full ACVCL method but to stand up the data
pipeline, baseline selectors (Random / Similarity / Hard Pair), and a
minimal TCC-style alignment loss so that future ACVCL selectors can be
dropped in behind the same API.

Research spec: see [`SETUP.md`](SETUP.md). Current status: see
[`SETUP_STATUS.md`](SETUP_STATUS.md).

## Environment

```bash
# venv already exists at .venv/ (Python 3.10.12)
source .venv/bin/activate
pip install -r requirements.txt            # already installed in this env
python scripts/env_check.py                # torch / CUDA / ffmpeg / HF
```

Compute: 1× NVIDIA H100 **MIG 3g.40gb slice** (≈ 42 GB usable), 4 CPU cores,
CUDA 12.1 PyTorch build. bf16 is supported.

## Datasets

Not committed to git (see `.gitignore` — `data/` is local-only, ~1.3 GB on
disk); this table is the index of what is actually present in this
environment.

| dataset     | what is here                                        | size  | path                                        |
|-------------|------------------------------------------------------|-------|----------------------------------------------|
| COIN        | full annotations JSON (11 827 videos, 180 tasks)     | 11 MB | `data/coin/COIN.json`                        |
| YouCook2    | split lists, duration CSVs, InternVideo features     | 765 MB| `data/youcook2/merve__YouCook2/splits/`, `data/youcook2/internvideo_features/` |
| HowTo100M   | HF loader + extensions metadata                      | 31 MB | `data/howto100m/HuggingFaceM4__howto100m/`   |
| Pouring (Sermanet)   | frames/videos + train/val/test pickles        | 475 MB| `data/pouring/sermanet_pouring/`             |
| SynCR       | synthetic train/test splits (HF croissant dataset)    | 24 MB | `data/syncr/`                                |
| CrossTask   | **not downloaded** — upstream hosts blocked          | 0     | `data/crosstask/` (empty), see `SETUP_STATUS.md` |

Inspect a dataset:

```bash
python scripts/inspect_dataset.py --dataset coin
python scripts/inspect_dataset.py --dataset youcook2
python scripts/inspect_dataset.py --dataset howto100m
```

## Baselines / third-party code

Pulled into `third_party/` for reference only (each is its own git
checkout, not a submodule, and is **not** tracked by this repo — see
`.gitignore`; do not merge into `src/`):

- `third_party/CrossTask/` — DmZhukov/CrossTask (code only).
- `third_party/lav/` — hadjisma/VideoAlignment (LAV).
- `third_party/tcc_ref/tcc/`, `third_party/tcc_Temporal_Cycle_Consistency_Loss.pytorch/` — google-research/tcc and a PyTorch port, used as reference for `src/losses/tcc.py`.
- `third_party/CARL_code/` — minghchen/CARL, contrastive action representation learning baseline.
- `third_party/SYNCR/` — synthetic-data generation code (CLEVRER / Habitat / Kubric) backing the `data/syncr` dataset.
- `third_party/MDS-VQA/` — multi-dataset video-QA baseline, reference only.
- `third_party/videosetdiff/` — project page/demo, reference only.
- `third_party/ActiveAD/` — placeholder, README only so far.

## Project code (`src/`, `scripts/`)

| path | purpose |
|---|---|
| `src/datasets/coin.py` | COIN dataset loader |
| `src/losses/tcc.py` | TCC-style temporal alignment loss |
| `src/selectors/{random,similarity,hard_pair}.py` | pair-selection strategies sharing a common `base.py` API |
| `scripts/env_check.py` | verify torch / CUDA / ffmpeg / HF are usable in this env |
| `scripts/inspect_dataset.py` | print stats for a given dataset |
| `scripts/prepare_pouring.py` | build train/val/test pickles for the Pouring dataset |
| `scripts/tcc_train_pouring.py` | train the TCC loss on Pouring (checkpoint → `outputs/tcc_pouring/`) |
| `scripts/exp7_pair_strategy_youcook2.py` | compare random / similarity / hard-pair selectors on YouCook2 (results → `outputs/exp7_youcook2/`) |
| `scripts/smoke_test_pairs.py`, `scripts/smoke_test_sims.py` | end-to-end smoke tests: pair sampling → temporal module → TCC loss → backward |

## Smoke tests

```bash
# A + C combined: pair sampling -> temporal module -> TCC loss -> backward
python scripts/smoke_test_pairs.py
```

Produces a table comparing `random`, `similarity`, `hard_pair` selectors on
loss convergence. See `SETUP_STATUS.md` and `REPRODUCTION_STATUS.md` for
latest run numbers and environment status.
