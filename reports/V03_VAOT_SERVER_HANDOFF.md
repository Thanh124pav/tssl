# V03 — VAOT reproduction: data, scripts, and what is left for the server

**Date:** 2026-11-10
**Status:** data and pipeline prepared on the workstation; **no VAOT paper number has been reproduced yet.** All training is deferred to the server.

## 1. What the paper needs

VAOT (alignment-only model of arXiv 2503.16832) is evaluated on Pouring, Penn Action and IKEA ASM. Targets (paper Table 2, VAOT rows):

| Dataset | Acc@0.1 | Acc@0.5 | Acc@1.0 | Progress | τ | AP@5 | AP@10 | AP@15 |
|---|---|---|---|---|---|---|---|---|
| Pouring | 91.80 | 92.88 | 94.63 | 91.63 | 88.28 | 91.34 | 90.56 | 90.29 |
| Penn Action (mean of 13 per-action models) | 83.96 | 85.35 | 86.92 | 84.31 | 88.99 | 81.62 | 81.03 | 80.68 |
| IKEA ASM | 33.73 | 36.42 | 38.64 | – | – | 31.49 | 31.92 | 32.01 |

Hyper-parameters (paper supplementary = upstream `train.py` defaults): 40 sampled frames on Pouring, 20 on Penn Action / IKEA ASM; lr 1e-5, weight decay 1e-5; one video pair per step; ε 0.07, ζ 0.5, α 0.3, r 0.02, ρ 0.35. The paper gives no epoch count; upstream `config.py` says 200.

## 2. Data

| Dataset | State on the workstation | Source |
|---|---|---|
| Pouring | downloaded and converted: 70 train / 14 val videos, 23,694 frames, phase labels 0–4, frame count equals label count for every video | CARL Google Drive copy of the TCC release |
| Penn Action | downloaded and converted: 2,106 videos (1,140 train / 966 val) in 13 per-action folders, 3.4 GB of frames | CARL Google Drive copy (videos + TCC phase labels) |
| IKEA ASM | **not downloaded** | official archives only; the multi-view RGB zip is 136 GB |

The original TCC bucket (`storage.googleapis.com/tcc-pouring`) returns HTTP 403, which is why the CARL copies are used. The earlier V01 audit looked at a different Pouring copy (Sermanet, all-zero labels); that blocker does not apply to this one.

`data/` is git-ignored, so the server has to fetch the data itself:

```bash
python reproductions/tssl/scripts/download_vaot_datasets.py --dest reproductions/tssl/data/downloads/vaot
python -I reproductions/tssl/scripts/prepare_vaot_dataset.py --src reproductions/tssl/data/downloads/vaot/pouring \
    --dst reproductions/tssl/data/final/vaot/pouring --single-action pouring
python -I reproductions/tssl/scripts/prepare_vaot_dataset.py --src reproductions/tssl/data/downloads/vaot/penn_action \
    --dst reproductions/tssl/data/final/vaot/penn_action
```

Resulting layout: `data/final/vaot/<dataset>/<action>/{train,val}/<video>/<idx>.jpg` and `data/final/vaot/<dataset>/npyrecords/<action>_{train,val}.npy`.

IKEA ASM still needs its own preparation step (select the Kallax_Drawer_Shelf videos, 61 train / 29 val per the LAV paper, and build 17-phase frame labels from the official action annotations). Nothing for it exists yet.

## 3. Training

```bash
bash reproductions/tssl/scripts/run_vaot.sh pouring pouring 40 --chunk 0 --workers 10
bash reproductions/tssl/scripts/run_vaot.sh penn_action <action> 20 --chunk 0 --workers 10   # once per action
```

- `scripts/vaot_train.py` wraps the upstream `AlignNet` unchanged (model, loss, sampler, augmentation, optimiser). `--chunk 0` is the exact upstream forward and needs a large GPU.
- `--chunk 16` (default) is a memory workaround for small GPUs: ResNet features are computed without a graph in groups of 16 images, then each group is recomputed and back-propagated with the feature gradient. The one numerical difference is that train-mode BatchNorm in the ResNet sees 16 images instead of the whole step.
- Checkpoints: every 1,000 steps plus a rolling `last.ckpt` under `data/vaot_runs/<dataset>/<action>/ckpts/`; re-running the same command resumes.
- `CONDA_ENV=<name>` overrides the environment name (default `deeplearning`).
- On the cluster these commands must be wrapped in an `sbatch` job per `RULE.md`.

Measured on the workstation (GTX 1650 4 GB, `--chunk 16`): about 6 s/step on Pouring with 40 frames, i.e. roughly 26 h for 200 epochs, and an estimated 9 days for all 13 Penn Action models. That is why training was dropped locally.

## 4. Evaluation

```bash
python reproductions/tssl/scripts/vaot_eval.py --ckpts reproductions/tssl/data/vaot_runs/pouring/pouring/ckpts/model_*.ckpt \
    --data_path reproductions/tssl/data/final/vaot/pouring/pouring \
    --labels_dir reproductions/tssl/data/final/vaot/pouring/npyrecords \
    --action pouring --dest reproductions/tssl/data/vaot_runs/pouring/eval
```

It calls the upstream metric functions (`repos/VAOT/evals`) and writes `results.json` with two protocols per checkpoint:

- **released** — what the published `evaluations.py` actually does: the SVM and the progression regressor are fitted on a fraction of the *validation* videos and scored on all validation videos (`train_embs = val_embs`), and progression targets go through the hard-coded `IDX_TO_CLASS` table (`{1: 5, 2: 13, …}`), which does not match phase ids 0–4.
- **standard** — TCC protocol: fitted on training videos, scored on validation videos, progression targets from the dataset's own phase ids.

Kendall's τ and AP@K use validation embeddings only and are the same in both. Which protocol produced the paper's numbers is not stated; report both.

Do not run evaluation and training on the same small GPU at once: on the workstation the training job died with `CUDA error: unknown error` at the moment an evaluation process started on the same 4 GB card.

## 5. Verification done locally

- **Pouring conversion:** 70 train / 14 val videos, labels {0..4}, zero frame/label length mismatches.
- **Penn Action conversion:** 2,106 videos, zero frame/label length mismatches. All 13 action folder names and their phase counts match upstream `evals/datasets.py` (`DATASET_TO_NUM_CLASSES`). Per-action counts are in `data/final/vaot/penn_action/manifest.json`.
- **Training loop (`--chunk 16`):** ran 12 steps in a smoke test and about 30 steps of the real Pouring run before it was killed by the CUDA error in §4; loss was 3.59 → 3.58, i.e. no learning signal has been observed yet. Checkpoint save and resume path were exercised only by the smoke test's save.
- **`--chunk 0`:** never executed (does not fit in 4 GB).
- **Evaluation script (`vaot_eval.py`):** never completed a run. The first attempts failed on bugs that are now fixed (wrong checkpoint name; chaining `.eval()` on `AlignNet`, whose `train()` returns None); the last attempt was stopped by the user while still computing embeddings. Treat it as untested: the metric code after the embedding step has not executed.
