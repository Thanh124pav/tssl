# S01 — Frozen Protocol & Config Registry

**Date:** 2026-10-28
**Packet:** S01

## 1. Sources of truth

- **Paper:** Ali et al., "Joint Self-Supervised Video Alignment and Action Segmentation", ICCV 2025 (arXiv:2503.16832)
- **Public repo:** https://github.com/trquhuytin/VASOT-ICCV25 (commit 3f738f3)
- **Official scripts:** `run_bf.sh`, `run_yti.sh`, README inline commands
- **Code defaults:** `train.py` argparse defaults

## 2. Desktop Assembly — Protocol Resolution

### Discrepancy analysis

| Parameter | README command | Code default | version_62534 hparams | Standalone ckpt |
|---|---|---|---|---|
| epochs | 30 | 15 | 30 | 100 (step=7600 = 100×76) |
| rho | 0.25 | 0.1 | 0.25 | unknown |
| radius_gw | 0.02 | 0.04 | 0.02 | unknown |
| layer_sizes | 512 128 40 | 64 128 40 | 512 128 40 | unknown |
| n_clusters | 22 | 8 | 22 | unknown |
| lambda_actions_train | 0.16 | 0.05 | 0.16 | unknown |
| ub_actions | -ua (true) | false | true | unknown |
| std_feats | not in README cmd | false | false | unknown |

### Decision: Follow README (public_repo_config)

The README states: "All hyperparameters are set in the subsequent scripts/commands according to the paper and should be consistent with the reported results."

The README command is:
```bash
python3 train.py -d desktop_assembly -ac all -c 22 -ne 30 --seed 0 \
    --rho 0.25 -lat 0.16 -vf 5 -lr 1e-3 -wd 1e-4 -r 0.02 -ls 512 128 40 -ua
```

The standalone `epoch=99-step=7600.ckpt` suggests 100 epochs was tried at some point, but the README explicitly uses 30. We follow the README.

**Chosen reproduction config (DA):** 30 epochs, rho=0.25, radius_gw=0.02, layers=[512,128,40], K=22, lat=0.16, -ua. **This matches the version_62534 run.**

### Trailing frame policy

34/129 DA videos have features one frame longer than GT. The loader indexes by GT length, so the trailing feature frame is dropped. This is consistent behavior and does not affect metrics since GT defines the evaluation window.

**Policy:** Accept GT-indexed truncation. Document it but do not adjust.

## 3. YouTube Instructions — Protocol Resolution

### Official config from `run_yti.sh`

```bash
python3 src/train.py -d YTI -ac ${activity} -c ${K} -ne 30 -g 0 --seed 0 \
    --rho 0.15 -lat 0.12 -r 0.04 -lr 1e-4 -wd 1e-5 -ls 3000 32 32 -x -1 -vf 5
```

| Activity | K (clusters) |
|---|---:|
| changing_tire | 11 |
| coffee | 10 |
| cpr | 7 |
| jump_car | 12 |
| repot | 8 |

### Key differences from DA

| Parameter | YTI | DA |
|---|---|---|
| layer_sizes | 3000 32 32 | 512 128 40 |
| learning_rate | 1e-4 | 1e-3 |
| weight_decay | 1e-5 | 1e-4 |
| rho | 0.15 | 0.25 |
| lambda_actions_train | 0.12 | 0.16 |
| radius_gw | 0.04 | 0.02 |
| exclude | -1 (exclude class -1) | None |
| ub_actions | false (not passed) | true (-ua) |
| std_feats | false | false |

### Discrepancy check with version_62534 hparams

The version_62534 hparams show `n_clusters=22, layer_sizes=[512,128,40], rho=0.25` — this is the **DA config**, not YTI. However, the dispatcher log shows YTI runs completed successfully. This means:
- YTI runs likely used different Trainer instances with per-run configs
- The hparams.yaml saved only the last-created Trainer's config (DA)
- The actual YTI config per run came from CLI args in the dispatch script

**Implication:** The version_62534 hparams do NOT represent YTI config. YTI runs used the correct `run_yti.sh` parameters. Logs confirm correct activity names and completion.

### Missing-video policy

`repot_0026` is missing from both features and GT — filtered consistently. 149 videos across 5 activities.

### Aggregation

**Chosen aggregation:** per-activity metrics (MoF, F1, mIoU) using `test_*_full` protocol, then arithmetic mean across 5 activities. This matches the paper's reporting style (Table in paper shows per-activity + mean).

## 4. 50 Salads — Already frozen

### FS mid (19 classes)
```bash
python3 train.py -d FS -ac all -c 19 -ne 30 --seed 0 --rho 0.15 -lat 0.15 -vf 5 -lr 1e-3 -wd 1e-4 -ua
```

### FSeval (12 classes)
```bash
python3 train.py -d FSeval -ac all -c 12 -ne 30 --seed 0 --rho 0.15 -lat 0.11 -vf 5 -lr 1e-3 -wd 1e-4 -ua
```

## 5. Breakfast — Already frozen (from earlier reproduction)

Per `run_bf.sh`: 10 activities, 50 epochs each, `-s` (standardize features), rho=0.2, radius_gw=0.04, layers=[64,128,40], lr=1e-3, wd=1e-4, -ae 0.6, -at 0.5.

## 6. Existing run validation

### DA seed-0 run: MATCHES chosen protocol
- version_62534 hparams confirm: layers=[512,128,40], K=22, rho=0.25, radius_gw=0.02, epochs=30
- **Verdict: DA run can be accepted as canonical if S02 runner audit passes.**

### YTI seed-0 runs: CONFIG UNCERTAIN
- Hparams file captured DA config, not YTI config
- Logs show correct activity names and reasonable metrics
- But we cannot confirm exact CLI args from hparams alone
- **Verdict: Provisionally accept; S02 runner should capture full CLI in manifest.**

## 7. Frozen config files

Machine-readable configs are in `reproductions/tssl/configs/vasot/baselines/`.
