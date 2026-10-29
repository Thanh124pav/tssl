# V02 — PL 1.x → PL 2.x port of VAOT and dedup of utils.py

**Date:** 2026-10-29
**Scope:** Code port only. No training is performed (V02 run cap = 1 tiny; the
degenerate Pouring labels in V01 make a smoke-only run uninformative. The port
is reusable the moment a labelled dataset arrives.)

## 1. Patches applied (`reproductions/tssl/repos/VAOT`)

| File | Change | Reason |
|---|---|---|
| `utils.py` | Deleted the duplicate second copy of the module (lines 250–498 of the original file). The file now has 249 lines, one definition per name. | PL 1.x utility file was concatenated with itself. Patching only the first copy would leave stale definitions below. |
| `utils.py` | `CheckpointEveryNSteps.on_batch_end(self, trainer, _)` → `on_train_batch_end(self, trainer, *args, **kwargs)`. | PL 2.x renamed the hook. The new signature matches PL 2.x's `(trainer, pl_module, outputs, batch, batch_idx)`. |
| `train.py` | `from pytorch_lightning.core.lightning import LightningModule` → `from pytorch_lightning import ... LightningModule`. | PL 2.x removed the `core.lightning` submodule import path. |
| `train.py` | `Trainer(gpus=[hparams.GPUS], …, resume_from_checkpoint=checkpoint_path)` → `Trainer(devices=[hparams.GPUS], accelerator='gpu', …)` and `trainer.fit(model, ckpt_path=checkpoint_path)`. | PL 2.x removed `gpus` and `resume_from_checkpoint`; resume moved to `fit(ckpt_path=…)`. |
| `models.py` | `models.resnet50(pretrained=pretrained)` → `models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)`. | `pretrained=` is deprecated in torchvision; the weights enum is the API. |
| `align_dataset.py` | `np.int(f)` → `int(f)`. | `np.int` removed in numpy ≥ 1.24. |
| `align_dataset_multi_action.py` | `np.int(f)` → `int(f)`. | Same. |

## 2. Validation

Import test after patches (acvcl env, PL 2.6.6, torchvision 0.18, numpy 2.x):

```
$ for mod in config models utils align_dataset align_dataset_test \
             align_dataset_multi_action asot train evaluations; do \
      python3 -I -c "import sys; sys.path.insert(0,'.'); import $mod; print('OK: $mod')"; \
    done
OK: config
OK: models
OK: utils
OK: align_dataset
OK: align_dataset_test
OK: align_dataset_multi_action
OK: asot
OK: train
OK: evaluations
```

Previously `train` and `evaluations` failed on `from pytorch_lightning.core.lightning import LightningModule`; both now import cleanly.

## 3. What is NOT in V02

- **No training smoke run.** The V01 audit showed `sermanet_pouring` labels are
  identically zero across all splits; a smoke run cannot measure anything the
  paper measures. Running training purely "to see the loop go" would consume
  GPU time without producing any signal and is therefore skipped.
- **No eval smoke run.** `evaluations.py` still has the hardcoded relative paths
  `./Data_Test/` and `./npyrecords/`; wiring those to explicit CLI args is a
  V03 problem, not a port problem, and V03 is blocked on data.
- **No reorganisation of `evals/` or `asot.py`.** Those modules were importable
  before the port and remain so; no PL 1.x surface in them.

## 4. Next step when data arrives

If a labelled VAOT dataset (Pouring with real phase labels, Penn Action, or
IKEA ASM) becomes available on the cluster:

1. Point `evaluations.py:134`'s `lab_name` and `labels` loader to the new
   `npyrecords/`.
2. Run `scripts/prepare_pouring_for_vaot.py` (or an equivalent for Penn/IKEA)
   to produce per-video frame subdirs.
3. Launch training as a V03 packet (own `run_id`, manifest, wandb project).

The port is on `main` as of commit `<F00 commit>`; no further code change is
expected before V03.
