# VAOT Source Provenance

- **Upstream:** https://github.com/trquhuytin/VAOT-ICCV25.git
- **Commit:** daec4a6d05fef827e19a69ce505423ee4a1ad0e8
- **Import date:** 2026-10-29
- **Method:** rsync --exclude='.git' (no nested .git)

## Compatibility Audit (acvcl env)

### Python: 3.6.13 → 3.10.22 — OK
No Python 3.6-specific constructs found.

### PyTorch Lightning: 1.5.10 → 2.6.6 — NEEDS PORTING
- `from pytorch_lightning.core.lightning import LightningModule` → `from pytorch_lightning import LightningModule`
- `Trainer(gpus=[...])` → `Trainer(devices=[...], accelerator='gpu')`
- `resume_from_checkpoint=` kwarg removed from Trainer
- `CheckpointEveryNSteps` callback in utils.py needs audit
- `CSVLogger` import path may differ

### torchvision: ResNet50 pretrained API
- `models.resnet50(pretrained=True)` → `models.resnet50(weights='IMAGENET1K_V1')`

### Missing pip packages
- easydict — INSTALLED in acvcl
- natsort — INSTALLED in acvcl
- dtw (dtw-python?) — eval only, not yet installed
- seaborn — eval only, not yet installed
- gputil — unused in core code, not needed

### Datasets (from code/paper)
- **Pouring** — hardcoded in evaluations.py as default
- **Penn Action** — multi-action variant in align_dataset_multi_action.py
- **IKEA ASM** — mentioned in eval CSV header comments
- All use raw video frames (jpg), NOT pre-extracted features

### Metrics (VAOT-specific, NOT shared with VASOT)
- Phase Classification (PC) at fractions 0.1, 0.5, 1.0
- Kendall's Tau (sqeuclidean distance)
- Phase Progression
- AP@5, AP@10, AP@15

## Import Test Results (2026-10-29)

| Module | Status | Notes |
|--------|--------|-------|
| config | OK | easydict installed |
| models | OK | |
| asot | OK | Shared ASOT solver |
| utils | OK | |
| align_dataset | OK | |
| train | FAIL | `from pytorch_lightning.core.lightning import LightningModule` — removed in PL 2.x |
| evaluations | FAIL | Same PL import issue |

### PL 2.x Porting Required (train.py, evaluations.py)
1. `from pytorch_lightning.core.lightning import LightningModule` → `from pytorch_lightning import LightningModule`
2. `Trainer(gpus=[...])` → `Trainer(devices=[...], accelerator='gpu')`
3. `resume_from_checkpoint=` kwarg removed
4. `on_batch_end` callback → `on_train_batch_end`
5. `models.resnet50(pretrained=True)` → `models.resnet50(weights='IMAGENET1K_V1')` (deprecation warning)
6. `np.int` → `int` (removed in numpy 1.24+, used in align_dataset.py line 56)

### Data Availability
- **Pouring:** frames at `/mnt/data/thanhpv/acvcl/data/pouring/sermanet_pouring/frames/{train,test}/`
  - FLAT structure (422 train frames, 376 test frames) — VAOT expects per-video directories
  - Need to check if this is per-video or if reorganization needed
- **Penn Action:** NOT FOUND
- **IKEA ASM:** NOT FOUND
- **npyrecords (labels):** NOT FOUND — evaluations.py expects `./npyrecords/pouring_val.npy`

### Code Issues Found
- `utils.py` has duplicated content (file appears concatenated twice, ~250 lines repeated)
- `evaluations.py` has hardcoded paths: `data_path = './Data_Test/'`, `lab_name = "pouring" + "_val"`
