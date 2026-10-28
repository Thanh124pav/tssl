# WANDB_SETUP

Project: **`acvcl-reproduction`** · entity: `hust_edu_vn`

## Credentials

The API key is read at run time from `/home/thanhpv/wandb.txt` by
`reproductions/env.sh`:

```bash
export WANDB_API_KEY="$(tr -d ' \t\r\n' < /home/thanhpv/wandb.txt)"
```

The key is never printed, committed, copied into the repo, or written into any
config, checkpoint or log. `wandb.txt` stays where it is; nothing in
`reproductions/` contains the literal value. W&B state is kept off `$HOME`:

| var | path |
|---|---|
| `WANDB_DIR` | `/mnt/data/thanhpv/acvcl/outputs/wandb` (gitignored) |
| `WANDB_CACHE_DIR` | `/mnt/data/thanhpv/cache/wandb` |
| `WANDB_CONFIG_DIR` | `/mnt/data/thanhpv/cache/wandb/config` |

`api.wandb.ai` is reachable from the compute node, so runs log online; no
offline queue is pending.

## Run naming and grouping

Runs are grouped by research direction via `wandb.init(group=...)`:

| group | runs |
|---|---|
| `data_selection` | `cifar10_<method>_frac<f>_seed<s>` |
| `visual_debiasing` | `<dataset>_<method>_seed<s>` |
| `robot_learning` | `<dataset>_<policy>_seed<s>` |

Tags carry the provenance, e.g. `cords,cifar10,official-config,frac0.1`.

## What gets logged

**Data selection** (`reproductions/data_selection/run_cords.py`) — CORDS has
native W&B support for train/val/test loss and accuracy plus "Data Samples
Encountered(in %)". The runner adds, on every logged step:

- `lr` — read from the live optimizer, so it tracks the cosine schedule
- `gpu_mem_peak_MB` — `torch.cuda.max_memory_allocated()`
- `selection_time_total_s`, `selection_time_last_s`, `selection_count` — from a
  wrapper around `AdaptiveDSSDataLoader.resample`, which is the single place
  every CORDS strategy re-selects its subset

and to the run summary: `final/best_test_acc`, `final/best_val_acc`,
`wall_time_s`, `selection_overhead_pct`, `fraction`, `select_every`, `seed`.

**Visual debiasing** (`reproductions/visual_debiasing/run_cnc.py`) — CnC ships
no W&B support, so the runner hooks `utils.logging.summarize_acc`, the one
function that reduces per-group counts, and logs `<caller>/avg_acc` and
`<caller>/worst_group_acc` (worst-group accuracy is the headline metric for
these papers). Summary carries `final_worst_group_acc` and
`best_worst_group_acc`.

Hyperparameters, epochs and seeds go into `wandb.config` as a flattened copy of
the paper's own config, so the published settings are visible per run.

## Deliberate limits

Datasets and checkpoints are **not** uploaded as artifacts — logging is metrics
only, so no large files leave the cluster. Logging frequency follows each
repo's own eval cadence (CORDS: once per epoch; CnC: once per evaluation pass)
rather than per-step, to keep traffic and shared-storage writes low.
