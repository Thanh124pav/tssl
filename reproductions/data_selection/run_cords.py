"""Run one CORDS supervised data-selection experiment with W&B logging.

Wraps the official `train_sl.TrainClassifier` and its shipped configs so the
published hyperparameters stay authoritative; only the knobs passed on the
command line are overridden.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "third_party" / "cords"))
sys.path.insert(0, str(HERE))

import cords_compat

cords_compat.install_all()

import torch  # noqa: E402
import wandb  # noqa: E402
from cords.utils.config_utils import load_config_data  # noqa: E402
from cords.utils.data.dataloader.SL.adaptive.adaptivedataloader import (  # noqa: E402
    AdaptiveDSSDataLoader,
)
from train_sl import TrainClassifier  # noqa: E402

SELECTION = {"total_s": 0.0, "count": 0, "last_s": 0.0}


def instrument_selection():
    original = AdaptiveDSSDataLoader.resample

    def timed_resample(self):
        start = time.perf_counter()
        out = original(self)
        elapsed = time.perf_counter() - start
        SELECTION["total_s"] += elapsed
        SELECTION["count"] += 1
        SELECTION["last_s"] = elapsed
        return out

    AdaptiveDSSDataLoader.resample = timed_resample


class InstrumentedTrainer(TrainClassifier):
    def optimizer_with_scheduler(self, model):
        optimizer, scheduler = super().optimizer_with_scheduler(model)
        self.captured_optimizer = optimizer
        return optimizer, scheduler


def instrument_wandb(trainer):
    original = wandb.log

    def enriched_log(payload, *args, **kwargs):
        payload = dict(payload)
        optimizer = getattr(trainer, "captured_optimizer", None)
        if optimizer is not None:
            payload["lr"] = optimizer.param_groups[0]["lr"]
        if torch.cuda.is_available():
            payload["gpu_mem_peak_MB"] = round(
                torch.cuda.max_memory_allocated() / 2**20, 1
            )
        payload["selection_time_total_s"] = round(SELECTION["total_s"], 3)
        payload["selection_time_last_s"] = round(SELECTION["last_s"], 3)
        payload["selection_count"] = SELECTION["count"]
        return original(payload, *args, **kwargs)

    wandb.log = enriched_log


def flatten(cfg, prefix=""):
    flat = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if hasattr(value, "items"):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--select-every", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datadir", default=str(REPO / "data" / "cords"))
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--project", default="acvcl-reproduction")
    parser.add_argument("--group", default="data_selection")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tags", default="")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config_data(args.config)
    cfg.dataset.datadir = args.datadir
    cfg.train_args.results_dir = args.results_dir
    cfg.ckpt.dir = args.results_dir
    cfg.ckpt.is_save = False
    if args.epochs is not None:
        cfg.train_args.num_epochs = args.epochs
        if cfg.scheduler.type == "cosine_annealing":
            cfg.scheduler.T_max = args.epochs
    if args.fraction is not None:
        cfg.dss_args.fraction = args.fraction
    if args.select_every is not None:
        cfg.dss_args.select_every = args.select_every
    cfg.train_args.wandb = not args.no_wandb
    cfg.train_args.print_every = max(1, cfg.train_args.num_epochs // 50)

    torch.manual_seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    instrument_selection()

    if not args.no_wandb:
        config = flatten(cfg)
        config["seed"] = args.seed
        wandb.init(
            project=args.project,
            name=args.run_name,
            group=args.group,
            tags=[t for t in args.tags.split(",") if t],
            config=config,
            dir=os.environ.get("WANDB_DIR", args.results_dir),
        )

    trainer = InstrumentedTrainer(cfg)
    if not args.no_wandb:
        instrument_wandb(trainer)

    wall_start = time.perf_counter()
    trn_acc, val_acc, tst_acc, best_acc, timing = trainer.train()
    wall_s = time.perf_counter() - wall_start

    summary = {
        "run_name": args.run_name,
        "method": cfg.dss_args.type,
        "dataset": cfg.dataset.name,
        "architecture": cfg.model.architecture,
        "fraction": cfg.dss_args.fraction,
        "select_every": cfg.dss_args.select_every,
        "epochs": cfg.train_args.num_epochs,
        "seed": args.seed,
        "final_test_acc": float(tst_acc[-1]) if len(tst_acc) else None,
        "best_test_acc": float(max(tst_acc)) if len(tst_acc) else None,
        "final_val_acc": float(val_acc[-1]) if len(val_acc) else None,
        "best_val_acc": float(max(val_acc)) if len(val_acc) else None,
        "final_train_acc": float(trn_acc[-1]) if len(trn_acc) else None,
        "cords_reported_best": [float(b) for b in best_acc] if best_acc is not None else None,
        "train_time_s_cords": float(timing[-1]) if len(timing) else None,
        "wall_time_s": round(wall_s, 2),
        "selection_time_total_s": round(SELECTION["total_s"], 3),
        "selection_count": SELECTION["count"],
        "selection_overhead_pct": (
            round(100.0 * SELECTION["total_s"] / wall_s, 2) if wall_s > 0 else None
        ),
        "gpu_mem_peak_MB": (
            round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if torch.cuda.is_available()
            else None
        ),
    }

    out_path = Path(args.results_dir) / f"{args.run_name}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if not args.no_wandb:
        wandb.summary.update(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
