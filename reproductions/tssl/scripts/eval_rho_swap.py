"""Post-hoc test of the segmentation temporal prior without retraining.

This distinguishes the inference-time effect of rho_seg from representation
changes caused by training with a different prior. It uses the same full-video
test loader and metrics as train.py, but never saves or modifies a checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parents[1] / "repos" / "VASOT" / "src"
sys.path.insert(0, str(SRC))

from train import VideoSSL  # noqa: E402
from video_dataset import VideoDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", default="FSeval")
    parser.add_argument("--activity", default="all")
    parser.add_argument("--rho-seg", type=float, required=True)
    parser.add_argument("--data-root", default=str(SRC.parent / "data"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = VideoSSL.load_from_checkpoint(args.ckpt, map_location="cpu", weights_only=False)
    trained_rho_seg = model.rho_seg
    model.rho_seg = args.rho_seg
    model.visualize = False
    dataset = VideoDataset(args.data_root, args.dataset, None, standardise=False,
                           random=False, action_class=[args.activity])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(accelerator=accelerator, devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    metrics = trainer.test(model, dataloaders=loader)[0]
    result = {"checkpoint": str(Path(args.ckpt).resolve()),
              "dataset": args.dataset, "activity": args.activity,
              "trained_rho_seg": trained_rho_seg,
              "eval_rho_seg": args.rho_seg, "accelerator": accelerator,
              "metrics": metrics}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
