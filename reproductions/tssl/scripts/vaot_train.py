#!/usr/bin/env python3
"""Train VAOT (official AlignNet) on one alignment dataset, on a small GPU.

The model, loss, sampler, augmentations and optimiser are the upstream ones
(`repos/VAOT/train.py::AlignNet`). Only the ResNet-50 forward/backward is
re-organised so it fits in a few GB of VRAM:

  1. frames go through `base_cnn` without a graph, `--chunk` images at a time;
  2. the embedder + ASOT loss run on those features and give dL/dfeatures;
  3. each chunk is recomputed with a graph and back-propagated with that
     gradient (BatchNorm momentum 0, so running stats update once per step).

The only numerical difference from upstream is that train-mode BatchNorm in
the ResNet sees `--chunk` images per batch instead of all 4*num_frames
images. Chunks are drawn from a random permutation of the step's frames so
each one mixes both videos and all time positions.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

VAOT = Path(__file__).resolve().parents[1] / "repos" / "VAOT"
sys.path.insert(0, str(VAOT))

from pytorch_lightning import Callback, Trainer, seed_everything  # noqa: E402
from pytorch_lightning.loggers import CSVLogger  # noqa: E402

from config import CONFIG  # noqa: E402
from train import AlignNet  # noqa: E402


class ChunkedAlignNet(AlignNet):
    chunk = 16

    def forward(self, x):
        if self.chunk <= 0 or not (self.training and torch.is_grad_enabled()):
            return super().forward(x)  # --chunk 0: exact upstream forward (needs a large GPU)
        num_frames = x.size(1) // self.hparams.config.DATA.NUM_CONTEXT
        b, s = x.shape[:2]
        flat = x.reshape(b * s, 1, *x.shape[2:])
        perm = torch.randperm(b * s, device=x.device)
        chunks = perm.split(self.chunk)
        with torch.no_grad():
            outs = [self.base_cnn(flat[idx]) for idx in chunks]
        feats = torch.empty((b * s, *outs[0].shape[2:]), device=x.device, dtype=outs[0].dtype)
        for idx, out in zip(chunks, outs):
            feats[idx] = out[:, 0]
        del outs
        feats = feats.reshape(b, s, *feats.shape[1:]).requires_grad_(True)
        self._pending = (flat, chunks, feats)
        return self.emb(feats, num_frames)

    def on_after_backward(self):
        pending = getattr(self, "_pending", None)
        if pending is None:
            return
        self._pending = None
        flat, chunks, feats = pending
        grad = feats.grad.reshape(-1, *feats.shape[2:])
        bns = [m for m in self.base_cnn.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        saved = [m.momentum for m in bns]
        for m in bns:
            m.momentum = 0.0
        try:
            for idx in chunks:
                out = self.base_cnn(flat[idx])[:, 0]
                (out * grad[idx]).sum().backward()
        finally:
            for m, mom in zip(bns, saved):
                m.momentum = mom


class StepCheckpoints(Callback):
    """Numbered checkpoint every `every` steps plus a rolling `last.ckpt`."""

    def __init__(self, dirpath, every, last_every):
        self.dir, self.every, self.last_every = Path(dirpath), every, last_every
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0, self.s0 = None, None

    def on_train_batch_end(self, trainer, *args, **kwargs):
        step = trainer.global_step
        if self.t0 is None:
            self.t0, self.s0 = time.time(), step
        if step % self.every == 0:
            trainer.save_checkpoint(self.dir / f"model_epoch={trainer.current_epoch}_step={step}.ckpt")
        if step % self.last_every == 0:
            tmp = self.dir / "last.tmp"
            trainer.save_checkpoint(tmp)
            os.replace(tmp, self.dir / "last.ckpt")
            rate = (time.time() - self.t0) / max(1, step - self.s0)
            print(f"[step {step}] epoch {trainer.current_epoch} {rate:.2f} s/step "
                  f"loss {float(trainer.callback_metrics.get('train_loss', float('nan'))):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="dir containing train/<video>/<frame>.jpg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_frames", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=CONFIG.TRAIN.EPOCHS)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=16,
                    help="images per ResNet pass; 0 = upstream full-batch forward (>=24 GB VRAM)")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--last_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Upstream train.py defaults (= paper Table 4: eps 0.07, zeta 0.5,
    # alpha 0.3, r 0.02, rho 0.35, lr 1e-5, wd 1e-5, one video pair per step).
    CONFIG.GPUS = 0
    CONFIG.SEED = args.seed
    CONFIG.DATA_PATH = args.data_path
    CONFIG.ROOT = args.out
    CONFIG.CKPT_PATH = os.path.join(args.out, "ckpts")
    CONFIG.TRAIN.NUM_FRAMES = CONFIG.EVAL.NUM_FRAMES = args.num_frames
    CONFIG.TRAIN.EPOCHS = args.epochs
    CONFIG.DATA.WORKERS = args.workers
    CONFIG.RESUME_FROM = None
    CONFIG.ALPHA_TRAIN, CONFIG.ALPHA_EVAL = 0.3, 0.6
    CONFIG.N_OT_TRAIN, CONFIG.N_OT_EVAL = [25, 1], [25, 1]
    CONFIG.EPS_TRAIN, CONFIG.EPS_EVAL = 0.07, 0.04
    CONFIG.RADIUS_GW = 0.02
    CONFIG.LAMBDA_FRAMES_TRAIN = CONFIG.LAMBDA_ACTIONS_TRAIN = 0.05
    CONFIG.LAMBDA_FRAMES_EVAL, CONFIG.LAMBDA_ACTIONS_EVAL = 0.05, 0.01
    CONFIG.RHO, CONFIG.ZETA = 0.35, 0.50
    CONFIG.STEP_SIZE = None
    CONFIG.UB_FRAMES = CONFIG.UB_ACTIONS = False

    seed_everything(args.seed)
    ChunkedAlignNet.chunk = args.chunk
    model = ChunkedAlignNet(CONFIG)
    ckpt_dir = Path(CONFIG.CKPT_PATH)
    last = ckpt_dir / "last.ckpt"
    trainer = Trainer(devices=[0], accelerator="gpu", max_epochs=args.epochs, max_steps=args.max_steps,
                      default_root_dir=args.out, deterministic=True,
                      callbacks=[StepCheckpoints(ckpt_dir, args.save_every, args.last_every)],
                      limit_val_batches=0, num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, logger=CSVLogger(save_dir=args.out, name="csv"),
                      log_every_n_steps=10)
    trainer.fit(model, ckpt_path=str(last) if last.exists() else None)
    trainer.save_checkpoint(ckpt_dir / f"model_epoch={trainer.current_epoch}_step={trainer.global_step}.ckpt")


if __name__ == "__main__":
    main()
