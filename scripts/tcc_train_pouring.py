"""Minimal TCC training on the Pouring dataset.

Loads sermanet/pouring MOVs, encodes with a frozen DINOv2 backbone, trains a
small temporal head, and optimizes the TCC cycle-back-regression loss across
pairs of videos. This is a port — NOT a full CARL reproduction — intended to
verify Experiment 1 of SETUP.md §6 end-to-end on real video on this H100 MIG
slice.

Metrics printed:
  * running training loss (should decrease)
  * frame-wise cycle consistency on val pairs
  * average per-video embedding norm (sanity)

The script is intentionally compact so it fits within our 4-CPU / 42 GB-GPU
budget. Set batch_size=2 pairs and NUM_FRAMES=32.
"""
from __future__ import annotations
import argparse, pickle, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_video

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses.tcc import tcc_cycle_regression_loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- data ---------------------------------- #
class PouringPairs(torch.utils.data.Dataset):
    """Returns a single (name, uint8 tensor T x H x W x 3) per __getitem__."""

    def __init__(self, root: Path, split: str, num_frames: int = 32, size: int = 224):
        with (root / f"{split}.pkl").open("rb") as f:
            self.items = pickle.load(f)
        self.root = root
        self.num_frames = num_frames
        self.size = size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        v, _, _ = read_video(str(self.root / it["video_file"]), pts_unit="sec")
        if v.shape[0] < 2:
            raise RuntimeError(f"empty video {it['name']}")
        # uniform-sample num_frames
        idxs = torch.linspace(0, v.shape[0] - 1, self.num_frames).long()
        v = v[idxs]                           # (T, H, W, 3) uint8
        v = v.permute(0, 3, 1, 2).float() / 255.0           # (T, 3, H, W)
        v = F.interpolate(v, size=(self.size, self.size),
                          mode="bilinear", align_corners=False)
        # Normalize using ImageNet stats (DINOv2 uses these)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        v = (v - mean) / std
        return it["name"], v


# ----------------------------- model --------------------------------- #
class Encoder(nn.Module):
    """Frozen DINOv2-small (CLS) + small temporal conv head."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small").eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        d = self.backbone.config.hidden_size            # 384
        self.temporal = nn.Sequential(
            nn.Conv1d(d, 256, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(256, embed_dim, kernel_size=3, padding=1),
        )

    @torch.no_grad()
    def _features(self, video: torch.Tensor) -> torch.Tensor:
        # video: (T, 3, H, W) float on device
        out = self.backbone(pixel_values=video).last_hidden_state[:, 0]  # (T, d)
        return out

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        f = self._features(video)                       # (T, d)
        h = self.temporal(f.T.unsqueeze(0)).squeeze(0).T  # (T, embed_dim)
        return h


# ----------------------------- eval ---------------------------------- #
@torch.no_grad()
def cycle_consistency(e1: torch.Tensor, e2: torch.Tensor) -> float:
    """Fraction of frames in e1 that cycle back to themselves via e2."""
    T = e1.shape[0]
    d12 = torch.cdist(e1, e2)
    i12 = d12.argmin(1)                                  # (T,)
    d21 = torch.cdist(e2[i12], e1)
    i21 = d21.argmin(1)                                  # (T,)
    return float((i21 == torch.arange(T, device=e1.device)).float().mean().item())


# ----------------------------- train --------------------------------- #
def train(root: Path, out_dir: Path, iters: int = 400, pairs_per_iter: int = 2,
          num_frames: int = 32, lr: float = 3e-4):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = PouringPairs(root, "train", num_frames=num_frames)
    val_ds   = PouringPairs(root, "val",   num_frames=num_frames)
    print(f"train videos: {len(train_ds)}  val videos: {len(val_ds)}")

    model = Encoder().to(DEVICE)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    rng = random.Random(0)

    # ---------- precompute val embeddings once per 50 iters -------------- #
    def eval_cycles() -> float:
        model.eval()
        if len(val_ds) < 2:
            return float("nan")
        scores = []
        for i in range(min(len(val_ds), 4)):
            for j in range(i + 1, min(len(val_ds), 4)):
                _, va = val_ds[i]
                _, vb = val_ds[j]
                ea = model(va.to(DEVICE))
                eb = model(vb.to(DEVICE))
                scores.append(cycle_consistency(ea, eb))
        model.train()
        return sum(scores) / max(len(scores), 1)

    train_losses = []
    val_curve = []
    t0 = time.time()
    for it in range(iters):
        model.train()
        total = 0.0
        for _ in range(pairs_per_iter):
            i, j = rng.sample(range(len(train_ds)), 2)
            _, va = train_ds[i]
            _, vb = train_ds[j]
            ea = model(va.to(DEVICE))
            eb = model(vb.to(DEVICE))
            # same number of frames in both -> toy TCC loss works directly
            loss = tcc_cycle_regression_loss(ea, eb)
            total = total + loss
        total = total / pairs_per_iter
        optim.zero_grad()
        total.backward()
        optim.step()
        train_losses.append(float(total.item()))
        if it % 20 == 0:
            cyc = eval_cycles()
            val_curve.append((it, cyc))
            print(f"it {it:4d}  loss={total.item():.3f}  val_cycle={cyc:.3f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    # final eval
    final_cyc = eval_cycles()
    print(f"\nfinal val cycle consistency: {final_cyc:.3f}")

    # save
    torch.save(model.state_dict(), out_dir / "tcc_pouring.pt")
    with (out_dir / "metrics.json").open("w") as f:
        import json
        json.dump({
            "iters": iters,
            "pairs_per_iter": pairs_per_iter,
            "num_frames": num_frames,
            "lr": lr,
            "train_losses": train_losses,
            "val_curve": val_curve,
            "final_val_cycle": final_cyc,
        }, f, indent=2)
    print(f"saved to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/thanhpv/acvcl/data/pouring/sermanet_pouring")
    ap.add_argument("--out",  default="/home/thanhpv/acvcl/outputs/tcc_pouring")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--pairs-per-iter", type=int, default=2)
    ap.add_argument("--num-frames", type=int, default=32)
    args = ap.parse_args()
    train(Path(args.root), Path(args.out), iters=args.iters,
          pairs_per_iter=args.pairs_per_iter, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
