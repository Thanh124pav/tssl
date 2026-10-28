"""Smoke-test the cross-video pipeline end-to-end on COIN annotations.

Pipeline:
    task-level same-task pair sampler
    -> random temporal embeddings (surrogate frame features)
    -> TCC cycle-back-regression loss
    -> backward step
    -> loss should decrease over a few hundred steps.

This does NOT download or decode videos — it stresses the selector / loss /
autograd plumbing so Tests A, C of SETUP.md Section 9 are exercised without
requiring the raw-video features (which are gated behind unreachable hosts).
"""
from __future__ import annotations
import random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.coin import CoinIndex
from src.selectors.random import RandomSameTaskSelector
from src.selectors.similarity import SimilaritySelector
from src.selectors.hard_pair import HardPairSelector
from src.losses.tcc import tcc_cycle_regression_loss


class TinyTemporal(nn.Module):
    def __init__(self, d_in=64, d_out=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Linear(64, d_out))

    def forward(self, x):     # x: (T, d_in)
        return self.net(x)


def build_surrogate_features(ds: CoinIndex, dim: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    task_proto = {t: rng.standard_normal(dim).astype("float32")
                  for t in ds.by_task}
    feats = {}
    for vid, v in ds.videos.items():
        T = max(8, min(32, int(v.duration // 4) or 16))
        base = task_proto[v.task_id]
        noise = rng.standard_normal((T, dim)).astype("float32") * 0.3
        drift = np.linspace(-0.5, 0.5, T, dtype="float32")[:, None]
        feats[vid] = base[None, :] + noise + drift * 0.2
    return feats


def run_selector(name, selector, ds, feats, model, steps=200, pairs_per_step=4,
                 device="cuda"):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = random.Random(0)
    losses = []
    for step in range(steps):
        pairs = selector.select(dataset=ds, num_pairs=pairs_per_step, rng=rng)
        batch_loss = 0.0
        n = 0
        for a, b in pairs:
            fa, fb = feats[a], feats[b]
            T = min(fa.shape[0], fb.shape[0], 16)
            fa, fb = fa[:T], fb[:T]
            ta = torch.from_numpy(fa).to(device)
            tb = torch.from_numpy(fb).to(device)
            ea, eb = model(ta), model(tb)
            loss = tcc_cycle_regression_loss(ea, eb)
            batch_loss = batch_loss + loss
            n += 1
        batch_loss = batch_loss / max(n, 1)
        opt.zero_grad(); batch_loss.backward(); opt.step()
        losses.append(float(batch_loss.item()))
    return losses


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    ds = CoinIndex(ROOT / "data" / "coin" / "COIN.json")
    print(f"[coin] {ds.num_videos} videos / {ds.num_tasks} tasks")

    feats = build_surrogate_features(ds, dim=64, seed=0)
    pooled = {k: v.mean(0) for k, v in feats.items()}

    selectors = [
        ("random",     RandomSameTaskSelector()),
        ("similarity", SimilaritySelector(pooled)),
        ("hard_pair",  HardPairSelector(pooled, pooled)),
    ]
    report = []
    for name, sel in selectors:
        torch.manual_seed(0)
        model = TinyTemporal().to(device)
        losses = run_selector(name, sel, ds, feats, model,
                              steps=200, pairs_per_step=4, device=device)
        head, tail = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
        report.append((name, head, tail))
        print(f"  [{name:11s}] loss  start={head:.3f}  end={tail:.3f}  "
              f"delta={head - tail:+.3f}")

    print("\n== summary ==")
    print(f"{'selector':12s} {'loss@start':>12s} {'loss@end':>10s} {'drop':>10s}")
    for n, h, t in report:
        print(f"{n:12s} {h:12.3f} {t:10.3f} {h - t:+10.3f}")


if __name__ == "__main__":
    main()
