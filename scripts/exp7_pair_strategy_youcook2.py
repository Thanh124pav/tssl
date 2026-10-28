"""Experiment 7: controlled pair-strategy comparison on YouCook2.

Uses real precomputed InternVideo-MM-L14 (768-D, 8 fps) features that are
grouped by recipe-id (task) on the sermanet mirror hosted on HF. Everything
is held constant between runs except the pair-selection strategy:
    * random same-task
    * cosine-nearest same-task
    * hard_pair (high semantic sim, low prediction sim)

We train a small Transformer-based temporal head with the TCC
cycle-regression loss and report:
    * loss convergence
    * frame-wise cycle-consistency on held-out same-task pairs
    * runtime
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses.tcc import tcc_cycle_regression_loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEAT_DIM = 768


# --------------------------- data index ------------------------------ #
def build_index(root: Path):
    """Return (features_dict, by_task_dict). features_dict[vid] -> (T, 768)."""
    features = {}
    by_task = {}
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        try:
            task_id = int(task_dir.name)
        except ValueError:
            continue
        for f in task_dir.glob("*.pth.tar"):
            vid = f.stem.replace(".pth", "")
            try:
                feat = torch.load(f, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if feat.ndim != 2 or feat.shape[1] != FEAT_DIM or feat.shape[0] < 8:
                continue
            features[vid] = feat.float()
            by_task.setdefault(task_id, []).append(vid)
    return features, by_task


def pool(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.mean(0), dim=-1)


# ------------------------- selectors --------------------------------- #
class RandomSel:
    name = "random"
    def pick(self, feats, by_task, rng):
        t = rng.choice([t for t, vs in by_task.items() if len(vs) >= 2])
        a, b = rng.sample(by_task[t], 2); return a, b


class SimilaritySel:
    name = "similarity"
    def __init__(self, feats): self.pooled = {k: pool(v) for k, v in feats.items()}
    def pick(self, feats, by_task, rng):
        t = rng.choice([t for t, vs in by_task.items() if len(vs) >= 2])
        vids = by_task[t]
        a = rng.choice(vids)
        va = self.pooled[a]
        b, bs = None, -1.0
        for cand in vids:
            if cand == a: continue
            s = float(va @ self.pooled[cand])
            if s > bs: b, bs = cand, s
        return a, b


class HardPairSel:
    name = "hard_pair"
    def __init__(self, feats, preds):
        self.feat_pool = {k: pool(v) for k, v in feats.items()}
        self.pred_pool = {k: pool(v) for k, v in preds.items()}
    def pick(self, feats, by_task, rng):
        t = rng.choice([t for t, vs in by_task.items() if len(vs) >= 2])
        vids = by_task[t]
        a = rng.choice(vids)
        b, bs = None, -1e9
        fa, pa = self.feat_pool[a], self.pred_pool[a]
        for cand in vids:
            if cand == a: continue
            sim  = float(fa @ self.feat_pool[cand])
            disc = float(pa @ self.pred_pool[cand])
            score = sim - disc
            if score > bs: b, bs = cand, score
        return a, b


# -------------------- temporal head ---------------------------------- #
class TemporalHead(nn.Module):
    def __init__(self, d_in=768, d_out=128):
        super().__init__()
        self.proj = nn.Linear(d_in, 256)
        enc = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512,
                                         batch_first=True, dropout=0.0, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, num_layers=2)
        self.out = nn.Linear(256, d_out)
    def forward(self, x):               # (T, 768)
        h = self.proj(x).unsqueeze(0)   # (1, T, 256)
        h = self.tr(h).squeeze(0)       # (T, 256)
        return self.out(h)              # (T, 128)


# ---------------- eval: pairwise cycle consistency ------------------- #
@torch.no_grad()
def eval_cycle(model, feats, by_task, val_pairs, max_T=64):
    model.eval()
    scores = []
    for a, b in val_pairs:
        xa = feats[a].to(DEVICE)
        xb = feats[b].to(DEVICE)
        if xa.shape[0] > max_T: xa = xa[torch.linspace(0, xa.shape[0]-1, max_T).long()]
        if xb.shape[0] > max_T: xb = xb[torch.linspace(0, xb.shape[0]-1, max_T).long()]
        ea, eb = model(xa), model(xb)
        T = min(ea.shape[0], eb.shape[0])
        ea, eb = ea[:T], eb[:T]
        d12 = torch.cdist(ea, eb)
        i12 = d12.argmin(1)
        d21 = torch.cdist(eb[i12], ea)
        i21 = d21.argmin(1)
        scores.append(float((i21 == torch.arange(T, device=DEVICE)).float().mean().item()))
    model.train()
    return sum(scores) / max(len(scores), 1)


# ------------------------ one run ------------------------------------ #
def run(name, selector, feats, by_task, val_pairs, iters=400,
        pairs_per_iter=2, num_frames=48, lr=1e-3):
    torch.manual_seed(0); random.seed(0)
    model = TemporalHead().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(42)
    losses, cycles = [], []
    t0 = time.time()
    for it in range(iters):
        model.train()
        total = 0.0
        for _ in range(pairs_per_iter):
            a, b = selector.pick(feats, by_task, rng)
            xa = feats[a].to(DEVICE); xb = feats[b].to(DEVICE)
            if xa.shape[0] > num_frames: xa = xa[torch.linspace(0, xa.shape[0]-1, num_frames).long()]
            if xb.shape[0] > num_frames: xb = xb[torch.linspace(0, xb.shape[0]-1, num_frames).long()]
            T = min(xa.shape[0], xb.shape[0], num_frames)
            if T < 8: continue
            xa, xb = xa[:T], xb[:T]
            ea, eb = model(xa), model(xb)
            total = total + tcc_cycle_regression_loss(ea, eb)
        total = total / pairs_per_iter
        opt.zero_grad(); total.backward(); opt.step()
        losses.append(float(total.item()))
        if (it + 1) % 50 == 0:
            cyc = eval_cycle(model, feats, by_task, val_pairs)
            cycles.append((it + 1, cyc))
            print(f"  [{name:11s}] it {it+1:4d}  loss={total.item():.3f}  val_cycle={cyc:.3f}")
    final_cyc = eval_cycle(model, feats, by_task, val_pairs)
    dt = time.time() - t0
    return {"name": name, "losses": losses, "cycle_curve": cycles,
            "final_cycle": final_cyc, "runtime_s": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/thanhpv/acvcl/data/youcook2/internvideo_features/internvideo_MM_L14_features")
    ap.add_argument("--out",  default="/home/thanhpv/acvcl/outputs/exp7_youcook2")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--val-pairs", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"loading features from {args.root}")
    feats, by_task = build_index(Path(args.root))
    tasks_ok = {t: v for t, v in by_task.items() if len(v) >= 2}
    print(f"loaded {len(feats)} videos across {len(by_task)} tasks "
          f"({len(tasks_ok)} tasks with >=2 videos)")
    if len(tasks_ok) < 2:
        raise SystemExit("not enough same-task pairs available")

    # Build held-out val pairs: pick N pairs with task size >=3; set them aside
    rng_val = random.Random(123)
    val_pairs = []
    multi = [t for t, v in tasks_ok.items() if len(v) >= 3]
    for _ in range(args.val_pairs):
        t = rng_val.choice(multi); a, b = rng_val.sample(tasks_ok[t], 2)
        val_pairs.append((a, b))

    # Shared pooled features for similarity + hard_pair
    selectors = [
        RandomSel(),
        SimilaritySel(feats),
        HardPairSel(feats, feats),   # use features as both repr + pred (toy)
    ]
    results = []
    for sel in selectors:
        print(f"\n[selector] {sel.name}")
        r = run(sel.name, sel, feats, tasks_ok, val_pairs,
                iters=args.iters, pairs_per_iter=2, num_frames=48, lr=1e-3)
        results.append(r)
        print(f"  final cycle = {r['final_cycle']:.3f}  ({r['runtime_s']:.0f}s)")

    print("\n== summary ==")
    print(f"{'selector':12s} {'loss@start':>11s} {'loss@end':>10s} {'final_cycle':>12s} {'runtime':>10s}")
    for r in results:
        h = sum(r["losses"][:20]) / 20; t = sum(r["losses"][-20:]) / 20
        print(f"{r['name']:12s} {h:11.3f} {t:10.3f} {r['final_cycle']:12.3f} {r['runtime_s']:9.0f}s")

    with (out / "results.json").open("w") as f:
        json.dump([{k: v for k, v in r.items() if k != "losses"} for r in results], f, indent=2)
    for r in results:
        np.save(out / f"losses_{r['name']}.npy", np.array(r["losses"]))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
