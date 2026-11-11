#!/usr/bin/env python3
"""What does softening the alignment target T do to the learning signal?

Alignment loss (train.py, Eq. 7):  L = -sum_j T_ij log P_ij,  P = softmax_j(cos/temp)
  = H(T_i) + KL(T_i || P_i);  dL/dlogit_ij = (P_ij - T_ij) / temp.
For a frozen checkpoint (and for the untrained network) this script builds the
pairwise OT target and softened / sharpened versions of it, and reports per
target, averaged over rows:
  supp    effective support exp(H(T)) in frames
  H       irreducible part of the loss
  KL      reducible part (what the gradient can remove)
  L1      sum_j |T - P|  (gradient magnitude times temp)
  same    target mass on frames of the same GT action   (compare with P's)
  phase   mass-weighted |relative position inside the GT segment, X vs Y|
          over same-action frames (0 = perfect intra-action alignment)
GT labels are used only to score the targets, never to build them.
"""
import argparse, json, random, sys
from pathlib import Path
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diag_common as dc  # noqa: E402


def rel_pos(gt):
    """Relative position (0..1) of each frame inside its contiguous GT segment."""
    n = gt.numel(); out = torch.zeros(n)
    s = 0
    for t in range(1, n + 1):
        if t == n or gt[t] != gt[s]:
            ln = t - s
            out[s:t] = (torch.arange(ln) + 0.5) / ln
            s = t
    return out


def rownorm(t, eps=1e-12):
    return t / t.sum(-1, keepdim=True).clamp_min(eps)


def blur(t, w, mx, my):
    b = F.avg_pool2d(t[None, None], kernel_size=2 * w + 1, stride=1, padding=w,
                     count_include_pad=False)[0, 0]
    return rownorm(b * mx[:, None] * my[None, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--init", action="store_true", help="untrained network instead of the checkpoint")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(2)
    model = dc.load_model(a.ckpt, init_seed=0 if a.init else None)
    data = dc.load_dataset(a.dataset, 256, a.activity, data_root=str(HERE.parent / "data" / "final"))
    names = sorted(data.video_fnames)
    rng = random.Random(0)
    pairs = [tuple(rng.sample(names, 2)) for _ in range(a.pairs)]
    emb = {}
    acc = {}

    def add(tag, t, p, same, dphase, rows):
        t = t[rows]; pp = p[rows]
        h = -(t * t.clamp_min(1e-12).log()).sum(-1)
        ce = -(t * pp.clamp_min(1e-12).log()).sum(-1)
        sm = (t * same[rows]).sum(-1)
        ph = (t * same[rows] * dphase[rows]).sum(-1) / sm.clamp_min(1e-12)
        has = sm > 1e-6
        d = acc.setdefault(tag, {k: [] for k in ("supp", "H", "KL", "L1", "same", "phase")})
        d["supp"].append(h.exp().mean().item()); d["H"].append(h.mean().item())
        d["KL"].append((ce - h).mean().item()); d["L1"].append((t - pp).abs().sum(-1).mean().item())
        d["same"].append(sm.mean().item())
        if has.any():
            d["phase"].append(ph[has].mean().item())

    with torch.no_grad():
        for nx, ny in pairs:
            for n in (nx, ny):
                if n not in emb:
                    e = dc.embed(model, data, n)
                    e["v"] = model._view(e["z"])
                    e["gtf"] = e["gt"].reshape(-1).long()
                    e["rp"] = rel_pos(e["gtf"])
                    emb[n] = e
            ex, ey = emb[nx], emb[ny]
            mx, my = ex["mask"][0].float(), ey["mask"][0].float()
            logits = (ex["v"] @ ey["v"].transpose(1, 2))[0] / model.temp
            p = logits.masked_fill(my[None, :] == 0, -1e4).softmax(-1)
            same = ((ex["gtf"][:, None] == ey["gtf"][None, :]) & (ex["gtf"][:, None] >= 0)).float() * my[None, :]
            dphase = (ex["rp"][:, None] - ey["rp"][None, :]).abs()
            rows = (mx > 0) & (ex["gtf"] >= 0)

            def plan(eps=None):
                t, _ = dc.align_plan(model, ex["v"], ey["v"], ex["mask"], ey["mask"], eps=eps)
                return rownorm(t[0] * my[None, :])

            t = plan()
            rows = rows & (t.sum(-1) > 0.5)
            u = rownorm(my[None, :].expand_as(t))
            add("P (model)", p, p, same, dphase, rows)
            add("pair OT (baseline T)", t, p, same, dphase, rows)
            add("sharpen T^2", rownorm(t ** 2), p, same, dphase, rows)
            add("hard top-1", F.one_hot(t.argmax(-1), t.shape[-1]).float(), p, same, dphase, rows)
            for w in (3, 6, 12):
                add(f"blur W={w}", blur(t, w, mx, my), p, same, dphase, rows)
            for lam in (0.1, 0.3, 0.5):
                add(f"mix uniform {lam}", (1 - lam) * t + lam * u, p, same, dphase, rows)
            for eps in (0.15, 0.30):
                add(f"eps {eps}", plan(eps), p, same, dphase, rows)
            px = dc.seg_plan(model, ex["v"], ex["mask"]); py = dc.seg_plan(model, ey["v"], ey["mask"])
            ind, valid = model.induced_alignment(px, py, mask_y=ey["mask"].float(), hard=True)
            add("hub (block-uniform)", ind[0], p, same, dphase, rows & valid[0])
            add("uniform", u, p, same, dphase, rows)

    res = {k: {m: (sum(v) / len(v) if v else float("nan")) for m, v in d.items()} for k, d in acc.items()}
    print(f"# {a.dataset}/{a.activity}  {'UNTRAINED' if a.init else Path(a.ckpt).parents[1].name}  pairs={len(pairs)}  temp={model.temp}")
    print(f"{'target':24s} {'supp':>7s} {'H':>6s} {'KL':>6s} {'L1':>6s} {'same':>6s} {'phase':>6s}")
    for k, d in res.items():
        print(f"{k:24s} {d['supp']:7.1f} {d['H']:6.3f} {d['KL']:6.3f} {d['L1']:6.3f} {d['same']:6.3f} {d['phase']:6.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
