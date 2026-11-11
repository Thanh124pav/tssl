#!/usr/bin/env python3
"""Scale of every spread regulariser (repos/VASOT/src/spread_reg.py) against
the alignment loss, on a frozen checkpoint and on the untrained network.

For random video pairs: gradient norm of each loss w.r.t. the frame
embeddings (through the unit normalisation), and the ratio to the gradient of
the alignment CE (Eq. 7 with the pairwise OT target). Used to pick one weight
per regulariser so the arms are compared at a similar strength.
  calib_spread.py --ckpt <last.ckpt> [--dataset FSeval] [--activity all] [--init]
"""
import argparse, random, sys
from pathlib import Path
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diag_common as dc  # noqa: E402  (puts repos/VASOT/src on the path)
from spread_reg import NAMES, spread_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--init", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    model = dc.load_model(a.ckpt, init_seed=0 if a.init else None)
    data = dc.load_dataset(a.dataset, 256, a.activity, data_root=str(HERE.parent / "data" / "final"))
    names = sorted(data.video_fnames)
    rng = random.Random(0)
    acc = {k: {"loss": [], "grad": [], "ratio": []} for k in ("align",) + NAMES}
    for _ in range(a.pairs):
        nx, ny = rng.sample(names, 2)
        with torch.no_grad():
            ex, ey = dc.embed(model, data, nx), dc.embed(model, data, ny)
            t, _ = dc.align_plan(model, ex["z"], ey["z"], ex["mask"], ey["mask"])
            ax = dc.seg_plan(model, ex["z"], ex["mask"])[0].argmax(-1)
            ay = dc.seg_plan(model, ey["z"], ey["mask"])[0].argmax(-1)
        mx, my = ex["mask"][0].bool(), ey["mask"][0].bool()
        ux = ex["z"][0].clone().requires_grad_(True)
        uy = ey["z"][0].clone().requires_grad_(True)

        def grad(loss):
            g = torch.autograd.grad(loss, (ux, uy), allow_unused=True)
            return torch.cat([x.reshape(-1) for x in g if x is not None]).norm().item()

        zx, zy = F.normalize(ux, dim=-1), F.normalize(uy, dim=-1)
        logp = (zx @ zy.T / model.temp).log_softmax(-1)
        l_align = -((t[0] * logp).sum(-1) * mx).mean()
        g_align = grad(l_align)
        acc["align"]["loss"].append(l_align.item()); acc["align"]["grad"].append(g_align); acc["align"]["ratio"].append(1.0)
        for name in NAMES:
            zx, zy = F.normalize(ux, dim=-1), F.normalize(uy, dim=-1)
            loss = spread_loss(name, [zx[mx], zy[my]], assigns=[ax[mx], ay[my]],
                               protos=F.normalize(model.clusters.detach(), dim=-1), temp=model.temp)
            g = grad(loss)
            acc[name]["loss"].append(loss.item()); acc[name]["grad"].append(g); acc[name]["ratio"].append(g / g_align)
    tag = "UNTRAINED" if a.init else Path(a.ckpt).parents[1].name
    print(f"# {a.dataset}/{a.activity} {tag} pairs={a.pairs}")
    print(f"{'loss':10s} {'value':>9s} {'|grad|':>9s} {'ratio':>8s}")
    for k, d in acc.items():
        m = lambda v: sum(v) / len(v)
        print(f"{k:10s} {m(d['loss']):9.3f} {m(d['grad']):9.4f} {m(d['ratio']):8.3f}")


if __name__ == "__main__":
    main()
