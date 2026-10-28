"""Post-hoc B-spline evaluation on a trained baseline checkpoint.

Loads a VASOT ckpt, runs the test set, and for each video replaces X with the
B-spline-smoothed X̂ before computing the segmentation OT.  Reports MoF/F1/mIoU.
This isolates whether B-spline *helps a well-trained encoder at inference* —
no training is done.

Usage:
  python eval_bspline_posthoc.py --ckpt <path> --dataset FS|FSeval|Breakfast \
      --tol 0.05 --kmax 32 [--space X|logits] [--arc-reparam]
"""
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import VideoSSL
from video_dataset import VideoDataset
import segmentation_ot
from metrics import ClusteringMetrics
from arc_reparam import arc_u, resample_uniform_u, inverse_warp_predictions
from bspline import smooth_X as bspline_smooth_X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--activity", nargs="+", default=["all"])
    ap.add_argument("--data-root", default="../data")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--kmax", type=int, default=32)
    ap.add_argument("--space", choices=["X", "logits", "none"], default="X")
    ap.add_argument("--arc-reparam", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    model = VideoSSL.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)

    data = VideoDataset(args.data_root, args.dataset, None, standardise=False,
                       random=False, action_class=args.activity)
    loader = DataLoader(data, batch_size=1, shuffle=False)

    mof = ClusteringMetrics(metric="mof")
    f1 = ClusteringMetrics(metric="f1")
    miou = ClusteringMetrics(metric="miou")

    seen = set()
    with torch.no_grad():
        for batch in loader:
            for half in batch:
                feats_raw, mask, gt, fname, _ = half
                fs = str(fname[0]) if isinstance(fname, (list, tuple)) else str(fname)
                if fs in seen:
                    continue
                seen.add(fs)
                feats_raw = feats_raw.to(device)
                mask = mask.to(device)
                gt = gt.to(device)
                B, T, _ = feats_raw.shape

                X = F.normalize(
                    model.mlp(feats_raw.reshape(-1, feats_raw.shape[-1])).reshape(B, T, -1), dim=-1)

                u = None
                if args.arc_reparam:
                    u = arc_u(X, mask)
                    X = resample_uniform_u(X, u)

                if args.space == "X":
                    X = bspline_smooth_X(X, mask, u_target=None, tol=args.tol, k_max=args.kmax)
                    X = F.normalize(X, dim=-1)
                elif args.space == "logits":
                    L = X @ model.clusters.T.unsqueeze(0)
                    L_s = bspline_smooth_X(L, mask, u_target=None, tol=args.tol, k_max=args.kmax)
                    X = F.normalize(L_s @ model.clusters.unsqueeze(0), dim=-1)

                tp = segmentation_ot.temporal_prior(T, model.n_clusters, model.rho, device)
                C = 1.0 - X @ model.clusters.T.unsqueeze(0)
                C = C + tp
                seg, _ = segmentation_ot.segment_ot(
                    C, mask, eps=model.train_eps, alpha=model.alpha_train, radius=model.radius_gw,
                    ub_frames=False, ub_actions=True,
                    lambda_frames=model.lambda_frames_train, lambda_actions=model.lambda_actions_train,
                    n_iters=model.n_ot_train, step_size=model.step_size,
                )
                seg = seg.argmax(dim=2)
                if u is not None:
                    seg = inverse_warp_predictions(seg, u)
                mof.update(seg, gt, mask)
                f1.update(seg, gt, mask)
                miou.update(seg, gt, mask)
    m, p2g = mof.compute(exclude_cls=None)
    f, _ = f1.compute(exclude_cls=None, pred_to_gt=p2g)
    i, _ = miou.compute(exclude_cls=None, pred_to_gt=p2g)
    print(f"space={args.space} arc={args.arc_reparam} tol={args.tol} kmax={args.kmax}  "
          f"MoF={m*100:.2f}  F1={f*100:.2f}  mIoU={i*100:.2f}")


if __name__ == "__main__":
    main()
