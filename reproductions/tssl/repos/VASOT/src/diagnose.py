"""Limitation diagnostics for VASOT.

For each video in a test set, compute:
  (1) T* row entropy — sparsity of segmentation assignment
  (2) Embedding smoothness ||X_{i+1} - X_i|| vs ground-truth action boundaries
  (3) Correlation between structural-prior cost |i-k|/N and semantic equality GT(i)==GT(k)
  (4) Hungarian cluster->class confusion matrix on predicted segments

Hypotheses probed:
  (A) T sub-representation      -> if row entropy << log(K), supervision is top-1-ish.
  (B) Intra-video loss          -> if ||dX|| spikes correlate weakly with GT boundaries, encoder is not learning boundaries.
  (C) Frame-indexing adversity  -> if correlation(|i-k|/N, [GT(i)==GT(k)]) is low, the step-function structural prior misleads OT.

Usage:
  python diagnose.py --ckpt <path> --dataset FS|FSeval|Breakfast --activity all [--n-clusters 19]

Outputs JSON to <ckpt_dir>/diagnostics.json plus histograms in <ckpt_dir>/diag_plots/.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import VideoSSL
from video_dataset import VideoDataset
import segmentation_ot


def row_entropy(T, mask):
    """T: [N,K]; returns array of row entropies for valid frames, normalized by log K."""
    T = T[mask]
    T = T / (T.sum(dim=1, keepdim=True).clamp_min(1e-12))
    H = -(T * (T.clamp_min(1e-12)).log()).sum(dim=1)
    return (H / np.log(max(T.shape[1], 2))).cpu().numpy()


def codebook_spectrum(clusters):
    """Return the sorted singular values of the codebook; collapse = long tail."""
    U, S, V = torch.linalg.svd(clusters, full_matrices=False)
    return S.cpu().numpy()


def cost_statistics(C_mat):
    """Return (mean, std, min, max) of the cost matrix (ignoring padding)."""
    v = C_mat.flatten().cpu().numpy()
    return float(v.mean()), float(v.std()), float(v.min()), float(v.max())


def smoothness(X, mask):
    """X: [N,D]; returns norms ||X_{i+1}-X_i|| for valid frames."""
    X = X[mask]
    diffs = X[1:] - X[:-1]
    return diffs.norm(dim=-1).cpu().numpy()


def gt_boundary_mask(gt, mask):
    gt = gt[mask].cpu().numpy()
    boundaries = (gt[1:] != gt[:-1]).astype(np.float32)
    return boundaries, gt


def pair_corr_index_vs_semantic(gt, mask, max_pairs=20000, rng=None):
    """Correlation(|i-k|/N, [GT(i)==GT(k)]) over random index pairs."""
    gt = gt[mask].cpu().numpy()
    N = len(gt)
    if N < 10:
        return np.nan, np.nan
    rng = rng or np.random.default_rng(0)
    n = min(max_pairs, N * (N - 1) // 2)
    i = rng.integers(0, N, size=n)
    k = rng.integers(0, N, size=n)
    valid = i != k
    i, k = i[valid], k[valid]
    d = np.abs(i - k) / N
    same = (gt[i] == gt[k]).astype(np.float32)
    # Pearson between distance and NOT-same (so positive corr means close -> same, far -> different)
    y = 1.0 - same
    if d.std() < 1e-9 or y.std() < 1e-9:
        return 0.0, len(i)
    r = np.corrcoef(d, y)[0, 1]
    return float(r), len(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, choices=["FS", "FSeval", "Breakfast", "YTI", "desktop_assembly"])
    ap.add_argument("--activity", nargs="+", default=["all"])
    ap.add_argument("--data-root", default="../data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-videos", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoSSL.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval()
    model.to(device)

    # Use test-time dataset (n_frames=None → full videos)
    data = VideoDataset(args.data_root, args.dataset, None, standardise=False,
                       random=False, action_class=args.activity)
    loader = DataLoader(data, batch_size=1, shuffle=False)

    entropies, smooths, boundary_corrs, index_corrs = [], [], [], []
    per_video = []
    with torch.no_grad():
        seen = set()
        for vi, batch in enumerate(loader):
            if args.max_videos and vi >= args.max_videos:
                break
            # dataset returns a pair; iterate both halves per batch
            for half in batch:
                features_raw, mask, gt, fname, _ = half
                fname_s = str(fname[0]) if isinstance(fname, (list, tuple)) else str(fname)
                if fname_s in seen:
                    continue
                seen.add(fname_s)
                features_raw = features_raw.to(device)
                mask = mask.to(device)
                gt = gt.to(device)
                B, T, _ = features_raw.shape

                X = F.normalize(
                    model.mlp(features_raw.reshape(-1, features_raw.shape[-1])).reshape(B, T, -1),
                    dim=-1,
                )
                temp_prior = segmentation_ot.temporal_prior(T, model.n_clusters, model.rho, device)
                C = 1.0 - X @ model.clusters.T.unsqueeze(0)
                C = C + temp_prior
                T_star, _ = segmentation_ot.segment_ot(
                    C, mask, eps=model.train_eps, alpha=model.alpha_train, radius=model.radius_gw,
                    ub_frames=False, ub_actions=True,
                    lambda_frames=model.lambda_frames_train, lambda_actions=model.lambda_actions_train,
                    n_iters=model.n_ot_train, step_size=model.step_size,
                )
                mb = mask[0]
                Hs = row_entropy(T_star[0], mb)
                S = smoothness(X[0], mb)
                bnd, gt_np = gt_boundary_mask(gt[0], mb)
                if len(bnd) == len(S):
                    bnd_vals = S[bnd > 0.5]
                    int_vals = S[bnd < 0.5]
                    if len(bnd_vals) > 1 and len(int_vals) > 1:
                        bc = (bnd_vals.mean() - int_vals.mean()) / (S.std() + 1e-9)
                    else:
                        bc = np.nan
                else:
                    bc = np.nan
                ic, n_pairs = pair_corr_index_vs_semantic(gt[0], mb)

                entropies.append(Hs.mean())
                smooths.append(S.mean())
                boundary_corrs.append(bc)
                index_corrs.append(ic)
                per_video.append({
                    "fname": fname_s,
                    "n_frames_valid": int(mb.sum().item()),
                    "row_entropy_mean": float(Hs.mean()),
                    "row_entropy_median": float(np.median(Hs)),
                    "smoothness_mean": float(S.mean()),
                    "boundary_z": float(bc) if not np.isnan(bc) else None,
                    "idx_vs_sem_corr": float(ic) if not np.isnan(ic) else None,
                    "n_pairs_corr": int(n_pairs) if not np.isnan(ic) else 0,
                })

    spec = codebook_spectrum(model.clusters.detach())
    summary = {
        "ckpt": args.ckpt,
        "dataset": args.dataset,
        "activity": args.activity,
        "n_videos": len(per_video),
        "T_entropy_mean_norm": float(np.nanmean(entropies)),
        "T_entropy_std_norm": float(np.nanstd(entropies)),
        "smoothness_mean": float(np.nanmean(smooths)),
        "boundary_z_mean": float(np.nanmean(boundary_corrs)),
        "idx_vs_sem_corr_mean": float(np.nanmean(index_corrs)),
        "idx_vs_sem_corr_std": float(np.nanstd(index_corrs)),
        "codebook_sv_top": float(spec[0]),
        "codebook_sv_min": float(spec[-1]),
        "codebook_cond": float(spec[0] / max(spec[-1], 1e-9)),
    }
    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / "diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(json.dumps({"summary": summary, "per_video": per_video}, indent=2))

    # histograms
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    axes[0].hist(entropies, bins=20); axes[0].set_title(f"T* row entropy / log(K)\nmean={summary['T_entropy_mean_norm']:.3f}"); axes[0].set_xlabel("per-video mean normed entropy")
    axes[1].hist(smooths, bins=20); axes[1].set_title("||X_{i+1}-X_i|| mean / video"); axes[1].set_xlabel("mean step norm")
    axes[2].hist([b for b in boundary_corrs if not np.isnan(b)], bins=20); axes[2].set_title(f"boundary z-score\nmean={summary['boundary_z_mean']:.2f}"); axes[2].set_xlabel("(||dX||@boundary - ||dX||@interior)/std")
    axes[3].hist([c for c in index_corrs if not np.isnan(c)], bins=20); axes[3].set_title(f"corr(|i-k|/N, [GT(i)!=GT(k)])\nmean={summary['idx_vs_sem_corr_mean']:.3f}"); axes[3].set_xlabel("Pearson r")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "diag_hists.png", dpi=120)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
