"""Audit B-spline reconstruction quality separately from downstream metrics."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "repos" / "VASOT" / "src"
sys.path.insert(0, str(SRC))

from bspline import evaluate_bspline, fit_bspline  # noqa: E402
from train import VideoSSL  # noqa: E402
from video_dataset import VideoDataset  # noqa: E402


def boundary_metrics(knots, gt, n_frames, window_frames):
    interior = knots[(knots > 0) & (knots < 1)]
    changes = torch.where(gt[1:] != gt[:-1])[0] + 1
    boundaries = changes.to(knots.dtype) / max(n_frames - 1, 1)
    window = window_frames / max(n_frames - 1, 1)
    if interior.numel() and boundaries.numel():
        distance = (interior[:, None] - boundaries[None, :]).abs()
        knot_precision = (distance.min(dim=1).values <= window).float().mean().item()
        boundary_coverage = (distance.min(dim=0).values <= window).float().mean().item()
    else:
        knot_precision = boundary_coverage = 0.0
    return knot_precision, boundary_coverage, int(boundaries.numel())


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", default="FSeval")
    parser.add_argument("--activity", default="all")
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--videos", type=int, default=20)
    parser.add_argument("--space", choices=["X", "logits", "posterior"], required=True)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--tol", type=float, required=True)
    parser.add_argument("--kmax", type=int, required=True)
    parser.add_argument("--boundary-window", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = VideoSSL.load_from_checkpoint(
        args.ckpt, map_location="cpu", weights_only=False
    ).eval()
    data = VideoDataset(
        str(SRC.parent / "data"), args.dataset, args.frames,
        standardise=False, random=False, action_class=[args.activity],
    )
    observations = []
    with torch.no_grad():
        for name in data.video_fnames[:args.videos]:
            raw, mask, gt, _, _ = data._load_video_data(name)
            valid_n = int(torch.as_tensor(mask).sum().item())
            x = F.normalize(model.mlp(raw[:valid_n]), dim=-1)
            logits = x @ model.clusters.T
            if args.space == "X":
                values = x
            elif args.space == "logits":
                values = logits
            else:
                values = torch.softmax(logits / model.temp, dim=-1)
            u = torch.linspace(0.0, 1.0, valid_n, dtype=values.dtype)
            start = time.perf_counter()
            knots, controls = fit_bspline(
                values, tol=args.tol, k_max=args.kmax, p=args.degree, u=u
            )
            fitted = evaluate_bspline(knots, controls, u, p=args.degree)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            err = (values - fitted).norm(dim=-1)
            scale = values.norm(dim=-1).square().mean().sqrt().clamp_min(1e-8)
            k_controls = int(knots.numel() - args.degree - 1)
            knot_precision, boundary_coverage, n_boundaries = boundary_metrics(
                knots, torch.as_tensor(gt[:valid_n]), valid_n, args.boundary_window
            )
            second_diff = fitted[2:] - 2 * fitted[1:-1] + fitted[:-2]
            observations.append({
                "video": str(name),
                "n_frames": valid_n,
                "n_boundaries": n_boundaries,
                "finite": bool(torch.isfinite(fitted).all()),
                "rmse": float(err.square().mean().sqrt()),
                "normalized_rmse": float(err.square().mean().sqrt() / scale),
                "max_error": float(err.max()),
                "q90_error": float(torch.quantile(err, 0.9)),
                "endpoint_max_error": float(err[[0, -1]].max()),
                "n_control_points": k_controls,
                "hit_kmax": k_controls >= args.kmax,
                "met_tolerance": float(err[1:-1].max()) <= args.tol if valid_n > 2 else True,
                "knot_boundary_precision": knot_precision,
                "boundary_knot_coverage": boundary_coverage,
                "mean_curvature": float(second_diff.norm(dim=-1).mean()) if valid_n > 2 else 0.0,
                "runtime_ms": elapsed_ms,
            })

    keys = [key for key in observations[0] if key not in {"video", "finite", "hit_kmax", "met_tolerance"}]
    summary = {key: finite_mean([item[key] for item in observations]) for key in keys}
    summary.update({
        "finite_fraction": finite_mean([item["finite"] for item in observations]),
        "hit_kmax_fraction": finite_mean([item["hit_kmax"] for item in observations]),
        "met_tolerance_fraction": finite_mean([item["met_tolerance"] for item in observations]),
    })
    result = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "dataset": args.dataset,
        "space": args.space,
        "degree": args.degree,
        "tol": args.tol,
        "kmax": args.kmax,
        "boundary_window_frames": args.boundary_window,
        "summary": summary,
        "videos": observations,
    }
    if not math.isclose(summary["finite_fraction"], 1.0):
        raise RuntimeError("non-finite B-spline fit detected")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "videos"}, indent=2))


if __name__ == "__main__":
    main()
