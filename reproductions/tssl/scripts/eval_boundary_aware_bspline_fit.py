"""Fit independent splines inside action segments and audit reconstruction.

GT boundaries provide an evaluation-only oracle ceiling. ASOT/posterior
boundaries are self-supervised candidates suitable for a later training path.
"""

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

import segmentation_ot  # noqa: E402
from bspline import evaluate_bspline, fit_bspline  # noqa: E402
from train import VideoSSL  # noqa: E402
from video_dataset import VideoDataset  # noqa: E402


def change_points(labels):
    return (torch.where(labels[1:] != labels[:-1])[0] + 1).tolist()


def prune_boundaries(boundaries, n_frames, min_segment):
    kept = []
    last = 0
    for boundary in boundaries:
        if boundary - last >= min_segment and n_frames - boundary >= min_segment:
            kept.append(int(boundary))
            last = int(boundary)
    return kept


def boundary_pr(gt_boundaries, pred_boundaries, tolerance=3):
    gt = torch.as_tensor(gt_boundaries, dtype=torch.float32)
    pred = torch.as_tensor(pred_boundaries, dtype=torch.float32)
    if not gt.numel() or not pred.numel():
        return 0.0, 0.0
    distance = (pred[:, None] - gt[None, :]).abs()
    return (
        float((distance.min(dim=1).values <= tolerance).float().mean()),
        float((distance.min(dim=0).values <= tolerance).float().mean()),
    )


def fit_piecewise(values, boundaries, degree, tol, capacity_ratio):
    fitted = torch.empty_like(values)
    total_controls = 0
    hit_capacity = False
    for start, end in zip([0] + boundaries, boundaries + [len(values)]):
        segment = values[start:end]
        local_degree = min(degree, max(len(segment) - 1, 1))
        local_kmax = max(local_degree + 1, int(math.ceil(len(segment) * capacity_ratio)))
        local_kmax = min(local_kmax, len(segment))
        if len(segment) <= 2:
            fitted[start:end] = segment
            total_controls += len(segment)
            continue
        u = torch.linspace(0.0, 1.0, len(segment), dtype=segment.dtype)
        knots, controls = fit_bspline(
            segment, tol=tol, k_max=local_kmax, p=local_degree, u=u
        )
        fitted[start:end] = evaluate_bspline(knots, controls, u, p=local_degree)
        n_controls = int(knots.numel() - local_degree - 1)
        total_controls += n_controls
        hit_capacity = hit_capacity or n_controls >= local_kmax
    return fitted, total_controls, hit_capacity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--space", choices=["X", "posterior"], required=True)
    parser.add_argument("--boundary-source", choices=["gt", "asot", "posterior"], required=True)
    parser.add_argument("--capacity-ratio", type=float, required=True)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--min-segment", type=int, default=4)
    parser.add_argument("--videos", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = VideoSSL.load_from_checkpoint(
        args.ckpt, map_location="cpu", weights_only=False
    ).eval()
    data = VideoDataset(
        str(SRC.parent / "data"), "FSeval", 256,
        standardise=False, random=False, action_class=["all"],
    )
    observations = []
    with torch.no_grad():
        for name in data.video_fnames[:args.videos]:
            raw, mask, gt, _, _ = data._load_video_data(name)
            n = int(torch.as_tensor(mask).sum())
            gt = torch.as_tensor(gt[:n], dtype=torch.long)
            x = F.normalize(model.mlp(raw[:n]), dim=-1)
            logits = x @ model.clusters.T
            posterior = torch.softmax(logits / model.temp, dim=-1)
            if args.boundary_source == "gt":
                labels = gt
            elif args.boundary_source == "posterior":
                labels = posterior.argmax(dim=-1)
            else:
                cost = 1.0 - x.unsqueeze(0) @ model.clusters.T.unsqueeze(0)
                cost += segmentation_ot.temporal_prior(
                    n, model.n_clusters, getattr(model, "rho_seg", model.rho), x.device
                )
                plan, _ = segmentation_ot.segment_ot(
                    cost, torch.ones(1, n, dtype=torch.bool), eps=model.train_eps,
                    alpha=model.alpha_train, radius=model.radius_gw,
                    ub_frames=False, ub_actions=True,
                    lambda_frames=model.lambda_frames_train,
                    lambda_actions=model.lambda_actions_train,
                    n_iters=model.n_ot_train, step_size=model.step_size,
                )
                labels = plan[0].argmax(dim=-1)
            boundaries = prune_boundaries(change_points(labels), n, args.min_segment)
            gt_boundaries = change_points(gt)
            values = x if args.space == "X" else posterior
            start_time = time.perf_counter()
            fitted, n_controls, hit_capacity = fit_piecewise(
                values, boundaries, args.degree, args.tol, args.capacity_ratio
            )
            runtime_ms = (time.perf_counter() - start_time) * 1000
            err = (values - fitted).norm(dim=-1)
            scale = values.norm(dim=-1).square().mean().sqrt().clamp_min(1e-8)
            precision, coverage = boundary_pr(gt_boundaries, boundaries)
            observations.append({
                "video": str(name),
                "n_frames": n,
                "n_segments": len(boundaries) + 1,
                "n_control_points": n_controls,
                "control_ratio": n_controls / n,
                "finite": bool(torch.isfinite(fitted).all()),
                "rmse": float(err.square().mean().sqrt()),
                "normalized_rmse": float(err.square().mean().sqrt() / scale),
                "max_error": float(err.max()),
                "q90_error": float(torch.quantile(err, 0.9)),
                "met_tolerance": float(err.max()) <= args.tol,
                "hit_local_capacity": hit_capacity,
                "boundary_precision": precision,
                "boundary_coverage": coverage,
                "runtime_ms": runtime_ms,
            })

    numeric = [key for key in observations[0] if key not in {"video", "finite", "met_tolerance", "hit_local_capacity"}]
    summary = {key: float(np.mean([item[key] for item in observations])) for key in numeric}
    summary.update({
        "finite_fraction": float(np.mean([item["finite"] for item in observations])),
        "met_tolerance_fraction": float(np.mean([item["met_tolerance"] for item in observations])),
        "hit_local_capacity_fraction": float(np.mean([item["hit_local_capacity"] for item in observations])),
    })
    result = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "space": args.space,
        "boundary_source": args.boundary_source,
        "capacity_ratio": args.capacity_ratio,
        "degree": args.degree,
        "tol": args.tol,
        "min_segment": args.min_segment,
        "summary": summary,
        "videos": observations,
    }
    if not all(math.isfinite(value) for value in summary.values()):
        raise RuntimeError("non-finite summary")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "videos"}, indent=2))


if __name__ == "__main__":
    main()
