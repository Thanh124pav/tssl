"""Evaluate predicted action boundaries on a deterministic 256-frame grid.

This is a diagnostic, not the paper's segment F1. Each video contributes its
own boundaries; predicted and GT transitions are matched one-to-one within a
specified tolerance in sampled-frame units.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "repos" / "VASOT" / "src"
sys.path.insert(0, str(SRC))

from train import VideoSSL  # noqa: E402
from video_dataset import VideoDataset  # noqa: E402
import segmentation_ot  # noqa: E402


def match_boundaries(pred, gt, tolerance):
    """Maximum-cardinality ordered matching, minimizing total displacement."""
    pred, gt = list(map(int, pred)), list(map(int, gt))
    n_pred, n_gt = len(pred), len(gt)
    dp = [[(0, 0) for _ in range(n_gt + 1)] for _ in range(n_pred + 1)]
    choice = [[None for _ in range(n_gt + 1)] for _ in range(n_pred + 1)]
    for i in range(n_pred - 1, -1, -1):
        for j in range(n_gt - 1, -1, -1):
            options = [(dp[i + 1][j], "skip_pred"),
                       (dp[i][j + 1], "skip_gt")]
            distance = abs(pred[i] - gt[j])
            if distance <= tolerance:
                matches, negative_distance = dp[i + 1][j + 1]
                options.append(((matches + 1, negative_distance - distance), "match"))
            dp[i][j], choice[i][j] = max(options, key=lambda item: item[0])
    displacements = []
    i = j = 0
    while i < n_pred and j < n_gt:
        if choice[i][j] == "match":
            displacements.append(abs(pred[i] - gt[j]))
            i += 1
            j += 1
        elif choice[i][j] == "skip_pred":
            i += 1
        else:
            j += 1
    return len(displacements), displacements


def predict(model, features, mask):
    features = features.unsqueeze(0)
    mask = mask.unsqueeze(0)
    n_frames = features.shape[1]
    embedding = F.normalize(model.mlp(features), dim=-1)
    embedding, warp = model._maybe_warp(embedding, mask)
    cost = 1.0 - embedding @ model.clusters.T.unsqueeze(0)
    cost += segmentation_ot.temporal_prior(n_frames, model.n_clusters,
                                           getattr(model, 'rho_seg', model.rho), features.device)
    plan, _ = segmentation_ot.segment_ot(
        cost, mask, eps=model.train_eps, alpha=model.alpha_train,
        radius=model.radius_gw, ub_frames=False, ub_actions=True,
        lambda_frames=model.lambda_frames_train,
        lambda_actions=model.lambda_actions_train,
        n_iters=model.n_ot_train, step_size=model.step_size,
    )
    labels = plan.argmax(dim=-1)[0]
    if warp is not None:
        from arc_reparam import inverse_warp_predictions
        labels = inverse_warp_predictions(labels.unsqueeze(0), warp)[0]
    return labels[mask[0]].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--activity", default="all")
    parser.add_argument("--data-root", default=str(SRC.parent / "data"))
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--tolerances", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = VideoSSL.load_from_checkpoint(args.ckpt, map_location="cpu", weights_only=False)
    model.eval()
    data = VideoDataset(args.data_root, args.dataset, args.frames,
                        standardise=False, random=False, action_class=[args.activity])
    totals = {t: {"tp": 0, "pred": 0, "gt": 0, "displacements": []} for t in args.tolerances}
    videos = []
    with torch.no_grad():
        for name in data.video_fnames:
            features, mask, gt, _, _ = data._load_video_data(name)
            mask = torch.as_tensor(mask, dtype=torch.bool)
            gt = gt[mask].cpu().numpy()
            pred = predict(model, features, mask)
            eligible = np.ones(len(gt) - 1, dtype=bool)
            if model.exclude_cls is not None:
                eligible &= (gt[1:] != model.exclude_cls) & (gt[:-1] != model.exclude_cls)
            pred_b = np.flatnonzero((pred[1:] != pred[:-1]) & eligible) + 1
            gt_b = np.flatnonzero((gt[1:] != gt[:-1]) & eligible) + 1
            videos.append({"name": name, "pred_boundaries": int(len(pred_b)),
                           "gt_boundaries": int(len(gt_b))})
            for tolerance, item in totals.items():
                tp, displacements = match_boundaries(pred_b, gt_b, tolerance)
                item["tp"] += tp
                item["pred"] += len(pred_b)
                item["gt"] += len(gt_b)
                item["displacements"].extend(displacements)

    metrics = {}
    for tolerance, item in totals.items():
        precision = item["tp"] / item["pred"] if item["pred"] else 0.0
        recall = item["tp"] / item["gt"] if item["gt"] else 0.0
        metrics[str(tolerance)] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": item["tp"], "pred_count": item["pred"], "gt_count": item["gt"],
            "median_displacement": float(np.median(item["displacements"]))
            if item["displacements"] else None,
        }
    output = {
        "checkpoint": str(Path(args.ckpt).resolve()), "dataset": args.dataset,
        "activity": args.activity, "n_videos": len(videos), "sampled_frames": args.frames,
        "excluded_label": model.exclude_cls,
        "fragmentation_ratio": sum(v["pred_boundaries"] for v in videos) /
        max(sum(v["gt_boundaries"] for v in videos), 1),
        "tolerances": metrics, "videos": videos,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "videos"}, indent=2))


if __name__ == "__main__":
    main()
