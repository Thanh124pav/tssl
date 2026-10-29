"""Evaluate action consistency of frame alignment on labelled action data.

This is an action-label agreement proxy, not a paper VAOT metric.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "repos" / "VASOT" / "src"
sys.path.insert(0, str(SRC))

import alignment_ot  # noqa: E402
from action_bridge import action_similarity  # noqa: E402
from train import VideoSSL  # noqa: E402
from video_dataset import VideoDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", default="FSeval")
    parser.add_argument("--activity", default="all")
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=20)
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
    names = data.video_fnames
    observations = []
    with torch.no_grad():
        for idx in range(min(args.pairs, len(names))):
            raw_x, mask_x, gt_x, _, _ = data._load_video_data(names[idx])
            raw_y, mask_y, gt_y, _, _ = data._load_video_data(names[(idx + 1) % len(names)])
            mask_x = torch.as_tensor(mask_x, dtype=torch.bool).unsqueeze(0)
            mask_y = torch.as_tensor(mask_y, dtype=torch.bool).unsqueeze(0)
            x = F.normalize(model.mlp(raw_x.unsqueeze(0)), dim=-1)
            y = F.normalize(model.mlp(raw_y.unsqueeze(0)), dim=-1)
            q_x = torch.softmax(x @ model.clusters.T.unsqueeze(0) / model.temp, dim=-1)
            q_y = torch.softmax(y @ model.clusters.T.unsqueeze(0) / model.temp, dim=-1)
            action_sim = action_similarity(q_x, q_y)
            nx, ny = x.shape[1], y.shape[1]
            cost = 1.0 - x @ y.transpose(1, 2)
            cost += model.action_bridge_cost_weight * (1.0 - action_sim)
            cost += alignment_ot.temporal_prior(
                nx, ny, getattr(model, "rho_align", model.rho), x.device
            )
            cost = torch.cat((torch.ones(1, 1, ny) * 0.5, cost), dim=1)
            cost = torch.cat((torch.ones(1, nx + 1, 1) * 0.5, cost), dim=2)
            transport, _ = alignment_ot.align_ot(
                cost, mask_x, mask_y, eps=model.train_eps,
                alpha=model.alpha_train, radius=model.radius_gw,
                ub_frames=model.ub_frames, ub_actions=model.ub_actions,
                lambda_frames=model.lambda_frames_train,
                lambda_actions=model.lambda_actions_train,
                n_iters=model.n_ot_train, step_size=model.step_size,
            )
            valid = mask_x.unsqueeze(2) * mask_y.unsqueeze(1)
            weighted = transport * valid
            mass = weighted.sum().clamp_min(1e-12)
            gt_x = torch.as_tensor(gt_x, dtype=torch.long)
            gt_y = torch.as_tensor(gt_y, dtype=torch.long)
            same_action = (gt_x[:, None] == gt_y[None, :]).to(weighted.dtype).unsqueeze(0)
            row_mass = weighted[0].sum(dim=-1)
            nonzero = row_mass > 1e-12
            rows = weighted[0, nonzero] / row_mass[nonzero, None]
            entropy = -(rows * rows.clamp_min(1e-12).log()).sum(dim=-1)
            observations.append({
                "x": str(names[idx]),
                "y": str(names[(idx + 1) % len(names)]),
                "gt_same_action_mass": float((weighted * same_action).sum() / mass),
                "posterior_similarity_mass": float((weighted * action_sim).sum() / mass),
                "top1_mass": float(rows.max(dim=-1).values.mean()),
                "effective_support": float(entropy.exp().mean()),
            })

    summary = {
        key: float(np.mean([item[key] for item in observations]))
        for key in ("gt_same_action_mass", "posterior_similarity_mass", "top1_mass", "effective_support")
    }
    result = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "dataset": args.dataset,
        "n_pairs": len(observations),
        "metric_warning": "GT same-action transport mass is a labelled proxy, not a paper VAOT metric.",
        "action_bridge_cost_weight": model.action_bridge_cost_weight,
        "action_bridge_loss_weight": model.action_bridge_loss_weight,
        "summary": summary,
        "pairs": observations,
    }
    if not all(math.isfinite(value) for value in summary.values()):
        raise RuntimeError("non-finite alignment proxy")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
