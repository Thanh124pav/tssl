"""Calibrate action-bridge cost scale on fixed video pairs (no training)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "repos" / "VASOT" / "src"
sys.path.insert(0, str(SRC))

from action_bridge import action_similarity  # noqa: E402
from train import VideoSSL  # noqa: E402
from video_dataset import VideoDataset  # noqa: E402


def stats(values):
    values = np.concatenate(values)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
    }


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
    visual_costs, action_costs = [], []
    names = data.video_fnames
    with torch.no_grad():
        for idx in range(min(args.pairs, len(names))):
            raw_x, mask_x, _, _, _ = data._load_video_data(names[idx])
            raw_y, mask_y, _, _, _ = data._load_video_data(names[(idx + 1) % len(names)])
            x = F.normalize(model.mlp(raw_x.unsqueeze(0)), dim=-1)
            y = F.normalize(model.mlp(raw_y.unsqueeze(0)), dim=-1)
            q_x = torch.softmax(x @ model.clusters.T.unsqueeze(0) / model.temp, dim=-1)
            q_y = torch.softmax(y @ model.clusters.T.unsqueeze(0) / model.temp, dim=-1)
            valid = torch.as_tensor(mask_x, dtype=torch.bool)[:, None] & torch.as_tensor(mask_y, dtype=torch.bool)[None, :]
            c_visual = (1.0 - x @ y.transpose(1, 2))[0][valid]
            c_action = (1.0 - action_similarity(q_x, q_y))[0][valid]
            visual_costs.append(c_visual.cpu().numpy())
            action_costs.append(c_action.cpu().numpy())

    result = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "dataset": args.dataset,
        "frames": args.frames,
        "pairs": min(args.pairs, len(names)),
        "visual_cost": stats(visual_costs),
        "action_cost": stats(action_costs),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
