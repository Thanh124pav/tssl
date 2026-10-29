"""Measure T_xa and T_xy support on fixed, sampled video pairs.

T_xy excludes the virtual row/column in the public solver. Its rows are
renormalised over real targets here; mean row mass before normalisation is
reported separately so virtual-frame use is not hidden.
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

from train import VideoSSL  # noqa: E402
from video_dataset import VideoDataset  # noqa: E402
import alignment_ot  # noqa: E402
import segmentation_ot  # noqa: E402


def support(plan):
    rows = plan[0]
    row_mass = rows.sum(dim=-1)
    valid = row_mass > 1e-8
    if valid.any():
        rows = rows[valid] / row_mass[valid].unsqueeze(-1)
        entropy = -(rows * rows.clamp_min(1e-12).log()).sum(dim=-1)
        top1 = rows.max(dim=-1).values
        entropy_norm = (entropy / np.log(rows.shape[-1])).mean().item()
        effective_support = entropy.exp().mean().item()
        top1_mass = top1.mean().item()
    else:
        entropy_norm = effective_support = top1_mass = float("nan")
    return {
        "row_entropy_norm": entropy_norm,
        "effective_support": effective_support,
        "top1_mass": top1_mass,
        "real_target_row_mass": row_mass.mean().item(),
        "nonzero_real_target_fraction": valid.float().mean().item(),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", default="FSeval")
    parser.add_argument("--activity", default="all")
    parser.add_argument("--data-root", default=str(SRC.parent / "data"))
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--eps", type=float, default=None,
                        help="Solver-only epsilon override; checkpoint weights stay fixed")
    parser.add_argument("--init-seed", type=int, default=None,
                        help="Recreate an untrained model from checkpoint hyperparameters with this seed")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.init_seed is None:
        model = VideoSSL.load_from_checkpoint(args.ckpt, map_location="cpu", weights_only=False)
    else:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        torch.manual_seed(args.init_seed)
        model = VideoSSL(**checkpoint["hyper_parameters"])
    model.eval()
    solver_eps = model.train_eps if args.eps is None else args.eps
    data = VideoDataset(args.data_root, args.dataset, args.frames,
                        standardise=False, random=False, action_class=[args.activity])
    names = data.video_fnames
    if len(names) < 2:
        raise ValueError("Need at least two videos")
    pair_names = [(names[i], names[(i + 1) % len(names)])
                  for i in range(min(args.pairs, len(names)))]
    observations = []

    with torch.no_grad():
        for name_x, name_y in pair_names:
            raw_x, mask_x, _, _, _ = data._load_video_data(name_x)
            raw_y, mask_y, _, _, _ = data._load_video_data(name_y)
            mask_x = torch.as_tensor(mask_x, dtype=torch.bool).unsqueeze(0)
            mask_y = torch.as_tensor(mask_y, dtype=torch.bool).unsqueeze(0)
            x = F.normalize(model.mlp(raw_x.unsqueeze(0)), dim=-1)
            y = F.normalize(model.mlp(raw_y.unsqueeze(0)), dim=-1)
            x, _ = model._maybe_warp(x, mask_x)
            y, _ = model._maybe_warp(y, mask_y)
            nx, ny = x.shape[1], y.shape[1]

            cost_xa = 1.0 - x @ model.clusters.T.unsqueeze(0)
            cost_xa += segmentation_ot.temporal_prior(nx, model.n_clusters,
                                                       getattr(model, 'rho_seg', model.rho), x.device)
            t_xa, _ = segmentation_ot.segment_ot(
                cost_xa, mask_x, eps=solver_eps, alpha=model.alpha_train,
                radius=model.radius_gw, ub_frames=False, ub_actions=True,
                lambda_frames=model.lambda_frames_train,
                lambda_actions=model.lambda_actions_train,
                n_iters=model.n_ot_train, step_size=model.step_size)

            cost_xy = 1.0 - x @ y.transpose(1, 2)
            cost_xy += alignment_ot.temporal_prior(nx, ny,
                                                   getattr(model, 'rho_align', model.rho), x.device)
            cost_xy = torch.cat((torch.ones(1, 1, ny) * 0.5, cost_xy), dim=1)
            cost_xy = torch.cat((torch.ones(1, nx + 1, 1) * 0.5, cost_xy), dim=2)
            t_xy, _ = alignment_ot.align_ot(
                cost_xy, mask_x, mask_y, eps=solver_eps,
                alpha=model.alpha_train, radius=model.radius_gw,
                ub_frames=model.ub_frames, ub_actions=model.ub_actions,
                lambda_frames=model.lambda_frames_train,
                lambda_actions=model.lambda_actions_train,
                n_iters=model.n_ot_train, step_size=model.step_size)

            observations.append({"x": name_x, "y": name_y,
                                 "T_xa": support(t_xa), "T_xy": support(t_xy)})

    summary = {}
    for key in ("T_xa", "T_xy"):
        summary[key] = {}
        for metric in observations[0][key]:
            values = np.asarray([o[key][metric] for o in observations], dtype=float)
            summary[key][metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    result = {"checkpoint": str(Path(args.ckpt).resolve()),
              "dataset": args.dataset, "activity": args.activity,
              "init_seed": args.init_seed,
              "solver_eps": solver_eps,
              "sampled_frames": args.frames, "n_pairs": len(observations),
              "summary": summary, "pairs": observations}
    result = json_safe(result)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
