"""Shared loaders for Phase G / Phase N frozen-checkpoint diagnostics.

Every helper here reuses the public VASOT solver and the checkpoint's own
training hyper-parameters. Nothing in this module trains, and ground-truth
labels are only ever returned for evaluation-side analysis.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "repos" / "VASOT" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train import VideoSSL  # noqa: E402
from video_dataset import VideoDataset  # noqa: E402
import alignment_ot  # noqa: E402
import segmentation_ot  # noqa: E402

DATA_ROOT = str(SRC.parent / "data")
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def load_model(ckpt, device="cpu", init_seed=None):
    """Frozen checkpoint, or an untrained model rebuilt from its hyper-parameters."""
    if init_seed is None:
        model = VideoSSL.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False)
    else:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        torch.manual_seed(init_seed)
        model = VideoSSL(**payload["hyper_parameters"])
    model.eval()
    model.to(device)
    return model


def load_dataset(dataset="FSeval", frames=256, activity="all", data_root=DATA_ROOT,
                 standardise=None):
    if standardise is None:
        standardise = dataset == "Breakfast"
    return VideoDataset(data_root, dataset, frames, standardise=standardise,
                        random=False, action_class=[activity])


@torch.no_grad()
def embed(model, data, name, device="cpu"):
    """Post-MLP, post-warp unit-norm frame embeddings plus mask and GT labels."""
    raw, mask, gt, _, _ = data._load_video_data(name)
    mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool).unsqueeze(0).to(device)
    z = F.normalize(model.mlp(raw.unsqueeze(0).to(device)), dim=-1)
    z, _ = model._maybe_warp(z, mask_t)
    return {"name": name, "z": z, "mask": mask_t, "gt": gt.to(device), "raw": raw.to(device)}


@torch.no_grad()
def embed_features(model, raw, device="cpu", mask=None):
    """Embed an explicitly supplied raw feature tensor (T, D) -> (1, T, d)."""
    raw = raw.to(device)
    if mask is None:
        mask = torch.ones(1, raw.shape[0], dtype=torch.bool, device=device)
    z = F.normalize(model.mlp(raw.unsqueeze(0)), dim=-1)
    z, _ = model._maybe_warp(z, mask)
    return z, mask


@torch.no_grad()
def seg_plan(model, z, mask, eps=None, rho_seg=None):
    """T_xa from the public segmentation solver with the checkpoint's parameters."""
    n = z.shape[1]
    rho = getattr(model, "rho_seg", model.rho) if rho_seg is None else rho_seg
    cost = 1.0 - z @ model.clusters.T.unsqueeze(0)
    cost = cost + segmentation_ot.temporal_prior(n, model.n_clusters, rho, z.device)
    plan, _ = segmentation_ot.segment_ot(
        cost, mask, eps=model.train_eps if eps is None else eps,
        alpha=model.alpha_train, radius=model.radius_gw,
        ub_frames=False, ub_actions=True,
        lambda_frames=model.lambda_frames_train,
        lambda_actions=model.lambda_actions_train,
        n_iters=model.n_ot_train, step_size=model.step_size)
    return plan


@torch.no_grad()
def align_plan(model, zx, zy, mask_x, mask_y, eps=None, rho_align=None, cost_xy=None):
    """T_xy from the public alignment solver (virtual row/column handled as in the repo)."""
    nx, ny = zx.shape[1], zy.shape[1]
    rho = getattr(model, "rho_align", model.rho) if rho_align is None else rho_align
    if cost_xy is None:
        cost_xy = 1.0 - zx @ zy.transpose(1, 2)
    cost_xy = cost_xy + alignment_ot.temporal_prior(nx, ny, rho, zx.device)
    cost_xy = torch.cat((torch.ones(1, 1, ny, device=zx.device) * 0.5, cost_xy), dim=1)
    cost_xy = torch.cat((torch.ones(1, nx + 1, 1, device=zx.device) * 0.5, cost_xy), dim=2)
    plan, trace = alignment_ot.align_ot(
        cost_xy, mask_x, mask_y, eps=model.train_eps if eps is None else eps,
        alpha=model.alpha_train, radius=model.radius_gw,
        ub_frames=model.ub_frames, ub_actions=model.ub_actions,
        lambda_frames=model.lambda_frames_train,
        lambda_actions=model.lambda_actions_train,
        n_iters=model.n_ot_train, step_size=model.step_size)
    return plan, trace


def plan_stats(plan, topk=(1, 5, 10)):
    """Row-conditional statistics. Rows are renormalised over real targets;
    the pre-normalisation row mass is reported so virtual mass is not hidden."""
    rows = plan[0] if plan.dim() == 3 else plan
    row_mass = rows.sum(dim=-1)
    valid = row_mass > 1e-8
    out = {"real_target_row_mass": row_mass.mean().item(),
           "nonzero_row_fraction": valid.float().mean().item(),
           "n_rows": int(rows.shape[0]), "n_cols": int(rows.shape[-1])}
    if not bool(valid.any()):
        return out
    r = rows[valid] / row_mass[valid].unsqueeze(-1)
    ent = -(r * r.clamp_min(1e-12).log()).sum(dim=-1)
    out["row_entropy_nats"] = ent.mean().item()
    out["row_entropy_norm"] = (ent / math.log(r.shape[-1])).mean().item()
    out["effective_support"] = ent.exp().mean().item()
    out["top1_mass"] = r.max(dim=-1).values.mean().item()
    for k in topk:
        if k <= r.shape[-1]:
            out[f"top{k}_mass"] = r.topk(k, dim=-1).values.sum(dim=-1).mean().item()
    return out


def gt_segments(gt):
    """[(start, end_exclusive, label)] over a 1-D label tensor."""
    labels = gt.detach().cpu().numpy()
    segs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segs.append((start, i, int(labels[start])))
            start = i
    return segs


def signed_boundary_distance(gt):
    """Signed frame distance to the nearest GT segment boundary (0 at a boundary)."""
    segs = gt_segments(gt)
    bounds = [s[0] for s in segs[1:]]
    n = len(gt)
    if not bounds:
        return np.full(n, np.inf)
    idx = np.arange(n)[:, None]
    b = np.asarray(bounds)[None, :]
    d = idx - b
    nearest = np.argmin(np.abs(d), axis=1)
    return d[np.arange(n), nearest].astype(float)


def monotone_warp(n, n_knots=5, severity=0.5, seed=0):
    """Monotone piecewise-linear time warp on [0, n-1].

    Returns source positions `t[j]` (float, in frame units of the original
    video) for each output frame j, i.e. the known index map of the warp.
    severity=0 is the identity; larger values let segment speeds differ more.
    Ordering is preserved by construction, so action order is preserved.
    """
    rng = np.random.default_rng(seed)
    knots = np.linspace(0.0, 1.0, n_knots + 1)
    widths = np.diff(knots)
    speeds = np.exp(rng.uniform(-severity, severity, size=n_knots))
    out_widths = widths * speeds
    out_widths = out_widths / out_widths.sum()
    out_knots = np.concatenate(([0.0], np.cumsum(out_widths)))
    u = np.linspace(0.0, 1.0, n)
    src = np.interp(u, out_knots, knots)  # inverse warp: output time -> source time
    return src * (n - 1), {"n_knots": n_knots, "severity": severity, "seed": seed,
                           "speed_ratios": speeds.tolist(),
                           "max_speed_ratio": float(speeds.max() / speeds.min())}


def resample_features(raw, src_pos):
    """Linear interpolation of a (T, D) feature tensor at float positions."""
    t = raw.shape[0]
    pos = np.clip(np.asarray(src_pos, dtype=float), 0.0, t - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = torch.as_tensor(pos - lo, dtype=raw.dtype, device=raw.device).unsqueeze(-1)
    return raw[lo] * (1 - w) + raw[hi] * w


def relative_phase(gt):
    """Within-GT-segment relative position in [0, 1] (evaluation only)."""
    phase = np.zeros(len(gt))
    for s, e, _ in gt_segments(gt):
        phase[s:e] = np.linspace(0.0, 1.0, e - s) if e - s > 1 else 0.5
    return phase


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def init_wandb(name, tags, config, group="diagnostics", project="vasot-repro"):
    """Best-effort W&B run; diagnostics must not fail because logging failed."""
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; skipping logging")
        return None
    try:
        return wandb.init(project=project, name=name, group=group, tags=list(tags),
                          config=config, job_type="diagnostic", reinit=True,
                          dir=os.environ.get("WANDB_DIR", str(ARTIFACTS / "wandb")))
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] init failed ({exc}); continuing offline")
        return None


def pick_device(prefer="cuda"):
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
