"""A03 - what is the geometry of the frames inside a single video?

Measured on frozen embeddings, with the untrained model and the raw features as
controls, so "training did this" is separable from "the features were like that".

  lag profile      mean cosine between frames as a function of |i - j|
  intrinsic dim    two-NN (Facco et al.) and PCA participation ratio
  trajectory       step size, turning angle between consecutive velocities
  revisits         fraction of frames whose nearest neighbour is far in time
  GT structure     silhouette of GT action labels, within/between scatter
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import diag_common as dc


def lag_profile(z, lags):
    sim = (z @ z.T).cpu().numpy()
    n = z.shape[0]
    out = []
    for lag in lags:
        if lag >= n:
            continue
        idx = np.arange(n - lag)
        out.append({"lag": int(lag), "mean_cosine": float(sim[idx, idx + lag].mean())})
    return out


def two_nn_intrinsic_dim(z):
    """Facco et al. 2017: the ratio of the two nearest-neighbour distances has
    a distribution whose slope in log space gives the intrinsic dimension."""
    d = torch.cdist(z, z)
    d.fill_diagonal_(float("inf"))
    vals, _ = d.sort(dim=-1)
    r1, r2 = vals[:, 0], vals[:, 1]
    ok = (r1 > 1e-9) & (r2 > r1)
    mu = (r2[ok] / r1[ok]).cpu().numpy()
    mu = np.sort(mu)
    n = len(mu)
    if n < 10:
        return float("nan")
    f = np.arange(1, n + 1) / n
    keep = f < 0.9  # discard the tail, as in the original estimator
    x = np.log(mu[keep])
    y = -np.log(1 - f[keep])
    denom = float((x * x).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def participation_ratio(z):
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(zc.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    s1, s2 = float(ev.sum()), float((ev ** 2).sum())
    pr = (s1 ** 2) / s2 if s2 > 0 else float("nan")
    ev = ev.flip(0)
    cum = torch.cumsum(ev, 0) / max(s1, 1e-12)
    return {"participation_ratio": pr,
            "dims_for_90pct_variance": int((cum < 0.9).sum().item() + 1),
            "top1_variance_fraction": float(ev[0] / max(s1, 1e-12)),
            "ambient_dim": int(z.shape[1])}


def trajectory_stats(z):
    step = (z[1:] - z[:-1]).norm(dim=-1)
    v = F.normalize(z[1:] - z[:-1], dim=-1)
    turn = (v[1:] * v[:-1]).sum(-1).clamp(-1, 1).arccos()
    return {"step_mean": float(step.mean()), "step_std": float(step.std(unbiased=False)),
            "turning_angle_mean_deg": float(turn.mean() * 180 / np.pi),
            "turning_angle_median_deg": float(turn.median() * 180 / np.pi),
            "path_length": float(step.sum()),
            "endpoint_distance": float((z[-1] - z[0]).norm()),
            "straightness": float((z[-1] - z[0]).norm() / step.sum().clamp_min(1e-9))}


def revisits(z, far=20):
    d = torch.cdist(z, z)
    d.fill_diagonal_(float("inf"))
    nn = d.argmin(dim=-1)
    lag = (nn - torch.arange(len(z), device=z.device)).abs()
    return {"nn_temporal_lag_mean": float(lag.float().mean()),
            "nn_temporal_lag_median": float(lag.float().median()),
            f"fraction_nn_farther_than_{far}_frames": float((lag > far).float().mean()),
            "fraction_nn_is_adjacent": float((lag <= 1).float().mean())}


def gt_structure(z, gt):
    labels = gt.cpu().numpy()
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {"n_gt_actions": int(len(uniq))}
    cent = torch.stack([z[torch.as_tensor(labels == a, device=z.device)].mean(0) for a in uniq])
    within = float(np.mean([float((z[torch.as_tensor(labels == a, device=z.device)] - cent[i]).norm(dim=-1).mean())
                            for i, a in enumerate(uniq)]))
    bd = torch.cdist(cent, cent)
    bd.fill_diagonal_(float("inf"))
    out = {"n_gt_actions": int(len(uniq)),
           "within_action_scatter": within,
           "nearest_between_centroid_distance": float(bd.min()),
           "mean_between_centroid_distance": float(bd[~torch.isinf(bd)].mean()),
           "separation_ratio_between_over_within": float(bd[~torch.isinf(bd)].mean() / max(within, 1e-9))}
    try:
        from sklearn.metrics import silhouette_score
        out["silhouette_gt_labels"] = float(silhouette_score(z.cpu().numpy(), labels, metric="cosine"))
    except Exception:  # noqa: BLE001
        pass
    return out


def analyse(z, gt, lags, far):
    return {"lag_profile": lag_profile(z, lags),
            "intrinsic_dim_two_nn": two_nn_intrinsic_dim(z),
            **participation_ratio(z),
            **trajectory_stats(z),
            **revisits(z, far),
            **gt_structure(z, gt)}


def mean_of(records, key):
    vals = [r[key] for r in records if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--videos", type=int, default=12)
    ap.add_argument("--init-seed", type=int, default=1)
    ap.add_argument("--far-frames", type=int, default=20)
    ap.add_argument("--lags", type=int, nargs="*",
                    default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = dc.pick_device("cuda" if args.device in ("auto", "cuda") else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.set_grad_enabled(False)  # nothing here trains; model.clusters carries requires_grad
    run = None if args.no_wandb else dc.init_wandb(
        f"A03_{args.tag}", ["A03", "diagnostic", args.dataset, "action_space"], vars(args))

    model = dc.load_model(args.ckpt, device)
    init_model = dc.load_model(args.ckpt, device, init_seed=args.init_seed)
    data = dc.load_dataset(args.dataset, args.frames, args.activity)
    names = data.video_fnames[: args.videos]

    arms = {"trained": [], "untrained": [], "raw_features": []}
    for name in names:
        rec = dc.embed(model, data, name, device)
        rec_i = dc.embed(init_model, data, name, device)
        raw = F.normalize(rec["raw"], dim=-1)
        arms["trained"].append({"video": name, **analyse(rec["z"][0], rec["gt"], args.lags, args.far_frames)})
        arms["untrained"].append({"video": name, **analyse(rec_i["z"][0], rec["gt"], args.lags, args.far_frames)})
        arms["raw_features"].append({"video": name, **analyse(raw, rec["gt"], args.lags, args.far_frames)})

    summary = {}
    for arm, recs in arms.items():
        keys = [k for k in recs[0] if k not in ("video", "lag_profile")]
        summary[arm] = {k: mean_of(recs, k) for k in keys}
        lag_means = {}
        for r in recs:
            for row in r["lag_profile"]:
                lag_means.setdefault(row["lag"], []).append(row["mean_cosine"])
        summary[arm]["lag_profile"] = [{"lag": k, "mean_cosine": float(np.mean(v))}
                                       for k, v in sorted(lag_means.items())]

    result = {"checkpoint": str(Path(args.ckpt).resolve()), "dataset": args.dataset,
              "frames": args.frames, "videos": names, "device": device,
              "embedding_dim": int(model.clusters.shape[-1]),
              "raw_feature_dim": int(arms["raw_features"][0]["ambient_dim"]),
              "summary": summary, "per_video": arms}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dc.json_safe(result), indent=2, allow_nan=False) + "\n")

    flat = {}
    for arm, s in summary.items():
        for k, v in s.items():
            if isinstance(v, (int, float)):
                flat[f"{arm}/{k}"] = v
        for row in s["lag_profile"]:
            flat[f"{arm}/lag{row['lag']}_cosine"] = row["mean_cosine"]
    print(json.dumps(dc.json_safe({k: v for k, v in flat.items() if "lag" not in k}), indent=2))
    print(json.dumps(dc.json_safe({k: v for k, v in flat.items() if "lag" in k}), indent=2))
    if run is not None:
        run.log(dc.json_safe(flat))
        run.summary.update(dc.json_safe(flat))
        run.finish()


if __name__ == "__main__":
    main()
