"""A02 - what structure do the frames matched to one frame form across n >= 3 videos?

For a reference frame i of video X and m other videos Y_1..Y_m, the solver
gives one matched frame per other video. This packet asks what the matched SET
looks like: is it a tight cluster, a line, or diffuse? Is it tighter than the
obvious baselines? And are the matches mutually consistent?

Baselines for the matched set, all with the same m points, one per video:
  matched      argmax of the real transport plan X -> Y_k
  same_time    the frame at the same relative index i/N in each Y_k
  same_action  a random frame of the same GT action in each Y_k (eval only)
  random       a uniformly random frame of each Y_k

Shape descriptors of a set: mean pairwise cosine (tightness), PCA spectrum of
the set (linearity = top-1 variance fraction, participation ratio), distance to
the reference frame, and the temporal spread of the matched indices.

Consistency: direct X->Z matching versus the composition X->Y->Z, in frames.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import diag_common as dc


def set_shape(points, reference=None):
    """points: (m, d). Returns tightness / linearity descriptors."""
    m = points.shape[0]
    pn = F.normalize(points, dim=-1)
    sim = pn @ pn.T
    off = sim[~torch.eye(m, dtype=torch.bool, device=points.device)]
    centroid = points.mean(0)
    radius = (points - centroid).norm(dim=-1)
    out = {"n_points": int(m),
           "mean_pairwise_cosine": float(off.mean()),
           "min_pairwise_cosine": float(off.min()),
           "mean_radius_from_centroid": float(radius.mean()),
           "max_radius_from_centroid": float(radius.max())}
    if m >= 3:
        pc = points - centroid
        cov = (pc.T @ pc) / (m - 1)
        ev = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
        tot = float(ev.sum())
        if tot > 1e-12:
            out["linearity_top1_variance_fraction"] = float(ev[0] / tot)
            out["top2_variance_fraction"] = float(ev[:2].sum() / tot)
            out["participation_ratio"] = float((tot ** 2) / float((ev ** 2).sum()))
    if reference is not None:
        rn = F.normalize(reference, dim=-1)
        out["mean_cosine_to_reference"] = float((pn @ rn).mean())
        out["mean_distance_to_reference"] = float((points - reference).norm(dim=-1).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--videos", type=int, default=6, help="group size, >= 3 (1 reference + m others)")
    ap.add_argument("--references", type=int, default=4, help="how many videos take the reference role")
    ap.add_argument("--stride", type=int, default=8, help="sample every k-th reference frame")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    assert args.videos >= 3, "this packet is about n >= 3 videos"
    device = dc.pick_device("cuda" if args.device in ("auto", "cuda") else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.set_grad_enabled(False)  # nothing here trains; model.clusters carries requires_grad
    rng = np.random.default_rng(args.seed)
    run = None if args.no_wandb else dc.init_wandb(
        f"A02_{args.tag}", ["A02", "diagnostic", args.dataset, "action_space"], vars(args))

    model = dc.load_model(args.ckpt, device)
    data = dc.load_dataset(args.dataset, args.frames, args.activity)
    names = data.video_fnames[: args.videos]
    recs = []
    for name in names:
        rec = dc.embed(model, data, name, device)
        t_xa = dc.seg_plan(model, rec["z"], rec["mask"])
        rec["post"] = t_xa[0] / t_xa[0].sum(-1, keepdim=True).clamp_min(1e-12)
        recs.append(rec)
    n = args.frames

    # all pairwise plans once; plans[(a, b)] is the argmax match a -> b
    plans, rows = {}, {}
    for a in range(len(recs)):
        for b in range(len(recs)):
            if a == b:
                continue
            T, _ = dc.align_plan(model, recs[a]["z"], recs[b]["z"], recs[a]["mask"], recs[b]["mask"])
            P = T[0] / T[0].sum(-1, keepdim=True).clamp_min(1e-12)
            plans[(a, b)] = P.argmax(-1).cpu().numpy()
            rows[(a, b)] = P

    arms = ("matched", "same_time", "same_action", "random")
    per_frame = {arm: [] for arm in arms}
    temporal, agreement = [], []
    frame_idx = list(range(0, n, args.stride))

    for a in range(min(args.references, len(recs))):
        others = [b for b in range(len(recs)) if b != a]
        gt_a = recs[a]["gt"].cpu().numpy()
        for i in frame_idx:
            if not bool(recs[a]["mask"][0, i]):
                continue
            ref = recs[a]["z"][0, i]
            picks = {arm: [] for arm in arms}
            matched_idx = []
            for b in others:
                gt_b = recs[b]["gt"].cpu().numpy()
                valid = recs[b]["mask"][0].cpu().numpy().astype(bool)
                j = int(plans[(a, b)][i])
                matched_idx.append(j)
                picks["matched"].append(recs[b]["z"][0, j])
                picks["same_time"].append(recs[b]["z"][0, i])
                cand = np.nonzero((gt_b == gt_a[i]) & valid)[0]
                picks["same_action"].append(
                    recs[b]["z"][0, int(rng.choice(cand))] if len(cand) else recs[b]["z"][0, i])
                picks["random"].append(recs[b]["z"][0, int(rng.integers(0, n))])
            for arm in arms:
                s = set_shape(torch.stack(picks[arm]), ref)
                s.update({"reference_video": names[a], "frame": i})
                per_frame[arm].append(s)
            mi = np.asarray(matched_idx, dtype=float)
            temporal.append({"frame": i, "reference_video": names[a],
                             "matched_index_mean": float(mi.mean()),
                             "matched_index_std": float(mi.std()),
                             "matched_index_bias_vs_i": float(mi.mean() - i),
                             "matched_index_range": float(mi.max() - mi.min())})
            agreement.append(float(np.mean([gt_a[i] == recs[b]["gt"].cpu().numpy()[int(plans[(a, b)][i])]
                                            for b in others])))

    def agg(recs_list):
        keys = {k for r in recs_list for k in r if isinstance(r.get(k), (int, float))}
        return {k: float(np.nanmean([r[k] for r in recs_list if k in r])) for k in keys
                if k not in ("frame",)}

    result = {"checkpoint": str(Path(args.ckpt).resolve()), "dataset": args.dataset,
              "frames": n, "group_videos": names, "group_size": len(recs),
              "points_per_set": len(recs) - 1, "reference_videos": args.references,
              "reference_frames_per_video": len(frame_idx), "device": device,
              "set_shape": {arm: agg(per_frame[arm]) for arm in arms},
              "temporal_spread": agg(temporal),
              "matched_gt_action_agreement": float(np.mean(agreement))}

    # cycle consistency: X->Z directly versus X->Y->Z
    cyc = []
    for a in range(min(args.references, len(recs))):
        for b in range(len(recs)):
            if b == a:
                continue
            for c in range(len(recs)):
                if c in (a, b):
                    continue
                direct = plans[(a, c)]
                composed = plans[(b, c)][plans[(a, b)]]
                d = np.abs(direct - composed)
                cyc.append({"a": names[a], "b": names[b], "c": names[c],
                            "mean_abs_frame_disagreement": float(d.mean()),
                            "median_abs_frame_disagreement": float(np.median(d)),
                            "fraction_within_4_frames": float((d <= 4).mean())})
    result["cycle_consistency"] = {
        "per_triple": cyc,
        "summary": {k: float(np.mean([t[k] for t in cyc]))
                    for k in ("mean_abs_frame_disagreement", "median_abs_frame_disagreement",
                              "fraction_within_4_frames")} if cyc else {}}

    # round-trip X -> Y -> X
    rt = []
    for a in range(min(args.references, len(recs))):
        for b in range(len(recs)):
            if b == a:
                continue
            back = plans[(b, a)][plans[(a, b)]]
            d = np.abs(back - np.arange(n))
            rt.append({"a": names[a], "b": names[b],
                       "mean_round_trip_error": float(d.mean()),
                       "fraction_within_4_frames": float((d <= 4).mean())})
    result["round_trip"] = {"per_pair": rt,
                            "summary": {k: float(np.mean([r[k] for r in rt]))
                                        for k in ("mean_round_trip_error", "fraction_within_4_frames")}}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dc.json_safe(result), indent=2, allow_nan=False) + "\n")

    flat = {}
    for arm, s in result["set_shape"].items():
        for k in ("mean_pairwise_cosine", "mean_radius_from_centroid",
                  "linearity_top1_variance_fraction", "participation_ratio",
                  "mean_cosine_to_reference"):
            if k in s:
                flat[f"set/{arm}/{k}"] = s[k]
    flat.update({f"temporal/{k}": v for k, v in result["temporal_spread"].items()})
    flat.update({f"cycle/{k}": v for k, v in result["cycle_consistency"]["summary"].items()})
    flat.update({f"roundtrip/{k}": v for k, v in result["round_trip"]["summary"].items()})
    flat["matched_gt_action_agreement"] = result["matched_gt_action_agreement"]
    print(json.dumps(dc.json_safe(flat), indent=2))
    if run is not None:
        run.log(dc.json_safe(flat))
        run.summary.update(dc.json_safe(flat))
        run.finish()


if __name__ == "__main__":
    main()
