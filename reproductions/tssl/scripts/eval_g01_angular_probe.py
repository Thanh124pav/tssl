"""G01 - within-action and cross-video action-centered geometry probe.

Two protocols, both diagnostic (no training, no GT in any score):

  P1 synthetic warp: a monotone piecewise-linear time warp of the SAME video
     gives an exact known index map, so true frame correspondence is available.
  P2 cross-video phase: frames of different videos sharing a GT action label,
     with GT within-segment relative phase used ONLY to define positives
     (|dphase| <= phase_pos) and hard negatives (|dphase| >= phase_neg).

Scores compared per frame pair, all on frozen embeddings:
  cosine          z_x . z_y                        (the VASOT baseline score)
  centered_dot    (z_x - c_x) . (z_y - c_y)        (unnormalised, centered)
  direction       u_x . u_y, u = (z-c)/||z-c||     (same-ray model)
  line            (u_x . u_y)^2                    (undirected-line ablation)
  radial          -|r_x - r_y|, r = ||z - c||      (radius-only control)
  blend_w         cosine + w * direction
Centers c are the action prototype chosen either by hard argmax of the T_xa
posterior or by the posterior-weighted (soft) mean of prototypes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import diag_common as dc

BLEND_WEIGHTS = (0.25, 0.5, 1.0)


def centers(post, clusters, mode):
    if mode == "hard":
        return clusters[post.argmax(dim=-1)]
    if mode == "soft":
        return post @ clusters
    if mode == "shuffled":
        # Control: a WRONG but FIXED action->centre mapping. The permutation is
        # drawn from a fixed generator so every video in a comparison gets the
        # same wrong mapping; drawing it per call would silently turn this into
        # an "inconsistent centre" control instead.
        k = post.shape[-1]
        g = torch.Generator(device="cpu").manual_seed(1234)
        perm = torch.randperm(k, generator=g).to(post.device)
        return clusters[perm][post.argmax(dim=-1)]
    if mode == "random":  # control: fixed random unit centers
        g = torch.Generator(device="cpu").manual_seed(0)
        rc = F.normalize(torch.randn(clusters.shape, generator=g), dim=-1).to(post.device)
        return rc[post.argmax(dim=-1)]
    raise ValueError(mode)


def geometry(z, post, clusters, mode, eps=1e-6):
    c = centers(post, clusters, mode)
    delta = z - c
    r = delta.norm(dim=-1)
    u = delta / (r.unsqueeze(-1) + eps)
    return {"z": z, "c": c, "u": u, "r": r}


def score_matrices(gx, gy):
    cos = gx["z"] @ gy["z"].T
    dirn = gx["u"] @ gy["u"].T
    out = {"cosine": cos,
           "centered_dot": (gx["z"] - gx["c"]) @ (gy["z"] - gy["c"]).T,
           "direction": dirn,
           "line": dirn ** 2,
           "radial": -(gx["r"].unsqueeze(1) - gy["r"].unsqueeze(0)).abs()}
    for w in BLEND_WEIGHTS:
        out[f"blend_w{w}"] = cos + w * dirn
    return out


def ranking_metrics(score, pos, cand):
    """AUROC/AUPRC over candidate pairs, Recall@k per query row.

    score, pos, cand: (Nx, Ny) float / bool arrays. `cand` marks the pairs that
    enter the comparison at all (the hard-negative pool plus positives).
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    s = score[cand]
    y = pos[cand]
    out = {"n_pairs": int(cand.sum()), "n_positive": int(y.sum())}
    if y.sum() == 0 or y.sum() == y.size:
        return out
    out["auroc"] = float(roc_auc_score(y, s))
    out["auprc"] = float(average_precision_score(y, s))
    rows = np.nonzero(cand.any(axis=1) & pos.any(axis=1))[0]
    r1, r5, ranks = [], [], []
    for i in rows:
        m = cand[i]
        sv = score[i][m]
        pv = pos[i][m]
        order = np.argsort(-sv)
        hit = np.nonzero(pv[order])[0]
        if hit.size == 0:
            continue
        ranks.append(int(hit[0]) + 1)
        r1.append(bool(pv[order[0]]))
        r5.append(bool(pv[order[:5]].any()))
    if ranks:
        out.update({"recall_at_1": float(np.mean(r1)), "recall_at_5": float(np.mean(r5)),
                    "median_rank_of_first_positive": float(np.median(ranks)),
                    "n_queries": len(ranks)})
    return out


def aggregate(records):
    keys = {k for r in records for k in r}
    out = {}
    for k in keys:
        vals = [r[k] for r in records if k in r and isinstance(r[k], (int, float))]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--videos", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=12, help="cross-video pairs for P2")
    ap.add_argument("--warp-severity", type=float, default=0.5)
    ap.add_argument("--warp-knots", type=int, default=5)
    ap.add_argument("--pos-tol", type=int, default=2, help="P1 positive window in frames")
    ap.add_argument("--neg-gap", type=int, default=16, help="P1 hard-negative minimum gap")
    ap.add_argument("--phase-pos", type=float, default=0.10)
    ap.add_argument("--phase-neg", type=float, default=0.50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = dc.pick_device("cuda" if args.device in ("auto", "cuda") else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    run = None if args.no_wandb else dc.init_wandb(
        f"G01_{args.tag}", ["G01", "diagnostic", args.dataset, "phaseG"], vars(args))

    model = dc.load_model(args.ckpt, device)
    data = dc.load_dataset(args.dataset, args.frames, args.activity)
    names = data.video_fnames[: args.videos]
    clusters = model.clusters.detach()

    recs = []
    for name in names:
        rec = dc.embed(model, data, name, device)
        t_xa = dc.seg_plan(model, rec["z"], rec["mask"])
        rows = t_xa[0]
        rec["post"] = rows / rows.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rec["phase"] = dc.relative_phase(rec["gt"])
        recs.append(rec)

    center_modes = ["hard", "soft", "shuffled", "random"]
    result = {"checkpoint": str(Path(args.ckpt).resolve()), "dataset": args.dataset,
              "activity": args.activity, "frames": args.frames, "videos": names,
              "device": device, "blend_weights": list(BLEND_WEIGHTS),
              "protocols": {}}

    # ---------------- P1: synthetic warp, exact known index map -------------
    p1 = {mode: [] for mode in center_modes}
    radius_report = []
    for vi, rec in enumerate(recs):
        src, meta = dc.monotone_warp(args.frames, args.warp_knots, args.warp_severity, seed=vi)
        raw_w = dc.resample_features(rec["raw"], src)
        zw, mask_w = dc.embed_features(model, raw_w, device)
        t_w = dc.seg_plan(model, zw, mask_w)
        post_w = t_w[0] / t_w[0].sum(dim=-1, keepdim=True).clamp_min(1e-12)
        gt = rec["gt"].detach().cpu().numpy()
        gt_w = gt[np.clip(np.round(src).astype(int), 0, args.frames - 1)]
        # candidate / label masks: query = original frame i, candidate = warped frame j
        idx = np.arange(args.frames)
        delta = np.abs(idx[:, None] - src[None, :])
        same_action = gt[:, None] == gt_w[None, :]
        pos = (delta <= args.pos_tol) & same_action
        cand = pos | (same_action & (delta >= args.neg_gap))
        for mode in center_modes:
            gx = geometry(rec["z"][0], rec["post"], clusters, mode)
            gy = geometry(zw[0], post_w, clusters, mode)
            if mode == "hard":
                radius_report.append({"video": rec["name"],
                                      "r_mean": float(gx["r"].mean()),
                                      "r_p05": float(torch.quantile(gx["r"], 0.05)),
                                      "frac_r_below_0p1": float((gx["r"] < 0.1).float().mean())})
            sm = score_matrices(gx, gy)
            entry = {"video": rec["name"], "warp": meta}
            for sname, S in sm.items():
                s = S.detach().cpu().numpy()
                m = ranking_metrics(s, pos, cand)
                # alignment error of the argmax over the full warped video
                full = s.argmax(axis=1)
                m["alignment_error_frames_median"] = float(np.median(np.abs(src[full] - idx)))
                entry[sname] = m
            p1[mode].append(entry)
    result["protocols"]["P1_synthetic_warp"] = {
        "description": "same video, monotone piecewise-linear warp, exact index map",
        "positive_window_frames": args.pos_tol, "hard_negative_gap_frames": args.neg_gap,
        "per_center_mode": {mode: {
            "per_video": p1[mode],
            "summary": {sname: aggregate([e[sname] for e in p1[mode]])
                        for sname in p1[mode][0] if sname not in ("video", "warp")}}
            for mode in center_modes},
        "radius_report": radius_report}

    # ---------------- P2: cross-video, GT phase labels (evaluation only) ----
    p2 = {mode: [] for mode in center_modes}
    npairs = min(args.pairs, len(recs))
    for i in range(npairs):
        j = (i + 1) % len(recs)
        rx, ry = recs[i], recs[j]
        gx_gt = rx["gt"].detach().cpu().numpy()
        gy_gt = ry["gt"].detach().cpu().numpy()
        same_action = gx_gt[:, None] == gy_gt[None, :]
        dphase = np.abs(rx["phase"][:, None] - ry["phase"][None, :])
        pos = same_action & (dphase <= args.phase_pos)
        cand = pos | (same_action & (dphase >= args.phase_neg))
        for mode in center_modes:
            gx = geometry(rx["z"][0], rx["post"], clusters, mode)
            gy = geometry(ry["z"][0], ry["post"], clusters, mode)
            sm = score_matrices(gx, gy)
            entry = {"x": rx["name"], "y": ry["name"]}
            for sname, S in sm.items():
                entry[sname] = ranking_metrics(S.detach().cpu().numpy(), pos, cand)
            p2[mode].append(entry)
    result["protocols"]["P2_cross_video_phase"] = {
        "description": "different videos, same GT action; GT relative phase defines "
                       "positives/hard negatives for evaluation only",
        "phase_pos": args.phase_pos, "phase_neg": args.phase_neg,
        "per_center_mode": {mode: {
            "per_pair": p2[mode],
            "summary": {sname: aggregate([e[sname] for e in p2[mode]])
                        for sname in p2[mode][0] if sname not in ("x", "y")}}
            for mode in center_modes}}

    # ---------------- sensitivity: prototype / feature perturbation ---------
    sens = []
    for sigma in (0.01, 0.05, 0.10):
        g = torch.Generator(device="cpu").manual_seed(7)
        noise = torch.randn(clusters.shape, generator=g).to(clusters.device) * sigma
        pert = F.normalize(clusters + noise, dim=-1)
        rows = []
        for i in range(min(6, npairs)):
            j = (i + 1) % len(recs)
            rx, ry = recs[i], recs[j]
            gx_gt = rx["gt"].detach().cpu().numpy()
            gy_gt = ry["gt"].detach().cpu().numpy()
            same_action = gx_gt[:, None] == gy_gt[None, :]
            dphase = np.abs(rx["phase"][:, None] - ry["phase"][None, :])
            pos = same_action & (dphase <= args.phase_pos)
            cand = pos | (same_action & (dphase >= args.phase_neg))
            gx = geometry(rx["z"][0], rx["post"], pert, "hard")
            gy = geometry(ry["z"][0], ry["post"], pert, "hard")
            sm = score_matrices(gx, gy)
            rows.append({k: ranking_metrics(v.detach().cpu().numpy(), pos, cand)
                         for k, v in sm.items()})
        sens.append({"prototype_noise_sigma": sigma,
                     "summary": {k: aggregate([r[k] for r in rows]) for k in rows[0]}})
    result["prototype_perturbation"] = sens

    # ---------------- cross-video angular dispersion (>=3 videos) ----------
    bins = np.linspace(0, 1, 6)
    disp = []
    for k in range(int(model.n_clusters)):
        for b in range(len(bins) - 1):
            per_video_u = []
            for rec in recs:
                g = geometry(rec["z"][0], rec["post"], clusters, "hard")
                hard = rec["post"].argmax(dim=-1).detach().cpu().numpy()
                m = (hard == k) & (rec["phase"] >= bins[b]) & (rec["phase"] < bins[b + 1])
                if m.sum() >= 3:
                    per_video_u.append(F.normalize(g["u"][torch.as_tensor(m, device=device)].mean(0), dim=-1))
            if len(per_video_u) >= 3:
                U = torch.stack(per_video_u)
                sim = U @ U.T
                off = sim[~torch.eye(len(U), dtype=torch.bool, device=device)]
                disp.append({"cluster": k, "phase_bin": b, "n_videos": len(U),
                             "mean_cross_video_direction_cosine": float(off.mean())})
    # control: same cluster, DIFFERENT phase bin, across videos. If this is as
    # high as the same-phase number, direction encodes action identity only.
    group_u = {}
    for rec in recs:
        g = geometry(rec["z"][0], rec["post"], clusters, "hard")
        hard = rec["post"].argmax(dim=-1).detach().cpu().numpy()
        for k in range(int(model.n_clusters)):
            for b in range(len(bins) - 1):
                m = (hard == k) & (rec["phase"] >= bins[b]) & (rec["phase"] < bins[b + 1])
                if m.sum() >= 3:
                    group_u.setdefault((k, b), []).append(
                        F.normalize(g["u"][torch.as_tensor(m, device=device)].mean(0), dim=-1))
    cross_diff = []
    for (k, b), us in group_u.items():
        for b2 in range(len(bins) - 1):
            if b2 == b or (k, b2) not in group_u:
                continue
            A = torch.stack(us)
            B = torch.stack(group_u[(k, b2)])
            cross_diff.append(float((A @ B.T).mean()))
    result["cross_video_angular_dispersion"] = {
        "per_group": disp,
        "mean_cross_video_direction_cosine": float(np.mean([d["mean_cross_video_direction_cosine"]
                                                            for d in disp])) if disp else None,
        "mean_cross_video_cross_phase_direction_cosine": float(np.mean(cross_diff)) if cross_diff else None,
        "n_groups": len(disp), "n_cross_phase_comparisons": len(cross_diff)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dc.json_safe(result), indent=2, allow_nan=False) + "\n")

    flat = {}
    for proto, key in (("P1_synthetic_warp", "P1"), ("P2_cross_video_phase", "P2")):
        summ = result["protocols"][proto]["per_center_mode"]["hard"]["summary"]
        for sname, m in summ.items():
            for metric in ("auroc", "recall_at_1", "alignment_error_frames_median"):
                if metric in m:
                    flat[f"{key}/hard/{sname}/{metric}"] = m[metric]
    cvd = result["cross_video_angular_dispersion"]
    if cvd["mean_cross_video_direction_cosine"] is not None:
        flat["angular/same_phase_cross_video_cosine"] = cvd["mean_cross_video_direction_cosine"]
    if cvd["mean_cross_video_cross_phase_direction_cosine"] is not None:
        flat["angular/cross_phase_cross_video_cosine"] = cvd["mean_cross_video_cross_phase_direction_cosine"]
    print(json.dumps(dc.json_safe(flat), indent=2))
    if run is not None:
        run.log(dc.json_safe(flat))
        run.summary.update(dc.json_safe(flat))
        run.finish()


if __name__ == "__main__":
    main()
