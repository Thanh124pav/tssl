"""A08 - audit of the fine-grained control-clustering proposal (2026-11-04).

The proposal is a chain of four links. Each link is tested as its own
precondition, so a failure localises instead of condemning the whole idea.

T1  RETRIEVAL RULE. For frame f of video X with action centre a_k, the line
    L = a_k + t(z_f - a_k). The corresponding frame in video Y is the candidate
    that is both similar (cosine / T_xy) and closest to L. Tested as a retrieval
    rule against cosine-only and T_xy-only, on synthetic monotone warps where
    the true correspondence is known exactly. Controls: random and shuffled
    centres, and the two-stage variant (top-m by similarity, then nearest to L).

T2  CONCURRENCY. Correspondence groups inside one segment each get a best-fit
    line through their matched frames across videos; those lines are claimed to
    meet at a new centre. Measured as: how well a line fits a group at all, and
    the least-squares common-point residual of the group lines, against a
    direction-randomised null.

T3  CROSS-RATIO. Four frames per video projected onto the video's principal
    axis give a cross-ratio. The claim needs it stable across videos and
    invariant under time reparameterisation. Both are measured, the second with
    the N-C1 monotone warps.

Link 4 (contrastive anti-collapse) needs no new run: G00 and A05 already
measure whether collapse occurs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import diag_common as dc


# ---------------------------------------------------------------- T1 ----------

def line_distance(cand, centre, through, eps=1e-9):
    """Distance from each candidate to the line centre + t*(through - centre)."""
    d = through - centre
    d = d / (d.norm() + eps)
    v = cand - centre
    par = (v @ d).unsqueeze(-1) * d
    return (v - par).norm(dim=-1), (v @ d)


def rank_metrics(scores, true_pos, src, n):
    """scores: (n_query, n_cand). true_pos: index map. Returns R@1/R@5 and error."""
    order = scores.argsort(dim=-1, descending=True)
    top1 = order[:, 0].cpu().numpy()
    err = np.abs(src[top1] - np.arange(n))
    r1 = float((err <= 2).mean())
    top5 = order[:, :5].cpu().numpy()
    r5 = float(np.mean([(np.abs(src[row] - i) <= 2).any() for i, row in enumerate(top5)]))
    return {"recall_at_1": r1, "recall_at_5": r5,
            "alignment_error_mean": float(err.mean()),
            "alignment_error_median": float(np.median(err))}


def t1_rules(model, zx, zy, post_x, clusters, src, n, centre_mode, t_xy=None, rng=None):
    """All retrieval rules on one (X, warped X) pair with a known index map."""
    hard = post_x.argmax(dim=-1)
    if centre_mode == "prototype":
        C = clusters[hard]
    elif centre_mode == "random":
        g = torch.Generator(device="cpu").manual_seed(0)
        C = F.normalize(torch.randn(clusters.shape, generator=g), dim=-1).to(zx.device)[hard]
    elif centre_mode == "shuffled":
        g = torch.Generator(device="cpu").manual_seed(1234)
        C = clusters[torch.randperm(clusters.shape[0], generator=g).to(zx.device)][hard]
    else:
        raise ValueError(centre_mode)

    cos = zx @ zy.T
    out = {}
    # line-only, plus the two combinations the proposal describes
    dist = torch.empty_like(cos)
    tpar = torch.empty_like(cos)
    for i in range(zx.shape[0]):
        dist[i], tpar[i] = line_distance(zy, C[i], zx[i])
    line_score = -dist
    ray_score = torch.where(tpar > 0, -dist, torch.full_like(dist, -1e3))

    def z(a):  # per-row standardisation so the sum is not dominated by scale
        return (a - a.mean(dim=-1, keepdim=True)) / (a.std(dim=-1, keepdim=True) + 1e-9)

    out["cosine"] = cos
    out["line"] = line_score
    out["ray"] = ray_score
    for w in (0.5, 1.0, 2.0):
        out[f"cosine+line_w{w}"] = z(cos) + w * z(line_score)
    if t_xy is not None:
        out["T_xy"] = t_xy
        for w in (0.5, 1.0, 2.0):
            out[f"T_xy+line_w{w}"] = z(t_xy) + w * z(line_score)
    res = {k: rank_metrics(v, None, src, n) for k, v in out.items()}
    # two-stage: keep the top-m by similarity, then take the nearest to the line
    for m in (5, 10, 25):
        for base_name, base in (("cosine", cos),) + (() if t_xy is None else (("T_xy", t_xy),)):
            keep = base.argsort(dim=-1, descending=True)[:, :m]
            masked = torch.full_like(line_score, -1e9)
            masked.scatter_(1, keep, line_score.gather(1, keep))
            res[f"{base_name}_top{m}_then_line"] = rank_metrics(masked, None, src, n)
    return res


# ---------------------------------------------------------------- T2 ----------

def fit_line(points):
    """Best-fit line: (point on line, unit direction, mean residual, variance frac)."""
    mu = points.mean(dim=0)
    X = points - mu
    _, s, vt = torch.linalg.svd(X, full_matrices=False)
    d = vt[0]
    resid = (X - (X @ d).unsqueeze(-1) * d).norm(dim=-1)
    var = (s ** 2)
    return mu, d, float(resid.mean()), float(var[0] / max(float(var.sum()), 1e-12))


def common_point(mus, ds):
    """Least-squares point minimising the sum of squared distances to the lines."""
    dev = mus[0].device
    dim = mus[0].shape[0]
    A = torch.zeros(dim, dim, device=dev, dtype=torch.float64)
    b = torch.zeros(dim, device=dev, dtype=torch.float64)
    for mu, d in zip(mus, ds):
        d = d.double()
        P = torch.eye(dim, device=dev, dtype=torch.float64) - torch.outer(d, d)
        A += P
        b += P @ mu.double()
    x = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
    res = []
    for mu, d in zip(mus, ds):
        v = x - mu.double()
        res.append(float((v - (v @ d.double()) * d.double()).norm()))
    return x.float(), float(np.mean(res))


# ---------------------------------------------------------------- T3 ----------

def cross_ratio(t):
    """Cross-ratio of four scalars, with a clamped denominator."""
    t1, t2, t3, t4 = t
    num = (t1 - t3) * (t2 - t4)
    den = (t2 - t3) * (t1 - t4)
    if abs(den) < 1e-8:
        return float("nan")
    return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--warp-severity", type=float, default=0.5)
    ap.add_argument("--positions", type=float, nargs=4, default=[0.1, 0.4, 0.6, 0.9])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = dc.pick_device("cuda" if args.device in ("auto", "cuda") else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)
    run = None if args.no_wandb else dc.init_wandb(
        f"A08_{args.tag}", ["A08", "diagnostic", args.dataset, "proposal_audit"], vars(args))

    model = dc.load_model(args.ckpt, device)
    data = dc.load_dataset(args.dataset, args.frames, args.activity)
    names = data.video_fnames[: args.videos]
    clusters = model.clusters.detach()
    n = args.frames

    recs = []
    for name in names:
        rec = dc.embed(model, data, name, device)
        t = dc.seg_plan(model, rec["z"], rec["mask"])
        rec["post"] = t[0] / t[0].sum(-1, keepdim=True).clamp_min(1e-12)
        rec["phase"] = dc.relative_phase(rec["gt"])
        recs.append(rec)

    result = {"checkpoint": str(Path(args.ckpt).resolve()), "dataset": args.dataset,
              "frames": n, "videos": names, "device": device,
              "warp_severity": args.warp_severity}

    # ---------------- T1: the retrieval rule --------------------------------
    t1 = {}
    for vi, rec in enumerate(recs):
        src, _ = dc.monotone_warp(n, 5, args.warp_severity, seed=vi)
        raw_w = dc.resample_features(rec["raw"], src)
        zy, mask_y = dc.embed_features(model, raw_w, device)
        T, _ = dc.align_plan(model, rec["z"], zy, rec["mask"], mask_y)
        t_xy = T[0] / T[0].sum(-1, keepdim=True).clamp_min(1e-12)
        for mode in ("prototype", "random", "shuffled"):
            res = t1_rules(model, rec["z"][0], zy[0], rec["post"], clusters, src, n, mode,
                           t_xy=t_xy, rng=rng)
            for rule, m in res.items():
                t1.setdefault(mode, {}).setdefault(rule, []).append(m)
    result["T1_retrieval_rule"] = {
        mode: {rule: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
               for rule, rows in rules.items()}
        for mode, rules in t1.items()}

    # ---------------- T2: do the group lines concur? ------------------------
    # correspondence groups: reference video 0, every 4th frame, matched into the
    # other videos by the solver's argmax
    plans = {}
    for b in range(1, len(recs)):
        T, _ = dc.align_plan(model, recs[0]["z"], recs[b]["z"], recs[0]["mask"], recs[b]["mask"])
        plans[b] = T[0].argmax(-1).cpu().numpy()
    gt0 = recs[0]["gt"].cpu().numpy()
    segs = dc.gt_segments(recs[0]["gt"])
    groups, per_segment = [], {}
    for s, e, lab in segs:
        if e - s < 8:
            continue
        idx = list(range(s, e, max((e - s) // 6, 1)))
        lines = []
        for i in idx:
            pts = [recs[0]["z"][0, i]] + [recs[b]["z"][0, int(plans[b][i])] for b in plans]
            P = torch.stack(pts)
            mu, d, resid, vfrac = fit_line(P)
            scale = float((P - P.mean(0)).norm(dim=-1).mean())
            groups.append({"segment": [s, e], "frame": i, "n_points": int(P.shape[0]),
                           "line_fit_residual": resid,
                           "line_fit_residual_normalised": resid / max(scale, 1e-9),
                           "line_variance_fraction": vfrac})
            lines.append((mu, d))
        if len(lines) >= 3:
            mus = [m for m, _ in lines]
            ds = [d for _, d in lines]
            pt, res = common_point(mus, ds)
            scale = float(torch.stack(mus).std(dim=0).norm())
            # null: same line points, randomised directions
            null = []
            for _ in range(8):
                rd = [F.normalize(torch.randn(ds[0].shape, device=device), dim=0) for _ in ds]
                null.append(common_point(mus, rd)[1])
            per_segment[f"{s}-{e}"] = {
                "n_lines": len(lines),
                "common_point_residual": res,
                "common_point_residual_normalised": res / max(scale, 1e-9),
                "null_residual": float(np.mean(null)),
                "ratio_real_over_null": res / max(float(np.mean(null)), 1e-12),
                "distance_to_action_prototype": float(
                    (pt - clusters[int(np.bincount(
                        recs[0]["post"][s:e].argmax(-1).cpu().numpy()).argmax())]).norm())}
    result["T2_concurrency"] = {
        "per_group_line_fit": {
            "n_groups": len(groups),
            "mean_line_variance_fraction": float(np.mean([g["line_variance_fraction"] for g in groups])),
            "mean_normalised_residual": float(np.mean([g["line_fit_residual_normalised"] for g in groups]))},
        "per_segment": per_segment,
        "summary": {k: float(np.mean([v[k] for v in per_segment.values()]))
                    for k in ("common_point_residual", "common_point_residual_normalised",
                              "null_residual", "ratio_real_over_null",
                              "distance_to_action_prototype")} if per_segment else {}}

    # ---------------- T3: cross-ratio stability and warp invariance ---------
    crs, crs_warp, crs_index = [], [], []
    for vi, rec in enumerate(recs):
        Z = rec["z"][0]
        mu, d, _, vfrac = fit_line(Z)
        t = ((Z - mu) @ d).cpu().numpy()
        pos = [int(p * (n - 1)) for p in args.positions]
        crs.append({"video": rec["name"], "line_variance_fraction": vfrac,
                    "cross_ratio": cross_ratio([t[p] for p in pos])})
        # the same four SEMANTIC frames after a monotone time warp
        src, _ = dc.monotone_warp(n, 5, args.warp_severity, seed=vi)
        raw_w = dc.resample_features(rec["raw"], src)
        zw, _ = dc.embed_features(model, raw_w, device)
        muw, dw, _, _ = fit_line(zw[0])
        tw = ((zw[0] - muw) @ dw).cpu().numpy()
        # where each original position landed in the warped video
        warped_pos = [int(np.argmin(np.abs(src - p))) for p in pos]
        crs_warp.append(cross_ratio([tw[p] for p in warped_pos]))
        crs_index.append(cross_ratio([float(p) for p in pos]))
    vals = np.array([c["cross_ratio"] for c in crs], dtype=float)
    wv = np.array(crs_warp, dtype=float)
    ok = np.isfinite(vals) & np.isfinite(wv)
    result["T3_cross_ratio"] = {
        "per_video": crs,
        "line_variance_fraction_mean": float(np.mean([c["line_variance_fraction"] for c in crs])),
        "cross_ratio_mean": float(np.nanmean(vals)),
        "cross_ratio_std_across_videos": float(np.nanstd(vals)),
        "cross_ratio_cv_across_videos": float(np.nanstd(vals) / max(abs(np.nanmean(vals)), 1e-9)),
        "cross_ratio_from_frame_index": float(np.nanmean(crs_index)),
        "warped_cross_ratio_mean": float(np.nanmean(wv)),
        "mean_abs_change_under_warp": float(np.nanmean(np.abs(wv[ok] - vals[ok]))),
        "mean_rel_change_under_warp": float(np.nanmean(
            np.abs(wv[ok] - vals[ok]) / np.maximum(np.abs(vals[ok]), 1e-9)))}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dc.json_safe(result), indent=2, allow_nan=False) + "\n")

    print("=== T1 retrieval rule (Recall@1 within 2 frames / alignment error) ===")
    for mode in result["T1_retrieval_rule"]:
        print(f"-- centres = {mode}")
        for rule, m in result["T1_retrieval_rule"][mode].items():
            print(f"   {rule:26s} R@1={m['recall_at_1']:.4f}  R@5={m['recall_at_5']:.4f}  "
                  f"err={m['alignment_error_mean']:7.2f}")
    print("\n=== T2 concurrency ===")
    print(json.dumps(dc.json_safe(result["T2_concurrency"]["per_group_line_fit"]), indent=1))
    print(json.dumps(dc.json_safe(result["T2_concurrency"]["summary"]), indent=1))
    print("\n=== T3 cross-ratio ===")
    print(json.dumps(dc.json_safe({k: v for k, v in result["T3_cross_ratio"].items()
                                   if k != "per_video"}), indent=1))
    if run is not None:
        flat = {f"T1/{mode}/{rule}/{k}": v
                for mode, rules in result["T1_retrieval_rule"].items()
                for rule, m in rules.items() for k, v in m.items()}
        flat.update({f"T2/{k}": v for k, v in result["T2_concurrency"]["summary"].items()})
        flat.update({f"T3/{k}": v for k, v in result["T3_cross_ratio"].items() if k != "per_video"})
        run.log(dc.json_safe(flat))
        run.summary.update(dc.json_safe(flat))
        run.finish()


if __name__ == "__main__":
    main()
