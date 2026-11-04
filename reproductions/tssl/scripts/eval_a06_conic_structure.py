"""A06 - is there conic (not just linear) structure in the action space?

A02 measured linearity and found the matched sets are LESS linear than random.
That rules out a ray/line, and says nothing about a conic: points on a circle
have low linearity too. This packet tests the conic hypothesis directly.

Method, designed so that parameter count cannot manufacture a result:

  * A conic in 2D has 5 free parameters, so it fits any 5 points exactly. Every
    fit here is therefore evaluated on HELD-OUT points. The PCA plane itself is
    estimated on the training half only.
  * Line, circle and conic are fitted in the SAME algebraic form, as nested
    constraints on the coefficient vector of [x^2, xy, y^2, x, y, 1], and scored
    with the SAME algebraic residual, so the three are directly comparable.
  * Every fit is repeated on a Gaussian null with the group's own covariance and
    the same point count. The reported statistic is the ratio real/null: a
    conic that fits real data no better than matched-covariance noise is not
    structure.

Groups are (video, GT action) frame sets, which is exactly the object the G02
hypothesis is about: "frames of action k from video v lie near an
action-centered circle with video-specific radius r_{v,k}".
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import diag_common as dc

MODELS = {"line": [3, 4, 5], "circle": None, "conic": [0, 1, 2, 3, 4, 5]}


def design(p):
    x, y = p[:, 0], p[:, 1]
    return np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], axis=1)


def fit_algebraic(p_train, model):
    """Least-norm solution of min ||D a|| with ||a|| = 1 under the model's
    constraint. line: only [x, y, 1]. circle: x^2 and y^2 share a coefficient
    and xy is dropped. conic: unconstrained."""
    D = design(p_train)
    if model == "line":
        sub = D[:, [3, 4, 5]]
        _, _, vt = np.linalg.svd(sub, full_matrices=False)
        a = np.zeros(6)
        a[[3, 4, 5]] = vt[-1]
    elif model == "circle":
        # x^2 + y^2 share one coefficient -> basis [x^2+y^2, x, y, 1]
        sub = np.stack([D[:, 0] + D[:, 2], D[:, 3], D[:, 4], D[:, 5]], axis=1)
        _, _, vt = np.linalg.svd(sub, full_matrices=False)
        c = vt[-1]
        a = np.array([c[0], 0.0, c[0], c[1], c[2], c[3]])
        a = a / (np.linalg.norm(a) + 1e-12)
    elif model == "conic":
        _, _, vt = np.linalg.svd(D, full_matrices=False)
        a = vt[-1]
    else:
        raise ValueError(model)
    return a


def residual(p_test, a):
    D = design(p_test)
    return float(np.sqrt((D @ a) ** 2).mean())


def plane_and_scale(points_train):
    """PCA plane and normalisation estimated on the training half ONLY."""
    mu = points_train.mean(axis=0)
    X = points_train - mu
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    basis = vt[:2]
    proj = X @ basis.T
    scale = float(np.sqrt((proj ** 2).sum(axis=1).mean())) or 1.0
    var = (s ** 2)
    return {"mu": mu, "basis": basis, "scale": scale,
            "plane_variance_fraction": float(var[:2].sum() / max(var.sum(), 1e-12))}


def project(points, pl):
    return ((points - pl["mu"]) @ pl["basis"].T) / pl["scale"]


def group_test(Z, rng, min_points, n_null=8):
    """Z: (n, d) numpy. Returns held-out residual ratios for line/circle/conic."""
    n = Z.shape[0]
    if n < min_points:
        return None
    idx = rng.permutation(n)
    half = n // 2
    tr, te = Z[idx[:half]], Z[idx[half:]]
    pl = plane_and_scale(tr)
    ptr, pte = project(tr, pl), project(te, pl)
    out = {"n_points": int(n), "n_train": int(half), "n_test": int(n - half),
           "plane_variance_fraction": pl["plane_variance_fraction"]}
    # Gaussian null with the same covariance, in the SAME plane, same counts
    cov = np.cov(ptr.T) if ptr.shape[0] > 2 else np.eye(2)
    null_res = {m: [] for m in MODELS}
    for _ in range(n_null):
        ntr = rng.multivariate_normal(ptr.mean(axis=0), cov, size=ptr.shape[0])
        nte = rng.multivariate_normal(ptr.mean(axis=0), cov, size=pte.shape[0])
        for m in MODELS:
            null_res[m].append(residual(nte, fit_algebraic(ntr, m)))
    for m in MODELS:
        r = residual(pte, fit_algebraic(ptr, m))
        nr = float(np.mean(null_res[m]))
        out[f"{m}_heldout_residual"] = r
        out[f"{m}_null_residual"] = nr
        out[f"{m}_ratio_real_over_null"] = float(r / max(nr, 1e-12))
    return out



def quadric_test(Z, rng, dims=(3, 5), n_null=8):
    """A conic is planar. If the group is not planar, the natural generalisation
    is a QUADRIC hypersurface in the top-m PCA dimensions: x^T A x + b^T x + c = 0,
    with m(m+3)/2 free parameters. Fitted on a training half in a basis estimated
    on that half only, scored on held-out points, against a matched-covariance
    Gaussian null with the same point count and the same number of parameters."""
    out = {}
    n = Z.shape[0]
    for m in dims:
        if n < 2 * (m * (m + 3) // 2 + 2):
            continue  # not enough points for an honest held-out fit
        idx = rng.permutation(n)
        half = n // 2
        tr, te = Z[idx[:half]], Z[idx[half:]]
        mu = tr.mean(axis=0)
        _, sv, vt = np.linalg.svd(tr - mu, full_matrices=False)
        basis = vt[:m]
        ptr = (tr - mu) @ basis.T
        pte = (te - mu) @ basis.T
        scale = float(np.sqrt((ptr ** 2).sum(axis=1).mean())) or 1.0
        ptr, pte = ptr / scale, pte / scale

        def design_q(p):
            cols = [p[:, i] * p[:, j] for i in range(m) for j in range(i, m)]
            cols += [p[:, i] for i in range(m)] + [np.ones(p.shape[0])]
            return np.stack(cols, axis=1)

        def fit_and_score(a_tr, a_te):
            _, _, v = np.linalg.svd(design_q(a_tr), full_matrices=False)
            coef = v[-1]
            return float(np.sqrt((design_q(a_te) @ coef) ** 2).mean())

        real = fit_and_score(ptr, pte)
        cov = np.cov(ptr.T)
        nulls = []
        for _ in range(n_null):
            ntr = rng.multivariate_normal(ptr.mean(axis=0), cov, size=ptr.shape[0])
            nte = rng.multivariate_normal(ptr.mean(axis=0), cov, size=pte.shape[0])
            nulls.append(fit_and_score(ntr, nte))
        nr = float(np.mean(nulls))
        var = sv ** 2
        out[f"quadric{m}d_heldout_residual"] = real
        out[f"quadric{m}d_null_residual"] = nr
        out[f"quadric{m}d_ratio_real_over_null"] = float(real / max(nr, 1e-12))
        out[f"quadric{m}d_subspace_variance_fraction"] = float(var[:m].sum() / max(var.sum(), 1e-12))
        out[f"quadric{m}d_n_params"] = int(m * (m + 3) // 2)
    return out


def radius_stats(Z, center, rng, n_null=8):
    """Core of the circle hypothesis: is ||z - center|| roughly constant?"""
    r = np.linalg.norm(Z - center, axis=1)
    cv = float(r.std() / max(r.mean(), 1e-12))
    # null: Gaussian with the group's covariance, same centre offset
    mu, cov = Z.mean(axis=0), np.cov(Z.T)
    cvs = []
    for _ in range(n_null):
        S = rng.multivariate_normal(mu, cov, size=Z.shape[0])
        rs = np.linalg.norm(S - center, axis=1)
        cvs.append(rs.std() / max(rs.mean(), 1e-12))
    null_cv = float(np.mean(cvs))
    return {"radius_mean": float(r.mean()), "radius_cv": cv,
            "radius_cv_null": null_cv,
            "radius_cv_ratio_real_over_null": float(cv / max(null_cv, 1e-12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--videos", type=int, default=12)
    ap.add_argument("--min-points", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=5, help="random train/test splits per group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = dc.pick_device("cuda" if args.device in ("auto", "cuda") else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)
    run = None if args.no_wandb else dc.init_wandb(
        f"A06_{args.tag}", ["A06", "diagnostic", args.dataset, "action_space", "conic"], vars(args))

    model = dc.load_model(args.ckpt, device)
    data = dc.load_dataset(args.dataset, args.frames, args.activity)
    names = data.video_fnames[: args.videos]
    clusters = model.clusters.detach().cpu().numpy()

    groups, radii, traj = [], [], []
    per_action_radius = {}
    for name in names:
        rec = dc.embed(model, data, name, device)
        t_xa = dc.seg_plan(model, rec["z"], rec["mask"])
        post = (t_xa[0] / t_xa[0].sum(-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
        Z = rec["z"][0].cpu().numpy()
        gt = rec["gt"].cpu().numpy()
        valid = rec["mask"][0].cpu().numpy().astype(bool)
        hard = post.argmax(axis=1)

        for a in np.unique(gt):
            m = (gt == a) & valid
            if m.sum() < args.min_points:
                continue
            G = Z[m]
            # centre choices: the assigned action prototype (the G02 claim) and
            # the group's own centroid (a strictly easier target)
            k = int(np.bincount(hard[m]).argmax())
            for cname, c in (("prototype", clusters[k]), ("group_centroid", G.mean(axis=0))):
                radii.append({"video": name, "gt_action": int(a), "centre": cname,
                              **radius_stats(G, c, rng)})
                if cname == "prototype":
                    per_action_radius.setdefault(int(a), []).append(
                        float(np.linalg.norm(G - c, axis=1).mean()))
            for rep in range(args.repeats):
                g = group_test(G, rng, args.min_points)
                if g:
                    g.update(quadric_test(G, rng))
                    g.update({"video": name, "gt_action": int(a), "repeat": rep,
                              "scope": "video_action_group"})
                    groups.append(g)
        # whole-video trajectory, as a second scope
        for rep in range(args.repeats):
            g = group_test(Z[valid], rng, args.min_points)
            if g:
                g.update(quadric_test(Z[valid], rng))
                g.update({"video": name, "gt_action": -1, "repeat": rep,
                          "scope": "whole_video"})
                groups.append(g)

    def agg(rows, keys):
        return {k: float(np.nanmean([r[k] for r in rows if k in r])) for k in keys}

    rkeys = ["line_heldout_residual", "line_null_residual", "line_ratio_real_over_null",
             "circle_heldout_residual", "circle_null_residual", "circle_ratio_real_over_null",
             "conic_heldout_residual", "conic_null_residual", "conic_ratio_real_over_null",
             "plane_variance_fraction", "n_points",
             "quadric3d_ratio_real_over_null", "quadric3d_subspace_variance_fraction",
             "quadric5d_ratio_real_over_null", "quadric5d_subspace_variance_fraction",
             "quadric3d_heldout_residual", "quadric3d_null_residual",
             "quadric5d_heldout_residual", "quadric5d_null_residual"]
    by_scope = {}
    for scope in ("video_action_group", "whole_video"):
        rows = [g for g in groups if g["scope"] == scope]
        if rows:
            present = [k for k in rkeys if any(k in r for r in rows)]
            by_scope[scope] = {"n_fits": len(rows), **agg(rows, present)}
            # how often does the conic beat the circle beat the line, held out?
            by_scope[scope]["frac_conic_better_than_circle"] = float(np.mean(
                [r["conic_heldout_residual"] < r["circle_heldout_residual"] for r in rows]))
            by_scope[scope]["frac_circle_better_than_line"] = float(np.mean(
                [r["circle_heldout_residual"] < r["line_heldout_residual"] for r in rows]))
            by_scope[scope]["frac_conic_beats_its_null"] = float(np.mean(
                [r["conic_ratio_real_over_null"] < 1.0 for r in rows]))
            by_scope[scope]["frac_circle_beats_its_null"] = float(np.mean(
                [r["circle_ratio_real_over_null"] < 1.0 for r in rows]))

    rad_keys = ["radius_mean", "radius_cv", "radius_cv_null", "radius_cv_ratio_real_over_null"]
    by_centre = {c: {"n_groups": len([r for r in radii if r["centre"] == c]),
                     **agg([r for r in radii if r["centre"] == c], rad_keys)}
                 for c in ("prototype", "group_centroid")}

    # the "video-specific radius" part of the G02 claim: for one action, does the
    # radius differ between videos by more than it varies inside a video?
    vs = []
    for a, vals in per_action_radius.items():
        if len(vals) >= 3:
            vs.append({"gt_action": a, "n_videos": len(vals),
                       "between_video_radius_std": float(np.std(vals)),
                       "mean_radius": float(np.mean(vals))})
    within = float(np.nanmean([r["radius_cv"] * r["radius_mean"]
                               for r in radii if r["centre"] == "prototype"]))
    result = {"checkpoint": str(Path(args.ckpt).resolve()), "dataset": args.dataset,
              "frames": args.frames, "videos": names, "device": device,
              "method": {
                  "fit": "nested algebraic models on [x^2, xy, y^2, x, y, 1]; line = 3 dof, "
                         "circle = 4 dof (equal x^2,y^2 and no xy), conic = 6 dof",
                  "evaluation": "held-out half; the PCA plane and the normalisation are "
                                "estimated on the training half only",
                  "null": "Gaussian with the training half's in-plane covariance, same counts, "
                          "fitted and scored identically; the statistic is real/null"},
              "by_scope": by_scope,
              "radius_constancy": by_centre,
              "video_specific_radius": {
                  "per_action": vs,
                  "mean_between_video_radius_std": float(np.mean([v["between_video_radius_std"] for v in vs])) if vs else None,
                  "mean_within_group_radius_std": within},
              "groups": groups, "radii": radii}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dc.json_safe(result), indent=2, allow_nan=False) + "\n")

    flat = {}
    for scope, s in by_scope.items():
        for k, v in s.items():
            flat[f"{scope}/{k}"] = v
    for c, s in by_centre.items():
        for k, v in s.items():
            flat[f"radius/{c}/{k}"] = v
    print(json.dumps(dc.json_safe(flat), indent=2))
    if run is not None:
        run.log(dc.json_safe(flat))
        run.summary.update(dc.json_safe(flat))
        run.finish()


if __name__ == "__main__":
    main()
