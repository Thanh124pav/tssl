"""G00 - transport and action-prototype geometry audit on frozen checkpoints.

Measures, for one checkpoint and one dataset:
  (a) T_xa and T_xy row-conditional statistics under a stated normalisation,
  (b) action-prototype separation / scatter / GT confusion against a random control,
  (c) T_xa posterior behaviour as a function of signed distance to GT boundaries,
  (d) per-video posterior temporal jitter, kept separate from checkpoint drift.

No training. GT labels are used for evaluation-side stratification only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import diag_common as dc


def prototype_geometry(clusters, label):
    """clusters: (K, d) tensor, assumed unit-norm by the training loop."""
    c = clusters.detach().float()
    k = c.shape[0]
    norms = c.norm(dim=-1)
    cn = F.normalize(c, dim=-1)
    cos = cn @ cn.T
    off = cos[~torch.eye(k, dtype=torch.bool, device=c.device)]
    # nearest-neighbour cosine per prototype (higher = less separated)
    nn_cos = (cos - torch.eye(k, device=c.device) * 2.0).max(dim=-1).values
    return {
        "which": label,
        "n_prototypes": k,
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std(unbiased=False).item(),
        "offdiag_cosine_mean": off.mean().item(),
        "offdiag_cosine_max": off.max().item(),
        "offdiag_cosine_min": off.min().item(),
        "nearest_cosine_mean": nn_cos.mean().item(),
        "nearest_cosine_max": nn_cos.max().item(),
        "nearest_margin_mean": (1.0 - nn_cos).mean().item(),
    }


def hungarian_drift(a, b):
    """Best-match cosine between two prototype sets (K, d)."""
    from scipy.optimize import linear_sum_assignment
    an, bn = F.normalize(a.float(), dim=-1), F.normalize(b.float(), dim=-1)
    sim = (an @ bn.T).detach().cpu().numpy()
    rows, cols = linear_sum_assignment(-sim)
    matched = sim[rows, cols]
    return {"matched_cosine_mean": float(matched.mean()),
            "matched_cosine_min": float(matched.min()),
            "matched_displacement_mean": float((1.0 - matched).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--videos", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--init-seed", type=int, default=1,
                    help="untrained control rebuilt from the checkpoint hyper-parameters")
    ap.add_argument("--compare-ckpt", default=None,
                    help="second checkpoint for prototype drift (optional)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = dc.pick_device("cuda" if args.device in ("auto", "cuda") else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))

    run = None if args.no_wandb else dc.init_wandb(
        f"G00_{args.tag}", ["G00", "diagnostic", args.dataset, "phaseG"], vars(args))

    model = dc.load_model(args.ckpt, device)
    init_model = dc.load_model(args.ckpt, device, init_seed=args.init_seed)
    data = dc.load_dataset(args.dataset, args.frames, args.activity)
    names = data.video_fnames[: args.videos]
    if len(names) < 2:
        raise SystemExit("need >= 2 videos")

    result = {"checkpoint": str(Path(args.ckpt).resolve()), "dataset": args.dataset,
              "activity": args.activity, "frames": args.frames,
              "n_videos": len(names), "device": device,
              "solver": {"eps": model.train_eps, "alpha": model.alpha_train,
                         "radius_gw": model.radius_gw, "rho_seg": getattr(model, "rho_seg", model.rho),
                         "rho_align": getattr(model, "rho_align", model.rho),
                         "n_ot_iters": list(model.n_ot_train),
                         "ub_frames": model.ub_frames, "ub_actions": model.ub_actions},
              "normalisation": {
                  "T_xa": "rows renormalised over K actions; entropy in nats; norm = entropy/log K",
                  "T_xy": "virtual row/col dropped by solver; rows renormalised over real Y frames; norm = entropy/log Ny"}}

    # ---- (b) prototype geometry -------------------------------------------
    geom = [prototype_geometry(model.clusters, "trained"),
            prototype_geometry(init_model.clusters, f"init_seed{args.init_seed}")]
    result["prototype_geometry"] = geom
    result["prototype_drift_trained_vs_init"] = hungarian_drift(init_model.clusters, model.clusters)
    if args.compare_ckpt:
        other = dc.load_model(args.compare_ckpt, device)
        result["prototype_drift_vs_compare_ckpt"] = hungarian_drift(model.clusters, other.clusters)
        result["compare_ckpt"] = str(Path(args.compare_ckpt).resolve())
        del other

    # ---- (a) + (c) + (d) per-video segmentation ----------------------------
    per_video, boundary_rows, jitter = [], [], []
    embeddings, gts, masks = [], [], []
    scatter_num, scatter_den = 0.0, 0
    confusion = {}
    for name in names:
        rec = dc.embed(model, data, name, device)
        embeddings.append(rec["z"])
        gts.append(rec["gt"])
        masks.append(rec["mask"])
        t_xa = dc.seg_plan(model, rec["z"], rec["mask"])
        rows = t_xa[0]
        row_mass = rows.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        post = rows / row_mass
        stats = dc.plan_stats(t_xa)
        stats.update({"video": name})
        per_video.append({"T_xa": stats})

        hard = post.argmax(dim=-1)
        gt = rec["gt"]
        valid = rec["mask"][0]
        # within-cluster scatter against the assigned prototype
        proto = model.clusters[hard]
        d = (rec["z"][0] - proto).norm(dim=-1)[valid]
        scatter_num += d.sum().item()
        scatter_den += int(valid.sum())
        # confusion: assigned cluster vs GT action
        for a, g in zip(hard[valid].tolist(), gt[valid].tolist()):
            confusion.setdefault(int(a), {}).setdefault(int(g), 0)
            confusion[int(a)][int(g)] += 1
        # posterior temporal jitter: argmax flips inside GT segments
        flips, inside = 0, 0
        for s, e, _ in dc.gt_segments(gt):
            if e - s < 2:
                continue
            seg = hard[s:e]
            flips += int((seg[1:] != seg[:-1]).sum())
            inside += int(e - s - 1)
        jitter.append({"video": name,
                       "within_segment_flip_rate": flips / inside if inside else float("nan"),
                       "n_gt_segments": len(dc.gt_segments(gt))})
        # boundary stratification
        dist = dc.signed_boundary_distance(gt)
        ent = -(post * post.clamp_min(1e-12).log()).sum(dim=-1)
        ent_norm = (ent / np.log(post.shape[-1])).detach().cpu().numpy()
        top1 = post.max(dim=-1).values.detach().cpu().numpy()
        segs = dc.gt_segments(gt)
        seg_of = np.zeros(len(gt), dtype=int)
        for i, (s, e, _) in enumerate(segs):
            seg_of[s:e] = i
        prev_next_mass = np.zeros(len(gt))
        # mass the posterior puts on the clusters that dominate the neighbouring GT segments
        dom = []
        for s, e, _ in segs:
            dom.append(int(torch.bincount(hard[s:e], minlength=post.shape[-1]).argmax()))
        pm = post.detach().cpu().numpy()
        for i in range(len(gt)):
            si = seg_of[i]
            nb = {dom[si - 1]} if si > 0 else set()
            if si + 1 < len(segs):
                nb.add(dom[si + 1])
            nb.discard(dom[si])  # neighbour mass only, never the own-segment cluster
            prev_next_mass[i] = float(pm[i, sorted(nb)].sum()) if nb else 0.0
        v = valid.detach().cpu().numpy()
        for i in np.nonzero(v)[0]:
            boundary_rows.append((float(dist[i]), float(ent_norm[i]), float(top1[i]),
                                  float(prev_next_mass[i])))

    result["within_cluster_scatter_mean_l2"] = scatter_num / max(scatter_den, 1)
    result["posterior_jitter"] = {
        "per_video": jitter,
        "within_segment_flip_rate_mean": float(np.nanmean([j["within_segment_flip_rate"] for j in jitter]))}

    # confusion summary: purity of hard assignment against GT actions
    total = sum(sum(v.values()) for v in confusion.values())
    purity = sum(max(v.values()) for v in confusion.values()) / max(total, 1)
    used = len(confusion)
    result["assignment"] = {"hard_assignment_purity": purity,
                            "clusters_used": used,
                            "n_clusters": int(model.n_clusters),
                            "confusion_counts": {str(k): {str(g): c for g, c in v.items()}
                                                 for k, v in confusion.items()}}

    # boundary table
    arr = np.asarray(boundary_rows)
    bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)]
    table = []
    absd = np.abs(arr[:, 0])
    # videos with no GT boundary at all produce inf distances; count them rather
    # than letting them vanish between the bins
    n_no_boundary = int(np.isinf(absd).sum())
    for lo, hi in bins:
        m = (absd >= lo) & (absd <= hi)
        if m.sum() == 0:
            continue
        table.append({"abs_dist_lo": lo, "abs_dist_hi": None if hi > 10 ** 8 else hi,
                      "n_frames": int(m.sum()),
                      "entropy_norm_mean": float(arr[m, 1].mean()),
                      "top1_mass_mean": float(arr[m, 2].mean()),
                      "neighbour_segment_mass_mean": float(arr[m, 3].mean())}
                     )
    result["boundary_stratification"] = table
    result["boundary_frames_without_any_gt_boundary"] = n_no_boundary

    # ---- aggregate T_xa plus T_xy on fixed adjacent pairs ------------------
    agg = {}
    for metric in per_video[0]["T_xa"]:
        vals = [v["T_xa"][metric] for v in per_video]
        if isinstance(vals[0], (int, float)):
            agg[metric] = float(np.nanmean(np.asarray(vals, dtype=float)))
    result["T_xa_summary"] = agg
    result["T_xa_per_video"] = per_video

    pair_stats = []
    npairs = min(args.pairs, len(names))
    for i in range(npairs):
        j = (i + 1) % len(names)
        t_xy, trace = dc.align_plan(model, embeddings[i], embeddings[j], masks[i], masks[j])
        s = dc.plan_stats(t_xy)
        s.update({"x": names[i], "y": names[j], "final_objective": float(trace[-1].item())})
        pair_stats.append(s)
    result["T_xy_summary"] = {m: float(np.nanmean([p[m] for p in pair_stats]))
                              for m in pair_stats[0] if isinstance(pair_stats[0][m], (int, float))}
    result["T_xy_per_pair"] = pair_stats

    # untrained control on the same videos/pairs
    init_xa, init_xy = [], []
    for name in names:
        rec = dc.embed(init_model, data, name, device)
        init_xa.append(dc.plan_stats(dc.seg_plan(init_model, rec["z"], rec["mask"])))
    init_emb = [dc.embed(init_model, data, n, device) for n in names]
    for i in range(npairs):
        j = (i + 1) % len(names)
        t, _ = dc.align_plan(init_model, init_emb[i]["z"], init_emb[j]["z"],
                             init_emb[i]["mask"], init_emb[j]["mask"])
        init_xy.append(dc.plan_stats(t))
    result["init_control"] = {
        "T_xa": {m: float(np.nanmean([s[m] for s in init_xa])) for m in init_xa[0]
                 if isinstance(init_xa[0][m], (int, float))},
        "T_xy": {m: float(np.nanmean([s[m] for s in init_xy])) for m in init_xy[0]
                 if isinstance(init_xy[0][m], (int, float))}}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dc.json_safe(result)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    flat = {f"T_xa/{k}": v for k, v in agg.items()}
    flat.update({f"T_xy/{k}": v for k, v in result["T_xy_summary"].items()})
    flat.update({f"init/T_xa/{k}": v for k, v in result["init_control"]["T_xa"].items()})
    flat.update({f"init/T_xy/{k}": v for k, v in result["init_control"]["T_xy"].items()})
    flat.update({"proto/offdiag_cosine_mean": geom[0]["offdiag_cosine_mean"],
                 "proto/nearest_margin_mean": geom[0]["nearest_margin_mean"],
                 "proto/init_offdiag_cosine_mean": geom[1]["offdiag_cosine_mean"],
                 "proto/drift_vs_init": result["prototype_drift_trained_vs_init"]["matched_displacement_mean"],
                 "assign/purity": purity, "assign/clusters_used": used,
                 "scatter/mean_l2": result["within_cluster_scatter_mean_l2"],
                 "jitter/within_segment_flip_rate": result["posterior_jitter"]["within_segment_flip_rate_mean"]})
    print(json.dumps(dc.json_safe(flat), indent=2))
    print(json.dumps(dc.json_safe(table), indent=2))
    if run is not None:
        import wandb
        run.log(dc.json_safe(flat))
        run.summary.update(dc.json_safe(flat))
        if table:
            tbl = wandb.Table(columns=list(table[0].keys()),
                              data=[[r[c] for c in table[0]] for r in table])
            run.log({"boundary_stratification": tbl})
        run.finish()


if __name__ == "__main__":
    main()
