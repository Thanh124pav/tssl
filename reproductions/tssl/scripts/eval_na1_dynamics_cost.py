"""N-A1 - is the frame-index structural cost an inadequate within-video structure?

Part 1 (diagnosis, frozen checkpoint): within-action vs inter-action roughness,
temporal-neighbour rank under the feature metric, and transition similarity,
stratified by GT action duration and distance to a GT boundary.

Part 2 (controlled swap): the SAME mirror-descent FGW solver (verified against
the public conv implementation in fgw_dense.check_equivalence) is run with the
cross-video cost, priors, marginals, budget and seeds unchanged, replacing only
C^X / C^Y by

  index      the structure the public filters implement (baseline)
  feature    within-video feature similarity / dissimilarity
  velocity   local-window transition (velocity) similarity
  shuffled   velocity structure with its frames permuted   (control)
  random     a random symmetric structure                  (control)
  none       alpha = 0, structural term removed            (control)

Correspondence error is measured against known monotone warps; real video pairs
are additionally reported with an action-agreement proxy, labelled as a proxy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import diag_common as dc
import alignment_ot
import fgw_dense


def velocity(z, window):
    """Local transition vectors v_i = z_i - z_{i-w}, edge-padded."""
    n = z.shape[0]
    idx = torch.clamp(torch.arange(n, device=z.device) - window, min=0)
    return z - z[idx]


def build_structure(kind, z, radius, scale_to=None, seed=0, window=4):
    """Returns (Cx, Cy) of shape (n, n), following the repo's convention:
    Cx is a within-X similarity, Cy a within-Y dissimilarity in [0, 1]."""
    n = z.shape[0]
    dev = z.device
    if kind == "index":
        return fgw_dense.index_structure(n, n, radius, dev)
    if kind == "feature":
        s = (F.normalize(z, dim=-1) @ F.normalize(z, dim=-1).T).clamp(-1, 1)
        sim = (s + 1) / 2
    elif kind == "velocity":
        v = F.normalize(velocity(z, window), dim=-1)
        sim = ((v @ v.T).clamp(-1, 1) + 1) / 2
    elif kind == "shuffled":
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(n, generator=g).to(dev)
        v = F.normalize(velocity(z, window), dim=-1)[perm]
        sim = ((v @ v.T).clamp(-1, 1) + 1) / 2
    elif kind == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.rand(n, n, generator=g).to(dev)
        sim = (a + a.T) / 2
    elif kind in ("band_feature", "band_velocity", "band_shuffled", "band_feature_shuffled"):
        # keep the index band (so sparsity matches the baseline exactly) and
        # reweight inside it by feature / transition similarity
        band, _ = fgw_dense.index_structure(n, n, radius, dev)
        band = (band > 0).float()
        if kind in ("band_feature", "band_feature_shuffled"):
            w = F.normalize(z, dim=-1)
            if kind == "band_feature_shuffled":
                g = torch.Generator(device="cpu").manual_seed(seed)
                w = w[torch.randperm(n, generator=g).to(dev)]
        else:
            w = F.normalize(velocity(z, window), dim=-1)
            if kind == "band_shuffled":
                g = torch.Generator(device="cpu").manual_seed(seed)
                w = w[torch.randperm(n, generator=g).to(dev)]
        sim = band * (((w @ w.T).clamp(-1, 1) + 1) / 2)
    else:
        raise ValueError(kind)
    sim = sim - torch.diag(torch.diag(sim))
    Cx = sim.clone()
    if scale_to is not None:  # match the index structure's mean entry
        Cx = Cx * (scale_to / Cx.mean().clamp_min(1e-12))
    Cy = 1.0 - sim
    Cy.fill_diagonal_(1.0)
    return Cx, Cy


def diagnose_video(z, gt, window=4):
    """Roughness, neighbour rank and transition similarity, stratified by GT."""
    zc = z
    step = (zc[1:] - zc[:-1]).norm(dim=-1)
    segs = dc.gt_segments(gt)
    bset = {s[0] for s in segs[1:]}
    inside = torch.tensor([i + 1 not in bset for i in range(len(step))], device=z.device)
    v = F.normalize(velocity(zc, window), dim=-1)
    tsim = (v[1:] * v[:-1]).sum(dim=-1)
    d = torch.cdist(zc, zc)
    d.fill_diagonal_(float("inf"))
    order = d.argsort(dim=-1)
    nxt = torch.arange(1, len(zc), device=z.device)
    rank_next = (order[:-1] == nxt.unsqueeze(1)).float().argmax(dim=-1) + 1
    durations = {}
    for s, e, _ in segs:
        durations.setdefault("short" if e - s < 16 else ("mid" if e - s < 48 else "long"), []).append((s, e))
    out = {
        "roughness_within_action": float(step[inside].mean()),
        "roughness_at_boundary": float(step[~inside].mean()) if (~inside).any() else None,
        "roughness_ratio_boundary_over_within": (float(step[~inside].mean() / step[inside].mean())
                                                 if (~inside).any() else None),
        "transition_similarity_within_action": float(tsim[inside[:len(tsim)]].mean()),
        "transition_similarity_at_boundary": (float(tsim[~inside[:len(tsim)]].mean())
                                              if (~inside[:len(tsim)]).any() else None),
        "temporal_neighbour_rank_mean": float(rank_next.float().mean()),
        "temporal_neighbour_rank_median": float(rank_next.float().median()),
        "temporal_neighbour_is_nn_fraction": float((rank_next == 1).float().mean()),
        "n_frames": int(len(zc)),
    }
    for bucket, spans in durations.items():
        vals = [float(step[s:max(e - 1, s + 1)].mean()) for s, e in spans if e - s > 1]
        if vals:
            out[f"roughness_duration_{bucket}"] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--warp-severity", type=float, default=0.5)
    ap.add_argument("--jitter-sigmas", type=float, nargs="*", default=[0.0, 0.02, 0.05])
    ap.add_argument("--structures", nargs="*",
                    default=["index", "feature", "velocity", "shuffled", "random", "none"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = dc.pick_device("cuda" if args.device in ("auto", "cuda") else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    run = None if args.no_wandb else dc.init_wandb(
        f"NA1_{args.tag}", ["N-A1", "diagnostic", args.dataset, "phaseN"], vars(args))

    model = dc.load_model(args.ckpt, device)
    data = dc.load_dataset(args.dataset, args.frames, args.activity)
    names = data.video_fnames[: args.videos]
    n = args.frames
    rho = getattr(model, "rho_align", model.rho)

    result = {"checkpoint": str(Path(args.ckpt).resolve()), "dataset": args.dataset,
              "frames": n, "videos": names, "device": device,
              "solver_equivalence": fgw_dense.check_equivalence(
                  n=64, m=64, radius=model.radius_gw, device=device,
                  eps=model.train_eps, alpha=model.alpha_train,
                  ub_frames=model.ub_frames, ub_actions=model.ub_actions,
                  lambda_frames=model.lambda_frames_train,
                  lambda_actions=model.lambda_actions_train,
                  n_iters=tuple(model.n_ot_train)),
              "structure_convention": "Cx = within-X similarity, Cy = within-Y dissimilarity; "
                                      "index variant reproduces the public conv filters"}

    recs = [dc.embed(model, data, name, device) for name in names]
    result["diagnosis"] = [dict(video=r["name"], **diagnose_video(r["z"][0], r["gt"], args.window))
                           for r in recs]
    result["diagnosis_summary"] = {
        k: float(np.mean([d[k] for d in result["diagnosis"] if d.get(k) is not None]))
        for k in result["diagnosis"][0] if k != "video"}

    # index-structure scale, so alternative structures can be scale-matched
    Cx_idx, _ = fgw_dense.index_structure(n + 1, n + 1, model.radius_gw, device)
    idx_scale = float(Cx_idx.mean())
    result["index_structure_mean_entry"] = idx_scale

    # ---- controlled structure swap on known-warp pairs ---------------------
    swap = {s: [] for s in args.structures}
    for vi, rec in enumerate(recs):
        src, meta = dc.monotone_warp(n, 5, args.warp_severity, seed=vi)
        raw_w = dc.resample_features(rec["raw"], src)
        zy, mask_y = dc.embed_features(model, raw_w, device)
        zx, mask_x = rec["z"], rec["mask"]
        base_cost = 1.0 - zx @ zy.transpose(1, 2)
        base_cost = base_cost + alignment_ot.temporal_prior(n, n, rho, device)
        cost = torch.cat((torch.ones(1, 1, n, device=device) * 0.5, base_cost), dim=1)
        cost = torch.cat((torch.ones(1, n + 1, 1, device=device) * 0.5, cost), dim=2)
        # NOTE: the jitter below perturbs the embeddings that the STRUCTURE is
        # built from, while the cross-video ground cost stays fixed. That is
        # deliberate - the question is how sensitive each structure is to
        # feature noise - and it is why the index structure, which does not
        # read features at all, is exactly invariant to it.
        for sigma in args.jitter_sigmas:
            if sigma > 0:
                g = torch.Generator(device="cpu").manual_seed(100 + vi)
                zx_j = F.normalize(zx + torch.randn(zx.shape, generator=g).to(device) * sigma, dim=-1)
                zy_j = F.normalize(zy + torch.randn(zy.shape, generator=g).to(device) * sigma, dim=-1)
            else:
                zx_j, zy_j = zx, zy
            for kind in args.structures:
                t0 = time.time()
                if kind == "none":
                    Cx = torch.zeros(n + 1, n + 1, device=device)
                    Cy = torch.zeros(n + 1, n + 1, device=device)
                    alpha = 0.0
                else:
                    alpha = model.alpha_train
                    # structures are built on the padded (virtual-row) index space
                    zx_pad = torch.cat((zx_j[0, :1], zx_j[0]), dim=0)
                    zy_pad = torch.cat((zy_j[0, :1], zy_j[0]), dim=0)
                    Cx, _ = build_structure(kind, zx_pad, model.radius_gw,
                                            scale_to=None if kind == "index" else idx_scale,
                                            seed=vi, window=args.window)
                    _, Cy = build_structure(kind, zy_pad, model.radius_gw, seed=vi + 1,
                                            window=args.window)
                T, trace = fgw_dense.align_ot_dense(
                    cost, Cx, Cy, mask_x, mask_y, eps=model.train_eps, alpha=alpha,
                    ub_frames=model.ub_frames, ub_actions=model.ub_actions,
                    lambda_frames=model.lambda_frames_train,
                    lambda_actions=model.lambda_actions_train,
                    n_iters=tuple(model.n_ot_train), step_size=model.step_size)
                runtime = time.time() - t0
                stats = dc.plan_stats(T)
                pred = T[0].argmax(dim=-1).detach().cpu().numpy()
                err = np.abs(src[pred] - np.arange(n))
                stats.update({"video": rec["name"], "structure": kind,
                              "structure_feature_jitter_sigma": sigma,
                              "alignment_error_mean": float(err.mean()),
                              "alignment_error_median": float(np.median(err)),
                              "final_objective": float(trace[-1].item()),
                              "runtime_s": runtime,
                              "structure_mean_entry": float(Cx.mean())})
                swap[kind].append(stats)

    result["structure_swap"] = {}
    for kind in args.structures:
        rows = swap[kind]
        result["structure_swap"][kind] = {
            "per_run": rows,
            "summary_by_jitter": {
                str(sig): {m: float(np.mean([r[m] for r in rows if r["structure_feature_jitter_sigma"] == sig]))
                           for m in ("alignment_error_mean", "alignment_error_median",
                                     "effective_support", "top1_mass", "runtime_s",
                                     "final_objective")}
                for sig in args.jitter_sigmas}}

    # ---- real pairs, action-agreement PROXY only --------------------------
    proxy = {s: [] for s in args.structures}
    for i in range(len(recs)):
        j = (i + 1) % len(recs)
        rx, ry = recs[i], recs[j]
        base = 1.0 - rx["z"] @ ry["z"].transpose(1, 2)
        base = base + alignment_ot.temporal_prior(n, n, rho, device)
        cost = torch.cat((torch.ones(1, 1, n, device=device) * 0.5, base), dim=1)
        cost = torch.cat((torch.ones(1, n + 1, 1, device=device) * 0.5, cost), dim=2)
        gx = rx["gt"].detach().cpu().numpy()
        gy = ry["gt"].detach().cpu().numpy()
        for kind in args.structures:
            if kind == "none":
                Cx = Cy = torch.zeros(n + 1, n + 1, device=device)
                alpha = 0.0
            else:
                alpha = model.alpha_train
                zx_pad = torch.cat((rx["z"][0, :1], rx["z"][0]), dim=0)
                zy_pad = torch.cat((ry["z"][0, :1], ry["z"][0]), dim=0)
                Cx, _ = build_structure(kind, zx_pad, model.radius_gw,
                                        scale_to=None if kind == "index" else idx_scale,
                                        seed=i, window=args.window)
                _, Cy = build_structure(kind, zy_pad, model.radius_gw, seed=i + 1,
                                        window=args.window)
            T, _ = fgw_dense.align_ot_dense(
                cost, Cx, Cy, rx["mask"], ry["mask"], eps=model.train_eps, alpha=alpha,
                ub_frames=model.ub_frames, ub_actions=model.ub_actions,
                lambda_frames=model.lambda_frames_train,
                lambda_actions=model.lambda_actions_train,
                n_iters=tuple(model.n_ot_train), step_size=model.step_size)
            pred = T[0].argmax(dim=-1).detach().cpu().numpy()
            proxy[kind].append({"x": rx["name"], "y": ry["name"],
                                "action_agreement_proxy": float((gx == gy[pred]).mean()),
                                **{k: v for k, v in dc.plan_stats(T).items()
                                   if k in ("effective_support", "top1_mass")}})
    result["real_pairs_action_agreement_proxy"] = {
        kind: {"per_pair": proxy[kind],
               "mean_action_agreement_proxy": float(np.mean([p["action_agreement_proxy"]
                                                             for p in proxy[kind]]))}
        for kind in args.structures}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dc.json_safe(result), indent=2, allow_nan=False) + "\n")

    flat = {f"diag/{k}": v for k, v in result["diagnosis_summary"].items()}
    for kind in args.structures:
        s = result["structure_swap"][kind]["summary_by_jitter"]["0.0"]
        flat[f"swap/{kind}/align_err"] = s["alignment_error_mean"]
        flat[f"swap/{kind}/support"] = s["effective_support"]
        flat[f"swap/{kind}/runtime_s"] = s["runtime_s"]
        flat[f"proxy/{kind}/action_agreement"] = \
            result["real_pairs_action_agreement_proxy"][kind]["mean_action_agreement_proxy"]
    print(json.dumps(dc.json_safe(flat), indent=2))
    if run is not None:
        run.log(dc.json_safe(flat))
        run.summary.update(dc.json_safe(flat))
        run.finish()


if __name__ == "__main__":
    main()
