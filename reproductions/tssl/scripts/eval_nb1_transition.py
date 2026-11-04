"""N-B1 - does alignment preserve action identity but not within-action phase?

Five measurements on a frozen checkpoint:

A  transport mass on the correct GT action vs on the correct phase, each
   against the "uniform inside the correct action" expectation, so phase
   correspondence is separated from mere same-action mass.
B  within-action phase separability of transition (velocity) features, for
   several windows and for directed vs symmetric variants, against the
   state-cosine baseline.
C  transport stability under small embedding perturbations.
D  gradient cosine between the segmentation CE and the alignment CE (and the
   E13 action-bridge CE) with respect to the shared encoder parameters.
E  one-factor cost comparison at equal solver budget: state-only (baseline),
   transition-only, state+transition, and the E13 action-posterior bridge.

GT action labels and GT relative phase are used for evaluation only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import diag_common as dc
import alignment_ot
import segmentation_ot
from action_bridge import action_similarity


def velocity(z, window, directed=True):
    n = z.shape[0]
    back = torch.clamp(torch.arange(n, device=z.device) - window, min=0)
    v = z - z[back]
    if directed:
        return F.normalize(v, dim=-1)
    fwd = torch.clamp(torch.arange(n, device=z.device) + window, max=n - 1)
    return F.normalize(torch.cat([v, z[fwd] - z], dim=-1), dim=-1)


def auroc(scores, pos, cand):
    from sklearn.metrics import roc_auc_score
    s, y = scores[cand], pos[cand]
    if y.sum() in (0, y.size):
        return None
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--windows", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--phase-pos", type=float, default=0.10)
    ap.add_argument("--phase-neg", type=float, default=0.50)
    ap.add_argument("--warp-severity", type=float, default=0.5)
    ap.add_argument("--bridge-weight", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = dc.pick_device("cuda" if args.device in ("auto", "cuda") else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    run = None if args.no_wandb else dc.init_wandb(
        f"NB1_{args.tag}", ["N-B1", "diagnostic", args.dataset, "phaseN"], vars(args))

    model = dc.load_model(args.ckpt, device)
    data = dc.load_dataset(args.dataset, args.frames, args.activity)
    names = data.video_fnames[: args.videos]
    n = args.frames
    rho = getattr(model, "rho_align", model.rho)

    recs = []
    for name in names:
        rec = dc.embed(model, data, name, device)
        t_xa = dc.seg_plan(model, rec["z"], rec["mask"])
        rec["post"] = t_xa[0] / t_xa[0].sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rec["phase"] = dc.relative_phase(rec["gt"])
        recs.append(rec)

    result = {"checkpoint": str(Path(args.ckpt).resolve()), "dataset": args.dataset,
              "frames": n, "videos": names, "device": device,
              "phase_pos": args.phase_pos, "phase_neg": args.phase_neg}

    # ---- A: action mass vs phase mass -------------------------------------
    a_rows = []
    for i in range(len(recs)):
        j = (i + 1) % len(recs)
        rx, ry = recs[i], recs[j]
        T, _ = dc.align_plan(model, rx["z"], ry["z"], rx["mask"], ry["mask"])
        P = (T[0] / T[0].sum(dim=-1, keepdim=True).clamp_min(1e-12)).detach().cpu().numpy()
        gx, gy = rx["gt"].cpu().numpy(), ry["gt"].cpu().numpy()
        same_action = (gx[:, None] == gy[None, :])
        dphase = np.abs(rx["phase"][:, None] - ry["phase"][None, :])
        same_phase = same_action & (dphase <= args.phase_pos)
        mass_action = float((P * same_action).sum(axis=1).mean())
        mass_phase = float((P * same_phase).sum(axis=1).mean())
        # expectation if mass were uniform inside the correct action
        # Per row: (mass this row puts on the correct action) x (fraction of that
        # action's frames that are also at the correct phase). Averaging the two
        # factors separately and multiplying would not be the same quantity.
        with np.errstate(invalid="ignore", divide="ignore"):
            frac_phase_in_action = np.where(same_action.sum(1) > 0,
                                            same_phase.sum(1) / np.maximum(same_action.sum(1), 1), np.nan)
        row_action_mass = (P * same_action).sum(axis=1)
        exp_phase_given_action = float(np.nanmean(row_action_mass * frac_phase_in_action))
        # mass-weighted mean |dphase| inside the correct action
        w = P * same_action
        wsum = w.sum(axis=1, keepdims=True)
        md = float(np.nanmean(np.where(wsum[:, 0] > 1e-9,
                                       (w * dphase).sum(axis=1) / np.maximum(wsum[:, 0], 1e-12), np.nan)))
        unif = float(np.nanmean(np.where(same_action.sum(1) > 0,
                                         (same_action * dphase).sum(1) / np.maximum(same_action.sum(1), 1), np.nan)))
        a_rows.append({"x": rx["name"], "y": ry["name"],
                       "mass_on_correct_action": mass_action,
                       "mass_on_correct_phase": mass_phase,
                       "expected_phase_mass_if_uniform_in_action": exp_phase_given_action,
                       "phase_mass_lift_over_uniform": mass_phase / max(exp_phase_given_action, 1e-9),
                       "mean_abs_dphase_under_transport": md,
                       "mean_abs_dphase_uniform_in_action": unif})
    result["A_action_vs_phase"] = {
        "per_pair": a_rows,
        "summary": {k: float(np.nanmean([r[k] for r in a_rows])) for k in a_rows[0]
                    if k not in ("x", "y")}}

    # ---- B: within-action transition separability -------------------------
    b_rows = []
    for i in range(len(recs)):
        j = (i + 1) % len(recs)
        rx, ry = recs[i], recs[j]
        gx, gy = rx["gt"].cpu().numpy(), ry["gt"].cpu().numpy()
        same_action = gx[:, None] == gy[None, :]
        dphase = np.abs(rx["phase"][:, None] - ry["phase"][None, :])
        pos = same_action & (dphase <= args.phase_pos)
        cand = pos | (same_action & (dphase >= args.phase_neg))
        row = {"x": rx["name"], "y": ry["name"],
               "state_cosine": auroc((rx["z"][0] @ ry["z"][0].T).cpu().numpy(), pos, cand)}
        for w in args.windows:
            for directed in (True, False):
                vx = velocity(rx["z"][0], w, directed)
                vy = velocity(ry["z"][0], w, directed)
                key = f"velocity_w{w}_{'directed' if directed else 'symmetric'}"
                row[key] = auroc((vx @ vy.T).cpu().numpy(), pos, cand)
        b_rows.append(row)
    result["B_transition_separability"] = {
        "per_pair": b_rows,
        "summary": {k: float(np.nanmean([r[k] for r in b_rows if r[k] is not None]))
                    for k in b_rows[0] if k not in ("x", "y")}}

    # ---- C: transport stability under embedding perturbation --------------
    c_rows = []
    for sigma in (0.01, 0.02, 0.05):
        for i in range(min(4, len(recs))):
            j = (i + 1) % len(recs)
            rx, ry = recs[i], recs[j]
            T0, _ = dc.align_plan(model, rx["z"], ry["z"], rx["mask"], ry["mask"])
            g = torch.Generator(device="cpu").manual_seed(13 + i)
            zx = F.normalize(rx["z"] + torch.randn(rx["z"].shape, generator=g).to(device) * sigma, dim=-1)
            zy = F.normalize(ry["z"] + torch.randn(ry["z"].shape, generator=g).to(device) * sigma, dim=-1)
            T1, _ = dc.align_plan(model, zx, zy, rx["mask"], ry["mask"])
            p0 = T0[0] / T0[0].sum(-1, keepdim=True).clamp_min(1e-12)
            p1 = T1[0] / T1[0].sum(-1, keepdim=True).clamp_min(1e-12)
            c_rows.append({"sigma": sigma,
                           "top1_agreement": float((p0.argmax(-1) == p1.argmax(-1)).float().mean()),
                           "row_l1_drift": float((p0 - p1).abs().sum(-1).mean()),
                           "argmax_shift_frames": float((p0.argmax(-1) - p1.argmax(-1)).abs().float().mean())})
    result["C_transport_stability"] = {
        "per_run": c_rows,
        "summary": {str(s): {k: float(np.mean([r[k] for r in c_rows if r["sigma"] == s]))
                             for k in ("top1_agreement", "row_l1_drift", "argmax_shift_frames")}
                    for s in (0.01, 0.02, 0.05)}}

    # ---- D: gradient cosine, segmentation CE vs alignment CE --------------
    params = [p for p in model.mlp.parameters() if p.requires_grad]

    def grads(loss):
        gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return torch.cat([g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=device)
                          for g, p in zip(gs, params)])

    d_rows = []
    num_eps = 1e-8
    for i in range(min(4, len(recs))):
        j = (i + 1) % len(recs)
        rawx, rawy = recs[i]["raw"], recs[j]["raw"]
        mask_x, mask_y = recs[i]["mask"], recs[j]["mask"]
        fx = F.normalize(model.mlp(rawx.unsqueeze(0)), dim=-1)
        fy = F.normalize(model.mlp(rawy.unsqueeze(0)), dim=-1)
        qx = torch.softmax(fx @ model.clusters.T.unsqueeze(0) / model.temp, dim=-1)
        qy = torch.softmax(fy @ model.clusters.T.unsqueeze(0) / model.temp, dim=-1)
        with torch.no_grad():
            opt_x = dc.seg_plan(model, fx, mask_x)
            opt_y = dc.seg_plan(model, fy, mask_y)
            opt_xy, _ = dc.align_plan(model, fx, fy, mask_x, mask_y)
        seg_loss = -(((opt_x * (qx + num_eps).log()) * mask_x[..., None]).sum(-1).mean()
                     + ((opt_y * (qy + num_eps).log()) * mask_y[..., None]).sum(-1).mean())
        codes = torch.softmax(fx @ fy.transpose(1, 2) / model.temp, dim=-1)
        align_loss = -((opt_xy * (codes + num_eps).log())
                       * mask_x.unsqueeze(2) * mask_y.unsqueeze(1)).sum(-1).mean()
        pred_bridge = torch.softmax(action_similarity(qx, qy) / model.action_bridge_temp, dim=-1)
        bridge_loss = -((opt_xy.detach() * (pred_bridge + num_eps).log())
                        * mask_x.unsqueeze(2) * mask_y.unsqueeze(1)).sum(-1).mean()
        gs, ga, gb = grads(seg_loss), grads(align_loss), grads(bridge_loss)
        cos = lambda a, b: float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())  # noqa: E731
        d_rows.append({"x": recs[i]["name"], "y": recs[j]["name"],
                       "seg_loss": float(seg_loss), "align_loss": float(align_loss),
                       "bridge_loss": float(bridge_loss),
                       "grad_norm_seg": float(gs.norm()), "grad_norm_align": float(ga.norm()),
                       "grad_norm_bridge": float(gb.norm()),
                       "cos_seg_align": cos(gs, ga), "cos_seg_bridge": cos(gs, gb),
                       "cos_align_bridge": cos(ga, gb)})
    result["D_gradient_conflict"] = {
        "per_pair": d_rows,
        "summary": {k: float(np.mean([r[k] for r in d_rows])) for k in d_rows[0]
                    if k not in ("x", "y")}}

    # ---- E: cost variants at equal solver budget --------------------------
    def run_cost(zx, zy, mask_x, mask_y, cost):
        cost = cost + alignment_ot.temporal_prior(cost.shape[1], cost.shape[2], rho, device)
        c = torch.cat((torch.ones(1, 1, cost.shape[2], device=device) * 0.5, cost), dim=1)
        c = torch.cat((torch.ones(1, c.shape[1], 1, device=device) * 0.5, c), dim=2)
        return alignment_ot.align_ot(
            c, mask_x, mask_y, eps=model.train_eps, alpha=model.alpha_train,
            radius=model.radius_gw, ub_frames=model.ub_frames, ub_actions=model.ub_actions,
            lambda_frames=model.lambda_frames_train, lambda_actions=model.lambda_actions_train,
            n_iters=tuple(model.n_ot_train), step_size=model.step_size)

    e_rows = []
    for i, rec in enumerate(recs):
        src, _ = dc.monotone_warp(n, 5, args.warp_severity, seed=i)
        raw_w = dc.resample_features(rec["raw"], src)
        zy, mask_y = dc.embed_features(model, raw_w, device)
        zx, mask_x = rec["z"], rec["mask"]
        t_w = dc.seg_plan(model, zy, mask_y)
        post_y = t_w[0] / t_w[0].sum(-1, keepdim=True).clamp_min(1e-12)
        vx = velocity(zx[0], 4).unsqueeze(0)
        vy = velocity(zy[0], 4).unsqueeze(0)
        state = 1.0 - zx @ zy.transpose(1, 2)
        trans = 1.0 - vx @ vy.transpose(1, 2)
        bridge = 1.0 - action_similarity(rec["post"].unsqueeze(0), post_y.unsqueeze(0))
        variants = {"state_only": state,
                    "transition_only": trans,
                    "state_plus_transition": 0.5 * state + 0.5 * trans,
                    "state_plus_action_bridge": state + args.bridge_weight * bridge}
        for vname, cost in variants.items():
            T, _ = run_cost(zx, zy, mask_x, mask_y, cost)
            pred = T[0].argmax(-1).cpu().numpy()
            err = np.abs(src[pred] - np.arange(n))
            s = dc.plan_stats(T)
            e_rows.append({"video": rec["name"], "variant": vname,
                           "alignment_error_mean": float(err.mean()),
                           "alignment_error_median": float(np.median(err)),
                           "effective_support": s.get("effective_support"),
                           "top1_mass": s.get("top1_mass")})
    result["E_cost_variants_on_known_warps"] = {
        "per_run": e_rows,
        "summary": {v: {k: float(np.mean([r[k] for r in e_rows if r["variant"] == v]))
                        for k in ("alignment_error_mean", "alignment_error_median",
                                  "effective_support", "top1_mass")}
                    for v in sorted({r["variant"] for r in e_rows})}}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dc.json_safe(result), indent=2, allow_nan=False) + "\n")

    flat = {f"A/{k}": v for k, v in result["A_action_vs_phase"]["summary"].items()}
    flat.update({f"B/{k}": v for k, v in result["B_transition_separability"]["summary"].items()})
    flat.update({f"C/sigma{s}/{k}": v for s, d in result["C_transport_stability"]["summary"].items()
                 for k, v in d.items()})
    flat.update({f"D/{k}": v for k, v in result["D_gradient_conflict"]["summary"].items()})
    flat.update({f"E/{v}/{k}": x for v, d in result["E_cost_variants_on_known_warps"]["summary"].items()
                 for k, x in d.items()})
    print(json.dumps(dc.json_safe(flat), indent=2))
    if run is not None:
        run.log(dc.json_safe(flat))
        run.summary.update(dc.json_safe(flat))
        run.finish()


if __name__ == "__main__":
    main()
