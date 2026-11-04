"""N-C1 - nonlinear temporal-warp stress test of VASOT's alignment cost.

For a frozen checkpoint, a video X is warped in time by a monotone map whose
index correspondence is known exactly. The oracle correspondence is then
scored under the EXACT terms the repo uses, decomposed into

    0.5 * alpha * <C^X T C^Y, T>      structural (GW) term
    (1 - alpha) * <1 - z_x z_y^T, T>  feature term
    (1 - alpha) * <rho * |i/N - j/M|, T>  diagonal index prior

so that a structural penalty on the correct correspondence can be separated
from a feature-cost or prior penalty. The solver is then run for real and its
output compared with the known map, with rho_align=0 and DTW controls and an
interpolation-artifact control that resamples without warping.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diag_common as dc
import alignment_ot


def oracle_plan(src, n, device):
    """One-hot plan X -> X' : row i picks the warped frame whose source
    position is closest to i. Rows sum to 1/N like the solver's output scale."""
    j = np.argmin(np.abs(src[None, :] - np.arange(n)[:, None]), axis=1)
    T = torch.zeros(1, n, len(src), device=device)
    T[0, torch.arange(n), torch.as_tensor(j, device=device)] = 1.0 / n
    return T, j


def decompose(model, zx, zy, T, rho):
    """Objective terms for a given plan, using the repo's own filters."""
    nx, ny = zx.shape[1], zy.shape[1]
    dev = zx.device
    Cv = alignment_ot.construct_Cv_filter(nx, model.radius_gw, dev)
    Ca = alignment_ot.construct_Ca_filter(ny, model.radius_gw, dev)
    struct = (alignment_ot.mult_Cv(Cv, alignment_ot.mult_Ca(Ca, T)) * T).sum().item()
    feat_cost = 1.0 - zx @ zy.transpose(1, 2)
    feat = (feat_cost * T).sum().item()
    prior = alignment_ot.temporal_prior(nx, ny, rho, dev)
    prior_term = (prior * T).sum().item()
    a = model.alpha_train
    return {"gw_structural_raw": struct,
            "gw_structural_weighted": 0.5 * a * struct,
            "feature_term_raw": feat,
            "feature_term_weighted": (1 - a) * feat,
            "index_prior_term_raw": prior_term,
            "index_prior_term_weighted": (1 - a) * prior_term,
            "total_weighted": 0.5 * a * struct + (1 - a) * (feat + prior_term)}


def dtw_path(cost):
    """Classic DTW over a cost matrix; returns, for each row, the mean matched column."""
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        ci = cost[i - 1]
        for j in range(1, m + 1):
            acc[i, j] = ci[j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    matched = {}
    for a, b in pairs:
        matched.setdefault(a, []).append(b)
    return np.array([np.mean(matched.get(a, [np.nan])) for a in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--knots", type=int, default=5)
    ap.add_argument("--severities", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-dtw", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = dc.pick_device("cuda" if args.device in ("auto", "cuda") else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    run = None if args.no_wandb else dc.init_wandb(
        f"NC1_{args.tag}", ["N-C1", "diagnostic", args.dataset, "phaseN"], vars(args))

    model = dc.load_model(args.ckpt, device)
    data = dc.load_dataset(args.dataset, args.frames, args.activity)
    names = data.video_fnames[: args.videos]
    rho_default = getattr(model, "rho_align", model.rho)
    n = args.frames

    rows = []
    for vi, name in enumerate(names):
        rec = dc.embed(model, data, name, device)
        zx, mask_x = rec["z"], rec["mask"]
        variants = [("identity", np.arange(n, dtype=float), {"type": "identity"}),
                    ("resample_only", np.arange(n, dtype=float) + 0.5, {"type": "half_frame_shift"})]
        for sev in args.severities:
            if sev == 0.0:
                continue
            src, meta = dc.monotone_warp(n, args.knots, sev, seed=vi)
            meta["type"] = "piecewise"
            variants.append((f"piecewise_s{sev}", src, meta))
            # global (affine) speed control at a comparable mean speed ratio
            g = np.linspace(0, n - 1, n) ** (1.0 + sev * 0.5)
            g = g / g.max() * (n - 1)
            variants.append((f"global_s{sev}", g, {"type": "global_power", "severity": sev}))

        for vname, src, meta in variants:
            raw_w = dc.resample_features(rec["raw"], src)
            zy, mask_y = dc.embed_features(model, raw_w, device)
            T_or, j_or = oracle_plan(src, n, device)
            entry = {"video": name, "variant": vname, "warp": meta,
                     "oracle_rho_default": decompose(model, zx, zy, T_or, rho_default),
                     "oracle_rho0": decompose(model, zx, zy, T_or, 0.0),
                     "oracle_index_error_frames": float(np.mean(np.abs(src[j_or] - np.arange(n))))}
            # the solver, for real
            for rho_name, rho in (("rho_default", rho_default), ("rho0", 0.0)):
                T, trace = dc.align_plan(model, zx, zy, mask_x, mask_y, rho_align=rho)
                stats = dc.plan_stats(T)
                # align_ot rescales rows to sum to 1; the oracle plan carries total
                # mass 1, so put the solver plan on the same scale before costing it.
                T_unit = T / max(float(T.sum()), 1e-12)
                pred = T[0].argmax(dim=-1).detach().cpu().numpy()
                err = np.abs(src[pred] - np.arange(n))
                stats.update({"alignment_error_mean": float(err.mean()),
                              "alignment_error_median": float(np.median(err)),
                              "final_objective": float(trace[-1].item()),
                              "objective_decrease": float((trace[0] - trace[-1]).item()),
                              "oracle_vs_solver_total_weighted":
                                  entry[f"oracle_{rho_name}"]["total_weighted"],
                              })
                # is the oracle cheaper than what the solver found?
                solver_terms = decompose(model, zx, zy, T_unit, rho)
                stats["solver_total_weighted"] = solver_terms["total_weighted"]
                stats["solver_gw_structural_weighted"] = solver_terms["gw_structural_weighted"]
                stats["oracle_minus_solver_objective"] = (
                    stats["oracle_vs_solver_total_weighted"] - stats["solver_total_weighted"])
                entry[f"solver_{rho_name}"] = stats
            if not args.skip_dtw:
                cost = (1.0 - zx @ zy.transpose(1, 2))[0].detach().cpu().numpy().astype(np.float64)
                m = dtw_path(cost)
                ok = np.isfinite(m)
                cols = np.clip(np.round(np.where(ok, m, 0)).astype(int), 0, len(src) - 1)
                err = np.abs(src[cols] - np.arange(n))
                entry["dtw"] = {"alignment_error_mean": float(err[ok].mean()),
                                "rows_covered_fraction": float(ok.mean())}
            rows.append(entry)

    # aggregate per variant
    variants = sorted({r["variant"] for r in rows})
    summary = {}
    for v in variants:
        sel = [r for r in rows if r["variant"] == v]
        summary[v] = {
            "n_videos": len(sel),
            "oracle_rho_default": {k: float(np.mean([s["oracle_rho_default"][k] for s in sel]))
                                   for k in sel[0]["oracle_rho_default"]},
            "oracle_rho0": {k: float(np.mean([s["oracle_rho0"][k] for s in sel]))
                            for k in sel[0]["oracle_rho0"]},
            "solver_rho_default": {k: float(np.mean([s["solver_rho_default"][k] for s in sel]))
                                   for k in sel[0]["solver_rho_default"]
                                   if isinstance(sel[0]["solver_rho_default"][k], (int, float))},
            "solver_rho0": {k: float(np.mean([s["solver_rho0"][k] for s in sel]))
                            for k in sel[0]["solver_rho0"]
                            if isinstance(sel[0]["solver_rho0"][k], (int, float))},
        }
        if "dtw" in sel[0]:
            summary[v]["dtw_alignment_error_mean"] = float(np.mean([s["dtw"]["alignment_error_mean"] for s in sel]))

    result = {"checkpoint": str(Path(args.ckpt).resolve()), "dataset": args.dataset,
              "frames": n, "videos": names, "device": device,
              "interpolation": "linear in raw feature space, monotone source positions",
              "alpha_train": model.alpha_train, "radius_gw": model.radius_gw,
              "rho_align_default": rho_default,
              "summary": summary, "per_run": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dc.json_safe(result), indent=2, allow_nan=False) + "\n")

    flat = {}
    for v, s in summary.items():
        flat[f"{v}/oracle_gw_structural"] = s["oracle_rho_default"]["gw_structural_weighted"]
        flat[f"{v}/oracle_feature"] = s["oracle_rho_default"]["feature_term_weighted"]
        flat[f"{v}/oracle_index_prior"] = s["oracle_rho_default"]["index_prior_term_weighted"]
        flat[f"{v}/oracle_total"] = s["oracle_rho_default"]["total_weighted"]
        flat[f"{v}/solver_align_err"] = s["solver_rho_default"]["alignment_error_mean"]
        flat[f"{v}/solver_align_err_rho0"] = s["solver_rho0"]["alignment_error_mean"]
        flat[f"{v}/solver_support"] = s["solver_rho_default"].get("effective_support")
        flat[f"{v}/oracle_minus_solver_objective"] = s["solver_rho_default"].get("oracle_minus_solver_objective")
        flat[f"{v}/oracle_minus_solver_objective_rho0"] = s["solver_rho0"].get("oracle_minus_solver_objective")
        if "dtw_alignment_error_mean" in s:
            flat[f"{v}/dtw_align_err"] = s["dtw_alignment_error_mean"]
    print(json.dumps(dc.json_safe(flat), indent=2))
    if run is not None:
        run.log(dc.json_safe(flat))
        run.summary.update(dc.json_safe(flat))
        run.finish()


if __name__ == "__main__":
    main()
