"""A04 - how confident, and how correct, is the frame -> segment assignment?

Two different "confidences" exist in the model and are reported separately:
  ot_posterior   top-1 of the row-normalised T_xa plan (what the pseudo-label uses)
  codes_softmax  top-1 of softmax(z a^T / temp) (what the CE loss predicts)

Correctness uses the repo's own protocol: a one-to-one Hungarian assignment
between predicted clusters and GT actions (metrics.pred_to_gt_match), fitted on
the pooled frames, exactly as eval_mof does. Clusters left unmatched count as
errors, which is the honest reading of that protocol.

Outputs reliability diagrams, ECE/MCE, the AUROC of confidence for predicting
its own correctness, and the same stratified by boundary distance, segment
duration and action.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diag_common as dc
from metrics import pred_to_gt_match


def expected_calibration_error(conf, correct, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    ece, mce, rows = 0.0, 0.0, []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0})
            continue
        acc = float(correct[m].mean())
        avg_conf = float(conf[m].mean())
        gap = abs(acc - avg_conf)
        ece += (m.sum() / len(conf)) * gap
        mce = max(mce, gap)
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(m.sum()),
                     "accuracy": acc, "mean_confidence": avg_conf,
                     "gap_conf_minus_acc": float(avg_conf - acc)})
    return float(ece), float(mce), rows


def summarise(conf, correct, label):
    from sklearn.metrics import roc_auc_score
    ece, mce, rows = expected_calibration_error(conf, correct)
    out = {"which": label, "n_frames": int(len(conf)),
           "accuracy": float(correct.mean()),
           "mean_confidence": float(conf.mean()),
           "overconfidence_conf_minus_acc": float(conf.mean() - correct.mean()),
           "ece": ece, "mce": mce, "reliability": rows}
    if 0 < correct.sum() < len(correct):
        out["auroc_confidence_predicts_correctness"] = float(roc_auc_score(correct, conf))
    # accuracy of the most/least confident deciles
    order = np.argsort(-conf)
    d = max(len(conf) // 10, 1)
    out["accuracy_top_decile_confidence"] = float(correct[order[:d]].mean())
    out["accuracy_bottom_decile_confidence"] = float(correct[order[-d:]].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--videos", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = dc.pick_device("cuda" if args.device in ("auto", "cuda") else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.set_grad_enabled(False)  # nothing here trains; model.clusters carries requires_grad
    run = None if args.no_wandb else dc.init_wandb(
        f"A04_{args.tag}", ["A04", "diagnostic", args.dataset, "action_space"], vars(args))

    model = dc.load_model(args.ckpt, device)
    data = dc.load_dataset(args.dataset, args.frames, args.activity)
    names = data.video_fnames[: args.videos]

    pred_all, gt_all, conf_ot, conf_sm, bdist, segdur, vid = [], [], [], [], [], [], []
    for name in names:
        rec = dc.embed(model, data, name, device)
        t_xa = dc.seg_plan(model, rec["z"], rec["mask"])
        post = t_xa[0] / t_xa[0].sum(-1, keepdim=True).clamp_min(1e-12)
        codes = torch.softmax(rec["z"][0] @ model.clusters.T / model.temp, dim=-1)
        valid = rec["mask"][0].cpu().numpy().astype(bool)
        gt = rec["gt"].cpu().numpy()
        segs = dc.gt_segments(rec["gt"])
        dur = np.zeros(len(gt))
        for s, e, _ in segs:
            dur[s:e] = e - s
        pred_all.append(post.argmax(-1).cpu().numpy()[valid])
        gt_all.append(gt[valid])
        conf_ot.append(post.max(-1).values.cpu().numpy()[valid])
        conf_sm.append(codes.max(-1).values.cpu().numpy()[valid])
        bdist.append(np.abs(dc.signed_boundary_distance(rec["gt"]))[valid])
        segdur.append(dur[valid])
        vid.append(np.full(int(valid.sum()), name))

    pred = np.concatenate(pred_all)
    gt = np.concatenate(gt_all)
    c_ot = np.concatenate(conf_ot)
    c_sm = np.concatenate(conf_sm)
    bd = np.concatenate(bdist)
    sd = np.concatenate(segdur)

    # repo protocol: one-to-one Hungarian cluster -> GT action
    pred_opt, gt_opt = pred_to_gt_match(pred, gt)
    mapping = dict(zip(pred_opt.tolist(), gt_opt.tolist()))
    mapped = np.array([mapping.get(int(p), -1) for p in pred])
    correct = (mapped == gt)
    # many-to-one (purity) protocol, for separating representation quality from
    # the one-to-one protocol's penalty on unmatched clusters
    major = {}
    for p in np.unique(pred):
        major[int(p)] = int(np.bincount(gt[pred == p]).argmax())
    correct_m2o = np.array([major[int(p)] == g for p, g in zip(pred, gt)])

    result = {"checkpoint": str(Path(args.ckpt).resolve()), "dataset": args.dataset,
              "frames": args.frames, "videos": names, "device": device,
              "n_clusters": int(model.n_clusters),
              "n_gt_actions": int(len(np.unique(gt))),
              "clusters_used": int(len(np.unique(pred))),
              "clusters_matched_one_to_one": int(len(mapping)),
              "protocol": {
                  "one_to_one": "metrics.pred_to_gt_match, as eval_mof uses; unmatched clusters are errors",
                  "many_to_one": "each cluster takes its majority GT action (purity)"},
              "accuracy_one_to_one": float(correct.mean()),
              "accuracy_many_to_one": float(correct_m2o.mean()),
              "frames_in_unmatched_clusters": float((mapped == -1).mean()),
              "calibration": {
                  "ot_posterior": summarise(c_ot, correct, "ot_posterior"),
                  "codes_softmax": summarise(c_sm, correct, "codes_softmax"),
                  "ot_posterior_many_to_one": summarise(c_ot, correct_m2o, "ot_posterior_m2o")}}

    # stratifications
    def strat(mask_fn, bins, label, values):
        rows = []
        for lo, hi in bins:
            m = mask_fn(values, lo, hi)
            if m.sum() == 0:
                continue
            rows.append({"lo": lo, "hi": None if hi > 1e8 else hi, "n": int(m.sum()),
                         "accuracy": float(correct[m].mean()),
                         "accuracy_many_to_one": float(correct_m2o[m].mean()),
                         "mean_ot_confidence": float(c_ot[m].mean()),
                         "mean_softmax_confidence": float(c_sm[m].mean()),
                         "overconfidence": float(c_ot[m].mean() - correct[m].mean())})
        return {label: rows}

    inside = lambda v, lo, hi: (v >= lo) & (v <= hi)  # noqa: E731
    result.update(strat(inside, [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)],
                        "by_boundary_distance", bd))
    result.update(strat(inside, [(0, 15), (16, 47), (48, 10 ** 9)],
                        "by_segment_duration", sd))
    per_action = []
    for a in np.unique(gt):
        m = gt == a
        per_action.append({"gt_action": int(a), "n": int(m.sum()),
                           "accuracy": float(correct[m].mean()),
                           "mean_ot_confidence": float(c_ot[m].mean()),
                           "overconfidence": float(c_ot[m].mean() - correct[m].mean())})
    result["by_gt_action"] = per_action

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dc.json_safe(result), indent=2, allow_nan=False) + "\n")

    flat = {"accuracy_one_to_one": result["accuracy_one_to_one"],
            "accuracy_many_to_one": result["accuracy_many_to_one"],
            "frames_in_unmatched_clusters": result["frames_in_unmatched_clusters"],
            "clusters_used": result["clusters_used"]}
    for k, v in result["calibration"].items():
        for m in ("accuracy", "mean_confidence", "overconfidence_conf_minus_acc", "ece", "mce",
                  "auroc_confidence_predicts_correctness", "accuracy_top_decile_confidence",
                  "accuracy_bottom_decile_confidence"):
            if m in v:
                flat[f"{k}/{m}"] = v[m]
    print(json.dumps(dc.json_safe(flat), indent=2))
    print(json.dumps(dc.json_safe(result["by_boundary_distance"]), indent=2))
    if run is not None:
        run.log(dc.json_safe(flat))
        run.summary.update(dc.json_safe(flat))
        run.finish()


if __name__ == "__main__":
    main()
