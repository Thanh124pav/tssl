#!/usr/bin/env python3
"""Video-alignment metrics for VASOT checkpoints on the feature benchmarks.

Uses the VAOT metric code (repos/VAOT/evals) on frozen frame embeddings, with
the per-frame action label standing in for the phase label:
  acc     phase classification: SVM fitted on even-indexed videos, scored on odd
  tau     Kendall's tau of cross-video nearest-neighbour order (all video pairs)
  ap@K    fine-grained frame retrieval, leave-one-video-out
Every video is sampled at --frames evenly spaced frames. A `raw` row (input
features, no encoder) is printed per dataset as the reference point.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "repos" / "VAOT"))
import diag_common as dc  # noqa: E402
from evals.kendalls_tau import evaluate_kendalls_tau  # noqa: E402
from evals.phase_classification import compute_ap, fit_svm, evaluate_svm  # noqa: E402


def va_metrics(embs, labs, names, tau_stride):
    tr, te = slice(0, None, 2), slice(1, None, 2)
    svm, _ = fit_svm(embs[tr], labs[tr])
    acc, _ = evaluate_svm(svm, embs[te], labs[te])
    _, tau = evaluate_kendalls_tau(embs, embs, stride=tau_stride, kt_dist="sqeuclidean")
    ap = compute_ap(embs, labs, names, [[]] * len(embs))
    return {"acc": 100 * acc, "tau": 100 * tau, "ap@5": 100 * ap[0], "ap@10": 100 * ap[1], "ap@15": 100 * ap[2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--activity", default="all")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--tau-stride", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = dc.load_dataset(args.dataset, args.frames, args.activity, data_root=str(HERE.parent / "data" / "final"))
    names = sorted(data.video_fnames)
    results = {}

    def collect(get):
        embs, labs = [], []
        for n in names:
            z, gt, mask = get(n)
            keep = mask & (gt >= 0)
            embs.append(z[keep].cpu().numpy()); labs.append(gt[keep].cpu().numpy())
        return embs, labs

    def raw(n):
        r, mask, gt, _, _ = data._load_video_data(n)
        return F.normalize(r.float(), dim=-1), gt, torch.as_tensor(np.asarray(mask), dtype=torch.bool)

    runs = [("raw", None)] + [tuple(c.split("=", 1)) for c in args.ckpts]
    for tag, path in runs:
        if path is None:
            embs, labs = collect(raw)
        else:
            model = dc.load_model(path)
            def enc(n, model=model):
                e = dc.embed(model, data, n)
                return model._view(e["z"])[0], e["gt"].reshape(-1), e["mask"][0]
            embs, labs = collect(enc)
        results[tag] = va_metrics(embs, labs, names, args.tau_stride)
        print(f"{tag:14s}" + "  ".join(f"{k} {v:6.2f}" for k, v in results[tag].items()), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
