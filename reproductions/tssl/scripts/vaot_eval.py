#!/usr/bin/env python3
"""Evaluate VAOT checkpoints with the upstream metric code (`repos/VAOT/evals`).

Two protocols are reported for every checkpoint, because the released
`evaluations.py` does not do what the TCC/LAV protocol describes:

  released  the released script as-is: classifier and progression regressor
            are fitted on a fraction of the *validation* videos and scored on
            all validation videos (`train_embs = val_embs`), and progression
            targets use the hard-coded `IDX_TO_CLASS` table.
  standard  TCC protocol: fitted on a fraction of the *training* videos,
            scored on validation videos; progression targets use the
            dataset's own phase ids.

Kendall's tau and AP@K only use validation embeddings and are identical in
both. Embeddings are cached next to the results.
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted

VAOT = Path(__file__).resolve().parents[1] / "repos" / "VAOT"
sys.path.insert(0, str(VAOT))

import utils  # noqa: E402
from config import CONFIG  # noqa: E402
from evals import phase_progression as pp  # noqa: E402
from evals.kendalls_tau import evaluate_kendalls_tau  # noqa: E402
from evals.phase_classification import compute_ap, evaluate_phase_classification  # noqa: E402
from train import AlignNet  # noqa: E402

RELEASED_IDX_TO_CLASS = dict(pp.IDX_TO_CLASS)


def load_model(ckpt, device):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = state["hyper_parameters"]["config"]
    model = AlignNet(cfg)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()  # AlignNet.train() returns None, so do not chain
    return model, cfg


@torch.no_grad()
def embed_split(model, cfg, split_dir, labels, device, batch):
    """Per-frame embeddings, same frame/context selection as align_dataset_test."""
    tf = utils.get_transforms(augment=False)
    ctx, stride = cfg.DATA.NUM_CONTEXT, cfg.DATA.CONTEXT_STRIDE
    embs, labs, names, paths = [], [], [], []
    for vid in natsorted(glob.glob(os.path.join(split_dir, "*"))):
        frames = np.array(sorted(glob.glob(os.path.join(vid, "*"))))
        steps = np.arange(len(frames))
        offs = np.arange(ctx - 1, -1, -1) * stride
        idx = np.maximum(0, steps[:, None] - offs)  # (T, ctx)
        out = []
        for i in range(0, len(frames), batch):
            sel = idx[i:i + batch].reshape(-1)
            x = tf(utils.get_pil_images(frames[sel])).unsqueeze(0).to(device)
            out.append(model(x)[0].float().cpu().numpy())
        e = np.concatenate(out)
        lab = labels[os.path.basename(vid)]["labels"]
        n = min(len(e), len(lab))
        embs.append(e[:n]); labs.append(lab[:n]); names.append(os.path.basename(vid)); paths.append(list(frames[:n]))
    return embs, labs, names, paths


def metrics(train, val, action, step, stride):
    t_embs, t_labs = train
    v_embs, v_labs, v_names, v_paths = val
    row = {}
    evaluate_phase_classification(step, t_embs, t_labs, v_embs, v_labs, act_name=action,
                                  CONFIG=CONFIG, csv_dict=row)
    res = {f"acc@{f}": 100 * row[f"PC ({f})"] for f in CONFIG.EVAL.CLASSIFICATION_FRACTIONS}
    _, prog = pp.evaluate_phase_progression({"embs": t_embs, "labels": t_labs},
                                            {"embs": v_embs, "labels": v_labs}, action, step, CONFIG)
    res["progress"] = 100 * prog[-1]
    _, tau = evaluate_kendalls_tau(v_embs, v_embs, stride=stride, kt_dist=CONFIG.EVAL.KENDALLS_TAU_DISTANCE)
    res["tau"] = 100 * tau
    ap = compute_ap(v_embs, v_labs, v_names, v_paths)
    res.update({f"ap@{k}": 100 * a for k, a in zip((5, 10, 15), ap)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data_path", required=True, help="dir with train/ and val/ video folders")
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda:0"
    os.makedirs(args.dest, exist_ok=True)
    labels = {s: np.load(os.path.join(args.labels_dir, f"{args.action}_{s}.npy"), allow_pickle=True).item()
              for s in ("train", "val")}
    out_path = os.path.join(args.dest, "results.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for ckpt in natsorted(args.ckpts):
        step = int(re.search(r"step=(\d+)", ckpt).group(1))
        if str(step) in results:
            continue
        cache = os.path.join(args.dest, f"embs_step{step}.npy")
        if os.path.exists(cache):
            emb = np.load(cache, allow_pickle=True).item()
        else:
            model, cfg = load_model(ckpt, device)
            CONFIG.update(cfg)
            emb = {s: embed_split(model, cfg, os.path.join(args.data_path, s), labels[s], device, args.batch)
                   for s in ("train", "val")}
            np.save(cache, emb, allow_pickle=True)
            del model
            torch.cuda.empty_cache()
        tr, va = emb["train"], emb["val"]
        pp.IDX_TO_CLASS = RELEASED_IDX_TO_CLASS
        released = metrics((va[0], va[1]), va, args.action, step, args.stride)
        pp.IDX_TO_CLASS = {i: i for i in range(64)}
        standard = metrics((tr[0], tr[1]), va, args.action, step, args.stride)
        results[str(step)] = {"released": released, "standard": standard}
        json.dump(results, open(out_path, "w"), indent=1)
        print(step, "released", {k: round(v, 2) for k, v in released.items()}, flush=True)
        print(step, "standard", {k: round(v, 2) for k, v in standard.items()}, flush=True)


if __name__ == "__main__":
    main()
