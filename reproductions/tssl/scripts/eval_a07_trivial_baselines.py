"""A07 - how much of VASOT's segmentation score needs VASOT?

The diagnostics say the representation is weak: within-action phase is near
chance, actions overlap heavily, and the one thing training encodes strongly is
absolute time. Yet the reported scores are competitive. This packet asks which
component actually produces the score, by scoring trivial controls under the
EXACT protocol A04 uses: pooled frames, one-to-one Hungarian cluster->GT
matching (metrics.pred_to_gt_match, as eval_mof does), plus a many-to-one
purity reading.

Arms, in increasing order of what they are allowed to use:

  uniform        frame i -> cluster floor(i/N * K). No model, no features, no
                 training. Pure position in the video.
  kmeans_raw     k-means on the raw features. Appearance only, no time, no
                 training of the encoder.
  kmeans_trained k-means on the trained embeddings. Appearance of a trained
                 encoder, but no OT decoder and no temporal prior.
  untrained_ot   untrained encoder + the full VASOT OT decoder with its
                 temporal prior. Decoder only, no learned representation.
  vasot          the trained checkpoint, full pipeline.
  vasot_rho0     trained checkpoint, temporal prior switched off at decode time.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import diag_common as dc
from metrics import pred_to_gt_match


def score(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    pred_opt, gt_opt = pred_to_gt_match(pred, gt)
    mapping = dict(zip(pred_opt.tolist(), gt_opt.tolist()))
    one_to_one = float(np.mean([mapping.get(int(p), -1) == g for p, g in zip(pred, gt)]))
    major = {int(p): int(np.bincount(gt[pred == p]).argmax()) for p in np.unique(pred)}
    many_to_one = float(np.mean([major[int(p)] == g for p, g in zip(pred, gt)]))
    return {"accuracy_one_to_one": one_to_one, "accuracy_many_to_one": many_to_one,
            "clusters_used": int(len(np.unique(pred))),
            "clusters_matched": int(len(mapping))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="FSeval")
    ap.add_argument("--activity", default="all")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--videos", type=int, default=20)
    ap.add_argument("--init-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = dc.pick_device("cuda" if args.device in ("auto", "cuda") else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.set_grad_enabled(False)
    run = None if args.no_wandb else dc.init_wandb(
        f"A07_{args.tag}", ["A07", "diagnostic", args.dataset, "action_space", "baselines"],
        vars(args))

    model = dc.load_model(args.ckpt, device)
    init_model = dc.load_model(args.ckpt, device, init_seed=args.init_seed)
    data = dc.load_dataset(args.dataset, args.frames, args.activity)
    names = data.video_fnames[: args.videos]
    K = int(model.n_clusters)
    n = args.frames

    preds = {k: [] for k in ("uniform", "kmeans_raw", "kmeans_trained",
                             "untrained_ot", "vasot", "vasot_rho0")}
    gts, Zs, RAWs = [], [], []
    for name in names:
        rec = dc.embed(model, data, name, device)
        rec_i = dc.embed(init_model, data, name, device)
        valid = rec["mask"][0].cpu().numpy().astype(bool)
        gts.append(rec["gt"].cpu().numpy()[valid])
        Zs.append(rec["z"][0].cpu().numpy()[valid])
        RAWs.append(rec["raw"].cpu().numpy()[valid])

        # position only
        preds["uniform"].append(np.minimum((np.arange(n) / n * K).astype(int), K - 1)[valid])
        # full pipeline, and with the decode-time prior removed
        for key, rho in (("vasot", None), ("vasot_rho0", 0.0)):
            t = dc.seg_plan(model, rec["z"], rec["mask"], rho_seg=rho)
            preds[key].append(t[0].argmax(-1).cpu().numpy()[valid])
        # decoder with an untrained encoder
        t = dc.seg_plan(init_model, rec_i["z"], rec_i["mask"])
        preds["untrained_ot"].append(t[0].argmax(-1).cpu().numpy()[valid])

    # k-means arms are fitted on the pooled frames, like the codebook would be
    from sklearn.cluster import KMeans
    for key, feats in (("kmeans_raw", RAWs), ("kmeans_trained", Zs)):
        pooled = np.concatenate(feats, axis=0)
        km = KMeans(n_clusters=K, n_init=10, random_state=args.seed).fit(pooled)
        off = 0
        for f in feats:
            preds[key].append(km.labels_[off:off + len(f)])
            off += len(f)

    gt_all = np.concatenate(gts)
    result = {"checkpoint": str(Path(args.ckpt).resolve()), "dataset": args.dataset,
              "frames": n, "videos": names, "n_frames_scored": int(len(gt_all)),
              "n_clusters": K, "n_gt_actions": int(len(np.unique(gt_all))),
              "device": device,
              "protocol": "pooled frames; one-to-one Hungarian as metrics.eval_mof; "
                          "many-to-one is per-cluster majority (purity)",
              "arms": {}}
    for key, chunks in preds.items():
        result["arms"][key] = score(np.concatenate(chunks), gt_all)

    # per-video scoring too, since pooling can hide variance
    per_video = {key: [] for key in preds}
    for key, chunks in preds.items():
        for p, g in zip(chunks, gts):
            per_video[key].append(score(p, g)["accuracy_one_to_one"])
    for key in preds:
        result["arms"][key]["per_video_one_to_one_mean"] = float(np.mean(per_video[key]))
        result["arms"][key]["per_video_one_to_one_std"] = float(np.std(per_video[key]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dc.json_safe(result), indent=2, allow_nan=False) + "\n")

    order = ["uniform", "kmeans_raw", "kmeans_trained", "untrained_ot", "vasot_rho0", "vasot"]
    print(f"{'arm':16s}{'1-to-1':>9s}{'many-1':>9s}{'per-video':>11s}{'used':>6s}")
    for k in order:
        a = result["arms"][k]
        print(f"{k:16s}{a['accuracy_one_to_one']:9.4f}{a['accuracy_many_to_one']:9.4f}"
              f"{a['per_video_one_to_one_mean']:11.4f}{a['clusters_used']:6d}")
    flat = {f"{k}/{m}": v for k, a in result["arms"].items() for m, v in a.items()}
    if run is not None:
        run.log(dc.json_safe(flat))
        run.summary.update(dc.json_safe(flat))
        run.finish()


if __name__ == "__main__":
    main()
