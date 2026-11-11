#!/usr/bin/env python3
"""Convert a CARL-format alignment dataset (Pouring / Penn Action) to the
layout the VAOT code reads.

Input  (downloaded by download_vaot_datasets.py):
    <src>/videos/*.mp4, <src>/train.pkl, <src>/val.pkl
    each pkl: list of dicts {name, video_file, frame_label (tensor), seq_len, ...}

Output:
    <dst>/<action>/{train,val}/<video_name>/<idx:04d>.jpg
    <dst>/npyrecords/<action>_{train,val}.npy   {video_name: {'labels': int array}}
    <dst>/manifest.json

Pouring has a single action ("pouring"); Penn Action is split by the
dataset's own action label so one encoder can be trained per action, as the
paper does. Idempotent: a video whose frame count already matches is skipped.
"""
import argparse
import collections
import io
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

# The pkl files come from a public Google Drive: only allow the constructors
# they legitimately need instead of running arbitrary pickle globals.
_ALLOWED = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch.storage", "_load_from_bytes"): lambda b: torch.load(io.BytesIO(b), weights_only=True),
}


ALIASES = {"bowl": "bowling", "jumping_jack": "jumping_jacks", "pushup": "pushups",
           "pullup": "pullups", "squat": "squats", "clean_jerk": "clean_and_jerk"}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_pkl(path):
    with open(path, "rb") as f:
        return _Restricted(f).load()


def extract(video, out_dir, expect):
    have = len(list(out_dir.glob("*.jpg"))) if out_dir.exists() else 0
    if have == expect:
        return have
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    n = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{n:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--single-action", default=None,
                    help="name to use when the dataset has one action (pouring)")
    args = ap.parse_args()

    manifest = {}
    records = collections.defaultdict(dict)
    for split in ("train", "val"):
        items = load_pkl(args.src / f"{split}.pkl")
        if isinstance(items, tuple):  # Penn Action: (videos, per-action index)
            items = items[0]
        for item in items:
            name = item["name"]
            labels = np.asarray(item["frame_label"]).astype(int).reshape(-1)
            # Penn Action names are "<id>_<action>"; use upstream evals/datasets.py spelling
            raw = name.split("_", 1)[1] if not args.single_action else None
            action = args.single_action or ALIASES.get(raw, raw)
            n = extract(args.src / item["video_file"], args.dst / action / split / name, len(labels))
            m = manifest.setdefault(action, {}).setdefault(split, {"videos": 0, "frames": 0, "mismatch": []})
            m["videos"] += 1
            m["frames"] += n
            if n != len(labels):
                m["mismatch"].append([name, n, len(labels)])
            records[(action, split)][name] = {"labels": labels}
    (args.dst / "npyrecords").mkdir(parents=True, exist_ok=True)
    for (action, split), rec in records.items():
        np.save(args.dst / "npyrecords" / f"{action}_{split}.npy", rec, allow_pickle=True)
        manifest[action][split]["label_values"] = sorted(
            int(v) for v in np.unique(np.concatenate([r["labels"] for r in rec.values()])))
    (args.dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for action, m in manifest.items():
        for split, s in m.items():
            print(action, split, s["videos"], "videos", s["frames"], "frames",
                  "labels", s["label_values"], "mismatch", len(s["mismatch"]))


if __name__ == "__main__":
    main()
