#!/usr/bin/env python3
"""V02 preprocessing — convert sermanet_pouring layout to the VAOT layout.

Reads:
  /mnt/data/thanhpv/acvcl/data/pouring/sermanet_pouring/
      videos/{train,test}/*.mov
      train.pkl, val.pkl, test.pkl   # CARL/TCC-style, per-frame `frame_label`

Writes:
  /mnt/data/thanhpv/tssl/reproductions/tssl/data/vaot/pouring/
      frames/{train,test}/<video_name>/<idx:04d>.jpg
      npyrecords/pouring_{train,val,test}.npy    # dict keyed by video name,
                                                 # value = {'labels': 1-D int array}

Idempotent: skips a video directory if the expected frame count already exists.
Dry-run mode prints the planned work without writing.

Reason this script exists:
  V01 (reports/V01_VAOT_DATA_AUDIT.md) found that disk frames are a ~1/6
  subsample of the videos. VAOT expects full frames per video, one subdir per
  video; this script produces that layout from the authoritative .mov files
  plus the per-frame labels that already live in the pkl.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

SRC_ROOT = Path("/mnt/data/thanhpv/acvcl/data/pouring/sermanet_pouring")
DST_ROOT = Path("/mnt/data/thanhpv/tssl/reproductions/tssl/data/vaot/pouring")

# which pkl split -> which videos/ subdir (val reuses test videos)
SPLIT_TO_VIDEO_DIR = {"train": "train", "val": "test", "test": "test"}


def load_pkl(split: str) -> list[dict]:
    with open(SRC_ROOT / f"{split}.pkl", "rb") as f:
        return pickle.load(f)


def extract_video_frames(mov_path: Path, out_dir: Path, expected: int, dry_run: bool) -> int:
    """Extract every frame from mov_path into out_dir/<idx:04d>.jpg. Returns count written."""
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) == expected:
        return expected  # idempotent skip
    if dry_run:
        return expected

    out_dir.mkdir(parents=True, exist_ok=True)
    # If a partial extraction exists, redo clean to avoid off-by-one surprises.
    for p in out_dir.glob("*.jpg"):
        p.unlink()

    cap = cv2.VideoCapture(str(mov_path))
    count = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{count:04d}.jpg"), img)
        count += 1
    cap.release()
    return count


def build_npyrecords(split: str, items: list[dict], dry_run: bool) -> Path:
    out_path = DST_ROOT / "npyrecords" / f"pouring_{split}.npy"
    rec = {}
    for it in items:
        labels = it["frame_label"]
        if hasattr(labels, "numpy"):
            labels = labels.numpy()
        rec[it["name"]] = {"labels": np.asarray(labels, dtype=np.int64)}
    if dry_run:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, rec, allow_pickle=True)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Report work without writing anything.")
    args = ap.parse_args()

    for split in ("train", "val", "test"):
        items = load_pkl(split)
        video_subdir = SPLIT_TO_VIDEO_DIR[split]
        video_root = SRC_ROOT / "videos" / video_subdir
        frames_root = DST_ROOT / "frames" / video_subdir
        total_frames = 0
        for it in items:
            name = it["name"]
            seq_len = int(it["seq_len"])
            mov = video_root / f"{name}.mov"
            if not mov.exists():
                print(f"[WARN] {split}: missing {mov}", file=sys.stderr)
                continue
            got = extract_video_frames(mov, frames_root / name, seq_len, args.dry_run)
            total_frames += got
            marker = "[DRY]" if args.dry_run else "[OK]"
            print(f"{marker} {split}/{name}: expected={seq_len} got={got}")
        rec_path = build_npyrecords(split, items, args.dry_run)
        print(f"  wrote npyrecords -> {rec_path} ({'dry' if args.dry_run else 'real'})")
        print(f"  total frames for split={split}: {total_frames}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
