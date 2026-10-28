"""Convert sermanet/pouring MOV files into the CARL/TCC-expected pickle format.

CARL's `datasets/pouring.py` reads:
    PATH_TO_DATASET/{train,val,test}.pkl   # list of dicts
    PATH_TO_DATASET/<video_file>           # path returned by read_video

Each dict needs: {id, name, video_file, seq_len, frame_label}.

The sermanet/pouring HF mirror contains:
    videos/train/pouring_XXX.mov
    videos/test/pouring_XXX.mov       (and some pouring-miss_XXX.mov)

We produce `train.pkl` from `videos/train/*.mov` and `val.pkl` from
`videos/test/*.mov` (TCC literature uses "val" as the eval set for Pouring).
Since this public mirror has no per-frame phase labels, `frame_label` is set
to zeros; this disables phase-classification + Kendall's Tau phase metrics
but *does not* affect TCC/SCL self-supervised training.
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path

import torch
from torchvision.io import read_video


def build_entry(video_path: Path, root: Path, idx: int):
    name = video_path.stem                   # pouring_003
    rel  = video_path.relative_to(root).as_posix()
    video, _, info = read_video(str(video_path), pts_unit="sec")
    seq_len = video.shape[0]
    frame_label = torch.zeros(seq_len, dtype=torch.long)
    return {
        "id":          idx,
        "name":        name,
        "video_file":  rel,
        "seq_len":     int(seq_len),
        "frame_label": frame_label,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/thanhpv/acvcl/data/pouring/sermanet_pouring")
    ap.add_argument("--out", default=None,
                    help="where to drop train.pkl/val.pkl (default = --root)")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out or args.root)
    out.mkdir(parents=True, exist_ok=True)

    train_vids = sorted((root / "videos" / "train").glob("*.mov"))
    test_vids  = sorted((root / "videos" / "test" ).glob("*.mov"))
    if not train_vids:
        sys.exit(f"no train videos under {root/'videos/train'}")

    train = [build_entry(p, root, i) for i, p in enumerate(train_vids)]
    val   = [build_entry(p, root, i) for i, p in enumerate(test_vids)]

    with (out / "train.pkl").open("wb") as f: pickle.dump(train, f)
    with (out / "val.pkl").open("wb") as f:   pickle.dump(val, f)
    # TCC/CARL also expects a test.pkl symlink/alias
    with (out / "test.pkl").open("wb") as f:  pickle.dump(val, f)

    print(f"wrote {len(train)} train / {len(val)} val entries to {out}")
    for e in (train[:2] + val[:2]):
        print(f"  {e['name']:25s}  seq_len={e['seq_len']:4d}  video_file={e['video_file']}")


if __name__ == "__main__":
    main()
