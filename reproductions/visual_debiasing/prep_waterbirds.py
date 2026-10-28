"""Materialize Waterbirds from the HuggingFace mirror into the group_DRO layout.

nlp.stanford.edu (the dataset's original home) is blocked by the cluster
network policy, so the images come from grodino/waterbirds instead. That
parquet keeps the original JPEG bytes and filenames, so the files written here
are byte-identical to the official tarball's, and the group counts are checked
against the published ones before anything is written.

Both correct-n-contrast and nsf_debiasing read this same directory, so it is
built once and symlinked rather than copied.
"""

import argparse
import csv
import os
from pathlib import Path

from datasets import Image, load_dataset

SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}

# Published Waterbirds group sizes, keyed (y, place): y=1 waterbird, place=1 water.
EXPECTED_GROUPS = {
    "train": {(0, 0): 3498, (0, 1): 184, (1, 0): 56, (1, 1): 1057},
    "validation": {(0, 0): 467, (0, 1): 466, (1, 0): 133, (1, 1): 133},
    "test": {(0, 0): 2255, (0, 1): 2255, (1, 0): 642, (1, 1): 642},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="/mnt/data/thanhpv/acvcl/data/waterbirds/waterbird_complete95_forest2water2",
    )
    parser.add_argument("--repo", default="grodino/waterbirds")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = os.environ.get("HF_HOME")

    rows = []
    img_id = 0
    for split, split_id in SPLIT_IDS.items():
        ds = load_dataset(args.repo, split=split, cache_dir=cache)
        species = ds.features["bird"].names
        counts = {}
        raw = ds.cast_column("image", Image(decode=False))
        for rec in raw:
            y = int(rec["label"])
            place = int(rec["place"])
            counts[(y, place)] = counts.get((y, place), 0) + 1
            class_dir = f"{int(rec['bird']) + 1:03d}.{species[int(rec['bird'])]}"
            name = rec["image"]["path"] or f"{img_id:06d}.jpg"
            rel = f"{class_dir}/{name}"
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(rec["image"]["bytes"])
            rows.append(
                {
                    "img_id": img_id,
                    "img_filename": rel,
                    "y": y,
                    "split": split_id,
                    "place": place,
                    "place_filename": "",
                }
            )
            img_id += 1

        if counts != EXPECTED_GROUPS[split]:
            raise SystemExit(
                f"group counts for {split} do not match published Waterbirds: "
                f"{counts} != {EXPECTED_GROUPS[split]}"
            )
        print(f"{split:11s} {len(ds):5d} images, groups verified {counts}")

    meta = out / "metadata.csv"
    with meta.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["img_id", "img_filename", "y", "split", "place", "place_filename"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {meta} with {len(rows)} rows")


if __name__ == "__main__":
    main()
