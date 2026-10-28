"""Read overnight sweep's `_status.tsv` and produce a Markdown summary table.

Grouped by (variant, dataset) with mean±range over seeds, bolds cells that
beat the baseline's mean by more than their own seed-range.
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def pct(x):
    try:
        return float(x) * 100.0
    except (TypeError, ValueError):
        return None


def main(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    agg = defaultdict(lambda: defaultdict(list))        # agg[dataset][variant] = [(mof,f1,miou),...]
    for r in rows:
        key = r["variant"], r["dataset"]
        trip = tuple(pct(r[k]) for k in ("mof", "f1", "miou"))
        agg[r["dataset"]][r["variant"]].append(trip)

    datasets = sorted(agg.keys())
    variants = sorted({v for d in agg.values() for v in d})

    def summarise(samples, idx):
        vals = [t[idx] for t in samples if t[idx] is not None]
        if not vals:
            return None
        return sum(vals) / len(vals), max(vals) - min(vals)

    out = []
    out.append(f"# Overnight sweep summary — {Path(path).parent.name}\n")
    for metric_i, metric_name in enumerate(["MoF", "F1", "mIoU"]):
        out.append(f"## {metric_name}\n")
        header = "| variant | " + " | ".join(datasets) + " |"
        sep = "|" + "---|" * (len(datasets) + 1)
        out.append(header)
        out.append(sep)
        base_means = {d: summarise(agg[d].get("baseline", []), metric_i) for d in datasets}
        for v in variants:
            cells = []
            for d in datasets:
                s = summarise(agg[d].get(v, []), metric_i)
                if s is None:
                    cells.append("—")
                    continue
                mean, rng = s
                bm = base_means.get(d)
                mark = ""
                if v != "baseline" and bm is not None and mean > bm[0] + rng:
                    mark = "**"
                cells.append(f"{mark}{mean:.2f}±{rng:.2f}{mark}")
            out.append(f"| {v} | " + " | ".join(cells) + " |")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
