#!/usr/bin/env python3
"""Tables for the MRT matrix: segmentation (mrt_matrix.tsv) and VA metrics
(va/*.json from run_va_matrix.sh, va_new/*.json from run_va_new.py).

Per benchmark and arm: mean over seeds, and the paired difference to the
baseline of the same seed (mean, per-seed sign count). A last row gives the
mean of the per-benchmark paired differences and on how many benchmarks it is > 0.

  collect_mrt_results.py [arm ...]        default: the hub/local arms
  SEEDS="0" collect_mrt_results.py ...    restrict to some seeds
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "artifacts" / "runs" / "current"
ARMS = sys.argv[1:] or ["hub", "tmpl", "local", "hub_local", "local_raw", "hub_local_raw"]
ARMS = ["baseline"] + [a for a in ARMS if a != "baseline"]
SEEDS = [int(s) for s in os.environ["SEEDS"].split()] if os.environ.get("SEEDS") else None
BENCH = ["fseval", "da", "yti_changing_tire", "yti_coffee", "yti_cpr", "yti_jump_car", "yti_repot"]


def table(title, data, metrics, arms):
    # data[bench][arm][seed] = {metric: value}
    for m in metrics:
        print(f"\n### {title}: {m} (mean over seeds | paired delta vs baseline, wins/seeds)\n")
        print("| bench | " + " | ".join(arms) + " |")
        print("|---|" + "---:|" * len(arms))
        deltas = defaultdict(list)
        for bench in BENCH:
            if bench not in data:
                continue
            base = data[bench].get("baseline", {})
            cells = []
            for a in arms:
                runs = data[bench].get(a, {})
                if not runs:
                    cells.append("-"); continue
                mean = np.mean([r[m] for r in runs.values()])
                if a == "baseline":
                    cells.append(f"{mean:.1f} (n={len(runs)})"); continue
                d = [runs[s][m] - base[s][m] for s in runs if s in base]
                if d:
                    deltas[a].append(np.mean(d))
                    cells.append(f"{mean:.1f} ({np.mean(d):+.1f}, {sum(x > 0 for x in d)}/{len(d)})")
                else:
                    cells.append(f"{mean:.1f}")
            print(f"| {bench} | " + " | ".join(cells) + " |")
        cells = ["" if a == "baseline" else
                 (f"{np.mean(deltas[a]):+.1f}, {sum(x > 0 for x in deltas[a])}/{len(deltas[a])} bench" if deltas[a] else "-")
                 for a in arms]
        print("| **mean delta** | " + " | ".join(cells) + " |")


def keep(seed):
    return SEEDS is None or seed in SEEDS


seg = defaultdict(lambda: defaultdict(dict))
for line in (RUNS / "mrt_matrix.tsv").read_text().splitlines():
    b, a, s, mof, f1, miou = line.split("\t")
    if keep(int(s)):
        seg[b.replace(":", "_")][a][int(s)] = {"MoF": 100 * float(mof), "F1": 100 * float(f1), "mIoU": 100 * float(miou)}
table("Segmentation", seg, ["MoF", "F1", "mIoU"], ARMS)

va = defaultdict(lambda: defaultdict(dict))
for d in ("va", "va_new"):
    for f in sorted((RUNS / d).glob("*.json")):
        stem = f.stem.split("__")[0]
        if "_s" not in stem or not stem.rsplit("_s", 1)[1].isdigit():
            continue
        bench, seed = stem.rsplit("_s", 1)
        if keep(int(seed)):
            for arm, r in json.loads(f.read_text()).items():
                va[bench][arm][int(seed)] = r
if va:
    table("Alignment", va, ["acc", "tau", "ap@5", "ap@10", "ap@15"], ARMS[:1] + ["raw"] + ARMS[1:])
