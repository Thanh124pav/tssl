"""Dataset sanity report.

Usage:
    python scripts/inspect_dataset.py --dataset coin
    python scripts/inspect_dataset.py --dataset crosstask   (if data available)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def inspect_coin():
    from src.datasets.coin import CoinIndex
    ds = CoinIndex(ROOT / "data" / "coin" / "COIN.json")
    tasks = ds.task_names()
    sizes = sorted([(len(vs), t, tasks.get(t,"?")) for t, vs in ds.by_task.items()],
                   reverse=True)
    print(f"COIN: {ds.num_videos} videos across {ds.num_tasks} tasks")
    print(f"  subset counts:")
    from collections import Counter
    c = Counter(v.subset for v in ds.videos.values())
    for k, n in c.items(): print(f"    {k}: {n}")
    print(f"  task-size distribution (top 5 / bottom 5):")
    for n, t, name in sizes[:5]:  print(f"    {t:3d}  {n:4d}  {name}")
    print("    ...")
    for n, t, name in sizes[-5:]: print(f"    {t:3d}  {n:4d}  {name}")
    import random
    rng = random.Random(0)
    print("  example same-task pairs:")
    for _ in range(3):
        a, b = ds.sample_same_task_pair(rng=rng)
        va, vb = ds.videos[a], ds.videos[b]
        print(f"    {va.task_name:25s}  {a}  {b}")
    # steps distribution
    nsteps = [len(v.steps) for v in ds.videos.values()]
    print(f"  steps per video: min={min(nsteps)}  mean={sum(nsteps)/len(nsteps):.1f}  max={max(nsteps)}")

def inspect_crosstask():
    base = ROOT / "data" / "crosstask"
    rels = list(base.glob("*release*/tasks_primary.txt")) + list(base.glob("tasks_primary.txt"))
    if not rels:
        print("CrossTask release files not present. See SETUP_STATUS.md — the "
              "official hosts are unreachable from this environment.")
        return
    print(f"CrossTask release root: {rels[0].parent}")
    print(open(rels[0]).read()[:400])

def inspect_youcook2():
    base = ROOT / "data" / "youcook2"
    print("YouCook2 (metadata-only; raw hosts unreachable):")
    for p in sorted(base.rglob("*.txt")) + sorted(base.rglob("*.csv")):
        try: n = sum(1 for _ in p.open())
        except Exception: n = -1
        print(f"  {p.relative_to(base)}  ({n} lines)")

def inspect_howto100m():
    base = ROOT / "data" / "howto100m"
    for p in sorted(base.rglob("*.json")) + sorted(base.rglob("*.py")):
        print(f"  {p.relative_to(base)}  ({p.stat().st_size} B)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["coin", "crosstask", "youcook2", "howto100m"])
    args = ap.parse_args()
    {"coin": inspect_coin, "crosstask": inspect_crosstask,
     "youcook2": inspect_youcook2, "howto100m": inspect_howto100m}[args.dataset]()

if __name__ == "__main__":
    main()
