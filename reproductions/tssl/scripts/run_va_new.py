#!/usr/bin/env python3
"""VA metrics (eval_va.py) for finished runs that have no VA score yet.

Scans mrt_matrix.tsv; for every benchmark and seed, the finished arms missing
from va/ and va_new/ are scored in one eval_va.py call and written to
va_new/<bench>_s<seed>__<n>.json. Incremental and safe to re-run.
  JOBS=3 run_va_new.py [arm ...]     default: every arm in the table
"""
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "artifacts" / "runs" / "current"
OUT = RUNS / "va_new"
OUT.mkdir(exist_ok=True)
PY = os.environ.get("TSSL_PYTHON", sys.executable)
only = set(sys.argv[1:])

done = defaultdict(set)   # (tag, seed) -> arms already scored
for d in ("va", "va_new"):
    for f in (RUNS / d).glob("*.json"):
        stem = f.stem.split("__")[0]
        if "_s" in stem and stem.rsplit("_s", 1)[1].isdigit():
            tag, seed = stem.rsplit("_s", 1)
            done[(tag, int(seed))] |= set(json.loads(f.read_text()))

todo = defaultdict(list)
for line in (RUNS / "mrt_matrix.tsv").read_text().splitlines():
    bench, arm, seed = line.split("\t")[:3]
    tag = bench.replace(":", "_")
    ck = RUNS / f"mrt_{tag}_{arm}_s{seed}" / "checkpoints" / "last.ckpt"
    if bench.startswith(("bf:", "bf4:")) or (only and arm not in only):
        continue
    if arm not in done[(tag, int(seed))] and ck.exists():
        todo[(bench, tag, int(seed))].append(f"{arm}={ck}")


def run(item):
    (bench, tag, seed), cks = item
    if bench == "fseval":
        ds = ["--dataset", "FSeval"]
    elif bench == "da":
        ds = ["--dataset", "desktop_assembly"]
    else:
        ds = ["--dataset", "YTI", "--activity", bench.split(":", 1)[1]]
    n = len(list(OUT.glob(f"{tag}_s{seed}__*.json")))
    out = OUT / f"{tag}_s{seed}__{n}.json"
    tmp = out.with_suffix(".tmp")
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with open(out.with_suffix(".log"), "w") as log:
        r = subprocess.run([PY, str(HERE / "eval_va.py"), *ds, "--ckpts", *cks, "--out", str(tmp)],
                           stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    if r.returncode == 0 and tmp.exists():
        res = json.loads(tmp.read_text()); res.pop("raw", None)
        out.write_text(json.dumps(res, indent=1)); tmp.unlink()
        return f"ok {tag} s{seed}: {' '.join(c.split('=')[0] for c in cks)}"
    return f"FAILED {tag} s{seed}"


with ThreadPoolExecutor(int(os.environ.get("JOBS", 3))) as ex:
    for msg in ex.map(run, sorted(todo.items(), key=lambda kv: (kv[0][2], kv[0][1]))):
        print(msg, flush=True)
