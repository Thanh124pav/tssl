#!/usr/bin/env python3
"""Download the VAOT video-alignment datasets (Pouring, Penn Action).

Source: the pre-processed copies published by CARL (minghchen/CARL_code,
CVPR 2022) on Google Drive. They are the TCC splits and TCC phase labels
(`train.pkl` / `val.pkl` carry per-frame `frame_label`); the original TCC
bucket (storage.googleapis.com/tcc-pouring) returns HTTP 403.

gdown is used only to list the folder; its per-file endpoint gets throttled
after ~60 files, so files are fetched from drive.usercontent.google.com.
Resumable: re-running skips files that already exist.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gdown
import requests

FOLDERS = {
    "pouring": "1hvA4bDqPnxjiVM4c4mxm-UPOuO1lAVxW",
    "penn_action": "1hPbkKSSM5NoQKKzvAr2bHDp-pCHZpnVj",
}
URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"


def fetch(item, out, retries=6):
    dst = out / item.path
    if dst.exists() and dst.stat().st_size > 0:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    for attempt in range(retries):
        try:
            with requests.get(URL.format(item.id), stream=True, timeout=60) as r:
                ctype = r.headers.get("content-type", "")
                if r.status_code != 200 or "text/html" in ctype:
                    raise RuntimeError(f"status={r.status_code} ctype={ctype}")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.rename(dst)
            return True
        except Exception as e:  # noqa: BLE001 - retry any transport error
            err = e
            time.sleep(min(120, 5 * 2 ** attempt))
    print(f"FAILED {item.path}: {err}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--datasets", nargs="+", default=list(FOLDERS), choices=list(FOLDERS))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    bad = 0
    for name in args.datasets:
        out = args.dest / name
        out.mkdir(parents=True, exist_ok=True)
        items = gdown.download_folder(id=FOLDERS[name], output=str(out), quiet=True, skip_download=True)
        # "._*" entries are macOS resource forks, not data
        items = [i for i in items if not Path(i.path).name.startswith("._")]
        print(f"== {name}: {len(items)} files listed", flush=True)
        ok = []
        with ThreadPoolExecutor(args.threads) as ex:
            for n, r in enumerate(ex.map(lambda i: fetch(i, out), items), 1):
                ok.append(r)
                if n % 100 == 0:
                    print(f"   {name}: {n}/{len(items)}", flush=True)
        bad += ok.count(False)
        print(f"== {name}: {sum(ok)}/{len(items)} present", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
