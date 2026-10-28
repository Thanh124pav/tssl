"""Lightweight loader for the COIN annotations JSON.

The annotations are shipped as a single `COIN.json` with:
  database[video_id] = {
    "recipe_type": int,     # task id
    "class": str,           # task name (e.g., "PutOnHairExtensions")
    "subset": "training" | "testing",
    "video_url": str,
    "start": float, "end": float, "duration": float,
    "annotation": [ {id, segment:[s,e], label}, ... ]
  }

This module does NOT decode videos. It only exposes task/video indices
suitable for cross-video pair selection (random / same-task).
"""
from __future__ import annotations
import json, random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class CoinVideo:
    vid: str
    task_id: int
    task_name: str
    subset: str
    duration: float
    steps: list  # list of (start, end, label)


class CoinIndex:
    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)
        data = json.loads(self.json_path.read_text())
        self.db = data["database"]
        self.videos: dict[str, CoinVideo] = {}
        self.by_task: dict[int, list[str]] = defaultdict(list)
        for vid, meta in self.db.items():
            cv = CoinVideo(
                vid=vid,
                task_id=int(meta["recipe_type"]),
                task_name=meta["class"],
                subset=meta["subset"],
                duration=float(meta.get("duration", 0.0)),
                steps=[(float(a["segment"][0]), float(a["segment"][1]), a["label"])
                        for a in meta.get("annotation", [])],
            )
            self.videos[vid] = cv
            self.by_task[cv.task_id].append(vid)

    @property
    def num_videos(self) -> int: return len(self.videos)

    @property
    def num_tasks(self) -> int: return len(self.by_task)

    def task_names(self) -> dict[int, str]:
        return {t: self.videos[vs[0]].task_name for t, vs in self.by_task.items() if vs}

    def sample_same_task_pair(self, rng: random.Random | None = None,
                              min_task_size: int = 2) -> tuple[str, str]:
        rng = rng or random
        tasks = [t for t, vs in self.by_task.items() if len(vs) >= min_task_size]
        t = rng.choice(tasks)
        a, b = rng.sample(self.by_task[t], 2)
        return a, b

    def iter_subset(self, subset: str) -> Iterable[CoinVideo]:
        for v in self.videos.values():
            if v.subset == subset:
                yield v
