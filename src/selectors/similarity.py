"""Nearest-neighbour pair selector using pre-pooled video features."""
from __future__ import annotations
import random
import numpy as np


class SimilaritySelector:
    """Picks top-k nearest neighbours (cosine) within the same task."""
    name = "similarity"

    def __init__(self, video_features: dict[str, np.ndarray]):
        self.features = video_features

    def _vec(self, vid: str) -> np.ndarray:
        v = self.features[vid]
        v = v.mean(0) if v.ndim == 2 else v
        return v / (np.linalg.norm(v) + 1e-8)

    def select(self, *, dataset, num_pairs: int, model=None,
               rng: random.Random | None = None, **_) -> list[tuple[str, str]]:
        rng = rng or random.Random(0)
        tasks = [t for t, vs in dataset.by_task.items() if len(vs) >= 2]
        out: list[tuple[str, str]] = []
        while len(out) < num_pairs:
            t = rng.choice(tasks)
            vids = [v for v in dataset.by_task[t] if v in self.features]
            if len(vids) < 2:
                continue
            a = rng.choice(vids)
            va = self._vec(a)
            best, best_s = None, -1.0
            for b in vids:
                if b == a:
                    continue
                s = float(va @ self._vec(b))
                if s > best_s:
                    best, best_s = b, s
            out.append((a, best))
        return out
