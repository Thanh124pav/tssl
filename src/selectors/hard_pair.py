"""Hard-pair selector: high semantic similarity but high model disagreement.

For the toy setup we score candidates as
    hard_score = cos(feat_a, feat_b) - cos(pred_a, pred_b)
where `pred_*` is any model-derived representation (e.g. temporal head output).
"""
from __future__ import annotations
import random
import numpy as np


class HardPairSelector:
    name = "hard_pair"

    def __init__(self, video_features: dict[str, np.ndarray],
                 video_preds: dict[str, np.ndarray] | None = None):
        self.features = video_features
        self.preds = video_preds or video_features  # fallback: use features twice

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        x = x.mean(0) if x.ndim == 2 else x
        return x / (np.linalg.norm(x) + 1e-8)

    def _score(self, a: str, b: str) -> float:
        sim = float(self._unit(self.features[a]) @ self._unit(self.features[b]))
        dis = float(self._unit(self.preds[a]) @ self._unit(self.preds[b]))
        return sim - dis

    def select(self, *, dataset, num_pairs: int, model=None,
               rng: random.Random | None = None, **_) -> list[tuple[str, str]]:
        rng = rng or random.Random(0)
        tasks = [t for t, vs in dataset.by_task.items() if len(vs) >= 2]
        out: list[tuple[str, str]] = []
        while len(out) < num_pairs:
            t = rng.choice(tasks)
            vids = [v for v in dataset.by_task[t]
                    if v in self.features and v in self.preds]
            if len(vids) < 2:
                continue
            a = rng.choice(vids)
            best, best_s = None, -1e9
            for b in vids:
                if b == a:
                    continue
                s = self._score(a, b)
                if s > best_s:
                    best, best_s = b, s
            out.append((a, best))
        return out
