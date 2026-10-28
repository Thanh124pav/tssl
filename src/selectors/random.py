"""Random same-task pair selector."""
from __future__ import annotations
import random


class RandomSameTaskSelector:
    name = "random"

    def select(self, *, dataset, num_pairs: int, model=None,
               rng: random.Random | None = None, **_) -> list[tuple[str, str]]:
        rng = rng or random.Random(0)
        return [dataset.sample_same_task_pair(rng=rng) for _ in range(num_pairs)]
