"""Common selector interface. Future ACVCL selectors conform to this API."""
from __future__ import annotations
from typing import Protocol, Any


class PairSelector(Protocol):
    """Returns a list of (vid_a, vid_b) pairs for cross-video learning."""

    def select(self, *, model: Any = None, dataset: Any, num_pairs: int,
               rng: Any = None, **kwargs) -> list[tuple[str, str]]: ...
