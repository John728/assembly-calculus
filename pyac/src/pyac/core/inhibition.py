from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def k_cap(
    input_vector: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(input_vector)
    if n == 0:
        raise ValueError("input_vector must be non-empty")
    if k <= 0:
        raise ValueError("k must be > 0")
    if k > n:
        raise ValueError("k must be <= len(input_vector)")

    jitter = rng.uniform(0, 1e-12, size=n)
    input_with_jitter = input_vector + jitter
    indices = np.argpartition(input_with_jitter, -k)[-k:]
    return np.sort(indices)


@dataclass
class InhibitionState:
    inhibited: dict[str, bool] = field(default_factory=dict)

    def inhibit(self, area_name: str) -> None:
        self.inhibited[area_name] = True

    def disinhibit(self, area_name: str) -> None:
        self.inhibited[area_name] = False

    def is_inhibited(self, area_name: str) -> bool:
        return self.inhibited.get(area_name, False)
