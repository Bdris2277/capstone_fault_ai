from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class PerUnitState:
    baseline_sum: np.ndarray | None = None
    baseline_count: int = 0

    def update_and_get_baseline(self, sample: np.ndarray, *, baseline_samples: int) -> np.ndarray | None:
        if self.baseline_sum is None:
            self.baseline_sum = np.zeros_like(sample, dtype=np.float64)

        if self.baseline_count < baseline_samples:
            self.baseline_sum += sample
            self.baseline_count += 1
            if self.baseline_count < baseline_samples:
                return None

        return (self.baseline_sum / max(self.baseline_count, 1)).astype(np.float64)


def select_features(raw_values: Iterable[float], *, drop_indices_0_based: list[int]) -> np.ndarray:
    arr = np.asarray(list(raw_values), dtype=np.float64)
    if drop_indices_0_based:
        arr = np.delete(arr, drop_indices_0_based)
    return arr


def to_per_unit(values: np.ndarray, *, per_unit_feature_count: int, baseline: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    count = max(0, min(per_unit_feature_count, result.shape[0]))
    if count == 0:
        return result

    denom = baseline[:count]
    denom = np.where(denom == 0, 1.0, denom)
    result[:count] = result[:count] / denom
    return result

