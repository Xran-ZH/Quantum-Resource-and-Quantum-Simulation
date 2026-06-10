from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int,
    rng: np.random.Generator,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> tuple[float, float, float]:
    """Bootstrap a statistic and return the statistic with asymmetric errors."""
    observed = float(statistic(data))
    bootstrap_biases = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.choice(data, len(data), replace=True)
        bootstrap_biases[index] = float(statistic(sample)) - observed

    lower, upper = np.quantile(bootstrap_biases, quantiles)
    return observed, -float(lower), float(upper)
