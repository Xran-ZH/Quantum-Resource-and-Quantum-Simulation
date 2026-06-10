from __future__ import annotations

import numpy as np


def kurtosis_statistic(sample: np.ndarray) -> float:
    """Return the fourth central moment divided by variance squared."""
    mean = np.mean(sample)
    variance = np.var(sample)
    if variance == 0:
        return float("nan")
    return float(np.mean((sample - mean) ** 4) / variance**2)


def r_squared_linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return linear-fit coefficients and coefficient of determination."""
    coefficients = np.polyfit(x, y, 1)
    fit = np.poly1d(coefficients)
    r_squared = float(np.corrcoef(y, fit(x))[0, 1] ** 2)
    return coefficients, r_squared
