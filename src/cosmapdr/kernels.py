# Author: Fenosoa  Randrianjatovo <RandrianjatovoFenosoa@gmail.com>
"""Low-dimensional kernel helpers for CosMAP."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit


def find_ab_(spread: float, min_dist: float) -> Tuple[float, float]:
    """Fit UMAP-style low-dimensional kernel parameters a and b."""
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return float(params[0]), float(params[1])


def find_ab__v1(spread: float, x_max: float = 1.0, num: int = 300, p0=(1.0, 1.0)) -> Tuple[float, float]:
    """Alternative fit kept for experimentation."""
    if spread <= 0:
        raise ValueError("spread temperature must be positive")

    def q(x, a, b):
        return 1.0 / (1.0 + a * np.power(x, 2.0 * b))

    x = np.linspace(0, spread * 3, num)
    y = np.exp(-(0.5 / spread) * x**2)
    (a_hat, b_hat), _ = curve_fit(q, x, y, p0=p0, bounds=(0, np.inf))
    return float(a_hat), float(b_hat)
