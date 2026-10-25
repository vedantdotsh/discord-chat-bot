"""Closed forms for the sums Aryabhata records in Gaṇitapada 19-22.

These are the same formulas used now. The useful comparison is not accuracy
(both are exact) but what a generic polynomial fit needs before it recovers
the sum-of-squares formula that the closed form states outright.
"""

from __future__ import annotations

import numpy as np


def sum_naturals(n: int) -> int:
    return n * (n + 1) // 2


def sum_squares(n: int) -> int:
    return n * (n + 1) * (2 * n + 1) // 6


def sum_cubes(n: int) -> int:
    triangular = n * (n + 1) // 2
    return triangular * triangular


def loop_sum(n: int, power: int) -> int:
    return sum(k**power for k in range(1, n + 1))


def fit_sum_squares(train_n: int, degree: int) -> np.ndarray:
    """Least squares of sum-of-squares on n=1..train_n as a degree-d polynomial.

    Columns are n, n^2, ..., n^degree. No intercept: the sum is 0 at n=0.
    """
    ns = np.arange(1, train_n + 1, dtype=float)
    targets = np.array([sum_squares(int(n)) for n in ns], dtype=float)
    columns = [ns**power for power in range(1, degree + 1)]
    design = np.stack(columns, axis=1)
    coef, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return coef


def predict_polynomial(n: int, coef: np.ndarray) -> float:
    value = float(n)
    return float(sum(coef[i] * value ** (i + 1) for i in range(len(coef))))
