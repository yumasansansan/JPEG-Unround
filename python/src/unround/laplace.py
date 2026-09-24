# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""A Laplace model of a component's AC coefficients (docs/math.md, 2).

For each frequency, the coefficients of all the component's blocks are taken as
draws from one Laplace distribution, whose scale is estimated from their bins by
maximum likelihood, in closed form. The MMSE centre of a coefficient is then its
conditional mean within its bin: the centre of its interval, moved towards 0.

Both are computed without cancellation, so that the scale is within a few units
in the last place of the maximum-likelihood estimate, and the shrinkage within a
few units of 2^-53 of its exact value (test_laplace checks both against 60-digit
decimals).
"""

import math
from fractions import Fraction
from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = ["centres", "scale_from_counts", "scales", "shrinkage"]

type Array = npt.NDArray[np.float64]

_SERIES_BELOW: Final = 1.0
_FROM_COMPLEMENT: Final = 0.5  # from this t on, log(1/t) is taken from 1 - t
_SERIES_TERMS: Final = 11


def _bernoulli(count: int) -> list[Fraction]:
    """B_0, ..., B_{count - 1}, exactly, from sum over j <= m of C(m + 1, j) B_j = 0."""
    numbers = [Fraction(1)]
    for m in range(1, count):
        numbers.append(-sum((math.comb(m + 1, j) * numbers[j] for j in range(m)), Fraction(0)) / (m + 1))
    return numbers


def _series_coefficients() -> tuple[float, ...]:
    # 1/2 - 1/rho + 1/(e^rho - 1) = sum over k >= 1 of B_2k / (2k)! rho^(2k - 1), which
    # converges for rho < 2 pi; below 1, eleven terms leave less than 2e-19.
    numbers = _bernoulli(2 * _SERIES_TERMS + 1)
    return tuple(float(numbers[2 * k] / math.factorial(2 * k)) for k in range(1, _SERIES_TERMS + 1))


_SERIES: Final = _series_coefficients()


def scale_from_counts(zeros: int, nonzeros: int, odd_sum: int, step: int) -> float:
    """The maximum-likelihood scale of one frequency, from the counts of its levels (docs/math.md, 2.1).

    zeros and nonzeros count the levels that are 0 and that are not, odd_sum is the sum of
    2|q| - 1 over the latter, and step is the quantization step. The counts, and A and D
    below, are integers and are computed exactly; each enters floating point once, rounded
    to the nearest double. The scale is 0 when every level is 0.
    """
    if odd_sum == 0:
        return 0.0
    quadratic = zeros + odd_sum + 2 * nonzeros  # A
    discriminant = zeros * zeros + 4 * quadratic * odd_sum  # D, up to about 2^80
    root = math.sqrt(float(discriminant))
    # t = exp(-Q / (2 scale)) is the root in (0, 1) of A t^2 + n0 t - S. log(1/t) is taken
    # from t itself below 1/2, and above it from 1 - t, which has a form without cancellation:
    # 8 S (n0 + n1) / ((sqrt(D) + 2S - n0) (n0 + sqrt(D))).
    t = float(2 * odd_sum) / (float(zeros) + root)
    if t < _FROM_COMPLEMENT:
        log_inverse = -math.log(t)
    else:
        numerator = float(8 * odd_sum * (zeros + nonzeros))
        complement = numerator / ((root + float(2 * odd_sum - zeros)) * (float(zeros) + root))
        log_inverse = -math.log1p(-complement)
    return float(step) / (2.0 * log_inverse)


def scales(coefficients: npt.ArrayLike, quant_table: npt.ArrayLike) -> Array:
    """The maximum-likelihood Laplace scale of each frequency, shape (8, 8), in coefficient units.

    coefficients are the quantized levels, shape (rows, columns, 8, 8), and quant_table
    the steps, shape (8, 8). The scale of a frequency whose levels are all 0 is 0. The DC
    coefficient follows no Laplace distribution; its scale is infinite, which makes its
    MMSE centre the centre of its interval.
    """
    levels = np.abs(np.asarray(coefficients, dtype=np.int64))
    table = np.asarray(quant_table, dtype=np.int64)
    zeros = np.count_nonzero(levels == 0, axis=(0, 1))
    sums = levels.sum(axis=(0, 1))  # at most 2^15 for each of fewer than 2^40 blocks: exact in int64
    result = np.empty((8, 8), dtype=np.float64)
    for v in range(8):
        for u in range(8):
            n0 = int(zeros[v, u])
            n1 = int(levels.shape[0] * levels.shape[1]) - n0
            result[v, u] = scale_from_counts(n0, n1, 2 * int(sums[v, u]) - n1, int(table[v, u]))
    result[0, 0] = np.inf
    return result


def shrinkage(rho: npt.ArrayLike) -> Array:
    """delta / Q = 1/2 - 1/rho + 1/(e^rho - 1): how far towards 0 an MMSE centre lies, in steps.

    rho is the step over the scale, from 0 (an infinite scale: delta is 0) to infinity (a
    scale of 0: delta is 1/2). Below 1 it is the Taylor series, whose terms the closed form
    would cancel; from 1 on, the closed form, whose terms are then at most 1.
    """
    rho = np.asarray(rho, dtype=np.float64)
    result = np.empty_like(rho)
    small = rho < _SERIES_BELOW
    r = rho[small]
    squared = r * r
    series = np.full_like(r, _SERIES[-1])
    for coefficient in reversed(_SERIES[:-1]):
        series = coefficient + squared * series
    result[small] = r * series
    r = rho[~small]
    # 1 / (e^r - 1) as e^-r / (1 - e^-r), which neither overflows nor divides by 0.
    result[~small] = 0.5 - 1.0 / r + np.exp(-r) / -np.expm1(-r)
    return result


def centres(coefficients: npt.ArrayLike, quant_table: npt.ArrayLike, scale: npt.ArrayLike | None = None) -> Array:
    """The MMSE centre of every coefficient, shape (rows, columns, 8, 8), in coefficient units.

    The centre of the level q with the step Q is sign(q) (|q| - delta / Q) Q, where delta
    comes from the scale of its frequency, which scales() estimates unless it is given.
    The level 0 has the centre 0, and the DC coefficient the centre of its interval, q Q.
    These are the coefficients of the level-shifted canvas; unround.model adds the level
    shift to DC.
    Each centre lies within its interval in floating point too: |q| - delta / Q rounds to
    within [|q| - 1/2, |q|], and so does its product with Q to within the interval.
    """
    levels = np.asarray(coefficients, dtype=np.float64)
    step = np.asarray(quant_table, dtype=np.float64)
    scale = scales(coefficients, quant_table) if scale is None else np.asarray(scale, dtype=np.float64)
    rho = np.full((8, 8), np.inf)
    positive = scale > 0.0
    rho[positive] = step[positive] / scale[positive]
    delta = shrinkage(rho)
    return np.asarray(np.sign(levels) * (np.abs(levels) - delta) * step, dtype=np.float64)
