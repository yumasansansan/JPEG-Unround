# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Laplace model, against 60-digit decimals: the scale, the shrinkage and the centres.

The derivations of docs/math.md, 2 are checked as identities at 60 digits, and the
implementation against them within bounds of rounding error (rounding.py).
"""

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import numpy.typing as npt
import pytest

import rounding
from unround import laplace

DIGITS = 60


def draws(scale: float, step: float, count: int, seed: int) -> npt.NDArray[np.int64]:
    values = np.random.default_rng(seed).laplace(0.0, scale, size=count)
    return np.rint(values / step).astype(np.int64)


def as_frequency(levels: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Levels of one frequency, as blocks of a single row: shape (1, n, 8, 8), all frequencies alike."""
    return np.broadcast_to(levels[np.newaxis, :, np.newaxis, np.newaxis], (1, levels.size, 8, 8)).copy()


def counts(levels: npt.NDArray[np.int64]) -> tuple[int, int, int]:
    """n0, n1, and S: the sum of 2|q| - 1 over the non-zero levels."""
    magnitudes = np.abs(levels)
    nonzero = magnitudes[magnitudes > 0]
    return int(np.count_nonzero(magnitudes == 0)), int(nonzero.size), int(np.sum(2 * nonzero - 1))


def log_likelihood(t: Decimal, levels: npt.NDArray[np.int64]) -> Decimal:
    """L(t) of docs/math.md, 2.1, with its constant n1 log(1/2)."""
    n0, n1, s = counts(levels)
    return n0 * (1 - t).ln() + s * t.ln() + n1 * (1 - t * t).ln() + n1 * Decimal("0.5").ln()


def exact_t(levels: npt.NDArray[np.int64]) -> Decimal:
    n0, n1, s = counts(levels)
    a = n0 + s + 2 * n1
    return 2 * Decimal(s) / (n0 + Decimal(n0 * n0 + 4 * a * s).sqrt())


CASES = [(3.0, 10.0), (10.0, 10.0), (40.0, 7.0), (1.0, 16.0), (200.0, 2.0), (5000.0, 1.0)]


@pytest.mark.parametrize(("scale", "step"), CASES)
def test_the_likelihood_is_that_of_the_bins(scale: float, step: float) -> None:
    # The sum of the logarithms of the bins' probabilities, 1 - t for q = 0 and
    # t^(2|q| - 1) (1 - t^2) / 2 for q != 0, is L(t).
    levels = draws(scale, step, 400, seed=int(scale * 10 + step))
    with localcontext() as context:
        context.prec = DIGITS
        t = Decimal("0.73")
        bins = sum(
            ((1 - t).ln() if q == 0 else (Decimal("0.5") * t ** (2 * abs(int(q)) - 1) * (1 - t * t)).ln())
            for q in levels
        )
        assert abs(bins - log_likelihood(t, levels)) < Decimal(10) ** -(DIGITS - 10)


@pytest.mark.parametrize(("scale", "step"), CASES)
def test_the_scale_is_the_maximum_of_the_likelihood(scale: float, step: float) -> None:
    levels = draws(scale, step, 5000, seed=int(scale * 10 + step))
    n0, n1, s = counts(levels)
    with localcontext() as context:
        context.prec = DIGITS
        t = exact_t(levels)
        # t is the root of the quadratic in (0, 1) ...
        a = n0 + s + 2 * n1
        assert 0 < t < 1
        assert abs(a * t * t + n0 * t - s) < Decimal(10) ** -(DIGITS - 10) * (a + n0 + s)
        # ... and the likelihood is less on either side of it.
        for factor in (Decimal("0.999999"), Decimal("1.000001")):
            assert log_likelihood(t * factor, levels) < log_likelihood(t, levels)
        exact = Decimal(step) / (-2 * t.ln())
    estimated = laplace.scales(as_frequency(levels), np.full((8, 8), step))[0, 1]
    # About ten roundings, none magnified (docs/math.md, 2.1): sixteen units of 2^-53 are taken.
    assert abs(Decimal(float(estimated)) - exact) <= Decimal(16 * rounding.U) * exact


@pytest.mark.parametrize(
    ("zeros", "nonzeros", "odd_sum", "step"),
    [
        (3_000_000, 1_000_000, 50_000_000_000, 1),  # D is about 2^73: beyond binary64's integers
        (4_000_000, 1, 1, 255),  # t near 0
        (1, 4_000_000, 60_000_000_000, 2),  # t near 1
        (0, 7, 7, 16),  # every level +-1
    ],
)
def test_the_scale_from_large_counts(zeros: int, nonzeros: int, odd_sum: int, step: int) -> None:
    # The counts, A and D are integers computed exactly, however large; D beyond 2^53 is where
    # floating point would already have rounded them.
    with localcontext() as context:
        context.prec = DIGITS
        a = zeros + odd_sum + 2 * nonzeros
        t = 2 * Decimal(odd_sum) / (zeros + Decimal(zeros * zeros + 4 * a * odd_sum).sqrt())
        exact = Decimal(step) / (-2 * t.ln())
    estimated = laplace.scale_from_counts(zeros, nonzeros, odd_sum, step)
    assert abs(Decimal(estimated) - exact) <= Decimal(16 * rounding.U) * exact


def test_the_scale_comes_near_the_true_one() -> None:
    # The estimate from 200 000 draws has a standard error of about 0.3 per cent here, and
    # 2 per cent is about seven of them.
    levels = draws(12.0, 8.0, 200_000, seed=5)
    estimated = laplace.scales(as_frequency(levels), np.full((8, 8), 8.0))[0, 1]
    assert abs(estimated - 12.0) <= 0.02 * 12.0


def test_all_zeros_have_the_scale_zero_and_dc_an_infinite_one() -> None:
    levels = np.zeros((2, 3, 8, 8), dtype=np.int64)
    levels[:, :, 0, 0] = 5
    result = laplace.scales(levels, np.full((8, 8), 10.0))
    assert result[0, 0] == np.inf
    assert np.all(result.ravel()[1:] == 0.0)


def exact_shrinkage(rho: float) -> Decimal:
    """1/2 - 1/rho + 1/(e^rho - 1) at 60 digits, for rho as the double it is."""
    with localcontext() as context:
        context.prec = DIGITS
        r = Decimal(rho)
        return Decimal("0.5") - 1 / r + 1 / (r.exp() - 1)


def test_the_series_is_that_of_the_shrinkage() -> None:
    # The coefficients B_2k / (2k)!: the series, summed in rationals to far more terms than the
    # implementation takes, meets the closed form at 40 digits.
    numbers = [Fraction(1)]
    for m in range(1, 41):
        numbers.append(-sum((math.comb(m + 1, j) * numbers[j] for j in range(m)), Fraction(0)) / (m + 1))
    rho = Fraction(1, 8)
    series = sum(numbers[2 * k] / math.factorial(2 * k) * rho ** (2 * k - 1) for k in range(1, 21))
    with localcontext() as context:
        context.prec = DIGITS
        value = Decimal(series.numerator) / Decimal(series.denominator)
        assert abs(value - exact_shrinkage(0.125)) < Decimal(10) ** -40


def test_the_shrinkage_is_within_a_few_units_of_its_exact_value() -> None:
    # The series below 1: a Horner sum of eleven terms, whose magnitudes sum to at most 0.09,
    # within gamma(22) of that. The closed form from 1 on: terms of at most 1, with exp and
    # expm1 within a few units in the last place. Within 16 units of 2^-53 everywhere.
    rho = np.concatenate((np.geomspace(1e-12, 1e4, 3001), [0.999999999, 1.0, 1.000000001, 745.0, 800.0]))
    values = laplace.shrinkage(rho)
    for r, value in zip(rho, values, strict=True):
        assert abs(Decimal(float(value)) - exact_shrinkage(float(r))) <= Decimal(16 * rounding.U), r
    assert laplace.shrinkage(0.0) == 0.0
    assert laplace.shrinkage(np.inf) == 0.5
    assert np.all(np.diff(values[:3001]) >= 0.0)


def mean_by_antiderivative(lower: Decimal, upper: Decimal, scale: Decimal) -> Decimal:
    """The mean of the density proportional to exp(-c / scale) on [lower, upper], 0 <= lower.

    From the antiderivatives: that of c e^(-c/b) is -b e^(-c/b) (c + b), and that of
    e^(-c/b) is -b e^(-c/b).
    """
    near, far = (-lower / scale).exp(), (-upper / scale).exp()
    return (near * (lower + scale) - far * (upper + scale)) / (near - far)


@pytest.mark.parametrize("scale", [0.5, 3.0, 11.0, 150.0, 1e5])
def test_the_centres_are_the_means_of_their_bins(scale: float) -> None:
    step = 12.0
    levels = np.zeros((1, 8, 8, 8), dtype=np.int64)
    levels[0, :, 0, 1] = [-3, -2, -1, 0, 1, 2, 5, 40]
    centres = laplace.centres(levels, np.full((8, 8), step), np.full((8, 8), scale))[0, :, 0, 1]
    for level, centre in zip(levels[0, :, 0, 1], centres, strict=True):
        magnitude = abs(int(level))
        if magnitude == 0:
            assert centre == 0.0  # the bin of 0 is symmetric
            continue
        with localcontext() as context:
            context.prec = DIGITS
            lower = (magnitude - Decimal("0.5")) * Decimal(step)
            upper = (magnitude + Decimal("0.5")) * Decimal(step)
            exact = mean_by_antiderivative(lower, upper, Decimal(scale))
            # The closed form of docs/math.md, 2.2 is that mean ...
            rho = Decimal(step) / Decimal(scale)
            closed = Decimal(step) * (magnitude - (Decimal("0.5") - 1 / rho + 1 / (rho.exp() - 1)))
            assert abs(closed - exact) < Decimal(10) ** -(DIGITS - 15) * upper
        # ... and the centre is within (2|q| + 17) U Q of it: the shrinkage within 16U, then a
        # subtraction from |q| and a product with Q, each rounding within U of its result.
        bound = Decimal((2 * magnitude + 17) * rounding.U * step)
        assert abs(Decimal(float(abs(centre))) - exact) <= bound, level
        assert (magnitude - 0.5) * step <= abs(centre) <= magnitude * step
        assert np.sign(centre) == np.sign(level)


def test_the_centres_of_zero_and_of_dc() -> None:
    levels = np.zeros((1, 2, 8, 8), dtype=np.int64)
    levels[0, :, 0, 0] = [-4, 9]
    levels[0, 1, 3, 3] = 2
    centres = laplace.centres(levels, np.full((8, 8), 5.0))
    np.testing.assert_array_equal(centres[0, :, 0, 0], [-20.0, 45.0])
    assert centres[0, 0, 3, 3] == 0.0
    # A frequency with a single non-zero level among two: its centre lies towards 0.
    assert 7.5 <= centres[0, 1, 3, 3] < 10.0
