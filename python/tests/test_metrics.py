# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""PSNR, PSNR-B, 8-bit output, and consistency with the intervals.

For pictures of integers the means are rationals, and are checked exactly; the
logarithms against 50-digit decimals, within a few units in the last place.
"""

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

import rounding
import synthetic
from unround import dct, metrics


def exact_psnr(ratio: Fraction) -> Decimal:
    """10 log10 of a rational, at 50 digits."""
    with localcontext() as context:
        context.prec = 50
        return 10 * (Decimal(ratio.numerator) / Decimal(ratio.denominator)).log10()


def test_the_mse_of_integers_is_exact() -> None:
    rng = np.random.default_rng(39)
    reference = rng.integers(0, 256, size=(13, 17), dtype=np.uint8)
    image = rng.integers(0, 256, size=(13, 17), dtype=np.uint8)
    exact = sum(Fraction((int(a) - int(b)) ** 2) for a, b in zip(image.ravel(), reference.ravel(), strict=True))
    assert metrics.mse(reference, image) == exact / reference.size


def test_psnr() -> None:
    reference = np.full((4, 4), 100, dtype=np.uint8)
    assert metrics.psnr(reference, reference) == math.inf
    image = reference.copy()
    image[0, 0] += 16  # a mean squared error of 16
    # 10 log10(65025 / 16): the ratio is a rational, rounded once, and log10 within a few units.
    value = metrics.psnr(reference, image)
    assert abs(Decimal(value) - exact_psnr(Fraction(255**2, 16))) <= Decimal(4 * rounding.U * value)
    with pytest.raises(ValueError, match="compared"):
        metrics.psnr(reference, image[:3])


def test_the_blocking_effect_factor_of_a_known_picture() -> None:
    # Steps of 8 between blocks across, none within them: 16 x 16 samples, one edge column and
    # one edge row. The edge pairs are 16 across that differ by 8 and 16 down that do not:
    # a mean of 1024 / 32 = 32 against 0, times log2(8) / log2(16) = 3/4: exactly 24.
    picture = np.zeros((16, 16), dtype=np.uint8)
    picture[:, 8:] = 8
    assert metrics.blocking_effect_factor(picture) == 24.0
    assert metrics.blocking_effect_factor(picture.astype(np.float64)) == 24.0


def test_no_blocking_in_a_smooth_picture() -> None:
    # Differences alike across and down, at the edges of blocks and within them.
    y, x = np.mgrid[0:32, 0:40]
    assert metrics.blocking_effect_factor(x + y) == 0.0
    assert metrics.blocking_effect_factor(np.zeros((5, 5), dtype=np.uint8)) == 0.0


def test_psnr_b_is_at_most_psnr() -> None:
    rng = np.random.default_rng(40)
    reference = rng.integers(0, 256, size=(24, 32))
    blocks = np.kron(rng.integers(-9, 10, size=(3, 4)), np.ones((8, 8), dtype=np.int64))
    blocky = np.clip(reference + blocks, 0, 255)
    assert metrics.blocking_effect_factor(blocky) > 0.0
    assert metrics.psnr_b(reference, blocky) < metrics.psnr(reference, blocky)


def test_quantize_rounds_half_away_from_zero_exactly_and_clamps() -> None:
    samples = np.array([-3.0, 0.49999999999999994, 0.5, 1.5, 2.5, 254.49999999999997, 254.5, 300.0])
    np.testing.assert_array_equal(metrics.quantize(samples), [0, 0, 1, 2, 3, 254, 255, 255])
    with pytest.raises(ValueError, match="finite"):
        metrics.quantize(np.array([1.0, np.nan]))


def test_consistency() -> None:
    problem, canvas = synthetic.problem(seed=41)
    centres = dct.inverse(problem.centres) + 128.0
    # The centres lie within their intervals exactly, and the picture of their canvas is
    # within the round trip of the DCT, and the rounding of the level shift, of them. That is
    # far less than the distance of any centre from the ends of its interval here, and so
    # every coefficient of the picture is inside.
    distance = np.minimum(problem.centres - problem.lower, problem.upper - problem.centres)
    shift = dct.blocks(np.full(problem.shape, 2.0 * 128.0 * rounding.U))
    reach = rounding.roundtrip_error(problem.centres) + np.abs(dct.BASIS) @ shift @ np.abs(dct.BASIS).T
    assert np.all(distance > reach)
    share, largest = metrics.consistency(problem, centres)
    assert share == 1.0
    assert largest == 0.0
    share, largest = metrics.consistency(problem, centres + np.kron(np.ones((2, 3)), np.eye(8) * 40.0))
    assert share < 1.0
    assert largest > 0.0
    assert canvas.shape == problem.shape
