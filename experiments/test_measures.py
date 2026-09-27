# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The measures of the experiments: PSNR, PSNR-B, 8-bit output, and consistency with the intervals.

    python -m pytest -c python/pyproject.toml experiments/test_measures.py

For pictures of integers the means are rationals, and are checked exactly; the
logarithms against 50-digit decimals, within a few units in the last place. The
intervals are those of a small file as the reference implementation reads it, through
the package's binding (UNROUND_LIBRARY).
"""

import io
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
from PIL import Image

import measures
from unround import native
from unround.settings import DataTerm, Settings

U = 2.0**-53


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
    assert measures.mse(reference, image) == exact / reference.size


def test_psnr() -> None:
    reference = np.full((4, 4), 100, dtype=np.uint8)
    assert measures.psnr(reference, reference) == math.inf
    image = reference.copy()
    image[0, 0] += 16  # a mean squared error of 16
    # 10 log10(65025 / 16): the ratio is a rational, rounded once, and log10 within a few units.
    value = measures.psnr(reference, image)
    assert abs(Decimal(value) - exact_psnr(Fraction(255**2, 16))) <= Decimal(4 * U * value)
    with pytest.raises(ValueError, match="compared"):
        measures.psnr(reference, image[:3])


def test_the_blocking_effect_factor_of_a_known_picture() -> None:
    # Steps of 8 between blocks across, none within them: 16 x 16 samples, one edge column and
    # one edge row. The edge pairs are 16 across that differ by 8 and 16 down that do not:
    # a mean of 1024 / 32 = 32 against 0, times log2(8) / log2(16) = 3/4: exactly 24.
    picture = np.zeros((16, 16), dtype=np.uint8)
    picture[:, 8:] = 8
    assert measures.blocking_effect_factor(picture) == 24.0
    assert measures.blocking_effect_factor(picture.astype(np.float64)) == 24.0


def test_no_blocking_in_a_smooth_picture() -> None:
    # Differences alike across and down, at the edges of blocks and within them.
    y, x = np.mgrid[0:32, 0:40]
    assert measures.blocking_effect_factor(x + y) == 0.0
    assert measures.blocking_effect_factor(np.zeros((5, 5), dtype=np.uint8)) == 0.0


def test_psnr_b_is_at_most_psnr() -> None:
    rng = np.random.default_rng(40)
    reference = rng.integers(0, 256, size=(24, 32))
    blocks = np.kron(rng.integers(-9, 10, size=(3, 4)), np.ones((8, 8), dtype=np.int64))
    blocky = np.clip(reference + blocks, 0, 255)
    assert measures.blocking_effect_factor(blocky) > 0.0
    assert measures.psnr_b(reference, blocky) < measures.psnr(reference, blocky)


def test_quantize_rounds_half_away_from_zero_exactly_and_clamps() -> None:
    samples = np.array([-3.0, 0.49999999999999994, 0.5, 1.5, 2.5, 254.49999999999997, 254.5, 300.0])
    np.testing.assert_array_equal(measures.quantize(samples), [0, 0, 1, 2, 3, 254, 255, 255])
    with pytest.raises(ValueError, match="finite"):
        measures.quantize(np.array([1.0, np.nan]))


def middles(height: int, width: int) -> native.Decoded:
    """A greyscale file of a ramp and noise, decoded to the middles of its intervals."""
    rng = np.random.default_rng(41)
    y, x = np.mgrid[0:height, 0:width]
    samples = np.clip(4.0 * x - 3.0 * y + 100.0 + rng.normal(0.0, 4.0, size=(height, width)), 0.0, 255.0)
    stream = io.BytesIO()
    Image.fromarray(samples.astype(np.uint8)).save(stream, format="JPEG", quality=40)
    return native.decode(stream.getvalue(), Settings(method="mmse", data=DataTerm(centres="midpoint")))


def test_consistency() -> None:
    # The middles lie half a step from the ends of their intervals, and the DCT of their
    # canvas, rounded both ways, far less than that from them: every coefficient is inside.
    decoded = middles(16, 24)
    (problem,) = decoded.problems
    share, largest = measures.consistency(problem, decoded.canvas[0])
    assert share == 1.0
    assert largest == 0.0
    spikes = decoded.canvas[0] + np.kron(np.ones((2, 3)), np.eye(8) * 40.0)
    share, largest = measures.consistency(problem, spikes)
    assert share < 1.0
    assert largest > 0.0


def test_the_excess_of_a_picture_pads_it_as_an_encoder_would() -> None:
    decoded = middles(16, 24)  # a canvas of 16 x 24
    (problem,) = decoded.problems
    canvas = decoded.canvas[0]
    # Whole blocks: the picture is the canvas.
    whole = measures.picture_excess(problem, canvas)
    np.testing.assert_array_equal(whole, measures.excess(problem, measures.block_dct(canvas)))
    # Cut: the last row and column repeated to the canvas.
    cut = canvas[:13, :20]
    padded = np.pad(cut, ((0, 3), (0, 4)), mode="edge")
    expected = measures.excess(problem, measures.block_dct(padded))
    np.testing.assert_array_equal(measures.picture_excess(problem, cut), expected)
    for shape in ((17, 24), (16, 25), (0, 8), (16,)):
        with pytest.raises(ValueError, match="does not fit"):
            measures.picture_excess(problem, np.zeros(shape))
