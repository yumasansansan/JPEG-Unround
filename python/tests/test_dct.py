# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The block DCT against the definition of the JPEG standard, computed in 70-digit decimals.

The tolerances are bounds of rounding error, derived in rounding.py.
"""

import io
from decimal import Decimal, localcontext

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

import rounding
from unround import dct, jpegio

# pi to 80 digits.
PI = Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986")
DIGITS = 70


def exact_cos(m: int) -> Decimal:
    """cos(pi m / 16) by its Taylor series, to about DIGITS digits."""
    with localcontext() as context:
        context.prec = DIGITS + 10
        x = PI * (m % 32) / 16
        term = total = Decimal(1)
        n = 0
        while abs(term) > Decimal(10) ** -(DIGITS + 5):
            n += 2
            term = -term * x * x / (n * (n - 1))
            total += term
        return +total


def exact_basis() -> list[list[Decimal]]:
    """The basis of the standard: C(k)/2 cos((2n + 1) k pi / 16), C(0) = 1/sqrt(2), C(k) = 1."""
    with localcontext() as context:
        context.prec = DIGITS
        first = (Decimal(1) / 8).sqrt()
        return [[first if k == 0 else exact_cos((2 * n + 1) * k) / 2 for n in range(8)] for k in range(8)]


EXACT = exact_basis()


def exact_forward(block: npt.NDArray[np.int64]) -> list[list[Decimal]]:
    """The DCT of the standard of an 8x8 block, F(v, u) = sum over y, x of B[v][y] B[u][x] f(y, x)."""
    with localcontext() as context:
        context.prec = DIGITS
        return [
            [
                sum((EXACT[v][y] * EXACT[u][x] * int(block[y, x]) for y in range(8) for x in range(8)), Decimal(0))
                for u in range(8)
            ]
            for v in range(8)
        ]


def exact_inverse(block: npt.NDArray[np.int64]) -> list[list[Decimal]]:
    """The inverse DCT of an 8x8 block of coefficients."""
    with localcontext() as context:
        context.prec = DIGITS
        return [
            [
                sum((EXACT[v][y] * EXACT[u][x] * int(block[v, u]) for v in range(8) for u in range(8)), Decimal(0))
                for x in range(8)
            ]
            for y in range(8)
        ]


def test_every_entry_of_the_basis_is_the_double_nearest_to_it() -> None:
    for k in range(8):
        for n in range(8):
            exact = EXACT[k][n]
            assert float(dct.BASIS[k, n]) == float(exact), (k, n)  # float() of a Decimal rounds to nearest


def test_the_basis_is_orthonormal() -> None:
    # Entries within U of their magnitude, and a dot product of 8 pairs whose magnitudes sum
    # to at most 1 (Cauchy-Schwarz): gamma(8) (1 + U)^2 for the sum, 2U + U^2 for the entries.
    u = rounding.U
    bound = rounding.gamma(8) * (1.0 + u) ** 2 + 2.0 * u + u * u
    assert np.abs(dct.BASIS @ dct.BASIS.T - np.eye(8)).max() <= bound
    assert np.abs(dct.BASIS.T @ dct.BASIS - np.eye(8)).max() <= bound


def sample_blocks() -> npt.NDArray[np.int64]:
    """Level-shifted blocks: random ones, and the extremes."""
    rng = np.random.default_rng(12)
    blocks = rng.integers(-128, 128, size=(1, 5, 8, 8))
    blocks[0, 2] = 127
    blocks[0, 3] = -128
    blocks[0, 4] = np.where(np.add.outer(np.arange(8), np.arange(8)) % 2 == 0, 127, -128)
    return blocks


def test_forward_is_the_dct_of_the_standard() -> None:
    blocks = sample_blocks()
    canvas = dct.from_blocks(blocks.astype(np.float64))
    result = dct.forward(canvas)
    bound = rounding.forward_error(canvas)
    for index in range(blocks.shape[1]):
        exact = exact_forward(blocks[0, index])
        for v in range(8):
            for u in range(8):
                difference = abs(Decimal(float(result[0, index, v, u])) - exact[v][u])
                assert difference <= Decimal(float(bound[0, index, v, u])), (index, v, u)


def test_inverse_is_the_inverse_dct_of_the_standard() -> None:
    rng = np.random.default_rng(13)
    table = np.array([[16, 11, 10, 16, 24, 40, 51, 61]] * 8) + np.arange(8)[:, np.newaxis]
    levels = rng.integers(-60, 61, size=(1, 3, 8, 8)) // (1 + np.add.outer(np.arange(8), np.arange(8)))
    coefficients = levels * table
    result = dct.inverse(coefficients.astype(np.float64))
    bound = rounding.inverse_error(coefficients)
    for index in range(coefficients.shape[1]):
        exact = exact_inverse(coefficients[0, index])
        for y in range(8):
            for x in range(8):
                difference = abs(Decimal(float(result[y, 8 * index + x])) - exact[y][x])
                assert difference <= Decimal(float(bound[y, 8 * index + x])), (index, y, x)


def test_inverse_undoes_forward() -> None:
    # The inverse of the computed coefficients rounds within inverse_error, and the forward
    # rounding reaches the samples through the exact inverse, at most |B^T| e |B|.
    rng = np.random.default_rng(14)
    canvas = rng.uniform(-128.0, 127.0, size=(16, 32))
    coefficients = dct.forward(canvas)
    carried = np.abs(dct.BASIS).T @ rounding.forward_error(canvas) @ np.abs(dct.BASIS)
    bound = rounding.inverse_error(coefficients) + dct.from_blocks(carried)
    assert np.all(np.abs(dct.inverse(coefficients) - canvas) <= bound)


def test_blocks_are_in_raster_order() -> None:
    canvas = np.arange(16 * 24, dtype=np.float64).reshape(16, 24)
    blocks = dct.blocks(canvas)
    assert blocks.shape == (2, 3, 8, 8)
    np.testing.assert_array_equal(blocks[1, 2], canvas[8:16, 16:24])
    np.testing.assert_array_equal(dct.from_blocks(blocks), canvas)


def test_a_canvas_is_whole_blocks() -> None:
    with pytest.raises(ValueError, match="whole blocks"):
        dct.blocks(np.zeros((12, 16)))


def test_the_coefficients_of_a_file_are_those_of_its_decoded_planes() -> None:
    # The convention on a file: the coefficients of its decoded plane, less 128, are q Q up to
    # libjpeg's inverse DCT and the rounding of the plane to 8 bits. That inverse DCT meets
    # IEEE 1180 (a mean square error of at most 0.06 per sample), and the rounding adds 1/12,
    # so the RMS is about sqrt(0.14) = 0.38 in blocks that are not clamped. This is a check of
    # the convention, not a bound: a transposed table, or a missing level shift, would make
    # the difference of the order of a step.
    y, x = np.mgrid[0:40, 0:56].astype(np.float64)
    samples = np.clip(128.0 + 50.0 * np.sin(x / 9.0) * np.cos(y / 7.0), 0.0, 255.0).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(samples).save(stream, format="JPEG", quality=50)
    data = stream.getvalue()
    component = jpegio.read(data).components[0]
    plane = jpegio.decode_planes(data).planes[0].astype(np.float64)
    assert plane.min() > 0.0  # no block is clamped
    assert plane.max() < 255.0
    difference = dct.forward(plane - 128.0) - component.coefficients * component.quant_table.astype(np.float64)
    assert np.sqrt(np.mean(difference**2)) < 0.5
    assert np.all(np.abs(difference) <= 0.5 * component.quant_table)
