# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The 8x8 block DCT of JPEG: the orthonormal DCT-II of every block (docs/math.md, 1.1).

A component's canvas is an array of H x W samples, level-shifted by 128, where H
and W are multiples of 8. Its coefficients are an array of shape
(H // 8, W // 8, 8, 8): block rows from the top, blocks from the left, and each
block in natural order, so that [by, bx, v, u] is the coefficient of vertical
frequency v and horizontal frequency u. That is the layout of
unround.jpegio.Component.coefficients.
"""

from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = ["BASIS", "BLOCK", "COSINES", "blocks", "forward", "from_blocks", "inverse"]

type Array = npt.NDArray[np.float64]

BLOCK: Final = 8

# Angles in steps of pi / 16: a quarter turn, a half, three quarters, and the period.
_QUARTER: Final = 8
_HALF: Final = 16
_THREE_QUARTERS: Final = 24
_PERIOD: Final = 32


COSINES: Final = (
    1.0,
    0.9807852804032304491261822,
    0.9238795325112867561281832,
    0.8314696123025452370787884,
    0.7071067811865475244008444,
    0.5555702330196022247428308,
    0.3826834323650897717284600,
    0.1950903220161282678482849,
    0.0,
)
"""cos(pi j / 16) for j = 0, ..., 8, to 25 digits.

Each literal is read as the double nearest to it, which is the double nearest to the
cosine. np.cos would not give that: pi / 16 is rounded before the cosine is taken, and
near pi / 2 the cosine magnifies that rounding several times (2.3 units in the last
place at j = 7). The other two implementations use the same constants.
"""


def _basis() -> Array:
    # cos(pi m / 16) for m = (2n + 1) k, from m reduced in integers: over a period of 32,
    # [0, 8] is as it is, (8, 16] and (16, 24] are negated, and (24, 32) is mirrored.
    frequency = np.arange(BLOCK)[:, np.newaxis]
    position = np.arange(BLOCK)[np.newaxis, :]
    m = ((2 * position + 1) * frequency) % _PERIOD
    j = np.where(
        m <= _QUARTER,
        m,
        np.where(m <= _HALF, _HALF - m, np.where(m <= _THREE_QUARTERS, m - _HALF, _PERIOD - m)),
    )
    sign = np.where((m > _QUARTER) & (m <= _THREE_QUARTERS), -1.0, 1.0)
    # Halving is exact, and sqrt(1/8) correctly rounded: every entry is the double nearest
    # to the entry of the exact basis.
    basis = np.asarray(sign * np.asarray(COSINES)[j] / 2, dtype=np.float64)
    basis[0, :] = np.sqrt(1.0 / BLOCK)
    return basis


BASIS: Final = _basis()
"""The 8x8 matrix whose row k is the basis vector of frequency k, each entry correctly rounded."""


def blocks[T: np.generic](canvas: npt.NDArray[T]) -> npt.NDArray[T]:
    """The canvas as its blocks, shape (H // 8, W // 8, 8, 8), of whatever numbers it holds."""
    height, width = canvas.shape
    if height % BLOCK or width % BLOCK:
        message = f"a canvas is whole blocks, and {height} x {width} is not"
        raise ValueError(message)
    return canvas.reshape(height // BLOCK, BLOCK, width // BLOCK, BLOCK).transpose(0, 2, 1, 3)


def from_blocks[T: np.generic](block_array: npt.NDArray[T]) -> npt.NDArray[T]:
    """The canvas of blocks of shape (rows, columns, 8, 8)."""
    rows, columns = block_array.shape[:2]
    return np.ascontiguousarray(block_array.transpose(0, 2, 1, 3)).reshape(rows * BLOCK, columns * BLOCK)


def forward(canvas: Array) -> Array:
    """D: the coefficients of every block of the canvas."""
    return np.asarray(BASIS @ blocks(canvas) @ BASIS.T, dtype=np.float64)


def inverse(coefficients: Array) -> Array:
    """D transposed, the inverse DCT: the canvas whose blocks have these coefficients."""
    return from_blocks(np.asarray(BASIS.T @ coefficients @ BASIS, dtype=np.float64))
