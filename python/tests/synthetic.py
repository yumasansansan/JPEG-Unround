# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small problems for the tests: pictures quantized by the DCT here, with fixed seeds."""

import numpy as np
import numpy.typing as npt

from unround import dct
from unround.model import DataTerm, Problem, make_problem

type Array = npt.NDArray[np.float64]

# The luminance table of the JPEG standard (Annex K), at quality 50, in natural order.
ANNEX_K: npt.NDArray[np.int64] = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.int64,
)


def picture(rows: int, columns: int, seed: int) -> Array:
    """A canvas of samples about 128: a ramp, a bright disc with a sharp edge, and a little noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:columns].astype(np.float64)
    ramp = 90.0 * (x / max(columns - 1, 1)) - 60.0 * (y / max(rows - 1, 1))
    centre_y, centre_x = rng.uniform(0.3, 0.7) * rows, rng.uniform(0.3, 0.7) * columns
    disc = np.where((y - centre_y) ** 2 + (x - centre_x) ** 2 < (0.25 * min(rows, columns)) ** 2, 70.0, 0.0)
    return np.asarray(ramp + disc + rng.normal(0.0, 2.0, size=(rows, columns)) + 108.0, dtype=np.float64)


def levels(canvas: Array, table: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """The quantized levels of a canvas of samples, level-shifted and rounded to the nearest as an encoder would."""
    return np.rint(dct.forward(canvas - 128.0) / np.asarray(table, dtype=np.float64)).astype(np.int64)


def problem(
    rows: int = 16, columns: int = 24, seed: int = 1, *, scale: float = 2.0, data: DataTerm | None = None
) -> tuple[Problem, Array]:
    """A problem, and the canvas whose quantization it is, with the Annex K table times scale.

    data are the options of G, DataTerm() unless given.
    """
    canvas = picture(rows, columns, seed)
    table = np.maximum(np.rint(ANNEX_K * scale), 1).astype(np.int64)
    return make_problem(levels(canvas, table), table, data), canvas
