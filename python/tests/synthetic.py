# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small problems for the tests: pictures quantized by the DCT here, with fixed seeds."""

import numpy as np
import numpy.typing as npt

from unround import colour, dct, frames
from unround.model import DataTerm, Problem, make_problem

type Array = npt.NDArray[np.float64]

# The chrominance table of the JPEG standard (Annex K), at quality 50, in natural order.
ANNEX_K_CHROMA: npt.NDArray[np.int64] = np.array(
    [[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99], [24, 26, 56, 99, 99, 99, 99, 99]]
    + [[47, 66, 99, 99, 99, 99, 99, 99]]
    + [[99] * 8] * 4,
    dtype=np.int64,
)

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


def colour_picture(rows: int, columns: int, seed: int) -> Array:
    """An RGB picture, (rows, columns, 3), about 128: ramps of their own in each channel, a disc
    of another colour with a sharp edge, and a little noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:columns].astype(np.float64)
    across, down = x / max(columns - 1, 1), y / max(rows - 1, 1)
    ramps = np.stack((90.0 * across - 40.0 * down, 60.0 * down - 30.0 * across, 50.0 * (across + down)), axis=-1)
    centre_y, centre_x = rng.uniform(0.3, 0.7) * rows, rng.uniform(0.3, 0.7) * columns
    inside = (y - centre_y) ** 2 + (x - centre_x) ** 2 < (0.25 * min(rows, columns)) ** 2
    disc = np.where(inside[:, :, np.newaxis], np.array([70.0, -50.0, 20.0]), 0.0)
    noise = rng.normal(0.0, 2.0, size=(rows, columns, 3))
    return np.asarray(ramps + disc + noise + 100.0, dtype=np.float64)


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


def colour_frame(
    rows: int = 20,
    columns: int = 30,
    seed: int = 1,
    *,
    ratio: tuple[int, int] = (2, 2),
    data: DataTerm | None = None,
) -> tuple[frames.Frame, Array]:
    """A frame of Y, Cb and Cr quantized as an encoder would, and the YCbCr canvas (3, H, W) it came from.

    The picture of rows x columns is converted to YCbCr and filled out to the canvas by
    repeating its last row and column; the chroma, with cells of ratio (down, across), are
    the means of their cells. Y takes the luminance table of Annex K times 2, and the chroma
    the chrominance table times 2. Every level is the nearest to its coefficient, so the
    canvas itself is in the constraint set.
    """
    down, across = ratio
    luma = (across, down)  # the sampling factors, (h, v)
    factors = [luma, (1, 1), (1, 1)]
    height = dct.BLOCK * down * -(-rows // (dct.BLOCK * down))
    width = dct.BLOCK * across * -(-columns // (dct.BLOCK * across))
    planes = colour.to_ycbcr(colour_picture(rows, columns, seed))
    canvas = np.pad(planes, ((0, 0), (0, height - rows), (0, width - columns)), mode="edge")
    problems = []
    for index, cells in enumerate([(1, 1), ratio, ratio]):
        table = np.maximum(np.rint((ANNEX_K if index == 0 else ANNEX_K_CHROMA) * 2.0), 1).astype(np.int64)
        blocks = (
            dct.BLOCK * -(-rows // (dct.BLOCK * cells[0])),
            dct.BLOCK * -(-columns // (dct.BLOCK * cells[1])),
        )
        shrunk = canvas[index].reshape(height // cells[0], cells[0], width // cells[1], cells[1]).mean(axis=(1, 3))
        problems.append(make_problem(levels(shrunk[: blocks[0], : blocks[1]], table), table, data))
    return frames.make_frame(problems, factors, (rows, columns)), canvas
