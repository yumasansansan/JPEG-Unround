# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""How good a reconstruction is: PSNR, PSNR-B, 8-bit samples, and how much of it lies in the intervals.

The measures of the experiments, in NumPy. PSNR-B is Yim and Bovik's PSNR with a
blocking effect factor (IEEE Transactions on Image Processing 20(1), 2011): the mean
squared error is increased by how much more the differences across block edges are
than those within blocks.

Pictures of integers -- 8-bit output, say -- have sums of squared differences that
are integers, and means that are rationals: those are computed exactly, and enter
floating point once, rounded to the nearest double, where a logarithm or the factor
of the BEF needs it. Pictures of floating-point samples are summed in binary64.

The coefficients of a picture are its block DCT by SciPy's orthonormal DCT-II, which
is the DCT of JPEG (docs/math.md, 1.1); their intervals are those of a component's
model as the reference implementation made it (unround.native.Problem), the canvas
not level-shifted, and DC's with 1024 added.
"""

import math
from fractions import Fraction
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
from scipy.fft import dctn

if TYPE_CHECKING:
    from unround.native import Problem

__all__ = [
    "block_dct",
    "blocking_effect_factor",
    "consistency",
    "excess",
    "mse",
    "picture_excess",
    "psnr",
    "psnr_b",
    "quantize",
]

type Array = npt.NDArray[np.float64]

BLOCK: Final = 8
_HALF: Final = 0.5
_AXES: Final = 2  # a picture's: height and width


def _integers(*arrays: npt.NDArray[np.generic]) -> bool:
    return all(np.issubdtype(array.dtype, np.integer) for array in arrays)


def _mean(differences: npt.NDArray[np.generic]) -> Fraction | float:
    """The mean of the squares: exactly, for integers; in binary64 otherwise."""
    if differences.size == 0:
        message = "no samples to take a mean of"
        raise ValueError(message)
    if np.issubdtype(differences.dtype, np.integer):
        wide = differences.astype(np.int64)  # squares of differences of 16-bit samples, summed: < 2^63
        return Fraction(int(np.sum(wide * wide)), differences.size)
    values = differences.astype(np.float64)
    return float(np.sum(values * values)) / differences.size


def mse(reference: npt.ArrayLike, image: npt.ArrayLike) -> Fraction | float:
    """The mean squared error: a Fraction for pictures of integers, a float otherwise."""
    first, second = np.asarray(reference), np.asarray(image)
    if first.shape != second.shape:
        message = f"pictures of {first.shape} and {second.shape} are not compared"
        raise ValueError(message)
    if _integers(first, second):
        return _mean(second.astype(np.int64) - first.astype(np.int64))
    return _mean(second.astype(np.float64) - first.astype(np.float64))


def _psnr_of(error: Fraction | float, peak: float) -> float:
    if error == 0:
        return math.inf
    if isinstance(error, Fraction) and float(peak).is_integer():
        return 10.0 * math.log10(float(Fraction(int(peak) ** 2) / error))
    return 10.0 * math.log10(peak * peak / float(error))


def psnr(reference: npt.ArrayLike, image: npt.ArrayLike, peak: float = 255.0) -> float:
    """The peak signal-to-noise ratio of image against reference, in dB."""
    return _psnr_of(mse(reference, image), peak)


def blocking_effect_factor(image: npt.ArrayLike, block: int = BLOCK) -> float:
    """Yim and Bovik's BEF of a greyscale picture whose blocks start at its top-left corner.

    The mean squared difference of neighbours across block edges, less that of neighbours
    within blocks, pooled across and down, times log2(block) / log2(min(height, width)),
    or 0 where it is not more.
    """
    samples = np.asarray(image)
    if not np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float64)
    height, width = int(samples.shape[0]), int(samples.shape[1])
    wide = samples.astype(np.int64) if np.issubdtype(samples.dtype, np.integer) else samples
    across = np.diff(wide, axis=1)  # pairs (i, j), (i, j + 1)
    down = np.diff(wide, axis=0)  # pairs (i, j), (i + 1, j)
    at_edge_across = (np.arange(width - 1) % block) == block - 1
    at_edge_down = (np.arange(height - 1) % block) == block - 1
    edges_across, edges_down = across[:, at_edge_across].ravel(), down[at_edge_down, :].ravel()
    inner_across, inner_down = across[:, ~at_edge_across].ravel(), down[~at_edge_down, :].ravel()
    if edges_across.size + edges_down.size == 0 or inner_across.size + inner_down.size == 0:
        return 0.0
    edges = _mean(np.concatenate((edges_across, edges_down)))
    inner = _mean(np.concatenate((inner_across, inner_down)))
    if edges <= inner:
        return 0.0
    return math.log2(block) / math.log2(min(height, width)) * float(edges - inner)


def psnr_b(reference: npt.ArrayLike, image: npt.ArrayLike, block: int = BLOCK, peak: float = 255.0) -> float:
    """PSNR-B of image against reference, in dB: the PSNR of the MSE plus the image's BEF."""
    error = mse(reference, image)
    blocking = blocking_effect_factor(image, block)
    if blocking == 0.0:
        return _psnr_of(error, peak)
    return _psnr_of(float(error) + blocking, peak)


def quantize(picture: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """The picture as 8-bit samples: rounded to the nearest, half away from zero, and clamped to 0-255.

    The fraction is compared with 1/2 exactly: floor(x + 1/2) would round
    0.49999999999999994 up, since the sum rounds to 1.
    """
    samples = np.asarray(picture, dtype=np.float64)
    if not np.all(np.isfinite(samples)):
        message = "a sample to quantize is not finite"
        raise ValueError(message)
    magnitude = np.abs(samples)
    whole = np.floor(magnitude)
    fraction = magnitude - whole  # exact: whole <= magnitude < whole + 1 (Sterbenz)
    rounded = np.sign(samples) * (whole + (fraction >= _HALF))
    return np.asarray(np.clip(rounded, 0.0, 255.0), dtype=np.uint8)


def block_dct(samples: npt.ArrayLike) -> Array:
    """The DCT of every 8 x 8 block of samples of whole blocks, (rows, columns, 8, 8)."""
    values = np.asarray(samples, dtype=np.float64)
    height, width = values.shape
    if height % BLOCK or width % BLOCK:
        message = f"samples of {values.shape} are not whole blocks"
        raise ValueError(message)
    blocks = values.reshape(height // BLOCK, BLOCK, width // BLOCK, BLOCK).transpose(0, 2, 1, 3)
    return np.asarray(dctn(blocks, axes=(2, 3), norm="ortho"), dtype=np.float64)


def excess(problem: Problem, coefficients: npt.ArrayLike) -> Array:
    """How far each coefficient lies outside its interval, in steps of its frequency (0 inside)."""
    values = np.asarray(coefficients, dtype=np.float64)
    beyond = np.maximum(np.maximum(problem.lower - values, values - problem.upper), 0.0)
    return np.asarray(beyond / problem.steps, dtype=np.float64)


def picture_excess(problem: Problem, picture: npt.ArrayLike) -> Array:
    """How far each coefficient of a picture lies outside its interval, in steps (0 inside).

    picture has the samples of one component, (height, width), no larger than its blocks.
    It is padded to them as libjpeg's encoder pads a component, repeating the last column
    and row: the coefficients are those that a file that encoded it would hold. Where the
    picture is not whole blocks, the blocks at its right and bottom edges hold the repeated
    samples, not the canvas's own.
    """
    samples = np.asarray(picture, dtype=np.float64)
    rows, columns = BLOCK * problem.lower.shape[0], BLOCK * problem.lower.shape[1]
    if samples.ndim != _AXES or not (0 < samples.shape[0] <= rows and 0 < samples.shape[1] <= columns):
        message = f"a picture of {samples.shape} does not fit blocks over {(rows, columns)}"
        raise ValueError(message)
    padded = np.pad(samples, ((0, rows - samples.shape[0]), (0, columns - samples.shape[1])), mode="edge")
    return excess(problem, block_dct(padded))


def consistency(problem: Problem, picture: npt.ArrayLike) -> tuple[float, float]:
    """The share of coefficients within their intervals, and the largest excess in steps.

    The picture is padded as picture_excess() pads it, and so this tells whether a file that
    encoded it would be read as the same one.
    """
    outside = picture_excess(problem, picture)
    return float(np.count_nonzero(outside == 0.0)) / outside.size, float(outside.max())
