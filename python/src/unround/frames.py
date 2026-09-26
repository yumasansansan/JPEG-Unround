# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The components of a file on one canvas (docs/math.md, 1.3, 4.4 and 6.6).

A Frame holds the problem of every component, with the ratio of its cells to the
canvas, and the canvas's shape. Its unknowns are one canvas per component, stacked:
shape (C, H, W). The coefficients of a component are the DCT of the means of its
cells, over its blocks; what no coefficient constrains -- the deviations within the
cells and the samples beyond the blocks -- is free. One component whose canvas is
its blocks has nothing free, and is the problem of unround.model with a channel
axis: every function here then does what its counterpart there does, operation for
operation.

Fields put their entries first and the channels second, a vector field (2, C, H, W)
and a tensor field (3, C, H, W), so that unround.operators applies to them as it is.
The regularizers take the channels coupled, pixel by pixel, or each on its own, with
a weight for each channel's differences (the weights TV and TGV hold).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from unround import dct, model
from unround.model import TGV, TV, Dual, Problem
from unround.operators import div, div2, grad, sym_grad

__all__ = [
    "FREE_CENTRE",
    "FREE_RADIUS",
    "Channel",
    "Frame",
    "Primal",
    "channel_weights",
    "conjugate",
    "data_term",
    "excess",
    "make_frame",
    "means",
    "one",
    "project",
    "project_tensors",
    "project_vectors",
    "prox",
    "spread",
    "start",
    "sums",
    "tensor_norms",
    "tgv_objective",
    "tgv_residual",
    "tgv_values",
    "total",
    "tv_objective",
    "tv_values",
    "vector_norms",
]

type Array = npt.NDArray[np.float64]

FREE_CENTRE: Final = 128.0
"""Where the samples beyond the blocks start, and the centre of their box in 6.6."""

FREE_RADIUS: Final = 255.0
"""R of 6.6 by default: the box of a solution whose samples lie within [0, 255]."""

_BLOCK: Final = dct.BLOCK


@dataclass(frozen=True, slots=True, eq=False)
class Channel:
    """A component on the canvas: its problem, for its own samples, and the ratio of its cells.

    ratio is (r_v, r_h), the canvas's samples per sample of the component down and across.
    """

    problem: Problem
    ratio: tuple[int, int] = (1, 1)

    @property
    def cells(self) -> int:
        """n: the samples of the canvas in a cell."""
        return self.ratio[0] * self.ratio[1]

    @property
    def extent(self) -> tuple[int, int]:
        """The rows and columns of the canvas that the component's blocks cover."""
        rows, columns = self.problem.shape
        return rows * self.ratio[0], columns * self.ratio[1]


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """The components of a file, on one canvas of shape (H, W)."""

    channels: tuple[Channel, ...]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        """Refuses a frame whose cells do not tile the canvas, or whose blocks go beyond it."""
        if not self.channels:
            message = "a frame has a channel at least"
            raise ValueError(message)
        height, width = self.shape
        for channel in self.channels:
            down, across = channel.ratio
            rows, columns = channel.extent
            if down < 1 or across < 1 or height % down or width % across:
                message = f"cells of {down} x {across} do not tile a canvas of {height} x {width}"
                raise ValueError(message)
            if rows > height or columns > width:
                message = f"blocks over {rows} x {columns} samples go beyond a canvas of {height} x {width}"
                raise ValueError(message)

    @property
    def samples(self) -> int:
        """The samples of all the channels' canvases, C H W."""
        return len(self.channels) * self.shape[0] * self.shape[1]

    def free(self, channel: Channel) -> bool:
        """Whether some of the channel's samples are free: cells of several samples, or samples beyond its blocks."""
        return channel.cells > 1 or channel.extent != self.shape


@dataclass(frozen=True, slots=True, eq=False)
class Primal:
    """A primal point of a frame.

    coefficients are those of every component, each within its intervals; canvas is
    (C, H, W); w is TGV's field, (2, C, H, W).
    """

    coefficients: tuple[Array, ...]
    canvas: Array
    w: Array | None = None


def one(problem: Problem) -> Frame:
    """The frame of one component, whose canvas is its blocks: nothing is free."""
    return Frame(channels=(Channel(problem),), shape=problem.shape)


def _blocks(size: int, ratio: int) -> int:
    """ceil(size / (8 ratio)): a component's blocks, as libjpeg counts them."""
    return -(-size // (_BLOCK * ratio))


def make_frame(problems: Sequence[Problem], factors: Sequence[tuple[int, int]], size: tuple[int, int]) -> Frame:
    """The frame of a file's components (docs/math.md, 1.3).

    problems are the components' (unround.model.make_problem), factors their sampling
    factors (h, v), across and down, and size the picture's rows and columns. One
    component's canvas is its blocks. Several share the picture rounded up to whole
    MCUs, and each needs the largest factors to be whole multiples of its own.
    """
    if not problems or len(problems) != len(factors):
        message = f"a sampling factor for each of the components, not {len(factors)} for {len(problems)}"
        raise ValueError(message)
    if len(problems) == 1:
        return one(problems[0])
    rows, columns = size
    most_across = max(across for across, _ in factors)
    most_down = max(down for _, down in factors)
    channels = []
    for problem, (across, down) in zip(problems, factors, strict=True):
        if across < 1 or down < 1 or most_across % across or most_down % down:
            message = (
                f"the sampling factors {across} x {down} do not divide {most_across} x {most_down}: "
                "only whole cells are taken"
            )
            raise ValueError(message)
        ratio = (most_down // down, most_across // across)
        blocks = (_BLOCK * _blocks(rows, ratio[0]), _BLOCK * _blocks(columns, ratio[1]))
        if problem.shape != blocks:
            message = f"a component of {problem.shape} samples, where a picture of {size} has {blocks}"
            raise ValueError(message)
        channels.append(Channel(problem, ratio))
    shape = (_BLOCK * most_down * _blocks(rows, most_down), _BLOCK * most_across * _blocks(columns, most_across))
    return Frame(channels=tuple(channels), shape=shape)


def sums[T: (np.float64, np.object_)](channel: Channel, canvas: npt.NDArray[T]) -> npt.NDArray[T]:
    """The sums of the cells of a canvas over the component's blocks, nu^-1 E S x (docs/math.md, 4.4).

    With cells of one sample, the samples themselves (a view).
    """
    region = canvas[: channel.extent[0], : channel.extent[1]]
    if channel.cells == 1:
        return region
    rows, columns = channel.problem.shape
    down, across = channel.ratio
    return np.asarray(region.reshape(rows, down, columns, across).sum(axis=(1, 3)), dtype=canvas.dtype)


def means[T: (np.float64, np.object_)](channel: Channel, canvas: npt.NDArray[T]) -> npt.NDArray[T]:
    """E S x: the means of the cells of a canvas over the component's blocks, the component's samples."""
    if channel.cells == 1:
        return sums(channel, canvas)
    return np.asarray(sums(channel, canvas) / channel.cells, dtype=canvas.dtype)


def spread[T: (np.float64, np.object_)](frame: Frame, channel: Channel, samples: npt.NDArray[T]) -> npt.NDArray[T]:
    """The component's samples on the canvas: each repeated over its cell, and 0 beyond the blocks.

    This is nu^-1 S^T E^T: with the inverse DCT before it, nu^-1 A^T.
    """
    result = np.zeros(frame.shape, dtype=samples.dtype)
    down, across = channel.ratio
    rows, columns = channel.extent
    result[:rows, :columns] = np.repeat(np.repeat(samples, down, axis=0), across, axis=1)
    return result


def _put(frame: Frame, channel: Channel, canvas: Array, own: Array, coefficients: Array) -> Array:
    """v + nu^-1 A^T (zeta - A v), where own are the means of v's cells and zeta the coefficients."""
    if not frame.free(channel):
        return dct.inverse(coefficients)
    result = np.array(canvas, dtype=np.float64)
    rows, columns = channel.extent
    if channel.cells == 1:
        result[:rows, :columns] = dct.inverse(coefficients)
    else:
        down, across = channel.ratio
        change = dct.inverse(coefficients) - own
        result[:rows, :columns] += np.repeat(np.repeat(change, down, axis=0), across, axis=1)
    return result


def _check_canvas(frame: Frame, canvas: Array) -> None:
    if canvas.shape != (len(frame.channels), *frame.shape):
        message = f"a canvas of the frame is {(len(frame.channels), *frame.shape)}, not {canvas.shape}"
        raise ValueError(message)


def prox(frame: Frame, canvas: Array, tau: float) -> tuple[tuple[Array, ...], Array]:
    """prox_{tau G}(v) of a canvas (C, H, W): every component's coefficients, and the canvas (docs/math.md, 4.4).

    The coefficients take the proximal map of unround.model with the step tau / n, and
    the canvas moves by their change, repeated over the cells; its free samples stay.
    """
    _check_canvas(frame, canvas)
    coefficients = []
    result = np.empty_like(canvas)
    for index, channel in enumerate(frame.channels):
        own = means(channel, canvas[index])
        chosen = model.prox(channel.problem, dct.forward(own), tau / channel.cells)
        coefficients.append(chosen)
        result[index] = _put(frame, channel, canvas[index], own, chosen)
    return tuple(coefficients), result


def project(frame: Frame, canvas: Array) -> Array:
    """The projection of a canvas (C, H, W) onto the quantization constraint set (docs/math.md, 4.4)."""
    _check_canvas(frame, canvas)
    result = np.empty_like(canvas)
    for index, channel in enumerate(frame.channels):
        own = means(channel, canvas[index])
        result[index] = _put(frame, channel, canvas[index], own, model.clip(channel.problem, dct.forward(own)))
    return result


def excess(frame: Frame, canvas: Array) -> tuple[Array, ...]:
    """How far each component's coefficients of a canvas lie outside their intervals, in steps (0 inside)."""
    _check_canvas(frame, canvas)
    return tuple(
        model.excess(channel.problem, dct.forward(means(channel, canvas[index])))
        for index, channel in enumerate(frame.channels)
    )


def start(frame: Frame, coefficients: Sequence[Array] | None = None) -> Primal:
    """The starting point: these coefficients, or else the data term's centres, clipped to their intervals.

    Each component's canvas is the inverse DCT of its coefficients, repeated over the cells,
    and FREE_CENTRE beyond its blocks (docs/math.md, 5).
    """
    if coefficients is not None and len(coefficients) != len(frame.channels):
        message = f"coefficients to start from for each of the {len(frame.channels)} components"
        raise ValueError(message)
    chosen = []
    canvas = np.empty((len(frame.channels), *frame.shape))
    for index, channel in enumerate(frame.channels):
        problem = channel.problem
        if coefficients is None:
            clipped = model.clip(problem, problem.centres)
        else:
            given = np.asarray(coefficients[index], dtype=np.float64)
            if given.shape != problem.lower.shape or not np.all(np.isfinite(given)):
                message = f"the coefficients to start from are finite, of shape {problem.lower.shape}"
                raise ValueError(message)
            clipped = model.clip(problem, given)
        chosen.append(clipped)
        if frame.free(channel):
            samples = spread(frame, channel, dct.inverse(clipped))
            rows, columns = channel.extent
            beyond = np.ones(frame.shape, dtype=bool)
            beyond[:rows, :columns] = False
            samples[beyond] = FREE_CENTRE
            canvas[index] = samples
        else:
            canvas[index] = dct.inverse(clipped)
    return Primal(coefficients=tuple(chosen), canvas=canvas)


def data_term(frame: Frame, coefficients: Sequence[Array]) -> float:
    """G at coefficients within their intervals: the sum of the components' data terms."""
    value = model.data_term(frame.channels[0].problem, coefficients[0])
    for channel, own in zip(frame.channels[1:], coefficients[1:], strict=True):
        value += model.data_term(channel.problem, own)
    return value


def _conjugate(frame: Frame, channel: Channel, xi: Array, radius: float) -> float:
    value = model.conjugate(channel.problem, dct.forward(sums(channel, xi)))
    if not frame.free(channel):
        return value
    zeta = xi - spread(frame, channel, means(channel, xi))
    beyond = np.ones(frame.shape, dtype=bool)
    beyond[: channel.extent[0], : channel.extent[1]] = False
    return value + (FREE_CENTRE * float(np.sum(zeta[beyond])) + radius * float(np.sum(np.abs(zeta))))


def conjugate(frame: Frame, xi: Array, radius: float = FREE_RADIUS) -> float:
    """G*(xi) of a canvas xi (C, H, W), bounded over the box of the radius where samples are free (docs/math.md, 6.6).

    For each component, g*(nu^-1 A xi), and where it has free samples, the most that
    <zeta, x - Pi x> reaches over the box: <zeta, m> + radius ||zeta||_1, zeta = xi - Pi xi.
    Where nothing is free this is G* itself.
    """
    _check_canvas(frame, xi)
    value = _conjugate(frame, frame.channels[0], xi[0], radius)
    for index, channel in enumerate(frame.channels[1:], start=1):
        value += _conjugate(frame, channel, xi[index], radius)
    return value


def channel_weights(frame: Frame, weights: tuple[float, ...] | None) -> Array:
    """The gamma of every channel, shape (C, 1, 1): 1 for each where None."""
    count = len(frame.channels)
    values = (1.0,) * count if weights is None else weights
    if len(values) != count or not all(0.0 < value < math.inf for value in values):
        message = f"a positive, finite weight for each of the {count} channels, not {values}"
        raise ValueError(message)
    return np.asarray(values, dtype=np.float64).reshape(count, 1, 1)


def vector_norms(q: Array, *, coupled: bool) -> Array:
    """The norm of a vector field of channels (2, C, H, W) at each pixel: (H, W) coupled, (C, H, W) not."""
    squares = q[0] * q[0] + q[1] * q[1]
    if not coupled:
        return np.asarray(np.sqrt(squares), dtype=np.float64)
    added = squares[0]
    for channel in squares[1:]:
        added = added + channel
    return np.asarray(np.sqrt(added), dtype=np.float64)


def tensor_norms(r: Array, *, coupled: bool) -> Array:
    """The Frobenius norm of a tensor field of channels (3, C, H, W), the off-diagonal entry counted twice."""
    squares = r[0] * r[0] + r[1] * r[1] + 2.0 * r[2] * r[2]
    if not coupled:
        return np.asarray(np.sqrt(squares), dtype=np.float64)
    added = squares[0]
    for channel in squares[1:]:
        added = added + channel
    return np.asarray(np.sqrt(added), dtype=np.float64)


def total(norms: Array) -> float:
    """The sum of the norms of every pixel, and of every channel where they are taken apart."""
    if norms.ndim == 2:  # noqa: PLR2004
        return float(np.sum(norms))
    value = float(np.sum(norms[0]))
    for channel in norms[1:]:
        value += float(np.sum(channel))
    return value


def project_vectors(p: Array, radius: float, *, coupled: bool) -> Array:
    """p projected onto the ball of the radius at every pixel, over all its channels where coupled."""
    return np.asarray(p / np.maximum(1.0, vector_norms(p, coupled=coupled) / radius), dtype=np.float64)


def project_tensors(r: Array, radius: float, *, coupled: bool) -> Array:
    """r projected onto the Frobenius ball of the radius at every pixel, over all its channels where coupled."""
    return np.asarray(r / np.maximum(1.0, tensor_norms(r, coupled=coupled) / radius), dtype=np.float64)


def tv_objective(frame: Frame, weights: TV, point: Primal) -> float:
    """P(x) of the TV model of several channels (docs/math.md, 4.4)."""
    gammas = channel_weights(frame, weights.channel_weights)
    variation = total(vector_norms(gammas * grad(point.canvas), coupled=weights.coupled))
    return weights.alpha * variation + data_term(frame, point.coefficients)


def tv_values(frame: Frame, weights: TV, point: Primal, dual: Dual, radius: float = FREE_RADIUS) -> tuple[float, float]:
    """The primal value, and the dual value for a p within the dual balls (docs/math.md, 6.1 and 6.6).

    Their difference is the gap, or where samples are free, the partial gap of the radius.
    """
    gammas = channel_weights(frame, weights.channel_weights)
    return tv_objective(frame, weights, point), -conjugate(frame, gammas * div(dual.p), radius)


def _field(point: Primal) -> Array:
    if point.w is None:
        message = "a primal point of TGV has its field w"
        raise ValueError(message)
    return point.w


def tgv_objective(frame: Frame, weights: TGV, point: Primal) -> float:
    """P(x, w) of the TGV model of several channels (docs/math.md, 4.4)."""
    gammas = channel_weights(frame, weights.channel_weights)
    w = _field(point)
    first = total(vector_norms(gammas * (grad(point.canvas) - w), coupled=weights.coupled))
    second = total(tensor_norms(gammas * sym_grad(w), coupled=weights.coupled))
    return weights.alpha1 * first + weights.alpha0 * second + data_term(frame, point.coefficients)


def tgv_values(
    frame: Frame, weights: TGV, point: Primal, dual: Dual, radius: float = FREE_RADIUS
) -> tuple[float, float, float]:
    """The primal value, a dual value, and the scaling that made the dual feasible (docs/math.md, 6.2 and 6.6).

    r is scaled by theta, the most that keeps div2 r within alpha1 in the norm of the
    channels, and the dual is taken at p = -div2 (theta r).
    """
    if dual.r is None:
        message = "a dual point of TGV has its field r"
        raise ValueError(message)
    gammas = channel_weights(frame, weights.channel_weights)
    divergence = div2(dual.r)
    largest = float(np.max(vector_norms(divergence, coupled=weights.coupled)))
    theta = 1.0 if largest <= weights.alpha1 else weights.alpha1 / largest
    dual_value = -conjugate(frame, gammas * div(-theta * divergence), radius)
    return tgv_objective(frame, weights, point), dual_value, theta


def tgv_residual(frame: Frame, weights: TGV, dual: Dual) -> float:
    """sum over the channels of gamma_c ||p_c + div2 r_c||_{2,1}: what the partial gap of 6.3 adds, over its radius.

    The partial gap for |w_c| <= radius at every pixel is P(x, w) + G*(gamma div p) (the
    box of free samples in it, 6.6) plus the radius times this (docs/math.md, 6.3 and 6.6).
    """
    if dual.r is None:
        message = "a dual point of TGV has its field r"
        raise ValueError(message)
    gammas = channel_weights(frame, weights.channel_weights)
    return total(vector_norms(gammas * (dual.p + div2(dual.r)), coupled=False))
