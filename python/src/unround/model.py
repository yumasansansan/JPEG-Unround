# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The model of one component without chroma subsampling (docs/math.md, 1.2, 4 and 6).

A Problem holds what the file says about the component: the interval of every
coefficient, the MMSE centres and the weights of the data term. G, the data term
with the quantization constraint, is separable in the coefficients, and so are
its proximal map (prox) and its conjugate (conjugate). The objectives of TV and
TGV, and the duality gaps that tell how far an iterate is from their least
value, are written with them.

Canvases hold the samples themselves, shape (H, W), and coefficients have the
shape (H // 8, W // 8, 8, 8) (unround.dct). JPEG's level shift, D(x - 128), moves
only the DC coefficient of a block, by 8 x 128 = 1024 exactly; it is added to the
DC intervals and centres, so that nothing is added to or taken from the samples,
and the result is the canvas itself (docs/math.md, 1.1). A primal point is kept as
its coefficients c, which lie in their intervals, with its canvas x = D^T c and,
for TGV, its vector field w; a dual point is the vector field p and, for TGV, the
tensor field r.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from unround import dct, laplace
from unround.operators import div, div2, grad, sym_grad, tensor_norms, vector_norms

__all__ = [
    "LEVEL_SHIFT_DC",
    "TGV",
    "TV",
    "Dual",
    "Primal",
    "Problem",
    "clip",
    "conjugate",
    "data_term",
    "excess",
    "make_problem",
    "project",
    "prox",
    "tgv_objective",
    "tgv_partial_gap",
    "tgv_values",
    "total_variation",
    "tv_objective",
    "tv_values",
]

type Array = npt.NDArray[np.float64]

_LEVEL_AXES: Final = 4  # block rows, block columns, v, u

LEVEL_SHIFT_DC: Final = 1024.0
"""What the level shift of 128 adds to the DC coefficient of a block: 128 times 8, exactly."""


@dataclass(frozen=True, slots=True, eq=False)
class Problem:
    """A component to reconstruct.

    lower and upper are the ends of the coefficients' intervals, and centres their MMSE
    centres, each of shape (rows, columns, 8, 8), for a canvas of samples that are not
    level-shifted: those of DC include LEVEL_SHIFT_DC. steps are the quantization steps Q
    and weights the weights of the data term, mu / Q^2 for AC and 0 for DC, each of shape
    (8, 8).
    """

    lower: Array
    upper: Array
    centres: Array
    steps: Array
    weights: Array

    @property
    def shape(self) -> tuple[int, int]:
        """The canvas: rows and columns of samples."""
        return int(self.lower.shape[0]) * dct.BLOCK, int(self.lower.shape[1]) * dct.BLOCK

    @property
    def samples(self) -> int:
        """The number of samples of the canvas."""
        return int(self.lower.shape[0] * self.lower.shape[1]) * dct.BLOCK * dct.BLOCK


@dataclass(frozen=True, slots=True)
class TV:
    """The weight of total variation (docs/math.md, 4.2)."""

    alpha: float = 1.0


@dataclass(frozen=True, slots=True)
class TGV:
    """The weights of second-order total generalized variation (docs/math.md, 4.3).

    alpha1 weights the first-order part, ||grad x - w||, and alpha0 the second, ||E w||.
    """

    alpha1: float = 1.0
    alpha0: float = 2.0


@dataclass(frozen=True, slots=True, eq=False)
class Primal:
    """A primal point: coefficients within their intervals, their canvas, and TGV's field w."""

    coefficients: Array
    canvas: Array
    w: Array | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Dual:
    """A dual point: the vector field p, and TGV's symmetric tensor field r."""

    p: Array
    r: Array | None = None


def make_problem(
    coefficients: npt.ArrayLike,
    quant_table: npt.ArrayLike,
    *,
    mu: float,
    slack: float = 0.0,
    scale: npt.ArrayLike | None = None,
) -> Problem:
    """The problem of a component with these quantized levels and quantization table.

    mu weights the data term, and slack widens every interval by that many steps on each
    side (docs/math.md, 1.2). scale is the Laplace scale of each frequency, which
    unround.laplace.scales() estimates unless it is given. The ends are
    ((q - 1/2) - slack) Q and ((q + 1/2) + slack) Q, in that order, and those of DC and its
    centre have LEVEL_SHIFT_DC added; without slack every one of them is exact.
    """
    if not mu >= 0.0 or not slack >= 0.0:
        message = f"mu and slack are at least 0, not {mu} and {slack}"
        raise ValueError(message)
    levels = np.asarray(coefficients, dtype=np.float64)
    steps = np.asarray(quant_table, dtype=np.float64)
    block = (dct.BLOCK, dct.BLOCK)
    if levels.ndim != _LEVEL_AXES or levels.shape[2:] != block or steps.shape != block:
        message = f"levels of shape (rows, columns, 8, 8) and a table of (8, 8), not {levels.shape} and {steps.shape}"
        raise ValueError(message)
    if not np.all(steps >= 1.0):
        message = "a quantization step is at least 1"
        raise ValueError(message)
    weights = mu / (steps * steps)
    weights[0, 0] = 0.0
    lower = ((levels - 0.5) - slack) * steps
    upper = ((levels + 0.5) + slack) * steps
    centres = laplace.centres(coefficients, quant_table, scale)
    for bound in (lower, upper, centres):
        bound[:, :, 0, 0] += LEVEL_SHIFT_DC
    return Problem(lower=lower, upper=upper, centres=centres, steps=steps, weights=weights)


def clip(problem: Problem, coefficients: Array) -> Array:
    """The coefficients clipped to their intervals."""
    return np.clip(coefficients, problem.lower, problem.upper)


def project(problem: Problem, canvas: Array) -> Array:
    """The projection of a canvas onto the quantization constraint set: D^T clip(D x)."""
    return dct.inverse(clip(problem, dct.forward(canvas)))


def excess(problem: Problem, coefficients: Array) -> Array:
    """How far each coefficient lies outside its interval, in steps (0 inside)."""
    below = np.maximum(problem.lower - coefficients, 0.0)
    above = np.maximum(coefficients - problem.upper, 0.0)
    return np.asarray((below + above) / problem.steps, dtype=np.float64)


def prox(problem: Problem, coefficients: Array, tau: float) -> Array:
    """The coefficients of prox_{tau G}(v), given those of v (docs/math.md, 4.1)."""
    scaled = tau * problem.weights
    return clip(problem, (coefficients + scaled * problem.centres) / (1.0 + scaled))


def data_term(problem: Problem, coefficients: Array) -> float:
    """G at coefficients within their intervals: (mu/2) times the sum of (c - centre)^2 / Q^2 over AC."""
    difference = coefficients - problem.centres
    return 0.5 * float(np.sum(problem.weights * difference * difference))


def conjugate(problem: Problem, coefficients: Array) -> float:
    """G*(xi), given the coefficients s = D xi (docs/math.md, 4.1)."""
    weights = np.broadcast_to(problem.weights, coefficients.shape)
    quadratic = weights > 0.0
    # Where the weight is 0, the largest of s c over the interval is at one of its ends.
    linear = np.maximum(coefficients * problem.lower, coefficients * problem.upper)
    m = np.where(quadratic, weights, 1.0)
    best = np.clip(problem.centres + coefficients / m, problem.lower, problem.upper)
    curved = coefficients * best - 0.5 * m * (best - problem.centres) ** 2
    return float(np.sum(np.where(quadratic, curved, linear)))


def total_variation(canvas: Array) -> float:
    """The isotropic total variation, ||grad x||_{2,1}."""
    return float(np.sum(vector_norms(grad(canvas))))


def tv_objective(problem: Problem, weights: TV, point: Primal) -> float:
    """P(x) of the TV model (docs/math.md, 4.2)."""
    return weights.alpha * total_variation(point.canvas) + data_term(problem, point.coefficients)


def tv_values(problem: Problem, weights: TV, point: Primal, dual: Dual) -> tuple[float, float]:
    """The primal and dual values of the TV model, for a dual p with |p| <= alpha (6.1).

    Their difference is the duality gap, which bounds how far P(x) is above its least value.
    """
    return tv_objective(problem, weights, point), -conjugate(problem, dct.forward(div(dual.p)))


def _field(point: Primal) -> Array:
    if point.w is None:
        message = "a primal point of TGV has its field w"
        raise ValueError(message)
    return point.w


def tgv_objective(problem: Problem, weights: TGV, point: Primal) -> float:
    """P(x, w) of the TGV model (docs/math.md, 4.3)."""
    w = _field(point)
    first = float(np.sum(vector_norms(grad(point.canvas) - w)))
    second = float(np.sum(tensor_norms(sym_grad(w))))
    return weights.alpha1 * first + weights.alpha0 * second + data_term(problem, point.coefficients)


def tgv_values(problem: Problem, weights: TGV, point: Primal, dual: Dual) -> tuple[float, float, float]:
    """The primal value, a dual value, and the scaling that made the dual feasible (6.2).

    The dual's r has |r|_F <= alpha0 (its p is not used). r is scaled by theta, the most
    that keeps |div2 r| within alpha1, and the dual is taken at p = -div2 (theta r), where
    it is finite.
    """
    if dual.r is None:
        message = "a dual point of TGV has its field r"
        raise ValueError(message)
    divergence = div2(dual.r)
    largest = float(np.max(vector_norms(divergence)))
    theta = 1.0 if largest <= weights.alpha1 else weights.alpha1 / largest
    dual_value = -conjugate(problem, dct.forward(div(-theta * divergence)))
    return tgv_objective(problem, weights, point), dual_value, theta


def tgv_partial_gap(problem: Problem, weights: TGV, point: Primal, dual: Dual, radius: float) -> float:
    """The gap restricted to |w| <= radius at every sample (docs/math.md, 6.3).

    It bounds how far P(x, w) is above its least value when the radius is at least the
    largest |w| of a solution.
    """
    if dual.r is None:
        message = "a dual point of TGV has its field r"
        raise ValueError(message)
    residual = float(np.sum(vector_norms(dual.p + div2(dual.r))))
    primal = tgv_objective(problem, weights, point)
    return primal + conjugate(problem, dct.forward(div(dual.p))) + radius * residual
