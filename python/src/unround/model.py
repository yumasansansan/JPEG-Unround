# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The model of one component without chroma subsampling (docs/math.md, 1.2, 4 and 6).

A Problem holds what the file says about the component: the interval of every
coefficient, and the centres and the weights of the data term. G, the data term
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

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from unround import dct, laplace
from unround.operators import div, div2, grad, sym_grad, tensor_norms, vector_norms

__all__ = [
    "LEVEL_SHIFT_DC",
    "TGV",
    "TV",
    "Centres",
    "DataTerm",
    "Dual",
    "Primal",
    "Problem",
    "beyond",
    "clip",
    "conjugate",
    "data_term",
    "excess",
    "make_problem",
    "project",
    "prox",
    "rule_mu",
    "tgv_objective",
    "tgv_partial_gap",
    "tgv_values",
    "total_variation",
    "tv_objective",
    "tv_values",
]

type Array = npt.NDArray[np.float64]
type Centres = Literal["mmse", "midpoint"]
"""The centres of the data term: the MMSE centres of the Laplace model, or the middles of the intervals."""

_LEVEL_AXES: Final = 4  # block rows, block columns, v, u

LEVEL_SHIFT_DC: Final = 1024.0
"""What the level shift of 128 adds to the DC coefficient of a block: 128 times 8, exactly."""


@dataclass(frozen=True, slots=True, eq=False)
class Problem:
    """A component to reconstruct.

    lower and upper are the ends of the coefficients' intervals, and centres the centres
    of the data term, each of shape (rows, columns, 8, 8), for a canvas of samples that are
    not level-shifted: those of DC include LEVEL_SHIFT_DC. steps are the quantization steps
    Q and weights the weights of the data term, each of shape (8, 8): mu / Q^p for AC, and
    that times the weight of DC (0 by default) for DC.

    With a slack that has a cost, lower and upper are the ends widened by the slack,
    inner_lower and inner_upper those of the file's own intervals, and costs, of shape
    (8, 8), what G charges for each unit a coefficient lies beyond them (docs/math.md, 1.2
    and 4.1); otherwise the three are None.
    """

    lower: Array
    upper: Array
    centres: Array
    steps: Array
    weights: Array
    inner_lower: Array | None = None
    inner_upper: Array | None = None
    costs: Array | None = None

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
    """The weight of total variation (docs/math.md, 4.2), and how it takes several channels (4.4).

    channel_weights are the gamma of the channels' differences, 1 for each where None;
    coupled takes the channels together, pixel by pixel, or each on its own. Neither
    matters to one channel.
    """

    alpha: float = 1.0
    channel_weights: tuple[float, ...] | None = None
    coupled: bool = True


@dataclass(frozen=True, slots=True)
class TGV:
    """The weights of second-order total generalized variation (docs/math.md, 4.3 and 4.4).

    alpha1 weights the first-order part, ||grad x - w||, and alpha0 the second, ||E w||.
    channel_weights and coupled are those of TV.
    """

    alpha1: float = 1.0
    alpha0: float = 2.0
    channel_weights: tuple[float, ...] | None = None
    coupled: bool = True


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


@dataclass(frozen=True, slots=True)
class DataTerm:
    """The options of G: the weights and the centres of the data term, and the intervals' slack.

    mu weights the data term; where it is None, it is mu_scale times the mean of the
    component's 64 steps to the power mu_power, which follows the quantization. slack widens
    every interval by that many steps on each side (docs/math.md, 1.2); slack_cost is what
    leaving the file's own interval costs, within the slack, per step (0: nothing). The
    weight of an AC coefficient is mu / Q^power, and that of DC dc_weight times it (4.1).
    centres are the data term's: "mmse", the MMSE centres of the Laplace model (2.2), or
    "midpoint", the middles of the intervals, q Q.
    """

    mu: float | None = 1e-3
    mu_scale: float = 1.0
    mu_power: float = 1.0
    slack: float = 0.0
    dc_weight: float = 0.0
    centres: Centres = "mmse"
    power: float = 2.0
    slack_cost: float = 0.0


def _powers(steps: Array, power: float) -> Array:
    """Q^power: exact for the powers 0, 1 and 2 of steps of 16 bits, and correctly rounded or near it otherwise."""
    if power == 2.0:  # noqa: PLR2004
        return np.asarray(steps * steps, dtype=np.float64)
    if power == 1.0:
        return np.asarray(steps, dtype=np.float64)
    return np.asarray(np.power(steps, power), dtype=np.float64)


def rule_mu(data: DataTerm, quant_table: npt.ArrayLike) -> float:
    """mu of the data term: data.mu, or where it is None, mu_scale times the mean step to the power mu_power.

    The mean of the 64 steps is exact: their sum is an integer below 2^22, and dividing it
    by 64 is exact. The power is 1 by default, which rounds nothing more.
    """
    if data.mu is not None:
        return data.mu
    steps = np.asarray(quant_table, dtype=np.float64)
    mean = float(np.sum(steps)) / float(steps.size)
    return data.mu_scale * (mean if data.mu_power == 1.0 else math.pow(mean, data.mu_power))


def make_problem(
    coefficients: npt.ArrayLike,
    quant_table: npt.ArrayLike,
    data: DataTerm | None = None,
    *,
    scale: npt.ArrayLike | None = None,
) -> Problem:
    """The problem of a component with these quantized levels and quantization table.

    data are the options of G, DataTerm() unless given. scale is the Laplace scale of each
    frequency for the MMSE centres, which unround.laplace.scales() estimates unless it is
    given. The ends are ((q - 1/2) - slack) Q and ((q + 1/2) + slack) Q, in that order, and
    those of DC and its centre have LEVEL_SHIFT_DC added; without slack every one of them
    is exact, and so is every middle.
    """
    data = DataTerm() if data is None else data
    if not (0.0 <= data.mu_scale < math.inf and -math.inf < data.mu_power < math.inf):
        message = (
            f"the scale of mu is at least 0 and finite, and its power finite, not {data.mu_scale} and {data.mu_power}"
        )
        raise ValueError(message)
    mu, slack, dc_weight = rule_mu(data, quant_table), data.slack, data.dc_weight
    if not (0.0 <= mu < math.inf and 0.0 <= slack < math.inf and 0.0 <= dc_weight < math.inf):
        message = f"mu, slack and the weight of DC are at least 0 and finite, not {mu}, {slack} and {dc_weight}"
        raise ValueError(message)
    if not 0.0 <= data.power < math.inf:
        message = f"the power of the steps in the weights is at least 0 and finite, not {data.power}"
        raise ValueError(message)
    if not 0.0 <= data.slack_cost < math.inf:
        message = f"the cost of the slack is at least 0 and finite, not {data.slack_cost}"
        raise ValueError(message)
    if data.centres not in {"mmse", "midpoint"}:
        message = f"the centres are mmse or midpoint, not {data.centres!r}"
        raise ValueError(message)
    if data.centres == "midpoint" and scale is not None:
        message = "a Laplace scale is for the MMSE centres, not the middles"
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
    weights = mu / _powers(steps, data.power)
    weights[0, 0] *= dc_weight
    lower = ((levels - 0.5) - slack) * steps
    upper = ((levels + 0.5) + slack) * steps
    # The middles q Q are exact: |q| <= 2^15 and Q < 2^16.
    centres = laplace.centres(coefficients, quant_table, scale) if data.centres == "mmse" else levels * steps
    for bound in (lower, upper, centres):
        bound[:, :, 0, 0] += LEVEL_SHIFT_DC
    if slack == 0.0 or data.slack_cost == 0.0:
        return Problem(lower=lower, upper=upper, centres=centres, steps=steps, weights=weights)
    inner_lower = (levels - 0.5) * steps
    inner_upper = (levels + 0.5) * steps
    for bound in (inner_lower, inner_upper):
        bound[:, :, 0, 0] += LEVEL_SHIFT_DC
    return Problem(
        lower=lower,
        upper=upper,
        centres=centres,
        steps=steps,
        weights=weights,
        inner_lower=inner_lower,
        inner_upper=inner_upper,
        costs=data.slack_cost / steps,
    )


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


def _inner(problem: Problem) -> tuple[Array, Array, Array]:
    if problem.inner_lower is None or problem.inner_upper is None or problem.costs is None:
        message = "a problem whose slack has a cost has its inner intervals and costs"
        raise ValueError(message)
    return problem.inner_lower, problem.inner_upper, problem.costs


def beyond(problem: Problem, coefficients: Array) -> Array:
    """How far each coefficient lies beyond the file's own interval (0 within it), in coefficient units."""
    if problem.costs is None:
        return np.asarray(
            np.maximum(problem.lower - coefficients, 0.0) + np.maximum(coefficients - problem.upper, 0.0),
            dtype=np.float64,
        )
    inner_lower, inner_upper, _ = _inner(problem)
    return np.asarray(
        np.maximum(inner_lower - coefficients, 0.0) + np.maximum(coefficients - inner_upper, 0.0), dtype=np.float64
    )


def prox(problem: Problem, coefficients: Array, tau: float) -> Array:
    """The coefficients of prox_{tau G}(v), given those of v (docs/math.md, 4.1).

    Where the slack has a cost, the minimizer of the quadratic that lies beyond the file's
    own interval moves back towards it by tau times the cost over 1 + tau weight, but not
    past its end; then it is clipped to the interval widened by the slack.
    """
    scaled = tau * problem.weights
    moved = (coefficients + scaled * problem.centres) / (1.0 + scaled)
    if problem.costs is None:
        return clip(problem, moved)
    inner_lower, inner_upper, costs = _inner(problem)
    shrink = tau * costs / (1.0 + scaled)
    above = np.maximum(inner_upper, moved - shrink)
    below = np.minimum(inner_lower, moved + shrink)
    return clip(problem, np.where(moved > inner_upper, above, np.where(moved < inner_lower, below, moved)))


def data_term(problem: Problem, coefficients: Array) -> float:
    """G at coefficients within their intervals: half the sum of weight (c - centre)^2, and the slack's cost."""
    difference = coefficients - problem.centres
    value = 0.5 * float(np.sum(problem.weights * difference * difference))
    if problem.costs is not None:
        value += float(np.sum(problem.costs * beyond(problem, coefficients)))
    return value


def conjugate(problem: Problem, coefficients: Array) -> float:
    """G*(xi), given the coefficients s = D xi (docs/math.md, 4.1)."""
    weights = np.broadcast_to(problem.weights, coefficients.shape)
    quadratic = weights > 0.0
    m = np.where(quadratic, weights, 1.0)
    if problem.costs is None:
        # Where the weight is 0, the largest of s c over the interval is at one of its ends.
        linear = np.maximum(coefficients * problem.lower, coefficients * problem.upper)
        best = np.clip(problem.centres + coefficients / m, problem.lower, problem.upper)
        curved = coefficients * best - 0.5 * m * (best - problem.centres) ** 2
        return float(np.sum(np.where(quadratic, curved, linear)))
    inner_lower, inner_upper, costs = _inner(problem)
    cost = np.broadcast_to(costs, coefficients.shape)
    # The largest of s c - (m/2) (c - centre)^2 - cost dist(c, inner) over the widened
    # interval: at the quadratic's own top where that is within the inner interval; beyond
    # it, at the top of the piece with the cost, but not before the inner end; with no
    # weight, at the end of the piece whose slope keeps its sign.
    top = problem.centres + coefficients / m
    up = np.minimum(np.maximum(problem.centres + (coefficients - cost) / m, inner_upper), problem.upper)
    down = np.maximum(np.minimum(problem.centres + (coefficients + cost) / m, inner_lower), problem.lower)
    curved = np.where(top > inner_upper, up, np.where(top < inner_lower, down, top))
    straight = np.where(
        coefficients > cost,
        problem.upper,
        np.where(
            coefficients > 0.0,
            inner_upper,
            np.where(coefficients < -cost, problem.lower, inner_lower),
        ),
    )
    best = np.where(quadratic, curved, straight)
    charged = cost * beyond(problem, best)
    value = coefficients * best - 0.5 * weights * (best - problem.centres) ** 2 - charged
    return float(np.sum(value))


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
