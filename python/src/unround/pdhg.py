# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The primal-dual hybrid gradient method of Chambolle and Pock for TV and TGV (docs/math.md, 5 and 6).

Each iterate's canvas is the output of the proximal map of G, and so lies in the
quantization constraint set: its coefficients are kept, and are the result. The
solvers stop at the first record where a tolerance of the options is met (the
duality gap per sample, the gap relative to the primal value, or TGV's partial
gap per sample), or after the most iterations the options allow. Every value that
shapes the iterations is an option; the models' defaults stand where none is given.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from unround import dct
from unround.model import (
    TGV,
    TV,
    Dual,
    Primal,
    Problem,
    clip,
    prox,
    tgv_partial_gap,
    tgv_values,
    tv_values,
)
from unround.operators import div, div2, grad, project_tensors, project_vectors, sym_grad
from unround.results import Recorder, Result

__all__ = [
    "STEP_PRODUCT",
    "TGV_ITERATIONS",
    "TGV_NORM_SQUARED",
    "TGV_RATIO",
    "TGV_RELAXATION",
    "TGV_TOLERANCE",
    "TV_ITERATIONS",
    "TV_NORM_SQUARED",
    "TV_RATIO",
    "TV_RELAXATION",
    "TV_TOLERANCE",
    "Initial",
    "Options",
    "Plan",
    "plan",
    "solve_tgv",
    "solve_tv",
    "start",
    "steps",
]

type Array = npt.NDArray[np.float64]
type Observer = Callable[[int, Primal, float], None]

TV_NORM_SQUARED: Final = 8.0
"""A bound of ||grad||^2 (docs/math.md, 3.3): TV's L^2 by default."""

TGV_NORM_SQUARED: Final = 0.5 * (17.0 + math.sqrt(33.0))
"""A bound of ||K||^2 for TGV's K(x, w) = (grad x - w, E w) (docs/math.md, 3.3): TGV's L^2 by default."""

STEP_PRODUCT: Final = 0.99
"""sigma tau L^2 by default, below 1 as the method's convergence asks (docs/math.md, 5)."""

# The models' defaults, chosen on the tuning images (docs/math.md, 6.5), for the weight
# alpha of TV, or alpha1 of TGV, of 1.
TV_RATIO: Final = 30.0
"""tau / sigma for TV, over alpha squared."""
TGV_RATIO: Final = 10.0
"""tau / sigma for TGV, over alpha1 squared."""
TV_TOLERANCE: Final = 2e-4
"""The gap per sample at which TV stops, times alpha."""
TGV_TOLERANCE: Final = 1e-2
"""The gap per sample at which TGV stops, times alpha1."""
TV_ITERATIONS: Final = 20000
"""The most iterations of TV."""
TGV_ITERATIONS: Final = 10000
"""The most iterations of TGV."""
TV_RELAXATION: Final = 1.9
"""The relaxation of TV's iterations (docs/math.md, 5)."""
TGV_RELAXATION: Final = 1.9
"""The relaxation of TGV's iterations."""

_RELAXATION_BELOW: Final = 2.0


@dataclass(frozen=True, slots=True)
class Options:
    """When a solver stops, its steps, and how often it records (docs/math.md, 5 and 6.4).

    iterations is the most iterations. The solver stops earlier, at the first record where
    one of these holds, each off at 0: the duality gap per sample is at most tolerance; the
    gap is at most relative_tolerance times the primal value; or, for TGV, the partial gap
    of the radius partial_radius (6.3) is at most partial_tolerance per sample. step_ratio
    is tau / sigma, relaxation the rho of the relaxed steps, in (0, 2), step_product
    sigma tau L^2, in (0, 1), and norm_squared the L^2 of the steps, a bound of ||K||^2
    (5): below ||K||^2 the method need not converge. Where they are None, they are the
    model's defaults (plan()), which follow the weight of the first-order term unless
    scale_with_weight is False. Every record_every iterations (0: none but the last), and
    after the last, the solver records the primal and dual values, and checks the
    tolerances; with partial_radius, which only TGV takes, it records the partial gap too.
    """

    iterations: int | None = None
    tolerance: float | None = None
    relative_tolerance: float = 0.0
    partial_tolerance: float = 0.0
    step_ratio: float | None = None
    relaxation: float | None = None
    step_product: float = STEP_PRODUCT
    norm_squared: float | None = None
    scale_with_weight: bool = True
    record_every: int = 10
    partial_radius: float | None = None


@dataclass(frozen=True, slots=True)
class Plan:
    """What a solver does: its Options, with the model's defaults in place of None."""

    iterations: int
    tolerance: float
    relative_tolerance: float
    partial_tolerance: float
    step_ratio: float
    relaxation: float
    step_product: float
    norm_squared: float
    record_every: int
    partial_radius: float | None


def plan(options: Options | None, weights: TV | TGV) -> Plan:
    """The options with the model's defaults in place of None, checked (docs/math.md, 5 and 6.5).

    The defaults were chosen with the weight alpha of TV, or alpha1 of TGV, of 1. The dual
    variables are in its units and the objective scales with it, so the ratio of the steps
    is the default over its square and the tolerance the default times it, unless
    scale_with_weight is False. What the method cannot run with raises a ValueError.
    """
    options = Options() if options is None else options
    if isinstance(weights, TV):
        iterations, tolerance, ratio, alpha = TV_ITERATIONS, TV_TOLERANCE, TV_RATIO, weights.alpha
        relaxation, norm_squared = TV_RELAXATION, TV_NORM_SQUARED
        if options.partial_radius is not None or options.partial_tolerance != 0.0:
            message = "TV has no partial gap"
            raise ValueError(message)
    else:
        iterations, tolerance, ratio, alpha = TGV_ITERATIONS, TGV_TOLERANCE, TGV_RATIO, weights.alpha1
        relaxation, norm_squared = TGV_RELAXATION, TGV_NORM_SQUARED
        if not weights.alpha0 > 0.0:
            message = f"the weight of the second-order term is positive, not {weights.alpha0}"
            raise ValueError(message)
    if not alpha > 0.0:
        message = f"the weight of the first-order term is positive, not {alpha}"
        raise ValueError(message)
    scale = alpha if options.scale_with_weight else 1.0
    settled = Plan(
        iterations=iterations if options.iterations is None else options.iterations,
        tolerance=tolerance * scale if options.tolerance is None else options.tolerance,
        relative_tolerance=options.relative_tolerance,
        partial_tolerance=options.partial_tolerance,
        step_ratio=ratio / (scale * scale) if options.step_ratio is None else options.step_ratio,
        relaxation=relaxation if options.relaxation is None else options.relaxation,
        step_product=options.step_product,
        norm_squared=norm_squared if options.norm_squared is None else options.norm_squared,
        record_every=options.record_every,
        partial_radius=options.partial_radius,
    )
    _check(settled)
    return settled


def _check(settled: Plan) -> None:
    """Refuses each value out of its range, all of them in one ValueError."""
    radius = settled.partial_radius
    ranges = (
        (settled.iterations >= 0, f"the most iterations are at least 0, not {settled.iterations}"),
        (settled.tolerance >= 0.0, f"the tolerance is at least 0, not {settled.tolerance}"),
        (settled.relative_tolerance >= 0.0, f"the relative tolerance is at least 0, not {settled.relative_tolerance}"),
        (settled.partial_tolerance >= 0.0, f"the partial tolerance is at least 0, not {settled.partial_tolerance}"),
        (
            0.0 < settled.step_ratio < math.inf,
            f"the ratio of the steps is positive and finite, not {settled.step_ratio}",
        ),
        (0.0 < settled.relaxation < _RELAXATION_BELOW, f"the relaxation is in (0, 2), not {settled.relaxation}"),
        (
            0.0 < settled.step_product < 1.0,
            f"the product of the steps and L^2 is in (0, 1), not {settled.step_product}",
        ),
        (0.0 < settled.norm_squared < math.inf, f"L^2 is positive and finite, not {settled.norm_squared}"),
        (settled.record_every >= 0, f"records are every 0 or more iterations, not {settled.record_every}"),
        (radius is None or radius >= 0.0, f"the radius of the partial gap is at least 0, not {radius}"),
        (
            settled.partial_tolerance == 0.0 or radius is not None,
            "the partial tolerance needs the partial gap's radius",
        ),
    )
    wrong = [message for holds, message in ranges if not holds]
    if wrong:
        message = "; ".join(wrong)
        raise ValueError(message)


def _stops(settled: Plan, samples: int, values: tuple[float, float], partial: float = np.nan) -> tuple[float, bool]:
    """The gap per sample of a record, and whether a tolerance stops the solver there."""
    primal, dual = values
    gap = primal - dual
    per_sample = gap / samples
    met = (
        (settled.tolerance > 0.0 and per_sample <= settled.tolerance)
        or (settled.relative_tolerance > 0.0 and gap <= settled.relative_tolerance * abs(primal))
        or (settled.partial_tolerance > 0.0 and partial / samples <= settled.partial_tolerance)
    )
    return per_sample, met


@dataclass(frozen=True, slots=True, eq=False)
class Initial:
    """Where a solver starts, where not at its default (docs/math.md, 5).

    coefficients are clipped to their intervals, and are the data term's centres unless
    given. w is TGV's field w, shape (2, H, W), and is the gradient of the start's canvas
    unless given. p, shape (2, H, W), and TGV's r, shape (3, H, W), are where the dual
    starts, projected onto their balls, and are 0 unless given. TV has no w and no r, and
    the subgradient method no dual. The solvers keep copies: what is given is not changed.
    """

    coefficients: Array | None = None
    w: Array | None = None
    p: Array | None = None
    r: Array | None = None


def steps(norm_squared: float, ratio: float, product: float = STEP_PRODUCT) -> tuple[float, float]:
    """tau and sigma, with tau / sigma = ratio and sigma tau norm_squared = product."""
    if not (0.0 < ratio < math.inf and 0.0 < norm_squared < math.inf and 0.0 < product < 1.0):
        message = (
            "a ratio and an L^2 that are positive and finite, and a product in (0, 1), "
            f"not {ratio}, {norm_squared} and {product}"
        )
        raise ValueError(message)
    return math.sqrt(ratio * product / norm_squared), math.sqrt(product / (ratio * norm_squared))


def start(problem: Problem, coefficients: Array | None = None) -> Primal:
    """The starting point: these coefficients, or else the data term's centres, clipped to their intervals."""
    if coefficients is None:
        chosen = clip(problem, problem.centres)
    else:
        given = np.asarray(coefficients, dtype=np.float64)
        if given.shape != problem.lower.shape or not np.all(np.isfinite(given)):
            message = f"the coefficients to start from are finite, of shape {problem.lower.shape}"
            raise ValueError(message)
        chosen = clip(problem, given)
    return Primal(coefficients=chosen, canvas=dct.inverse(chosen))


def first_coefficients(first: Initial | None, *, with_w: bool, with_dual: bool = True) -> Array | None:
    """The coefficients to start from, of first; and a check that it gives only what the method has."""
    if first is None:
        return None
    if not with_w and (first.w is not None or first.r is not None):
        message = "TV has no field w or r to start from"
        raise ValueError(message)
    if not with_dual and first.p is not None:
        message = "the subgradient method has no dual to start from"
        raise ValueError(message)
    return first.coefficients


def _field(given: Array, shape: tuple[int, ...], name: str) -> Array:
    """A copy of a field to start from, the solver's own, checked."""
    copy = np.array(given, dtype=np.float64)
    if copy.shape != shape:
        message = f"{name} is a field of shape {shape}, not {copy.shape}"
        raise ValueError(message)
    if not np.all(np.isfinite(copy)):
        message = f"the field {name} to start from is finite"
        raise ValueError(message)
    return copy


def _first_p(first: Initial | None, shape: tuple[int, int], radius: float) -> Array:
    if first is None or first.p is None:
        return np.zeros((2, *shape))
    return project_vectors(_field(first.p, (2, *shape), "p"), radius)


def _due(iteration: int, options: Plan) -> bool:
    return iteration == options.iterations or (options.record_every > 0 and iteration % options.record_every == 0)


def solve_tv(
    problem: Problem,
    weights: TV,
    options: Options | None = None,
    *,
    first: Initial | None = None,
    observe: Observer | None = None,
) -> Result:
    """Minimizes alpha ||grad x||_{2,1} + G(x) (docs/math.md, 4.2 and 5).

    options are Options() unless given, with the defaults of TV (plan()). first is where to
    start, the data term's centres and p = 0 unless given. observe is called with each
    recorded iteration, its point and its gap per sample, which is what the tolerance is
    compared with.

    Each iteration takes the proximal steps from the current point (x, p) to (x~, p~) and
    moves the current point to rho (x~, p~) + (1 - rho) (x, p). What is recorded, what is
    observed and what is returned are (x~, p~): x~ is an output of the proximal map of G, in
    the constraint set, and p~ is in the dual ball, even where the current point is not.
    """
    settled = plan(options, weights)
    tau, sigma = steps(settled.norm_squared, settled.step_ratio, settled.step_product)
    rho = settled.relaxation
    point = start(problem, first_coefficients(first, with_w=False))
    coefficients, canvas = point.coefficients, point.canvas
    x = canvas
    p = _first_p(first, problem.shape, weights.alpha)
    p_out = p
    recorder = Recorder()
    recorder.record(0, tv_values(problem, weights, point, Dual(p_out)))
    converged = False
    iteration = 0
    while iteration < settled.iterations and not converged:
        recorder.resume()
        iteration += 1
        coefficients = prox(problem, dct.forward(x + tau * div(p)), tau)
        canvas = dct.inverse(coefficients)
        p_out = project_vectors(p + sigma * grad(2.0 * canvas - x), weights.alpha)
        if rho == 1.0:
            x, p = canvas, p_out
        else:
            x = rho * canvas + (1.0 - rho) * x
            p = rho * p_out + (1.0 - rho) * p
        recorder.pause()
        if _due(iteration, settled):
            point = Primal(coefficients=coefficients, canvas=canvas)
            values = tv_values(problem, weights, point, Dual(p_out))
            recorder.record(iteration, values)
            gap, converged = _stops(settled, problem.samples, values)
            if observe is not None:
                observe(iteration, point, gap)
    return Result(
        primal=Primal(coefficients=coefficients, canvas=canvas),
        dual=Dual(p_out),
        iterations=iteration,
        converged=converged,
        history=recorder.history(),
    )


def solve_tgv(
    problem: Problem,
    weights: TGV,
    options: Options | None = None,
    *,
    first: Initial | None = None,
    observe: Observer | None = None,
) -> Result:
    """Minimizes alpha1 ||grad x - w||_{2,1} + alpha0 ||E w||_{F,1} + G(x) (docs/math.md, 4.3 and 5).

    options are Options() unless given, with the defaults of TGV (plan()). first is where to
    start: the data term's centres, w the gradient of their canvas, and p and r 0, unless
    given. The gap is that of the feasible dual (6.2). The steps are relaxed as solve_tv's
    are, and what is recorded, observed and returned are the outputs of the proximal steps.
    """
    settled = plan(options, weights)
    tau, sigma = steps(settled.norm_squared, settled.step_ratio, settled.step_product)
    rho = settled.relaxation
    point = start(problem, first_coefficients(first, with_w=True))
    coefficients, canvas = point.coefficients, point.canvas
    w = grad(canvas) if first is None or first.w is None else _field(first.w, (2, *problem.shape), "w")
    if first is None or first.r is None:
        r = np.zeros((3, *problem.shape))
    else:
        r = project_tensors(_field(first.r, (3, *problem.shape), "r"), weights.alpha0)
    x, w_out = canvas, w
    p = _first_p(first, problem.shape, weights.alpha1)
    p_out, r_out = p, r
    recorder = Recorder()

    def record(iteration: int, point: Primal, dual: Dual) -> tuple[float, bool]:
        primal_value, dual_value, theta = tgv_values(problem, weights, point, dual)
        partial = np.nan
        if settled.partial_radius is not None:
            partial = tgv_partial_gap(problem, weights, point, dual, settled.partial_radius)
        recorder.record(iteration, (primal_value, dual_value), theta, partial)
        return _stops(settled, problem.samples, (primal_value, dual_value), partial)

    record(0, Primal(coefficients=coefficients, canvas=canvas, w=w_out), Dual(p_out, r_out))
    converged = False
    iteration = 0
    while iteration < settled.iterations and not converged:
        recorder.resume()
        iteration += 1
        coefficients = prox(problem, dct.forward(x + tau * div(p)), tau)
        canvas = dct.inverse(coefficients)
        w_out = w + tau * (p + div2(r))
        extrapolated, extrapolated_w = 2.0 * canvas - x, 2.0 * w_out - w
        p_out = project_vectors(p + sigma * (grad(extrapolated) - extrapolated_w), weights.alpha1)
        r_out = project_tensors(r + sigma * sym_grad(extrapolated_w), weights.alpha0)
        if rho == 1.0:
            x, w, p, r = canvas, w_out, p_out, r_out
        else:
            x = rho * canvas + (1.0 - rho) * x
            w = rho * w_out + (1.0 - rho) * w
            p = rho * p_out + (1.0 - rho) * p
            r = rho * r_out + (1.0 - rho) * r
        recorder.pause()
        if _due(iteration, settled):
            point = Primal(coefficients=coefficients, canvas=canvas, w=w_out)
            gap, converged = record(iteration, point, Dual(p_out, r_out))
            if observe is not None:
                observe(iteration, point, gap)
    return Result(
        primal=Primal(coefficients=coefficients, canvas=canvas, w=w_out),
        dual=Dual(p_out, r_out),
        iterations=iteration,
        converged=converged,
        history=recorder.history(),
    )
