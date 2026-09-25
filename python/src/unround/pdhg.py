# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The primal-dual hybrid gradient method of Chambolle and Pock for TV and TGV (docs/math.md, 5 and 6).

Each iterate's canvas is the output of the proximal map of G, and so lies in the
quantization constraint set: its coefficients are kept, and are the result. The
solvers stop when the duality gap per sample is within the tolerance, or after
the most iterations the options allow.
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
"""A bound of ||grad||^2 (docs/math.md, 3.3)."""

TGV_NORM_SQUARED: Final = 0.5 * (17.0 + math.sqrt(33.0))
"""A bound of ||K||^2 for TGV's K(x, w) = (grad x - w, E w) (docs/math.md, 3.3)."""

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

_THETA: Final = 0.99
_RELAXATION_BELOW: Final = 2.0


@dataclass(frozen=True, slots=True)
class Options:
    """How long a solver runs, the ratio of its steps, and how often it records.

    iterations is the most iterations. The solver stops earlier when the duality gap per
    sample is at most tolerance (0: it never does). step_ratio is tau / sigma, and
    relaxation the rho of the relaxed steps, in (0, 2) (docs/math.md, 5). Where they are
    None, they are the model's defaults (plan()). Every record_every iterations, and after
    the last, the solver records the primal and dual values, and checks the tolerance.
    partial_radius, for TGV, also records the partial gap of that radius (6.3).
    """

    iterations: int | None = None
    tolerance: float | None = None
    step_ratio: float | None = None
    relaxation: float | None = None
    record_every: int = 10
    partial_radius: float | None = None


@dataclass(frozen=True, slots=True)
class Plan:
    """What a solver does: its Options, with the model's defaults in place of None."""

    iterations: int
    tolerance: float
    step_ratio: float
    relaxation: float
    record_every: int
    partial_radius: float | None


def plan(options: Options | None, weights: TV | TGV) -> Plan:
    """The options with the model's defaults in place of None (docs/math.md, 5 and 6.4).

    The defaults were chosen with the weight alpha of TV, or alpha1 of TGV, of 1. The dual
    variables are in its units and the objective scales with it, so the ratio of the steps
    is the default over its square and the tolerance the default times it.
    """
    options = Options() if options is None else options
    if isinstance(weights, TV):
        iterations, tolerance, ratio, alpha = TV_ITERATIONS, TV_TOLERANCE, TV_RATIO, weights.alpha
        relaxation = TV_RELAXATION
    else:
        iterations, tolerance, ratio, alpha = TGV_ITERATIONS, TGV_TOLERANCE, TGV_RATIO, weights.alpha1
        relaxation = TGV_RELAXATION
    if not alpha > 0.0:
        message = f"the weight of the first-order term is positive, not {alpha}"
        raise ValueError(message)
    if options.relaxation is not None:
        relaxation = options.relaxation
    if not 0.0 < relaxation < _RELAXATION_BELOW:
        message = f"the relaxation is in (0, 2), not {relaxation}"
        raise ValueError(message)
    return Plan(
        iterations=iterations if options.iterations is None else options.iterations,
        tolerance=tolerance * alpha if options.tolerance is None else options.tolerance,
        step_ratio=ratio / (alpha * alpha) if options.step_ratio is None else options.step_ratio,
        relaxation=relaxation,
        record_every=options.record_every,
        partial_radius=options.partial_radius,
    )


@dataclass(frozen=True, slots=True, eq=False)
class Initial:
    """Where a solver starts, where not at its default.

    coefficients are clipped to their intervals, and are the MMSE centres unless given. w
    is TGV's field w, shape (2, H, W), and is the gradient of the start's canvas unless
    given; TV has no w.
    """

    coefficients: Array | None = None
    w: Array | None = None


def steps(norm_squared: float, ratio: float) -> tuple[float, float]:
    """tau and sigma, with tau / sigma = ratio and sigma tau norm_squared = 0.99."""
    if not ratio > 0.0:
        message = f"the ratio of the steps is positive, not {ratio}"
        raise ValueError(message)
    return math.sqrt(ratio * _THETA / norm_squared), math.sqrt(_THETA / (ratio * norm_squared))


def start(problem: Problem, coefficients: Array | None = None) -> Primal:
    """The starting point: these coefficients, or else the MMSE centres, clipped to their intervals."""
    chosen = clip(problem, problem.centres if coefficients is None else coefficients)
    return Primal(coefficients=chosen, canvas=dct.inverse(chosen))


def first_coefficients(first: Initial | None, *, with_w: bool) -> Array | None:
    """The coefficients to start from, of first; and a check that it gives w only where the model has one."""
    if first is None:
        return None
    if first.w is not None and not with_w:
        message = "TV has no field w to start from"
        raise ValueError(message)
    return first.coefficients


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
    start, the MMSE centres unless given. observe is called with each recorded iteration,
    its point and its gap per sample, which is what the tolerance is compared with.

    Each iteration takes the proximal steps from the current point (x, p) to (x~, p~) and
    moves the current point to rho (x~, p~) + (1 - rho) (x, p). What is recorded, what is
    observed and what is returned are (x~, p~): x~ is an output of the proximal map of G, in
    the constraint set, and p~ is in the dual ball, even where the current point is not.
    """
    settled = plan(options, weights)
    tau, sigma = steps(TV_NORM_SQUARED, settled.step_ratio)
    rho = settled.relaxation
    point = start(problem, first_coefficients(first, with_w=False))
    coefficients, canvas = point.coefficients, point.canvas
    x = canvas
    p = np.zeros((2, *problem.shape))
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
            gap = (values[0] - values[1]) / problem.samples
            converged = gap <= settled.tolerance
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
    start: the MMSE centres, and w the gradient of their canvas, unless given. The gap is
    that of the feasible dual (6.2). The steps are relaxed as solve_tv's are, and what is
    recorded, observed and returned are the outputs of the proximal steps.
    """
    settled = plan(options, weights)
    tau, sigma = steps(TGV_NORM_SQUARED, settled.step_ratio)
    rho = settled.relaxation
    point = start(problem, first_coefficients(first, with_w=True))
    coefficients, canvas = point.coefficients, point.canvas
    if first is None or first.w is None:
        w = grad(canvas)
    else:
        w = np.array(first.w, dtype=np.float64)  # a copy: the solver's own
        if w.shape != (2, *problem.shape):
            message = f"w is a field of shape {(2, *problem.shape)}, not {w.shape}"
            raise ValueError(message)
    x, w_out = canvas, w
    p = np.zeros((2, *problem.shape))
    r = np.zeros((3, *problem.shape))
    p_out, r_out = p, r
    recorder = Recorder()

    def record(iteration: int, point: Primal, dual: Dual) -> float:
        primal_value, dual_value, theta = tgv_values(problem, weights, point, dual)
        partial = np.nan
        if settled.partial_radius is not None:
            partial = tgv_partial_gap(problem, weights, point, dual, settled.partial_radius)
        recorder.record(iteration, (primal_value, dual_value), theta, partial)
        return (primal_value - dual_value) / problem.samples

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
            gap = record(iteration, point, Dual(p_out, r_out))
            converged = gap <= settled.tolerance
            if observe is not None:
                observe(iteration, point, gap)
    return Result(
        primal=Primal(coefficients=coefficients, canvas=canvas, w=w_out),
        dual=Dual(p_out, r_out),
        iterations=iteration,
        converged=converged,
        history=recorder.history(),
    )
