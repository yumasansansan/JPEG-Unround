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

__all__ = ["TGV_NORM_SQUARED", "TV_NORM_SQUARED", "Options", "solve_tgv", "solve_tv", "start", "steps"]

type Array = npt.NDArray[np.float64]
type Observer = Callable[[int, Primal], None]

TV_NORM_SQUARED: Final = 8.0
"""A bound of ||grad||^2 (docs/math.md, 3.3)."""

TGV_NORM_SQUARED: Final = 0.5 * (17.0 + math.sqrt(33.0))
"""A bound of ||K||^2 for TGV's K(x, w) = (grad x - w, E w) (docs/math.md, 3.3)."""

_THETA: Final = 0.99


@dataclass(frozen=True, slots=True)
class Options:
    """How long a solver runs, the ratio of its steps, and how often it records.

    iterations is the most iterations. The solver stops earlier when the duality gap per
    sample is at most tolerance (0: it never does). step_ratio is tau / sigma. Every
    record_every iterations, and after the last, the solver records the primal and dual
    values, and checks the tolerance. partial_radius, for TGV, also records the partial
    gap of that radius (docs/math.md, 6.3).
    """

    iterations: int = 1000
    tolerance: float = 0.0
    step_ratio: float = 1.0
    record_every: int = 10
    partial_radius: float | None = None


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


def _due(iteration: int, options: Options) -> bool:
    return iteration == options.iterations or (options.record_every > 0 and iteration % options.record_every == 0)


def solve_tv(
    problem: Problem,
    weights: TV,
    options: Options | None = None,
    *,
    first: Array | None = None,
    observe: Observer | None = None,
) -> Result:
    """Minimizes alpha ||grad x||_{2,1} + G(x) (docs/math.md, 4.2 and 5).

    options are Options() unless given. first are the coefficients to start from, the MMSE
    centres unless given. observe is called with each recorded iteration and its point.
    """
    options = Options() if options is None else options
    tau, sigma = steps(TV_NORM_SQUARED, options.step_ratio)
    point = start(problem, first)
    coefficients, canvas = point.coefficients, point.canvas
    extrapolated = canvas
    p = np.zeros((2, *problem.shape))
    recorder = Recorder()
    recorder.record(0, tv_values(problem, weights, point, Dual(p)))
    converged = False
    iteration = 0
    while iteration < options.iterations and not converged:
        recorder.resume()
        iteration += 1
        p = project_vectors(p + sigma * grad(extrapolated), weights.alpha)
        following = prox(problem, dct.forward(canvas + tau * div(p)), tau)
        following_canvas = dct.inverse(following)
        extrapolated = 2.0 * following_canvas - canvas
        coefficients, canvas = following, following_canvas
        recorder.pause()
        if _due(iteration, options):
            point = Primal(coefficients=coefficients, canvas=canvas)
            values = tv_values(problem, weights, point, Dual(p))
            recorder.record(iteration, values)
            converged = values[0] - values[1] <= options.tolerance * problem.samples
            if observe is not None:
                observe(iteration, point)
    return Result(
        primal=Primal(coefficients=coefficients, canvas=canvas),
        dual=Dual(p),
        iterations=iteration,
        converged=converged,
        history=recorder.history(),
    )


def solve_tgv(
    problem: Problem,
    weights: TGV,
    options: Options | None = None,
    *,
    first: Array | None = None,
    observe: Observer | None = None,
) -> Result:
    """Minimizes alpha1 ||grad x - w||_{2,1} + alpha0 ||E w||_{F,1} + G(x) (docs/math.md, 4.3 and 5).

    options are Options() unless given. first are the coefficients to start from, the MMSE
    centres unless given, and w starts as the gradient of their canvas. The gap is that of
    the feasible dual (6.2).
    """
    options = Options() if options is None else options
    tau, sigma = steps(TGV_NORM_SQUARED, options.step_ratio)
    point = start(problem, first)
    coefficients, canvas = point.coefficients, point.canvas
    w = grad(canvas)
    extrapolated, extrapolated_w = canvas, w
    p = np.zeros((2, *problem.shape))
    r = np.zeros((3, *problem.shape))
    recorder = Recorder()

    def record(iteration: int, point: Primal, dual: Dual) -> bool:
        primal_value, dual_value, theta = tgv_values(problem, weights, point, dual)
        partial = np.nan
        if options.partial_radius is not None:
            partial = tgv_partial_gap(problem, weights, point, dual, options.partial_radius)
        recorder.record(iteration, (primal_value, dual_value), theta, partial)
        return primal_value - dual_value <= options.tolerance * problem.samples

    record(0, Primal(coefficients=coefficients, canvas=canvas, w=w), Dual(p, r))
    converged = False
    iteration = 0
    while iteration < options.iterations and not converged:
        recorder.resume()
        iteration += 1
        p = project_vectors(p + sigma * (grad(extrapolated) - extrapolated_w), weights.alpha1)
        r = project_tensors(r + sigma * sym_grad(extrapolated_w), weights.alpha0)
        following = prox(problem, dct.forward(canvas + tau * div(p)), tau)
        following_canvas = dct.inverse(following)
        following_w = w + tau * (p + div2(r))
        extrapolated = 2.0 * following_canvas - canvas
        extrapolated_w = 2.0 * following_w - w
        coefficients, canvas, w = following, following_canvas, following_w
        recorder.pause()
        if _due(iteration, options):
            point = Primal(coefficients=coefficients, canvas=canvas, w=w)
            converged = record(iteration, point, Dual(p, r))
            if observe is not None:
                observe(iteration, point)
    return Result(
        primal=Primal(coefficients=coefficients, canvas=canvas, w=w),
        dual=Dual(p, r),
        iterations=iteration,
        converged=converged,
        history=recorder.history(),
    )
