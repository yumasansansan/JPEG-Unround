# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""A subgradient method of jpeg2png's kind, for the TV model (docs/math.md, 7).

Normalized subgradient steps whose length falls as one over the square root of
the iteration, FISTA's extrapolation, and a projection onto the quantization
constraint set after every step. It is here to be compared with: it has no
convergence guarantee, and no gap to tell how far it is from the least value.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from unround import dct
from unround.model import TV, Primal, Problem, clip, tv_objective
from unround.operators import div, grad, vector_norms
from unround.pdhg import Initial, first_coefficients, start
from unround.results import Recorder, Result

__all__ = ["Options", "solve_tv", "subgradient"]

type Array = npt.NDArray[np.float64]
type Observer = Callable[[int, Primal, float], None]


@dataclass(frozen=True, slots=True)
class Options:
    """How many iterations the method takes, and how often it records the objective."""

    iterations: int = 50
    record_every: int = 1


def subgradient(problem: Problem, weights: TV, canvas: Array) -> Array:
    """A subgradient of alpha ||grad x||_{2,1} plus the data term, at any canvas.

    Where the gradient of x is 0, the subgradient of its norm taken is 0.
    """
    gradient = grad(canvas)
    norms = vector_norms(gradient)
    directions = np.divide(gradient, norms, out=np.zeros_like(gradient), where=norms > 0.0)
    data = dct.inverse(problem.weights * (dct.forward(canvas) - problem.centres))
    return np.asarray(-weights.alpha * div(directions) + data, dtype=np.float64)


def solve_tv(
    problem: Problem,
    weights: TV,
    options: Options | None = None,
    *,
    first: Initial | None = None,
    observe: Observer | None = None,
) -> Result:
    """Minimizes the TV model's objective within the quantization constraint set.

    options are Options() unless given. first is where to start, the MMSE centres unless
    given. The history's dual values are -inf: the method has none.
    """
    options = Options() if options is None else options
    point = start(problem, first_coefficients(first, with_w=False))
    coefficients, canvas = point.coefficients, point.canvas
    extrapolated = canvas
    momentum = 1.0
    radius = math.sqrt(problem.samples) / 2.0
    recorder = Recorder()
    recorder.record(0, (tv_objective(problem, weights, point), -math.inf))
    iteration = 0
    while iteration < options.iterations:
        recorder.resume()
        direction = subgradient(problem, weights, extrapolated)
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            recorder.pause()
            break
        step = radius / math.sqrt(1.0 + iteration)
        iteration += 1
        following = clip(problem, dct.forward(extrapolated - (step / length) * direction))
        following_canvas = dct.inverse(following)
        following_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        extrapolated = following_canvas + ((momentum - 1.0) / following_momentum) * (following_canvas - canvas)
        coefficients, canvas, momentum = following, following_canvas, following_momentum
        recorder.pause()
        if iteration == options.iterations or (options.record_every > 0 and iteration % options.record_every == 0):
            point = Primal(coefficients=coefficients, canvas=canvas)
            recorder.record(iteration, (tv_objective(problem, weights, point), -math.inf))
            if observe is not None:
                observe(iteration, point, math.inf)
    return Result(
        primal=Primal(coefficients=coefficients, canvas=canvas),
        dual=None,
        iterations=iteration,
        converged=False,
        history=recorder.history(),
    )
