# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""A subgradient method of jpeg2png's kind, for the TV model (docs/math.md, 7).

Normalized subgradient steps whose length falls as a power of the iteration (by
default one over its square root), FISTA's extrapolation, and a projection onto
the quantization constraint set after every step. It is here to be compared
with: it has no convergence guarantee, and no gap to tell how far it is from the
least value.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

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

_SQUARE_ROOT: Final = 0.5


@dataclass(frozen=True, slots=True)
class Options:
    """How many iterations the method takes, its steps, and how often it records the objective.

    The step after n iterations moves the canvas by step sqrt(N) / (1 + n)^decay along the
    normalized subgradient, N the number of samples, and momentum turns FISTA's
    extrapolation on (docs/math.md, 7). Every record_every iterations (0: none but the
    last), and after the last, the method records the objective.
    """

    iterations: int = 50
    record_every: int = 1
    step: float = 0.5
    decay: float = 0.5
    momentum: bool = True


def subgradient(problem: Problem, weights: TV, canvas: Array) -> Array:
    """A subgradient of alpha ||grad x||_{2,1} plus the data term, at any canvas.

    Where the gradient of x is 0, the subgradient of its norm taken is 0.
    """
    gradient = grad(canvas)
    norms = vector_norms(gradient)
    directions = np.divide(gradient, norms, out=np.zeros_like(gradient), where=norms > 0.0)
    data = dct.inverse(problem.weights * (dct.forward(canvas) - problem.centres))
    return np.asarray(-weights.alpha * div(directions) + data, dtype=np.float64)


def _power(base: float, exponent: float) -> float:
    """base^exponent, and for 1/2 the square root, which is correctly rounded."""
    return math.sqrt(base) if exponent == _SQUARE_ROOT else math.pow(base, exponent)


def solve_tv(
    problem: Problem,
    weights: TV,
    options: Options | None = None,
    *,
    first: Initial | None = None,
    observe: Observer | None = None,
) -> Result:
    """Minimizes the TV model's objective within the quantization constraint set.

    options are Options() unless given. first is where to start, the data term's centres
    unless given. The history's dual values are -inf: the method has none.
    """
    options = Options() if options is None else options
    ranges = (
        (options.iterations >= 0, f"the iterations are at least 0, not {options.iterations}"),
        (options.record_every >= 0, f"records are every 0 or more iterations, not {options.record_every}"),
        (0.0 < options.step < math.inf, f"the step is positive and finite, not {options.step}"),
        (0.0 <= options.decay < math.inf, f"the decay is at least 0 and finite, not {options.decay}"),
    )
    wrong = [message for holds, message in ranges if not holds]
    if wrong:
        message = "; ".join(wrong)
        raise ValueError(message)
    point = start(problem, first_coefficients(first, with_w=False, with_dual=False))
    coefficients, canvas = point.coefficients, point.canvas
    extrapolated = canvas
    momentum = 1.0
    radius = options.step * math.sqrt(problem.samples)
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
        step = radius / _power(1.0 + iteration, options.decay)
        iteration += 1
        following = clip(problem, dct.forward(extrapolated - (step / length) * direction))
        following_canvas = dct.inverse(following)
        following_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        if options.momentum:
            extrapolated = following_canvas + ((momentum - 1.0) / following_momentum) * (following_canvas - canvas)
        else:
            extrapolated = following_canvas
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
