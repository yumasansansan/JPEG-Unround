# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The subgradient method of jpeg2png's kind: its subgradient, and its iterates."""

import math

import numpy as np
import numpy.typing as npt
import pytest

import rounding
import synthetic
from unround import dct, model, operators, pdhg, subgradient
from unround.model import TV, DataTerm, Primal

type Complex = npt.NDArray[np.complex128]


def complex_objective(problem: model.Problem, weights: TV, canvas: Complex) -> complex:
    """The TV objective without the constraint, in complex arithmetic, for the complex step."""
    gradient = operators.grad(canvas)
    variation = np.sum(np.sqrt(gradient[0] * gradient[0] + gradient[1] * gradient[1]))
    coefficients = dct.BASIS @ dct.blocks(canvas) @ dct.BASIS.T  # dct.forward keeps to float64
    difference = coefficients - problem.centres
    return complex(weights.alpha * variation + 0.5 * np.sum(problem.weights * difference * difference))


def test_the_subgradient_is_the_derivative_where_the_objective_is_smooth() -> None:
    # The complex step, Im f(x + i h v) / h with h = 1e-20, gives the directional derivative
    # without the cancellation of a difference quotient: its error is of order h^2 and of the
    # rounding of the terms. The canvas has no zero gradient but at its last sample, where
    # both differences are 0 whatever x is, the term is constant, and the subgradient taken
    # is 0.
    problem, canvas = synthetic.problem(rows=8, columns=16, seed=31, data=DataTerm(mu=40.0))
    rng = np.random.default_rng(32)
    canvas = canvas + rng.normal(0.0, 5.0, size=canvas.shape)
    weights = TV(alpha=1.3)
    direction = subgradient.subgradient(problem, weights, canvas)
    h = 1e-20
    for _ in range(10):
        v = rng.normal(size=canvas.shape)
        derivative = complex_objective(problem, weights, canvas + 1j * h * v).imag / h
        inner = float(np.sum(direction * v))
        # The terms' own derivatives, alpha |grad v| and w |c - centre| |D v|, bound what each
        # term contributes; each is computed within a few roundings, and both sums within
        # gamma(n) of the sums of their magnitudes.
        coefficients = dct.forward(canvas)
        magnitude = weights.alpha * float(np.sum(operators.vector_norms(operators.grad(v))))
        magnitude += float(np.sum(problem.weights * np.abs(coefficients - problem.centres) * np.abs(dct.forward(v))))
        allowance = (rounding.gamma(4 * canvas.size) + 16.0 * rounding.U) * magnitude
        allowance += rounding.gamma(canvas.size) * float(np.sum(np.abs(direction * v)))
        assert abs(derivative - inner) <= allowance


def test_the_iterates_stay_within_the_constraint_set_and_go_down() -> None:
    problem, _ = synthetic.problem(seed=33)
    checked: list[int] = []

    def observe(iteration: int, point: Primal, gap: float) -> None:
        checked.append(iteration)
        assert gap == math.inf  # the method has no dual, and no gap
        reach = rounding.roundtrip_error(point.coefficients)
        assert np.all(model.excess(problem, dct.forward(point.canvas)) * problem.steps <= reach)

    result = subgradient.solve_tv(problem, TV(), subgradient.Options(iterations=50, record_every=5), observe=observe)
    assert result.iterations == 50
    assert checked == list(range(5, 51, 5))
    assert result.history.primal[-1] < result.history.primal[0]
    assert not result.converged
    assert result.dual is None


@pytest.mark.parametrize(
    "options",
    [
        subgradient.Options(iterations=4, record_every=0),
        subgradient.Options(iterations=4, record_every=0, step=0.25, decay=1.0, momentum=False),
        subgradient.Options(iterations=4, record_every=0, step=1.0, decay=0.0),
    ],
)
def test_the_iterates_follow_the_options(options: subgradient.Options) -> None:
    # The scheme of docs/math.md, 7, written out with the same operations in the same order:
    # equal to the last bit.
    problem, _ = synthetic.problem(seed=34)
    weights = TV()
    result = subgradient.solve_tv(problem, weights, options)
    canvas = pdhg.start(problem).canvas
    extrapolated, t = canvas, 1.0
    for n in range(options.iterations):
        direction = subgradient.subgradient(problem, weights, extrapolated)
        length = float(np.linalg.norm(direction))
        power = math.sqrt(1.0 + n) if options.decay == 0.5 else math.pow(1.0 + n, options.decay)
        h = options.step * math.sqrt(problem.samples) / power
        following = dct.inverse(model.clip(problem, dct.forward(extrapolated - (h / length) * direction)))
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        extrapolated = following + ((t - 1.0) / t_next) * (following - canvas) if options.momentum else following
        canvas, t = following, t_next
    assert result.iterations == options.iterations
    np.testing.assert_array_equal(result.primal.canvas, canvas)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        (subgradient.Options(iterations=-1), "iterations"),
        (subgradient.Options(record_every=-1), "records"),
        (subgradient.Options(step=0.0), "step"),
        (subgradient.Options(step=math.inf), "step"),
        (subgradient.Options(decay=-0.5), "decay"),
        (subgradient.Options(decay=math.nan), "decay"),
    ],
)
def test_the_options_out_of_their_ranges_are_refused(options: subgradient.Options, match: str) -> None:
    problem, _ = synthetic.problem(seed=35)
    with pytest.raises(ValueError, match=match):
        subgradient.solve_tv(problem, TV(), options)


def test_the_method_has_no_dual_to_start_from() -> None:
    problem, _ = synthetic.problem(seed=35)
    with pytest.raises(ValueError, match="no dual"):
        subgradient.solve_tv(problem, TV(), first=pdhg.Initial(p=np.zeros((2, *problem.shape))))
