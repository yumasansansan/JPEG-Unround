# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The model: the constraint set and its projection, the prox, the conjugate, and weak duality.

What holds exactly is checked exactly. What is computed in floating point is checked
within bounds of its rounding, computed from the arrays at hand (rounding.py).
"""

import math

import numpy as np
import numpy.typing as npt
import pytest

import rounding
import synthetic
from unround import dct, model, operators
from unround.model import TGV, TV, Dual, Primal

type Array = npt.NDArray[np.float64]


def inside(problem: model.Problem, rng: np.random.Generator) -> Array:
    """Coefficients drawn from their intervals, clipped so that they lie within them exactly."""
    return model.clip(problem, np.asarray(rng.uniform(problem.lower, problem.upper), dtype=np.float64))


def test_the_intervals_and_the_weights() -> None:
    levels = np.zeros((1, 1, 8, 8), dtype=np.int64)
    levels[0, 0, 0, 0], levels[0, 0, 2, 1] = 3, -2
    table = np.full((8, 8), 10)
    problem = model.make_problem(levels, table, mu=0.5, slack=0.25)
    # Every one of these is exact in binary floating point.
    assert problem.lower[0, 0, 0, 0] == 22.5
    assert problem.upper[0, 0, 0, 0] == 37.5
    assert problem.lower[0, 0, 2, 1] == -27.5
    assert problem.upper[0, 0, 2, 1] == -12.5
    assert problem.weights[0, 0] == 0.0
    assert problem.weights[2, 1] == 0.5 / 100.0
    assert problem.shape == (8, 8)
    assert problem.samples == 64


def test_without_slack_the_intervals_are_exact() -> None:
    # (q - 1/2) Q is exact for integers of the sizes a file holds, and so are the intervals.
    levels = np.array([-2048, -1, 0, 1, 2047])[np.newaxis, :, np.newaxis, np.newaxis] * np.ones((1, 1, 8, 8), int)
    problem = model.make_problem(levels, np.full((8, 8), 255), mu=1.0)
    exact = (levels.astype(object) * 2 - 1) * 255
    assert [float(value) / 2 for value in exact.ravel()] == problem.lower.ravel().tolist()


@pytest.mark.parametrize(
    ("levels", "table", "mu", "slack", "match"),
    [
        (np.zeros((2, 8, 8)), np.ones((8, 8)), 1.0, 0.0, "shape"),
        (np.zeros((1, 1, 8, 8)), np.ones((4, 4)), 1.0, 0.0, "shape"),
        (np.zeros((1, 1, 8, 8)), np.zeros((8, 8)), 1.0, 0.0, "at least 1"),
        (np.zeros((1, 1, 8, 8)), np.ones((8, 8)), -1.0, 0.0, "at least 0"),
        (np.zeros((1, 1, 8, 8)), np.ones((8, 8)), 1.0, math.nan, "at least 0"),
    ],
)
def test_make_problem_refuses(levels: Array, table: Array, mu: float, slack: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        model.make_problem(levels, table, mu=mu, slack=slack)


def test_the_centres_lie_within_their_intervals_exactly() -> None:
    problem, _ = synthetic.problem(seed=1)
    assert np.all(problem.lower <= problem.centres)
    assert np.all(problem.centres <= problem.upper)


def euclidean(values: Array) -> float:
    return float(np.sqrt(np.sum(values * values)))


def projection_error(problem: model.Problem, canvas: Array) -> Array:
    """A bound of |model.project(problem, x) - P(x)|, sample by sample.

    The forward DCT rounds within forward_error, which clip does not increase; the exact
    inverse carries that through |B^T| e |B|, and the inverse rounds within inverse_error.
    """
    basis = np.abs(dct.BASIS)
    carried = dct.from_blocks(np.asarray(basis.T @ rounding.forward_error(canvas) @ basis, dtype=np.float64))
    return rounding.inverse_error(model.clip(problem, dct.forward(canvas))) + carried


def test_the_projection_lands_in_the_set_is_idempotent_and_is_nearest() -> None:
    problem, canvas = synthetic.problem(seed=2)
    rng = np.random.default_rng(20)
    v = canvas + rng.normal(0.0, 30.0, size=canvas.shape)
    projected = model.project(problem, v)
    coefficients = model.clip(problem, dct.forward(v))
    # Its coefficients are within their intervals up to the round trip of the DCT.
    assert np.all(
        model.excess(problem, dct.forward(projected)) * problem.steps <= rounding.roundtrip_error(coefficients)
    )
    # P is 1-Lipschitz: projecting again moves it at most twice its own error, plus the new one.
    first = euclidean(projection_error(problem, v))
    again = model.project(problem, projected)
    assert euclidean(again - projected) <= 2.0 * first + euclidean(projection_error(problem, projected))
    # And it is nearest: |P(v) - v| <= |z - v| for every z in the set, up to the errors of the
    # two points and of the two computed distances.
    near = euclidean(projected - v)
    for _ in range(50):
        c = inside(problem, rng)
        other = dct.inverse(c)
        far = euclidean(other - v)
        allowance = first + euclidean(rounding.inverse_error(c)) + rounding.gamma(v.size + 1) * (near + far)
        assert near <= far + allowance


def prox_objective(problem: model.Problem, c: Array, e: Array, tau: float) -> float:
    """1/2 |c - e|^2 + tau G(c), for c within the intervals."""
    return 0.5 * float(np.sum((c - e) ** 2)) + tau * model.data_term(problem, c)


@pytest.mark.parametrize("tau", [0.01, 1.0, 1e4])
def test_the_prox_minimizes_its_objective(tau: float) -> None:
    problem, canvas = synthetic.problem(seed=3, mu=5.0)
    rng = np.random.default_rng(21)
    e = dct.forward(canvas + rng.normal(0.0, 20.0, size=canvas.shape))
    c = model.prox(problem, e, tau)
    assert np.all(model.excess(problem, c) == 0.0)
    # The computed value is within delta of the exact one: five roundings of the fraction,
    # each within U of |e| + tau w |centre|, over 1 + tau w; clip does not increase it.
    w = np.broadcast_to(problem.weights, c.shape)
    delta = 6.0 * rounding.U * (np.abs(e) + tau * w * np.abs(problem.centres)) / (1.0 + tau * w)
    # Then J(c) <= J(c*) + sum of |dJ(c*)| delta + (1 + tau w) delta^2 / 2, and each J is
    # evaluated within (gamma(2n) + 4U) of its value, its parts being non-negative.
    slope = np.abs(c - e) + tau * w * np.abs(c - problem.centres) + (1.0 + tau * w) * delta
    suboptimal = float(np.sum(slope * delta + 0.5 * (1.0 + tau * w) * delta * delta))
    evaluation = rounding.gamma(2 * c.size) + 4.0 * rounding.U
    best = prox_objective(problem, c, e, tau)
    for _ in range(50):
        other = inside(problem, rng)
        value = prox_objective(problem, other, e, tau)
        assert best <= value + suboptimal + evaluation * (best + value)
    # The optimality condition, coefficient by coefficient: where c* is inside its interval,
    # (e - c*) / tau is the gradient of the data term there. The residual at c is within
    # delta (1/tau + w), plus the rounding of the residual itself.
    interior = (c - problem.lower > delta) & (problem.upper - c > delta)
    residual = (e - c) / tau - w * (c - problem.centres)
    allowance = delta * (1.0 / tau + w) + 4.0 * rounding.U * (np.abs(e - c) / tau + w * np.abs(c - problem.centres))
    assert np.all(np.abs(residual[interior]) <= allowance[interior])


@pytest.mark.parametrize("mu", [0.0, 2.0])
def test_the_conjugate_is_the_largest_of_its_objective(mu: float) -> None:
    problem, _ = synthetic.problem(rows=8, columns=8, seed=4, mu=mu)
    rng = np.random.default_rng(22)
    w = np.broadcast_to(problem.weights, problem.lower.shape)
    for _ in range(5):
        s = rng.normal(0.0, 3.0, size=problem.lower.shape)
        value = model.conjugate(problem, s)
        own = rounding.conjugate_error(problem, s)
        # Fenchel-Young: no point of the set gives more than the conjugate.
        for _ in range(200):
            c = inside(problem, rng)
            left = float(np.sum(s * c)) - model.data_term(problem, c)
            left_error = (rounding.gamma(2 * c.size) + 4.0 * rounding.U) * (
                float(np.sum(np.abs(s * c))) + model.data_term(problem, c)
            )
            assert left <= value + own + left_error
        # And a grid over each interval comes within reach of it. Its points are within 4U of the
        # ends' magnitudes of the interval; the largest on the grid is within (m/2) spacing^2 of
        # the largest of the interval where that is inside it, and within the slope times 4U of
        # the ends where it is at an end.
        grid = np.linspace(0.0, 1.0, 20001)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        c = problem.lower + grid * (problem.upper - problem.lower)
        values = s * c - 0.5 * w * (c - problem.centres) ** 2
        best = float(np.sum(np.max(values, axis=0)))
        spacing = (problem.upper - problem.lower) / 20000.0
        ends = np.maximum(np.abs(problem.lower), np.abs(problem.upper))
        slope = np.abs(s) + w * (problem.upper - problem.lower)
        reach = float(np.sum(0.5 * w * spacing * spacing + slope * 4.0 * rounding.U * ends))
        evaluation = (rounding.gamma(problem.lower.size) + 6.0 * rounding.U) * rounding.conjugate_magnitude(problem, s)
        assert abs(best - value) <= reach + evaluation + own
    # G*(0) is minus the least of G, which is 0 at the centres, and exactly so in floating point:
    # the centres lie within their intervals, and every term is a product with 0.
    assert model.conjugate(problem, np.zeros_like(problem.lower)) == 0.0


def test_the_tv_dual_value_is_never_above_the_primal_value() -> None:
    # No allowance is made for rounding: the gaps of these random points exceed it by many
    # orders of magnitude, and so a failure can only be an error of the formulas.
    problem, _ = synthetic.problem(seed=5)
    rng = np.random.default_rng(23)
    weights = TV(alpha=1.5)
    for _ in range(20):
        c = inside(problem, rng)
        point = Primal(coefficients=c, canvas=dct.inverse(c))
        p = operators.project_vectors(rng.normal(0.0, 2.0, size=(2, *problem.shape)), weights.alpha)
        primal, dual = model.tv_values(problem, weights, point, Dual(p))
        assert dual < primal


def test_the_tgv_dual_is_made_feasible_and_is_never_above_the_primal_value() -> None:
    # As for TV, the gaps here are far above any rounding.
    problem, _ = synthetic.problem(seed=6)
    rng = np.random.default_rng(24)
    weights = TGV(alpha1=1.0, alpha0=2.0)
    for _ in range(20):
        c = inside(problem, rng)
        canvas = dct.inverse(c)
        point = Primal(coefficients=c, canvas=canvas, w=operators.grad(canvas) + rng.normal(size=(2, *problem.shape)))
        r = operators.project_tensors(rng.normal(0.0, 3.0, size=(3, *problem.shape)), weights.alpha0)
        primal, dual, theta = model.tgv_values(problem, weights, point, Dual(np.zeros((2, *problem.shape)), r))
        assert 0.0 < theta <= 1.0
        # The scaled divergence is within alpha1: its norms, computed within 4U of their size.
        largest = float(np.max(operators.vector_norms(operators.div2(theta * r))))
        assert largest <= weights.alpha1 * (1.0 + 4.0 * rounding.U)
        assert dual < primal


def test_the_partial_gap_adds_the_residual_of_the_dual() -> None:
    # The same operations in the same order: equal to the last bit.
    problem, canvas = synthetic.problem(seed=7)
    rng = np.random.default_rng(25)
    weights = TGV()
    c = model.clip(problem, dct.forward(canvas))
    point = Primal(coefficients=c, canvas=dct.inverse(c), w=operators.grad(dct.inverse(c)))
    p = operators.project_vectors(rng.normal(size=(2, *problem.shape)), weights.alpha1)
    r = operators.project_tensors(rng.normal(size=(3, *problem.shape)), weights.alpha0)
    primal = model.tgv_objective(problem, weights, point)
    conjugate = model.conjugate(problem, dct.forward(operators.div(p)))
    residual = float(np.sum(operators.vector_norms(p + operators.div2(r))))
    assert model.tgv_partial_gap(problem, weights, point, Dual(p, r), 0.0) == primal + conjugate + 0.0 * residual
    assert model.tgv_partial_gap(problem, weights, point, Dual(p, r), 3.0) == primal + conjugate + 3.0 * residual
