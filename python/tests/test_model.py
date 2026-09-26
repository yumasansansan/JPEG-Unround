# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The model: the constraint set and its projection, the prox, the conjugate, and weak duality.

What holds exactly is checked exactly. What is computed in floating point is checked
within bounds of its rounding, computed from the arrays at hand (rounding.py).
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import rounding
import synthetic
from unround import dct, model, operators
from unround.model import TGV, TV, DataTerm, Dual, Primal

type Array = npt.NDArray[np.float64]


def inside(problem: model.Problem, rng: np.random.Generator) -> Array:
    """Coefficients drawn from their intervals, clipped so that they lie within them exactly."""
    return model.clip(problem, np.asarray(rng.uniform(problem.lower, problem.upper), dtype=np.float64))


def test_the_intervals_and_the_weights() -> None:
    levels = np.zeros((1, 1, 8, 8), dtype=np.int64)
    levels[0, 0, 0, 0], levels[0, 0, 2, 1] = 3, -2
    table = np.full((8, 8), 10)
    problem = model.make_problem(levels, table, DataTerm(mu=0.5, slack=0.25))
    # Every one of these is exact in binary floating point. The DC interval carries the level
    # shift, 8 x 128.
    assert problem.lower[0, 0, 0, 0] == 22.5 + 1024.0
    assert problem.upper[0, 0, 0, 0] == 37.5 + 1024.0
    assert problem.centres[0, 0, 0, 0] == 30.0 + 1024.0
    assert problem.lower[0, 0, 2, 1] == -27.5
    assert problem.upper[0, 0, 2, 1] == -12.5
    assert problem.weights[0, 0] == 0.0
    assert problem.weights[2, 1] == 0.5 / 100.0
    assert problem.shape == (8, 8)
    assert problem.samples == 64


def test_without_slack_the_intervals_are_exact() -> None:
    # (q - 1/2) Q is exact for integers of the sizes a file holds, and so are the intervals.
    levels = np.array([-2048, -1, 0, 1, 2047])[np.newaxis, :, np.newaxis, np.newaxis] * np.ones((1, 1, 8, 8), int)
    problem = model.make_problem(levels, np.full((8, 8), 255), DataTerm(mu=1.0))
    exact = (levels.astype(object) * 2 - 1) * 255
    exact[:, :, 0, 0] += 2 * 1024
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
        model.make_problem(levels, table, DataTerm(mu=mu, slack=slack))


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"dc_weight": -1.0}, "at least 0"),
        ({"dc_weight": math.inf}, "finite"),
        ({"dc_weight": math.nan}, "at least 0"),
        ({"mu": math.inf}, "finite"),
        ({"centres": "median"}, "mmse or midpoint"),
        ({"power": -1.0}, "power"),
        ({"power": math.inf}, "power"),
        ({"slack_cost": -1.0}, "cost of the slack"),
        ({"slack_cost": math.nan}, "cost of the slack"),
        ({"mu": None, "mu_scale": -1.0}, "scale of mu"),
        ({"mu": None, "mu_power": math.inf}, "power"),
    ],
)
def test_make_problem_refuses_the_options_of_g(options: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        model.make_problem(np.zeros((1, 1, 8, 8)), np.ones((8, 8)), DataTerm(**options))
    with pytest.raises(ValueError, match="Laplace scale"):
        model.make_problem(np.zeros((1, 1, 8, 8)), np.ones((8, 8)), DataTerm(centres="midpoint"), scale=np.ones((8, 8)))


def test_the_weight_of_dc_and_the_middles() -> None:
    levels = np.zeros((1, 1, 8, 8), dtype=np.int64)
    levels[0, 0, 0, 0], levels[0, 0, 2, 1] = 3, -2
    table = np.full((8, 8), 10)
    problem = model.make_problem(levels, table, DataTerm(mu=0.5, dc_weight=2.0, centres="midpoint"))
    # mu / Q^2 rounds once, and doubling it is exact; the middles q Q are exact, and DC's has
    # the level shift.
    assert problem.weights[0, 0] == 2.0 * (0.5 / 100.0)
    assert problem.weights[2, 1] == 0.5 / 100.0
    assert problem.centres[0, 0, 0, 0] == 30.0 + 1024.0
    assert problem.centres[0, 0, 2, 1] == -20.0
    assert problem.centres[0, 0, 1, 1] == 0.0
    # The power of the steps: mu / Q, and mu / Q^3, which 10^3 holds exactly.
    assert model.make_problem(levels, table, DataTerm(mu=0.5, power=1.0)).weights[2, 1] == 0.5 / 10.0
    assert model.make_problem(levels, table, DataTerm(mu=0.5, power=3.0)).weights[2, 1] == 0.5 / 1000.0
    # The MMSE centre of a level that is not 0 lies nearer 0 than the middle.
    mmse = model.make_problem(levels, table, DataTerm(mu=0.5))
    assert -20.0 < mmse.centres[0, 0, 2, 1] < -15.0
    assert mmse.weights[0, 0] == 0.0


@pytest.mark.parametrize("centres", ["mmse", "midpoint"])
def test_the_centres_lie_within_their_intervals_exactly(centres: model.Centres) -> None:
    problem, _ = synthetic.problem(seed=1, data=DataTerm(centres=centres))
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


@pytest.mark.parametrize("dc_weight", [0.0, 1.0])
@pytest.mark.parametrize("tau", [0.01, 1.0, 1e4])
def test_the_prox_minimizes_its_objective(tau: float, dc_weight: float) -> None:
    problem, canvas = synthetic.problem(seed=3, data=DataTerm(mu=5.0, dc_weight=dc_weight))
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


def test_mu_follows_the_steps_where_it_is_not_given() -> None:
    # mu = scale times the mean step to the power: the mean of a table of 10s and a 74, and
    # of Annex K's table, exact, and their powers.
    table = np.full((8, 8), 10)
    table[0, 0] = 74
    assert model.rule_mu(DataTerm(mu=None, mu_scale=5.0), table) == 5.0 * 11.0
    assert model.rule_mu(DataTerm(mu=None, mu_scale=5.0, mu_power=2.0), table) == 5.0 * 121.0
    assert model.rule_mu(DataTerm(mu=2.5, mu_scale=5.0), table) == 2.5
    mean = float(np.sum(synthetic.ANNEX_K)) / 64.0
    assert mean == 3688.0 / 64.0
    problem = model.make_problem(np.zeros((1, 1, 8, 8)), synthetic.ANNEX_K, DataTerm(mu=None, mu_scale=3.0))
    assert problem.weights[2, 1] == (3.0 * mean) / (13.0 * 13.0)


def test_a_slack_costs_only_where_it_is_given() -> None:
    levels = np.zeros((1, 1, 8, 8), dtype=np.int64)
    levels[0, 0, 0, 0], levels[0, 0, 2, 1] = 3, -2
    table = np.full((8, 8), 10)
    for terms in (DataTerm(slack=0.5), DataTerm(slack_cost=2.0), DataTerm()):
        plain = model.make_problem(levels, table, terms)
        assert (plain.inner_lower, plain.inner_upper, plain.costs) == (None, None, None)
    costly = model.make_problem(levels, table, DataTerm(slack=0.5, slack_cost=2.0))
    assert costly.inner_lower is not None
    assert costly.inner_upper is not None
    assert costly.costs is not None
    # The file's own intervals, exactly, and the cost per unit of a coefficient, 2 / Q.
    assert costly.inner_lower[0, 0, 0, 0] == 25.0 + 1024.0
    assert costly.inner_upper[0, 0, 2, 1] == -15.0
    assert costly.lower[0, 0, 2, 1] == -30.0
    assert costly.costs[2, 1] == 2.0 / 10.0


@pytest.mark.parametrize("tau", [0.01, 1.0, 1e4])
def test_the_prox_with_a_costly_slack_meets_its_optimality_conditions(tau: float) -> None:
    # With a slack of half a step that costs 0.3 per step, the minimizer of
    # 1/2 (c - e)^2 + tau g(c) over the widened interval has, coefficient by coefficient,
    # r = (e - c) - tau m (c - centre) in tau cost times the subdifferential of the distance
    # to the file's own interval, plus its normal cone: 0 within the interval, tau cost
    # beyond it, [0, tau cost] at its upper end, and at least tau cost at the widened one;
    # and alike below. c is within delta of the exact minimizer: the fraction's roundings,
    # and three of the shift by tau cost / (1 + tau m); r is computed within 4U of its parts.
    problem, canvas = synthetic.problem(seed=3, data=DataTerm(mu=5.0, slack=0.5, slack_cost=0.3))
    assert problem.inner_lower is not None
    assert problem.inner_upper is not None
    assert problem.costs is not None
    rng = np.random.default_rng(27)
    e = dct.forward(canvas + rng.normal(0.0, 40.0, size=canvas.shape))
    c = model.prox(problem, e, tau)
    assert np.all(model.excess(problem, c) == 0.0)
    m = np.broadcast_to(problem.weights, c.shape)
    force = tau * np.broadcast_to(problem.costs, c.shape)
    moved = (e + tau * m * problem.centres) / (1.0 + tau * m)
    delta = 6.0 * rounding.U * (np.abs(e) + tau * m * np.abs(problem.centres)) / (1.0 + tau * m)
    delta += 4.0 * rounding.U * (np.abs(moved) + force / (1.0 + tau * m))
    r = (e - c) - tau * m * (c - problem.centres)
    allowance = (1.0 + tau * m) * delta + 4.0 * rounding.U * (np.abs(e - c) + tau * m * np.abs(c - problem.centres))
    low, high = problem.inner_lower, problem.inner_upper
    within = (c > low) & (c < high)
    above = (c > high) & (c < problem.upper)
    below = (c > problem.lower) & (c < low)
    assert np.all(np.abs(r[within]) <= allowance[within])
    assert np.all(np.abs(r[above] - force[above]) <= allowance[above])
    assert np.all(np.abs(r[below] + force[below]) <= allowance[below])
    at = c == high
    assert np.all((r[at] >= -allowance[at]) & (r[at] <= force[at] + allowance[at]))
    at = c == low
    assert np.all((r[at] <= allowance[at]) & (r[at] >= -force[at] - allowance[at]))
    at = c == problem.upper
    assert np.all(r[at] >= force[at] - allowance[at])
    at = c == problem.lower
    assert np.all(r[at] <= -force[at] + allowance[at])


def test_the_cost_draws_coefficients_back() -> None:
    # With a cost of 20 per step, tau = 10 moves a coefficient beyond its own interval back
    # by 10 cost / (1 + 10 m): about a unit or more. Those beyond by less come to the end of
    # their interval; those beyond by more stay beyond.
    costly, canvas = synthetic.problem(seed=3, data=DataTerm(mu=5.0, slack=0.5, slack_cost=20.0))
    free, _ = synthetic.problem(seed=3, data=DataTerm(mu=5.0, slack=0.5))
    rng = np.random.default_rng(27)
    e = dct.forward(canvas + rng.normal(0.0, 40.0, size=canvas.shape))
    kept = np.count_nonzero(model.beyond(costly, model.prox(costly, e, 10.0)))
    loose = np.count_nonzero(model.beyond(costly, model.prox(free, e, 10.0)))
    assert 0 < kept < loose


def test_the_conjugate_with_a_costly_slack_is_the_largest_of_its_objective() -> None:
    # As test_the_conjugate_is_the_largest_of_its_objective, with the slack's cost: a grid of
    # 4 x 5000 steps over each widened interval of a slack of half a step holds the ends of
    # the file's own interval, where the largest can lie, to the rounding of the grid points.
    problem, _ = synthetic.problem(rows=8, columns=8, seed=4, data=DataTerm(mu=2.0, slack=0.5, slack_cost=0.3))
    assert problem.costs is not None
    assert problem.inner_lower is not None
    assert problem.inner_upper is not None
    rng = np.random.default_rng(28)
    w = np.broadcast_to(problem.weights, problem.lower.shape)
    cost = np.broadcast_to(problem.costs, problem.lower.shape)
    for _ in range(5):
        s = rng.normal(0.0, 3.0, size=problem.lower.shape)
        value = model.conjugate(problem, s)
        own = rounding.conjugate_error(problem, s) + (rounding.gamma(s.size) + 8.0 * rounding.U) * float(
            np.sum(cost * (problem.upper - problem.lower))
        )
        for _ in range(200):
            c = np.clip(rng.uniform(problem.lower, problem.upper), problem.lower, problem.upper)
            left = float(np.sum(s * c)) - model.data_term(problem, c)
            left_error = (rounding.gamma(3 * c.size) + 6.0 * rounding.U) * (
                float(np.sum(np.abs(s * c))) + model.data_term(problem, c)
            )
            assert left <= value + own + left_error
        grid = np.linspace(0.0, 1.0, 20001)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        c = problem.lower + grid * (problem.upper - problem.lower)
        distance = np.maximum(problem.inner_lower - c, 0.0) + np.maximum(c - problem.inner_upper, 0.0)
        values = s * c - 0.5 * w * (c - problem.centres) ** 2 - cost * distance
        best = float(np.sum(np.max(values, axis=0)))
        spacing = (problem.upper - problem.lower) / 20000.0
        ends = np.maximum(np.abs(problem.lower), np.abs(problem.upper))
        slope = np.abs(s) + w * (problem.upper - problem.lower) + cost
        reach = float(np.sum(0.5 * w * spacing * spacing + slope * 8.0 * rounding.U * ends))
        evaluation = (rounding.gamma(problem.lower.size) + 8.0 * rounding.U) * (
            rounding.conjugate_magnitude(problem, s) + float(np.sum(cost * (problem.upper - problem.lower)))
        )
        assert abs(best - value) <= reach + evaluation + own


@pytest.mark.parametrize(("mu", "dc_weight"), [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0)])
def test_the_conjugate_is_the_largest_of_its_objective(mu: float, dc_weight: float) -> None:
    problem, _ = synthetic.problem(rows=8, columns=8, seed=4, data=DataTerm(mu=mu, dc_weight=dc_weight))
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
