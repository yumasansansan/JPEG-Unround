# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The primal-dual method: its steps, its iterates within the constraint set, and its gaps.

The iterates are checked against the constraint set within the round trip of the DCT
(rounding.py). How far the gaps fall in a number of iterations is not a theorem but a
measurement on these problems, kept as a check that the method still does as well:
each threshold is about ten times what was measured.
"""

import dataclasses

import numpy as np
import pytest

import rounding
import synthetic
from unround import dct, model, operators, pdhg, subgradient
from unround.model import TGV, TV, Dual, Primal


def test_the_steps_keep_their_ratio_and_their_product() -> None:
    # Each step is a square root of a product and a quotient: within 3U, and their quotient
    # and product within 8U.
    for ratio in (0.1, 1.0, 37.0):
        tau, sigma = pdhg.steps(pdhg.TGV_NORM_SQUARED, ratio)
        assert abs(tau / sigma - ratio) <= 8.0 * rounding.U * ratio
        assert abs(tau * sigma * pdhg.TGV_NORM_SQUARED - 0.99) <= 8.0 * rounding.U
    with pytest.raises(ValueError, match="positive"):
        pdhg.steps(8.0, 0.0)


def test_the_defaults_scale_with_the_weight() -> None:
    # What is None is the model's default: the ratio over the weight squared, the tolerance
    # times it; what is given is kept.
    tv = pdhg.plan(None, TV(1.0))
    assert (tv.iterations, tv.tolerance, tv.step_ratio) == (pdhg.TV_ITERATIONS, pdhg.TV_TOLERANCE, pdhg.TV_RATIO)
    assert tv.relaxation == pdhg.TV_RELAXATION
    tgv = pdhg.plan(pdhg.Options(), TGV(1.0, 2.0))
    assert (tgv.iterations, tgv.tolerance, tgv.step_ratio) == (pdhg.TGV_ITERATIONS, pdhg.TGV_TOLERANCE, pdhg.TGV_RATIO)
    assert tgv.relaxation == pdhg.TGV_RELAXATION
    scaled = pdhg.plan(None, TV(4.0))
    assert scaled.step_ratio == pdhg.TV_RATIO / 16.0
    assert scaled.tolerance == pdhg.TV_TOLERANCE * 4.0
    given = pdhg.plan(pdhg.Options(iterations=7, tolerance=0.0, step_ratio=2.5, record_every=3), TGV(0.5, 1.0))
    assert (given.iterations, given.tolerance, given.step_ratio, given.record_every) == (7, 0.0, 2.5, 3)
    with pytest.raises(ValueError, match="positive"):
        pdhg.plan(None, TV(0.0))
    # The relaxed steps converge for a relaxation in (0, 2) only.
    assert pdhg.plan(pdhg.Options(relaxation=1.9), TV(1.0)).relaxation == 1.9
    for relaxation in (0.0, 2.0, -1.0):
        with pytest.raises(ValueError, match="relaxation"):
            pdhg.plan(pdhg.Options(relaxation=relaxation), TV(1.0))


def test_the_start_is_the_mmse_decoder() -> None:
    problem, _ = synthetic.problem(seed=11)
    point = pdhg.start(problem)
    np.testing.assert_array_equal(point.coefficients, problem.centres)
    np.testing.assert_array_equal(point.canvas, dct.inverse(problem.centres))


def within_the_set(problem: model.Problem, point: Primal) -> bool:
    """The canvas's coefficients are within their intervals, up to the round trip of the DCT."""
    reach = rounding.roundtrip_error(point.coefficients)
    return bool(np.all(model.excess(problem, dct.forward(point.canvas)) * problem.steps <= reach))


@pytest.mark.parametrize(("ratio", "threshold"), [(1.0, 4e-9), (30.0, 1e-5)])
def test_tv_converges_within_the_constraint_set(ratio: float, threshold: float) -> None:
    # Measured: a gap per sample of 3.7e-10 after 4000 iterations with the ratio 1, and 9.6e-7
    # with 30.
    problem, _ = synthetic.problem(seed=12)
    seen: list[int] = []
    gaps: list[float] = []

    def observe(iteration: int, point: Primal, gap: float) -> None:
        seen.append(iteration)
        gaps.append(gap)
        assert within_the_set(problem, point)

    options = pdhg.Options(iterations=4000, tolerance=0.0, step_ratio=ratio, record_every=200)
    result = pdhg.solve_tv(problem, TV(1.0), options, observe=observe)
    history = result.history
    assert seen == list(range(200, 4001, 200))
    # The observer is given the gap per sample of each record, as the tolerance is compared with.
    assert gaps == list(history.gap[1:] / problem.samples)
    assert np.all(model.excess(problem, result.primal.coefficients) == 0.0)
    assert np.all(np.diff(history.seconds) >= 0.0)
    assert history.gap[-1] / problem.samples < threshold
    # The gap is never below 0 by more than the rounding of the two values, bounded at the last
    # iterate, where it is least.
    assert result.dual is not None
    allowance = rounding.tv_values_error(problem, TV(1.0), result.primal, result.dual)
    assert history.gap[-1] >= -allowance
    assert np.all(history.gap[:-1] > 0.0)


@pytest.mark.parametrize(("relaxation", "threshold"), [(1.5, 3e-5), (1.9, 3e-5)])
def test_relaxed_tv_keeps_to_the_constraint_set(relaxation: float, threshold: float) -> None:
    # Measured: a gap per sample of 3.4e-6 with the relaxation 1.5 and 2.8e-6 with 1.9 after
    # 1000 iterations, against 1.0e-5 without. The current point of a relaxed step may leave
    # the constraint set; what is observed and returned, the proximal steps' outputs, may not.
    problem, _ = synthetic.problem(seed=18)

    def observe(iteration: int, point: Primal, gap: float) -> None:  # noqa: ARG001
        assert within_the_set(problem, point)

    options = pdhg.Options(iterations=1000, tolerance=0.0, step_ratio=30.0, relaxation=relaxation, record_every=100)
    result = pdhg.solve_tv(problem, TV(1.0), options, observe=observe)
    history = result.history
    assert history.gap[-1] / problem.samples < threshold
    assert np.all(model.excess(problem, result.primal.coefficients) == 0.0)
    # The dual returned is a projection's output, within the ball up to rounding: dividing by
    # a norm taken within 2U makes a vector longer by at most 3U, and taking its norm here
    # adds 2U more, to first order.
    assert result.dual is not None
    assert np.all(operators.vector_norms(result.dual.p) <= 1.0 + 6.0 * rounding.U)
    allowance = rounding.tv_values_error(problem, TV(1.0), result.primal, result.dual)
    assert history.gap[-1] >= -allowance
    assert np.all(history.gap[:-1] > 0.0)


@pytest.mark.parametrize(("relaxation", "measured"), [(1.0, 130), (1.9, 80)])
def test_tv_stops_at_its_tolerance(relaxation: float, measured: int) -> None:
    # Measured: 130 iterations unrelaxed, 80 with the relaxation 1.9.
    problem, _ = synthetic.problem(seed=13)
    options = pdhg.Options(iterations=5000, tolerance=0.05, step_ratio=10.0, relaxation=relaxation)
    result = pdhg.solve_tv(problem, TV(1.0), options)
    assert result.converged
    assert result.iterations == measured
    assert result.history.gap[-1] <= 0.05 * problem.samples


def test_tgv_converges_within_the_constraint_set() -> None:
    # Measured: a gap per sample of 1.9e-4 after 6000 iterations, from 1.6e-2 after 500, and the
    # scaling of the dual 1. The gaps here are at least 0.07, and the values about 3e3, far
    # above their rounding: a gap below 0 could only be an error of the formulas.
    problem, _ = synthetic.problem(seed=14)
    options = pdhg.Options(iterations=6000, tolerance=0.0, step_ratio=10.0, record_every=500)
    result = pdhg.solve_tgv(problem, TGV(1.0, 2.0), options)
    history = result.history
    assert np.all(history.gap > 0.0)
    assert history.gap[-1] / problem.samples < 2e-3
    assert history.gap[-1] < 0.05 * history.gap[1]
    assert history.scaling[-1] > 0.9
    assert np.all(model.excess(problem, result.primal.coefficients) == 0.0)
    assert within_the_set(problem, result.primal)
    assert result.primal.w is not None
    assert result.dual is not None
    assert result.dual.r is not None


def test_relaxed_tgv_keeps_to_the_constraint_set() -> None:
    # Measured: a gap per sample of 1.1e-7 after 6000 iterations with the relaxation 1.9,
    # against 2.8e-6 without, and the scaling of the dual 1.
    problem, _ = synthetic.problem(seed=19)

    def observe(iteration: int, point: Primal, gap: float) -> None:  # noqa: ARG001
        assert within_the_set(problem, point)

    options = pdhg.Options(iterations=6000, tolerance=0.0, step_ratio=10.0, relaxation=1.9, record_every=500)
    result = pdhg.solve_tgv(problem, TGV(1.0, 2.0), options, observe=observe)
    history = result.history
    assert np.all(history.gap > 0.0)
    assert history.gap[-1] / problem.samples < 1e-6
    assert history.scaling[-1] > 0.9
    assert np.all(model.excess(problem, result.primal.coefficients) == 0.0)


def test_tgv_starts_from_the_field_given() -> None:
    # No iteration: the one record is the objective at the start, with w as given, or the
    # gradient of the start's canvas; and the field given is not changed.
    problem, _ = synthetic.problem(seed=17)
    options = pdhg.Options(iterations=0)
    begin = pdhg.start(problem)
    zero = np.zeros((2, *problem.shape))
    result = pdhg.solve_tgv(problem, TGV(), options, first=pdhg.Initial(w=zero))
    assert result.history.primal[0] == model.tgv_objective(problem, TGV(), dataclasses.replace(begin, w=zero))
    assert result.primal.w is not None
    assert result.primal.w is not zero
    np.testing.assert_array_equal(zero, 0.0)
    default = pdhg.solve_tgv(problem, TGV(), options)
    gradient = operators.grad(begin.canvas)
    assert default.history.primal[0] == model.tgv_objective(problem, TGV(), dataclasses.replace(begin, w=gradient))
    with pytest.raises(ValueError, match="shape"):
        pdhg.solve_tgv(problem, TGV(), options, first=pdhg.Initial(w=np.zeros(problem.shape)))
    with pytest.raises(ValueError, match="TV has no field w"):
        pdhg.solve_tv(problem, TV(), options, first=pdhg.Initial(w=zero))


def test_the_partial_gap_is_recorded_for_tgv() -> None:
    problem, _ = synthetic.problem(seed=15)
    options = pdhg.Options(iterations=20, tolerance=0.0, record_every=10, partial_radius=100.0)
    result = pdhg.solve_tgv(problem, TGV(), options)
    assert result.history.partial_gap.shape == (3,)
    assert np.all(np.isfinite(result.history.partial_gap))
    no_radius = pdhg.solve_tv(problem, TV(), pdhg.Options(iterations=20, tolerance=0.0, record_every=10))
    assert np.all(np.isnan(no_radius.history.partial_gap))


def test_tv_reaches_below_the_subgradient_method() -> None:
    # Both minimize the same objective over the same set, so the primal-dual method's dual
    # value is a lower bound of what the subgradient method can reach (by 11.6, far above
    # rounding). And in as many iterations its primal value goes below the subgradient
    # method's (measured: 3167.97 against 3169.95).
    problem, _ = synthetic.problem(seed=16)
    weights = TV(1.0)
    options = pdhg.Options(iterations=300, tolerance=0.0, step_ratio=10.0, record_every=300)
    primal_dual = pdhg.solve_tv(problem, weights, options)
    stepped = subgradient.solve_tv(problem, weights, subgradient.Options(iterations=300, record_every=300))
    assert primal_dual.history.dual[-1] < stepped.history.primal[-1]
    assert primal_dual.history.primal[-1] < stepped.history.primal[-1]
    assert stepped.history.dual[-1] == -np.inf
    assert isinstance(primal_dual.dual, Dual)
